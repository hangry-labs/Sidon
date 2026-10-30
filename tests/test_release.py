# SPDX-License-Identifier: MIT
# Adapted from faster-qwen3-tts-studio; see licenses/faster-qwen-upstream-MIT.txt.
"""Exercise both release transitions and failure recovery against disposable Git repos."""
from pathlib import Path
import shutil
import subprocess
import tomllib
from unittest.mock import patch

import pytest

from scripts.release import FILES, check_origin, prepare, run

ROOT = Path(__file__).resolve().parents[1]


def documents():
    return {
        'VERSION': '1.0.0-snapshot\n',
        'pyproject.toml': '[project]\nname = "sidon"\nversion = "1.0.0.dev0"\n',
        FILES[2]: '__version__ = "1.0.0-snapshot"\n',
        'uv.lock': 'version = 1\n\n[[package]]\nname = "sidon"\nversion = "1.0.0.dev0"\nsource = { editable = "." }\n',
        'README.md': 'Development version **v1.0.0-snapshot**, targeting the first **v1.0.0** release.\n\n'
            '## Version History\n\n<details open markdown="1">\n<summary><strong>v1.0.0-snapshot development notes</strong></summary>\n\n'
            '### v1.0.0-snapshot\n\nRolling commands.\n\n- Preserve this improvement.\n- Preserve UTF-8: 🎧 · café.\n\n</details>\n\n## Attribution\n\nKeep this.\n',
        'docs/dockerhub.md': 'Development version **v1.0.0-snapshot**, targeting the first **v1.0.0** release.\n\n'
            '**Publication status:** Rolling images only.\n\n## API\n',
    }


def write_documents(root, docs=None):
    for name, content in (docs or documents()).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode('utf-8'))


def test_checkout_versions_work_for_both_snapshot_and_stable():
    actual = {name: (ROOT / name).read_text(encoding='utf-8') for name in FILES}
    version = actual['VERSION'].strip()
    package = version.removesuffix('-snapshot') + ('.dev0' if version.endswith('-snapshot') else '')
    assert tomllib.loads(actual['pyproject.toml'])['project']['version'] == package
    project = [p for p in tomllib.loads(actual['uv.lock'])['package'] if p['name'] == 'sidon']
    assert len(project) == 1 and project[0]['version'] == package
    assert f'__version__ = "{version}"' in actual[FILES[2]]
    assert f'### v{version}\n' in actual['README.md']
    if version.endswith('-snapshot'):
        prepare(actual)


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_promotion_preserves_notes_unicode_and_jekyll(newline):
    original = {name: text.replace('\n', newline) for name, text in documents().items()}
    plan, release, following = prepare(original)
    assert plan == dict(tag='v1.0.0', version='1.0.0', next_snapshot='1.0.1-snapshot')
    for line in original['README.md'].splitlines():
        if line.startswith('- ') or line == 'Keep this.':
            assert line in release['README.md'] and line in following['README.md']
    assert '<details markdown="1">' in release['README.md']
    assert '<details open markdown="1">' in following['README.md']
    assert following['README.md'].count('<details') == following['README.md'].count('</details>') == 2
    assert following['README.md'].index('### v1.0.1-snapshot') < following['README.md'].index('### v1.0.0' + newline)
    assert 'sidon:v1.0.0_tiny' in release['README.md']
    assert 'sidon:latest' not in release['README.md']
    assert 'sha256:' not in release['README.md']
    assert 'Source release **v1.0.0**.' in release['docs/dockerhub.md']
    assert 'Development version **v1.0.1-snapshot**.' in following['docs/dockerhub.md']
    assert tomllib.loads(release['uv.lock'])['package'][0]['version'] == '1.0.0'
    assert tomllib.loads(following['uv.lock'])['package'][0]['version'] == '1.0.1.dev0'
    assert release['VERSION'] == '1.0.0' + newline
    _, second, _ = prepare(following, '1.1.0-snapshot')
    assert '### v1.0.0' + newline in second['README.md']


def test_mixed_newlines_use_history_local_newline_and_preserve_other_sections():
    original = documents()
    original['README.md'] = original['README.md'].replace('## Version History\n', '## Version History\r\n')
    _, release, following = prepare(original)
    assert '## Version History\r\n' in release['README.md']
    assert '\n### v1.0.1-snapshot\n' in following['README.md']
    assert following['README.md'].endswith('\n## Attribution\n\nKeep this.\n')


@pytest.mark.parametrize('value', ['1.0.0-snapshot', '0.9.9-snapshot', '1.1', '1.1-snapshot', '01.1.0-snapshot', '$(id)'])
def test_ambiguous_old_or_invalid_next_version_rejected(value):
    with pytest.raises(ValueError):
        prepare(documents(), value)


@pytest.mark.parametrize('file,old,new', [
    ('pyproject.toml', '1.0.0.dev0', '2.0.0.dev0'),
    ('uv.lock', '1.0.0.dev0', '2.0.0.dev0'),
    (FILES[2], '1.0.0-snapshot', '2.0.0-snapshot'),
    ('README.md', '### v1.0.0-snapshot', '### Missing'),
    ('docs/dockerhub.md', '**Publication status:**', '**Unknown:**'),
])
def test_drift_fails_before_mutation(file, old, new):
    original = documents()
    original[file] = original[file].replace(old, new)
    with pytest.raises(ValueError):
        prepare(original)


def test_canonical_fetch_and_push_remote_required():
    for value in ('https://github.com/hangry-labs/Sidon', 'https://github.com/hangry-labs/Sidon.git', 'git@github.com:hangry-labs/Sidon.git'):
        check_origin(value)
    for value in ('https://github.com/other/Sidon', 'https://example.invalid/hangry-labs/Sidon', ''):
        with pytest.raises(ValueError):
            check_origin(value)


def test_dry_run_is_pure_even_with_dirty_checkout(tmp_path, capsys):
    write_documents(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in FILES}
    answers = ['main', '', 'https://github.com/hangry-labs/Sidon', 'https://github.com/hangry-labs/Sidon', '', ' M README.md']
    with patch('scripts.release.git', side_effect=answers), patch('scripts.release.subprocess.run') as command:
        run(tmp_path, dry_run=True, show_diff=True)
    assert not command.called
    assert before == {name: (tmp_path / name).read_bytes() for name in FILES}
    assert 'Both transformations validated in memory' in capsys.readouterr().out


@pytest.mark.skipif(shutil.which('git') is None, reason='Runtime image omits Git; host CI exercises real refs')
@pytest.mark.parametrize('failure', ['', 'validation', 'push'])
def test_real_git_transitions_and_failure_recovery(tmp_path, capsys, failure):
    root = tmp_path / 'checkout'
    root.mkdir()
    write_documents(root)

    def git(*args, cwd=root):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True, encoding='utf-8').strip()

    git('init', '--quiet', '--initial-branch=main')
    git('config', 'user.name', 'Release test')
    git('config', 'user.email', 'release@example.invalid')
    git('config', 'core.autocrlf', 'false')
    git('add', '.')
    git('commit', '--quiet', '-m', 'Initial snapshot')
    initial = git('rev-parse', 'HEAD')
    remote = tmp_path / 'remote.git'
    git('clone', '--bare', '--quiet', str(root), str(remote), cwd=tmp_path)
    git('remote', 'add', 'origin', str(remote))
    real_run = subprocess.run
    checks, pushes = [], []

    def release_command(command, *args, **kwargs):
        if command[0] == 'task':
            assert tuple(command) == ('task', 'validate')
            checks.append(command)
            if failure == 'validation':
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        assert command[0] == 'git'
        if command[1] == 'push':
            assert command == ['git', 'push', '--atomic', 'origin', 'main', 'v1.0.0']
            pushes.append(command)
            if failure == 'push':
                raise subprocess.CalledProcessError(1, command)
        return real_run(command, *args, **kwargs)

    with patch('scripts.release.check_origin'), patch('scripts.release.subprocess.run', side_effect=release_command):
        if failure:
            with pytest.raises(subprocess.CalledProcessError):
                run(root)
        else:
            run(root)
    assert len(checks) == 1
    assert git('status', '--porcelain') == ''
    if failure == 'validation':
        assert not pushes and git('rev-parse', 'HEAD') == initial
        assert git('tag', '--list') == ''
        assert git('show', 'HEAD:VERSION') == '1.0.0-snapshot'
        return
    assert len(pushes) == 1
    assert git('cat-file', '-t', 'refs/tags/v1.0.0') == 'tag'
    assert git('show', 'v1.0.0:VERSION') == '1.0.0'
    assert git('show', 'HEAD:VERSION') == '1.0.1-snapshot'
    assert 'version = "1.0.0"' in git('show', 'v1.0.0:uv.lock')
    assert 'version = "1.0.1.dev0"' in git('show', 'HEAD:uv.lock')
    if failure == 'push':
        assert git('rev-parse', 'HEAD', cwd=remote) == initial
        assert 'do not run task release again' in capsys.readouterr().err
        real_run(pushes[0], cwd=root, check=True)
    assert git('rev-parse', 'HEAD', cwd=remote) == git('rev-parse', 'HEAD')
    assert git('rev-parse', 'v1.0.0^{commit}', cwd=remote) == git('rev-parse', 'v1.0.0^{commit}')
