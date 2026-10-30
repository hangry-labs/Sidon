# SPDX-License-Identifier: MIT
# Adapted from faster-qwen3-tts-studio; see licenses/faster-qwen-upstream-MIT.txt.
"""Promote a qualified snapshot, then atomically push its tag and next snapshot."""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

FILES = ('VERSION', 'pyproject.toml', 'src/sidon/__init__.py', 'uv.lock',
         'README.md', 'docs/dockerhub.md')
VALIDATION = [('task', 'validate')]
ORIGIN = 'hangry-labs/Sidon'
VERSION = r'(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)'


def replace_once(pattern, replacement, text):
    result, count = re.subn(pattern, replacement, text, flags=re.M)
    if count != 1:
        raise ValueError('Expected exactly one version field: ' + pattern)
    return result


def set_versions(original, display):
    result = dict(original)
    stable = display.removesuffix('-snapshot')
    package = stable + ('.dev0' if display.endswith('-snapshot') else '')
    result['VERSION'] = display + ('\r\n' if original['VERSION'].endswith('\r\n') else '\n')
    result['pyproject.toml'] = replace_once(r'^version = "[^"\r\n]+"(?=\r?$)',
                                          f'version = "{package}"', original['pyproject.toml'])
    result[FILES[2]] = replace_once(r'^__version__ = "[^"\r\n]+"(?=\r?$)',
                                  f'__version__ = "{display}"', original[FILES[2]])
    result['uv.lock'] = replace_once(r'(^name = "sidon"\r?\nversion = ")[^"\r\n]+(")',
                                     lambda match: match[1] + package + match[2], original['uv.lock'])
    return result


def banner(document, text):
    return replace_once(r'^(?:Development version|Source release) \*\*v[^\r\n]+(?=\r?$)', text, document)


def prepare(original, next_version=''):
    """Validate and transform BOTH stages without commands, writes or ref changes."""
    snapshot = original['VERSION'].strip()
    if not re.fullmatch(VERSION + '-snapshot', snapshot):
        raise ValueError('VERSION must use X.Y.Z-snapshot.')
    stable = snapshot.removesuffix('-snapshot')
    if tomllib.loads(original['pyproject.toml'])['project']['version'] != stable + '.dev0':
        raise ValueError('Package version does not match VERSION.')
    if not re.search(r'^__version__ = "' + re.escape(snapshot) + r'"\r?$', original[FILES[2]], re.M):
        raise ValueError('Runtime version does not match VERSION.')
    packages = [p for p in tomllib.loads(original['uv.lock'])['package'] if p['name'] == 'sidon']
    if len(packages) != 1 or packages[0]['version'] != stable + '.dev0' or packages[0].get('source') != {'editable': '.'}:
        raise ValueError('uv.lock project version/source does not match VERSION.')
    pieces = stable.split('.')
    next_snapshot = next_version or f'{pieces[0]}.{pieces[1]}.{int(pieces[2]) + 1}-snapshot'
    if not re.fullmatch(VERSION + '-snapshot', next_snapshot):
        raise ValueError('NEXT_VERSION must use X.Y.Z-snapshot.')
    next_stable = next_snapshot.removesuffix('-snapshot')
    if tuple(map(int, next_stable.split('.'))) <= tuple(map(int, stable.split('.'))):
        raise ValueError('The next snapshot must be newer than the release.')

    readme = original['README.md']
    heading = re.search(r'^### v' + re.escape(snapshot) + r'(\r?\n)', readme, re.M)
    if not heading:
        raise ValueError('README Version History is missing the snapshot heading.')
    openings = list(re.finditer(r'^<details[^>]*>', readme[:heading.start()], re.M))
    end = readme.find('</details>', heading.end())
    if not openings or end < 0:
        raise ValueError('Snapshot history must live in a details section.')
    start, opening = openings[-1].start(), openings[-1].group()
    end += len('</details>')
    section = readme[start:end]
    bullet = re.search(r'^- ', section, re.M)
    if not bullet:
        raise ValueError('Snapshot history needs release-note bullets.')
    newline = heading[1]
    stable_opening = re.sub(r'\sopen(?=[\s>])', '', opening)
    release_section = newline.join([
        stable_opening, f'<summary><strong>v{stable} release notes</strong></summary>', '',
        f'### v{stable}', '',
        f'Source release `v{stable}`, Python package `{stable}`. The source tag triggers',
        'container publication. Verify full/tiny/source registry indexes before announcing',
        'availability, then record the top-level OCI digests here. No digest is assumed.', '',
        '**Full image (after successful publication)**', '', '```bash',
        f'docker run -d --name sidon --restart unless-stopped --gpus all -p 127.0.0.1:7868:7868 -v sidon_data:/app/persistent hangrylabs/sidon:v{stable}',
        '```', '', '**Tiny image (after successful publication)**', '', '```bash',
        f'docker run -d --name sidon --restart unless-stopped --gpus all -p 127.0.0.1:7868:7868 -v sidon_data:/app/persistent hangrylabs/sidon:v{stable}_tiny',
        '```', '', f'Matching sources: `v{stable}_sources`. For GHCR, replace `hangrylabs/sidon`',
        'with `ghcr.io/hangry-labs/sidon`. Keep the same persistent volume on upgrades.', '',
    ]) + newline + section[bullet.start():]
    release = set_versions(original, stable)
    release['README.md'] = banner(readme[:start] + release_section + readme[end:], f'Source release **v{stable}**.')
    release['docs/dockerhub.md'] = banner(original['docs/dockerhub.md'], f'Source release **v{stable}**.')
    release['docs/dockerhub.md'] = replace_once(
        r'^\*\*Publication status:\*\*[^\r\n]+',
        f'**Publication status:** source release `v{stable}` prepares the versioned full/tiny/source tags. Availability requires successful CI and verified registry indexes.',
        release['docs/dockerhub.md'])
    following = set_versions(release, next_snapshot)
    following['README.md'] = banner(release['README.md'], f'Development version **v{next_snapshot}**.')
    new_section = newline.join([
        opening, f'<summary><strong>v{next_snapshot} development notes</strong></summary>', '',
        f'### v{next_snapshot}', '', f'Development after `v{stable}`. Rolling `latest` and `latest_tiny` follow',
        '`main` after successful publication; immutable stable tags remain unchanged.', '',
        '```bash',
        'docker run -d --name sidon --restart unless-stopped --gpus all -p 127.0.0.1:7868:7868 -v sidon_data:/app/persistent hangrylabs/sidon:latest',
        'docker run -d --name sidon --restart unless-stopped --gpus all -p 127.0.0.1:7868:7868 -v sidon_data:/app/persistent hangrylabs/sidon:latest_tiny',
        '```', '', '- No changes recorded yet.', '', '</details>', '', '',
    ])
    history_start = following['README.md'].index(release_section)
    following['README.md'] = following['README.md'][:history_start] + new_section + following['README.md'][history_start:]
    following['docs/dockerhub.md'] = banner(release['docs/dockerhub.md'], f'Development version **v{next_snapshot}**.')
    # Stable publication status remains explicit until registry digests are verified.
    return dict(tag='v' + stable, version=stable, next_snapshot=next_snapshot), release, following


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True, encoding='utf-8').strip()


def check_origin(url):
    if url not in (f'https://github.com/{ORIGIN}', f'https://github.com/{ORIGIN}.git',
                   f'git@github.com:{ORIGIN}.git', f'ssh://git@github.com/{ORIGIN}.git'):
        raise ValueError('origin must be the canonical hangry-labs/Sidon repository.')


def run(root, *, dry_run=False, next_version='', skip_validation=False, show_diff=False):
    original_bytes = {file: (root / file).read_bytes() for file in FILES}
    original = {file: content.decode('utf-8') for file, content in original_bytes.items()}
    plan, release, following = prepare(original, next_version)
    if git(root, 'branch', '--show-current') != 'main':
        raise ValueError('Release promotion must run from main.')
    if git(root, 'tag', '--list', plan['tag']):
        raise ValueError('The release tag already exists. Tags are immutable.')
    check_origin(git(root, 'remote', 'get-url', '--all', 'origin'))
    check_origin(git(root, 'remote', 'get-url', '--push', '--all', 'origin'))
    if git(root, 'ls-files', '--', '.ai', 'AGENTS.md', 'todo'):
        raise ValueError('Private context must not be tracked before release.')
    dirty = git(root, 'status', '--porcelain', '--untracked-files=all')
    if dirty and not dry_run:
        raise ValueError('The working tree must be clean before a real release.')
    push = ['git', 'push', '--atomic', 'origin', 'main', plan['tag']]
    print(json.dumps({**plan, 'dry_run': dry_run, 'working_tree_clean': not bool(dirty),
                      'changed_files': list(FILES), 'validation': [] if skip_validation else [' '.join(c) for c in VALIDATION],
                      'push': ' '.join(push)}, indent=2))
    if show_diff:
        for label, before, after in [('release', original, release), ('next snapshot', release, following)]:
            for file in FILES:
                print(''.join(difflib.unified_diff(before[file].splitlines(True), after[file].splitlines(True),
                    fromfile=file, tofile=label + '/' + file)), end='')
    if dry_run:
        print('Both transformations validated in memory. No files, builds, commits, tags or remote refs changed.')
        return
    subprocess.run(['git', 'fetch', 'origin', 'main', '--tags'], cwd=root, check=True)
    if git(root, 'rev-parse', 'HEAD') != git(root, 'rev-parse', 'refs/remotes/origin/main'):
        raise ValueError('main must match origin/main before release.')
    if git(root, 'tag', '--list', plan['tag']):
        raise ValueError('The release tag exists on the fetched history.')
    committed = False
    try:
        for file in FILES:
            (root / file).write_bytes(release[file].encode('utf-8'))
        if not skip_validation:
            for command in VALIDATION:
                subprocess.run(command, cwd=root, check=True)
        subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
        subprocess.run(['git', 'add', '--', *FILES], cwd=root, check=True)
        subprocess.run(['git', 'commit', '-m', 'release: ' + plan['tag']], cwd=root, check=True)
        committed = True
        subprocess.run(['git', 'tag', '-a', plan['tag'], '-m', 'Release ' + plan['tag']], cwd=root, check=True)
        for file in FILES:
            (root / file).write_bytes(following[file].encode('utf-8'))
        subprocess.run(['git', 'add', '--', *FILES], cwd=root, check=True)
        subprocess.run(['git', 'commit', '-m', 'chore: start ' + plan['next_snapshot']], cwd=root, check=True)
    except BaseException:
        if not committed:
            for file in FILES:
                (root / file).write_bytes(original_bytes[file])
            subprocess.run(['git', 'reset', '--', *FILES], cwd=root, check=True)
        else:
            print('Promotion stopped after the release commit. Inspect local refs before continuing; no refs were pushed.', file=sys.stderr)
        raise
    try:
        subprocess.run(push, cwd=root, check=True)
    except subprocess.CalledProcessError:
        print('Release refs are prepared locally, but push failed. Inspect local and remote refs; '
              'do not run task release again. Retry the prepared refs with:\n  ' + ' '.join(push), file=sys.stderr)
        raise
    print('Release tag and next snapshot pushed. GitHub Actions will publish the images.')
    print('This task does not wait for builds or create/lock a GitHub Release. Verify registry indexes and promote digest-pinned docs afterward.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--next-version', default='')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--show-diff', action='store_true')
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], dry_run=args.dry_run, next_version=args.next_version,
        skip_validation=args.skip_validation, show_diff=args.show_diff)
