# Dependency security review

Reviewed on 2026-10-30 against the repository's 14 open Dependabot alerts.
Three alerts report the same PyTorch advisory in different manifests. The
remaining alerts came from older development lock entries and pytest. Most of
those transitive packages were already patched in the separate Docker runtime
lock; both dependency views now include the reviewed fixes.

| Package | Patched minimum for reviewed alerts | Locked version |
| --- | --- | --- |
| certifi | 2026.10.18 | 2026.10.23 |
| filelock | 4.2.1 | 4.2.4 |
| fsspec | 2026.6.0 | 2026.9.0 |
| setuptools | 84.1.0 | 84.2.1 |
| idna | 3.15 | 3.20 |
| Pygments | 2.20.0 | 2.21.0 (development) |
| pytest | 9.0.3 | 9.0.3 (development/CI) |
| PyTorch | 2.14.1, verified by compiler regression test | 2.14.1+cu130 |

The pytest pin is updated in the project, packaged test stage and GitHub workflow.
The regression check in `tests/test_packaging.py` checks these reviewed minimums
in both locks and test runners. This records a specific review, not a guarantee
against subsequently disclosed vulnerabilities.

## GPU stack change

The patched PyTorch wheel uses CUDA 13.0, with Triton 3.8.0 and TorchAudio 2.11.1.
GPU deployments need a CUDA 13-compatible host driver; a host CUDA toolkit install
is not required by the Docker image. The model assets, SHA256 verification,
restoration defaults and complete-chunk API behavior are unchanged.

Upstream's [PyTorch advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
concerns JIT compilation of Python code with malformed container type annotations.
Sidon's upload APIs accept audio, and model loading is restricted to pinned,
checksum-verified exports. The dependency is upgraded rather than relying on that
exposure distinction. [TorchAudio 2.11.1](https://github.com/pytorch/audio/releases/tag/v2.11.1)
declares compatibility with PyTorch 2.11 and later.

### Advisory version correction

The advisory lists 2.13.0 as patched. The upstream reproducer still caused a
segmentation fault (exit 139) in the 2.13.0+cu130 candidate. Its source does not
contain the [container-annotation fix](https://github.com/pytorch/pytorch/commit/b90c94991cdf8b87c8f7439f79518e0ef2c4ca4f).
The selected 2.14.1 release contains that fix. A child-process regression test
checks that the malformed annotation is rejected with a Python exception rather
than crashing, so package-version metadata alone cannot pass this check.

The changed wheel's bundled native libraries and build recipes are checked by
the corresponding-source policy. Notices, model grants and runtime source
obligations remain separate; see [licensing](licensing.md).

GitHub alerts are evaluated against the default branch. Local lock updates do not
change their remote state; they can close after the fixes reach that branch and
GitHub rescans it.
