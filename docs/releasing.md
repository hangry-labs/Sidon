# Releasing Sidon

Sidon follows the same snapshot promotion process as Hangry Labs
faster-qwen3-tts-studio. The first stable target is `v1.0.0`; release versions use
three components. Runtime qualification happens before promotion; the release
task runs lightweight source and policy checks, and publishing CI builds and
tests the packaged runtime.

```powershell
task validate
task release DRY_RUN=1
# After reviewing the candidate and committing/pushing preparation changes:
task release
```

`DRY_RUN=1` executes both stable and next-snapshot transformations in memory. It
does not fetch, build, write files or create refs, and works with a dirty checkout.
For a detailed diff, run `uv run --no-project --python 3.13 python scripts/release.py --dry-run --show-diff`.

A real release requires clean `main`, the canonical `hangry-labs/Sidon` fetch and
push remote, synchronization with `origin/main`, and an unused stable Git tag.
It updates `VERSION`, runtime/package metadata, `uv.lock`, README history and the
Docker Hub publication status; commits the release; creates an annotated tag;
then commits the next snapshot and atomically pushes `main` plus the tag.
The default next version after `1.0.0` is `1.0.1-snapshot`. Override it with
`NEXT_VERSION=1.1.0-snapshot`. `SKIP_VALIDATION=1` only skips the lightweight checks
when already completed for the exact candidate. It does not qualify inference.

The task returns after the push. It does not wait for CI, create a GitHub Release,
upload Docker Hub prose, or invent image digests. If validation fails before the
release commit, original files are restored. If it fails after a commit/tag, inspect
local and remote refs before continuing. For a failed final push, retry the exact
`git push --atomic origin main vX.Y.Z` command printed by the task; do not rerun
promotion or replace the tag. Published tags are immutable.

## Qualify the candidate

- Review the complete tracked diff, attribution and release-note bullets. Exclude
  private context, credentials, local jobs, model caches and generated build files.
- Run `task validate`, `task test`, and `task check-runtime-lock`. Build Python
  distributions with `uv build` and inspect their contents and license metadata.
- Build Docker `checks`, `baked`, `tiny`, `sources` and `evidence` targets. Inspect
  the actual packaged application, pinned model hashes, dependency inventories,
  matching source manifest and absence of user audio/private files.
- Check packaged GPU enhancement, separation, split/enhance, format round trips,
  direct/async APIs and queue recovery. Check CPU restoration and tiny-cache
  behavior when those runtime inputs changed. Preserve existing data volumes.
- Keep listening acceptance and shared-host timing observations scoped to the
  tested material. See [validation](validation.md) and [security](security.md).

## Verify publication before announcing

The tag build publishes fixed `vX.Y.Z`, `vX.Y.Z_tiny` and `vX.Y.Z_sources` images
to Docker Hub and GHCR. The next snapshot on `main` publishes moving `latest`,
`latest_tiny` and `latest_sources`. Stable tag builds do not move rolling tags.
Sources publish before binaries, and each binary references a unique source image
through `app.hangrylabs.sources`. See [packaging](docker-packaging.md).

Wait for both workflows and check the publication evidence. Compare full, tiny
and source top-level OCI index digests in both official registries:

```bash
docker buildx imagetools inspect hangrylabs/sidon:v1.0.0
docker buildx imagetools inspect hangrylabs/sidon:v1.0.0_tiny
docker buildx imagetools inspect hangrylabs/sidon:v1.0.0_sources
docker buildx imagetools inspect ghcr.io/hangry-labs/sidon:v1.0.0
docker buildx imagetools inspect ghcr.io/hangry-labs/sidon:v1.0.0_tiny
docker buildx imagetools inspect ghcr.io/hangry-labs/sidon:v1.0.0_sources
```

Use the first `Digest:` value (the OCI index), not a platform child or attestation
digest. Verify public pull access. Add verified `tag@sha256:...` deployment
references to README and Docker Hub documentation while leaving rolling tags
unpinned. This is a later documentation commit; never move the stable tag to
include it. Confirm notices and corresponding sources remain available, then
create the GitHub Release and update Docker Hub prose as separate publication steps.
Prepared first-release prose is in [v1.0.0 notes](releases/v1.0.0.md); replace its
preparation notice only after stable publication has been verified.
