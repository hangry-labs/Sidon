# Docker packaging and publication

The Dockerfile supplies `baked` (all pinned CPU/CUDA weights), `tiny` (download on first use), `sources`, `evidence` and `checks` targets. The default target is tiny; Compose explicitly selects baked.

```bash
docker build --target checks .
docker build --target baked -t sidon:local .
docker build --target tiny -t sidon:tiny .
docker build --target sources -t sidon:sources .
docker buildx build --target evidence --output type=local,dest=build-evidence .
```

Both product targets use the same hashed CPython 3.13 / Linux x86-64 runtime and pinned Python base digest. FFmpeg/ffprobe provide additional audio decoding and compressed exports; their Debian codec dependencies are included in the notice/source inventory. The baked target verifies every downloaded asset against `sidon.inference.assets.ASSETS` and retains `/app/baked-models/manifest.json`. Baked weights are outside the data volume and remain available when a fresh volume is attached.

The optimized CUDA runtime includes GCC and C headers to build Triton's small
launcher locally. `/app/persistent/triton` caches compiled kernels. No external
download is required for compilation, including on a fresh offline volume.
`SIDON_RESTORATION_BACKEND=torchscript` selects the reference restoration path.

`/app/licenses/runtime/inventory.json` records actual installed Python and Debian packages, their notices and bundled native-file hashes. `runtime-constraints.txt` records exact Python package versions. Supplemental notices for wheel omissions live in `THIRD_PARTY_NOTICES.md`.

The source target resolves exact Debian source versions from signed apt indexes and matches reviewed wheel native hashes to `scripts/runtime_sources.toml`. Missing notices, changed native hashes, unreviewed copyleft packages or unavailable matching sources fail the build. Downloaded source archives are size/hash checked. The image includes source packages, patches/build recipes, notices, a manifest and SHA256SUMS under `/sources`.

To extract sources without executing anything in that image:

```bash
docker create --name sidon-source-export sidon:sources /unused
docker cp sidon-source-export:/sources ./corresponding-sources
docker rm sidon-source-export
```

Redistribute the matching sources alongside a binary image. Preserve the immutable `app.hangrylabs.sources` image label on published binaries, publish the source image before binaries, and keep sources available for the required license period. Do not substitute a later moving source tag when an earlier build's sources are needed. Dynamically linked libraries remain replaceable; no application restriction prohibits reverse engineering needed to debug modifications to LGPL components. NVIDIA binaries retain their installed redistribution terms.

## GitHub setup

The workflow publishes only for `hangry-labs/Sidon`. Configure Docker Hub secrets `DOCKER_USERNAME` and `DOCKER_TOKEN` with access to `hangrylabs/sidon`; GitHub's job token handles GHCR. PRs run validation and packaged runtime tests without registry credentials or publication.

Main publishes `latest`, `latest_tiny`, `latest_sources`. Stable `vX.Y` / `vX.Y.Z` Git tags publish their corresponding full, `_tiny` and `_sources` tags. Every run has an immutable `sources-<commit>-<run>-<attempt>` source identity. Existing stable tags are rejected rather than replaced. A manual dispatch can select a stable tag or opt into moving latest tags.

The workflow attaches SBOM/provenance to application images, checks both registries' top-level index digests and retains evidence as a workflow artifact. After the first successful stable publication, add readable tags with their verified top-level OCI index digests to stable documentation examples. No digest is fabricated before publication.

`docs/dockerhub.md` is the prepared Docker Hub long description. It is not uploaded automatically by the image workflow; update it when publication is verified. See [validation](validation.md) for the actual local qualification results.
