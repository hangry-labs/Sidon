# syntax=docker/dockerfile:1
# Shared image/source structure adapted from faster-qwen3-tts-studio (MIT).
FROM python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS dependencies
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/persistent/huggingface SIDON_DATA_DIR=/app/persistent/studio \
    SIDON_HOST=0.0.0.0 SIDON_PORT=7868
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.runtime.txt ./
RUN python -m pip install --only-binary=:all: --require-hashes -r requirements.runtime.txt

FROM dependencies AS model-assets
COPY scripts/prefetch_models.py ./scripts/
COPY src/sidon/__init__.py ./src/sidon/
COPY src/sidon/inference/__init__.py src/sidon/inference/assets.py ./src/sidon/inference/
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface python -u scripts/prefetch_models.py /app/baked-models

FROM dependencies AS app
COPY pyproject.toml README.md VERSION LICENSE RUNTIME_NOTICES.md THIRD_PARTY_NOTICES.md ./
COPY licenses/ ./licenses/
COPY src/ ./src/
RUN python -m pip install --no-build-isolation --no-deps . && python -m pip check
RUN python -c "from importlib.metadata import distributions; print('\n'.join(sorted({d.metadata['Name'] + '==' + d.version for d in distributions()})))" > /app/runtime-constraints.txt
COPY scripts/collect_runtime_licenses.py ./scripts/
COPY docs/licensing.md ./docs/
RUN python scripts/collect_runtime_licenses.py --output /app/licenses/runtime
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=dev
ARG SOURCE_IMAGE=local-build
LABEL org.opencontainers.image.source="https://github.com/hangry-labs/Sidon" \
    org.opencontainers.image.title="Hangry Labs Sidon" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.version="${IMAGE_VERSION}" \
    app.hangrylabs.sources="${SOURCE_IMAGE}"
EXPOSE 7868
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7868/healthz',timeout=3)"]
CMD ["python", "-m", "sidon.web.app"]

FROM model-assets AS baked
COPY --from=app /app /app
RUN python -m pip install --no-build-isolation --no-deps . && python -m pip check
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=dev
ARG SOURCE_IMAGE=local-build
LABEL org.opencontainers.image.source="https://github.com/hangry-labs/Sidon" \
    org.opencontainers.image.title="Hangry Labs Sidon" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.version="${IMAGE_VERSION}" \
    app.hangrylabs.sources="${SOURCE_IMAGE}"
ENV SIDON_MODEL_DIR=/app/baked-models HF_HUB_OFFLINE=1
EXPOSE 7868
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7868/healthz',timeout=3)"]
CMD ["python", "-m", "sidon.web.app"]

FROM app AS source-assets
COPY scripts/collect_runtime_sources.py scripts/runtime_sources.toml ./scripts/
RUN python -c "from pathlib import Path; import re; files=list(Path('/etc/apt/sources.list.d').glob('*.sources')); assert files; [p.write_text(re.sub(r'^Types: deb$', 'Types: deb deb-src', p.read_text(), flags=re.M)) for p in files]" \
    && apt-get update && python scripts/collect_runtime_sources.py --output /sources
COPY --from=app /app/licenses/runtime /sources/notices
COPY Dockerfile requirements.runtime.in requirements.runtime.txt /sources/build/
COPY scripts/collect_runtime_licenses.py scripts/collect_runtime_sources.py scripts/runtime_sources.toml /sources/build/
COPY LICENSE RUNTIME_NOTICES.md THIRD_PARTY_NOTICES.md docs/licensing.md /sources/
COPY licenses/ /sources/project-notices/

FROM scratch AS sources
COPY --from=source-assets /sources /sources
ARG VCS_REF=unknown
ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/hangry-labs/Sidon" \
    org.opencontainers.image.title="Sidon corresponding sources" \
    org.opencontainers.image.revision="${VCS_REF}" org.opencontainers.image.version="${IMAGE_VERSION}"

FROM scratch AS evidence
COPY --from=app /app/licenses/runtime /runtime
COPY --from=app /app/runtime-constraints.txt /runtime-constraints.txt
COPY --from=model-assets /app/baked-models/manifest.json /models/manifest.json
COPY --from=source-assets /sources/manifest.json /sources/manifest.json
COPY --from=source-assets /sources/SHA256SUMS /sources/SHA256SUMS

FROM app AS checks
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY requirements.runtime.in Dockerfile ./
RUN python -m pip install --only-binary=:all: pytest==8.4.2
RUN --network=none python -m pytest -q -p no:cacheprovider tests

FROM app AS tiny
ENV HF_HUB_OFFLINE=0
