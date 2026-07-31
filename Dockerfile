# The architecture-specific image is pinned by manifest digest. Do not replace
# this with `latest` or an unqualified version tag.
ARG VLLM_IMAGE="vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404@sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.title="Inkling-Small Ampere research environment"
LABEL org.opencontainers.image.description="Pinned four-A100 INT8 research environment"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace/inkling-small-ampere

COPY pyproject.toml README.md LICENSE NOTICE Makefile ./
COPY src ./src
RUN python -m pip install --no-deps .

COPY configs ./configs
COPY docs ./docs
COPY manifests ./manifests
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The upstream image starts `vllm serve` directly. This research image also
# contains inspection and conversion tools, so leave command selection explicit.
ENTRYPOINT []
CMD ["/bin/bash"]
