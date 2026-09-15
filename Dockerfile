# syntax=docker/dockerfile:1.7

# The default build is genuinely CPU-only and multi-architecture. The NVIDIA
# Compose overlay replaces both images and the Python requirement profile.
# Keeping CUDA image names out of the defaults is important: BuildKit resolves
# FROM images before it runs a stage, which previously made Apple Docker pull
# several gigabytes of CUDA even when the GPU overlay was not selected.
ARG PYTHON_BUILDER_IMAGE=ubuntu:22.04
ARG PYTHON_RUNTIME_IMAGE=ubuntu:22.04

FROM node:22-bullseye-slim AS frontend-deps
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

FROM frontend-deps AS frontend-builder
ARG MEMORIA_INTERNAL_API_URL=http://127.0.0.1:8000
ARG NEXT_PUBLIC_API_URL=
ENV NEXT_TELEMETRY_DISABLED=1 \
    MEMORIA_INTERNAL_API_URL=${MEMORIA_INTERNAL_API_URL} \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
COPY frontend/ ./
RUN npm run build

FROM ${PYTHON_BUILDER_IMAGE} AS python-builder
ARG MEMORIA_REQUIREMENTS=requirements-cpu.txt
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel
COPY backend/requirements*.txt /tmp/requirements/
COPY backend/dependency_smoke.py /tmp/requirements/dependency_smoke.py
RUN --mount=type=cache,target=/root/.cache/pip \
    test -f "/tmp/requirements/${MEMORIA_REQUIREMENTS}" \
    && /opt/venv/bin/pip install \
        -r "/tmp/requirements/${MEMORIA_REQUIREMENTS}" \
    && /opt/venv/bin/pip install --no-deps \
        -r /tmp/requirements/requirements-managed-no-deps.txt
# Validate both the upstream-metadata override and the CPU/GPU package boundary
# while the image is still building, before a bad multi-gigabyte image can run.
RUN case "${MEMORIA_REQUIREMENTS}" in \
        requirements-cpu.txt) profile=cpu ;; \
        requirements-gpu.txt) profile=gpu ;; \
        *) echo "Unsupported requirements profile: ${MEMORIA_REQUIREMENTS}" >&2; exit 2 ;; \
    esac \
    && /opt/venv/bin/python /tmp/requirements/dependency_smoke.py "$profile"

FROM node:22-bullseye-slim AS node-runtime

FROM ${PYTHON_RUNTIME_IMAGE} AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/var/lib/memoria/home \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXT_TELEMETRY_DISABLED=1 \
    XDG_CACHE_HOME=/var/lib/memoria/cache \
    TORCH_HOME=/var/lib/memoria/cache/torch \
    HF_HOME=/var/lib/memoria/cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libheif1 \
        libjpeg-turbo8 \
        libraw20 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        supervisor \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" memoria \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/memoria --shell /usr/sbin/nologin memoria \
    && install -d -o memoria -g memoria \
        /app /app/backend/data /var/lib/memoria

COPY --from=python-builder /opt/venv /opt/venv
COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node

COPY --chown=memoria:memoria backend/ /app/backend/
COPY --chown=memoria:memoria docker/ /app/docker/
COPY --from=frontend-builder --chown=memoria:memoria \
    /build/frontend/.next/standalone/ /app/frontend/

RUN chmod 0755 /app/docker/entrypoint.sh \
    && install -d -o memoria -g memoria /app/frontend/.next/cache

WORKDIR /app
USER memoria

EXPOSE 8000 5004

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null \
        && curl --fail --silent --show-error http://127.0.0.1:5004/ >/dev/null \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker/entrypoint.sh"]
