# syntax=docker/dockerfile:1.9@sha256:fe40cf4e92cd0c467be2cfc30657a680ae2398318afd50b0c80585784c604f28

# This file is lightly modified from the original at https://hynek.me/articles/docker-uv/

FROM ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 AS uv
FROM rclone/rclone:1.75.1@sha256:45401ad7410db1d67ffdb58e19059ad20b0d8e0285a60e38bbec55cc1019c7a5 AS rclone

### Use a separate build container for better reuse.
FROM ubuntu:noble@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3 AS build

SHELL ["sh", "-exc"]

### Start Build Prep.
# - Deadsnakes PPA to get newer python versions
# - build-essential/gcc packages are needed to compile numcodecs on ARM64.
RUN <<EOT
apt-get update -qy && \
apt-get install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt-get update -qy && \
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    build-essential \
    ca-certificates \
    python3-setuptools \
    python3.14-dev \
    build-essential \
    gcc \
EOT

COPY --from=uv /uv /usr/local/bin/uv

# - Silence uv complaining about not being able to use hard links,
# - tell uv to byte-compile packages for faster application startups,
# - prevent uv from accidentally downloading isolated Python builds,
# - pick a Python,
# - and finally declare `/app` as the target for `uv sync`.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.14 \
    UV_PROJECT_ENVIRONMENT=/app 


# Needed to be able to compile numcodecs on ARM64
ENV DISABLE_NUMCODECS_SSE2=1 \
    DISABLE_NUMCODECS_AVX2=1 \
    CFLAGS="-Wno-implicit-function-declaration" 

### End Build Prep

# Since there's no point in shipping lock files, we move them
# into a directory that is NOT copied into the runtime image.
# The trailing slash makes COPY create `/_lock/` automagically.
COPY pyproject.toml /_lock/
COPY uv.lock /_lock/

# Synchronize DEPENDENCIES without the application itself.
# This layer is cached until uv.lock or pyproject.toml change.
# You can create `/app` using `uv venv` in a separate `RUN`
# step to have it cached, but with uv it's so fast, it's not worth
# it, so we let `uv sync` create it for us automagically.
RUN --mount=type=cache,target=/root/.cache <<EOT
cd /_lock
uv sync \
    --locked \
    --no-dev \
    --no-install-project
EOT

# Now install the APPLICATION from `/src` without any dependencies.
# `/src` will NOT be copied into the runtime container.
# As of uv 0.4.11, you can also use
# `cd /src && uv sync --locked --no-dev --no-editable` instead.
COPY . /src
RUN --mount=type=cache,target=/root/.cache \
    uv pip install \
        --python=$UV_PROJECT_ENVIRONMENT \
        --no-deps \
        /src

##########################################################################

FROM ubuntu:noble@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3
SHELL ["sh", "-exc"]

# Optional: add the application virtualenv to search path.
ENV PATH=/app/bin:$PATH

# Don't run app as root.
RUN <<EOT
groupadd -r -g 999 app
useradd -r -u 999 -d /app -g 999 -N app
EOT

ENTRYPOINT ["python", "src/reformatters/__main__.py"]
# See <https://hynek.me/articles/docker-signals/>.
STOPSIGNAL SIGINT

# Install ca-certificates before copying over deadsnakes PPA sources, otherwise the
# apt update from the deadsnakes PPA can't verify HTTPS.
RUN <<EOT
apt-get update -qy && \
apt-get install -qyy \
    -o APT::Install-Suggests=false \
    ca-certificates
EOT

# Copy the deadsnakes PPA sources file including embedded GPG key
COPY --from=build /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-noble.sources /etc/apt/sources.list.d/

# Copy rclone from official image (Noble's version is too old for `rclone copyurl --urls` flag)
COPY --from=rclone /usr/local/bin/rclone /usr/bin/rclone

# Note how the runtime dependencies differ from build-time ones.
# Notably, there is no uv either!
RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Suggests=false \
    python3.14 \
    libpython3.14 \

apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOT

# Copy the pre-built `/app` directory to the runtime container
# and change the ownership to user app and group app in one step.
COPY --from=build --chown=app:app /app /app

COPY . /app/

USER app:app
WORKDIR /app

# Light test that the app can be run
RUN <<EOT
python -V
python src/reformatters/__main__.py --help
EOT
