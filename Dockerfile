# Python 3.12.15 / Debian Bookworm, pinned multi-platform image.
FROM python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Cache dependency installation separately from project files.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 1000 researcher \
    && chown researcher:researcher /app

COPY --chown=researcher:researcher . /app

USER researcher
EXPOSE 8888

# Jupyter retains its default token authentication.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/app"]
