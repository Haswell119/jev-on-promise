# syntax=docker/dockerfile:1
#
# Sextant decision engine: non-neural, deterministic, no network needed at inference.
#
#   docker build -t sextant:dev .
#   docker run --rm -p 8080:8080 sextant:dev                                   # HTTP API
#   docker run --rm --network none -i sextant:dev decide < examples/request_basic.json
#   docker run --rm --network none -i sextant:dev validate < examples/request_basic.json
#
# The model artifact (model/*.json) and the lexicon assets are compiled into the
# binary, so the runtime image contains nothing but the executable, curl for the
# HEALTHCHECK, and an unprivileged user. Offline builds: run `cargo vendor` and
# ship the `.cargo/config.toml` it prints inside the build context; `COPY . .`
# picks both up and the builder stage then needs no network at all.

# ---------------------------------------------------------------- builder ----
FROM rust:1.94-bookworm AS builder

WORKDIR /src

# rust-toolchain.toml (channel = "stable") is excluded via .dockerignore so the
# build uses the toolchain that ships with this image instead of downloading
# whatever "stable" resolves to on build day.
COPY . .

RUN cargo build --release --locked --bin sextant \
    && strip target/release/sextant \
    && ./target/release/sextant --version

# ---------------------------------------------------------------- runtime ----
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="sextant" \
      org.opencontainers.image.description="Sextant: non-neural, deterministic, calibrated decision engine (System One compatible API)" \
      org.opencontainers.image.source="https://github.com/haswell119/jev-on-promise" \
      org.opencontainers.image.licenses="Apache-2.0"

# curl is only used by the HEALTHCHECK; ca-certificates keeps curl usable for
# operators who exec into the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 sextant \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/sextant \
               --shell /usr/sbin/nologin sextant

COPY --from=builder /src/target/release/sextant /usr/local/bin/sextant

USER sextant:sextant
WORKDIR /home/sextant

# `serve` reads these; override with -e or with explicit flags after `serve`.
ENV SEXTANT_ADDR=0.0.0.0:8080 \
    SEXTANT_THREADS=0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/sextant"]
CMD ["serve"]
