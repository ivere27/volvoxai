FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    git \
    python3 \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Install Wasm32 target for clang (if we were using clang natively, but here we just need basic build tools)
RUN apt-get update && apt-get install -y clang lld && rm -rf /var/lib/apt/lists/*

# Install Rust and Naga
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    cargo install naga-cli

WORKDIR /workspace

ARG VOLVOXAI_VERSION=unknown
ARG VOLVOXAI_GIT_COMMIT=unknown
ARG VOLVOXAI_BUILD_DATE=unknown

LABEL org.opencontainers.image.title="VolvoxAI build environment" \
      org.opencontainers.image.version="${VOLVOXAI_VERSION}" \
      org.opencontainers.image.revision="${VOLVOXAI_GIT_COMMIT}" \
      org.opencontainers.image.created="${VOLVOXAI_BUILD_DATE}"

CMD ["/bin/bash"]
