# Update the image/snapshot together with the WASM runtime-library hashes in
# tools/release_profiles.mjs. Floating Ubuntu packages invalidate that closure.
FROM ubuntu:22.04@sha256:829f6df217bcbae2b371026e81711d1a787c61b2967ad09d015063663ebafbf7

# The minimal base has no CA bundle. Bootstrap it from a checksum-pinned Ubuntu
# package so the first snapshot request can verify TLS without a live apt mirror.
ADD --checksum=sha256:6e8cdcc8c86103acd4fc14649eac62ff2037108389074a7b167567af33c32245 \
    https://snapshot.ubuntu.com/ubuntu/20260911T000000Z/pool/main/c/ca-certificates/ca-certificates_20260601~22.04.1_all.deb \
    /tmp/ca-certificates.deb
RUN dpkg-deb --extract /tmp/ca-certificates.deb /tmp/ca-bootstrap && \
    mkdir -p /etc/ssl/certs && \
    cat /tmp/ca-bootstrap/usr/share/ca-certificates/mozilla/*.crt > /etc/ssl/certs/ca-certificates.crt && \
    rm -rf /tmp/ca-bootstrap /tmp/ca-certificates.deb && \
    sed -i -E 's#http://(archive|security).ubuntu.com/ubuntu/?#https://snapshot.ubuntu.com/ubuntu/20260911T000000Z/#g' /etc/apt/sources.list

RUN apt-get update --error-on=any && apt-get install -y \
    ca-certificates \
    curl \
    build-essential \
    cmake \
    git \
    gnupg \
    python3 \
    python3-pip \
    protobuf-compiler \
    libvulkan-dev \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Install Wasm32 target for clang (if we were using clang natively, but here we just need basic build tools)
RUN apt-get update --error-on=any && apt-get install -y clang lld && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04's Clang 14 accepts -mrelaxed-simd but predates the final
# i32x4.relaxed_dot intrinsic. The release recipe therefore uses LLVM 17 for
# every WASM object and pins its executable, loader, shared-library,
# and resource-header inputs in tools/release_profiles.mjs.
ARG LLVM_RELEASE=1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" \
        > /etc/apt/sources.list.d/llvm-17.list && \
    apt-get update --error-on=any && \
    apt-get install -y clang-17="${LLVM_RELEASE}" lld-17="${LLVM_RELEASE}" && \
    rm -rf /var/lib/apt/lists/*

# Install Rust and Naga
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    cargo install naga-cli --version 30.0.0 --locked

# npm's typecheck regenerates API descriptors with protoc and Python protobuf.
COPY tools/requirements-codegen.txt /opt/volvoxai/requirements-codegen.txt
RUN python3 -m pip install --no-cache-dir -r /opt/volvoxai/requirements-codegen.txt

# Use the upstream, statically linked Binaryen executable. The release recipe
# pins its identity; npm optional dependencies and PATH never select this tool.
ADD --checksum=sha256:195ddc94f9bc89f45abdabb0b9eea86023d727ba90eac8b35b80f2544fc30572 \
    https://github.com/WebAssembly/binaryen/releases/download/version_132/binaryen-version_132-x86_64-linux.tar.gz \
    /tmp/binaryen.tar.gz
RUN mkdir -p /opt/volvoxai/bin && \
    tar -xzf /tmp/binaryen.tar.gz -C /opt/volvoxai/bin --strip-components=2 \
        binaryen-version_132/bin/wasm-opt && \
    echo '1014958e6f20d412f1542320b43970214b0fb1ed780595e8f7c0d8761ed53725  /opt/volvoxai/bin/wasm-opt' \
        | sha256sum --check --strict && \
    /opt/volvoxai/bin/wasm-opt --version && \
    rm /tmp/binaryen.tar.gz

WORKDIR /workspace

ARG VOLVOXAI_VERSION=unknown
ARG VOLVOXAI_GIT_COMMIT=unknown
ARG VOLVOXAI_BUILD_DATE=unknown

LABEL org.opencontainers.image.title="VolvoxAI build environment" \
      org.opencontainers.image.version="${VOLVOXAI_VERSION}" \
      org.opencontainers.image.revision="${VOLVOXAI_GIT_COMMIT}" \
      org.opencontainers.image.created="${VOLVOXAI_BUILD_DATE}"

CMD ["/bin/bash"]
