FROM ubuntu:24.04

USER root

ARG JULIA_VERSION="1.11.2"
ARG PYTHON_VERSION="3.11"

# Number of threads used by Julia
# See: https://docs.julialang.org/en/v1/manual/multi-threading/
ENV JULIA_NUM_THREADS="4"
# Skip interactive prompts in apt commands
ENV DEBIAN_FRONTEND="noninteractive"

RUN apt update && \
    apt install -y \
        curl \
        fonts-ipafont \
        git \
        wget

# Install Oh My Zsh
# See: https://ohmyz.sh
ARG OH_MY_ZSH_INSTALLATION_SCRIPT_URL="https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh"
RUN apt update && \
    apt install -y zsh && \
    sh -c "$(wget ${OH_MY_ZSH_INSTALLATION_SCRIPT_URL} -O -)"

# Install Git LFS
# See: https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md
ARG GIT_LFS_INSTALLATION_SCRIPT_URL="https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh"
RUN apt update && \
    curl -s ${GIT_LFS_INSTALLATION_SCRIPT_URL} | bash && \
    apt install -y git-lfs

# Install Julia
# See: https://docs.julialang.org/en/v1/manual/installation/
ARG JULIA_INSTALLATION_SCRIPT_URL="https://install.julialang.org"
ARG JULIAUP_DIR="/root/.juliaup"
ARG JULIAUP="${JULIAUP_DIR}/bin/juliaup"
RUN curl -fsSL ${JULIA_INSTALLATION_SCRIPT_URL} | \
    sh -s -- --yes --default-channel=release -p=${JULIAUP_DIR} && \
    chmod +x ${JULIAUP} && \
    ${JULIAUP} add ${JULIA_VERSION} && \
    ${JULIAUP} default ${JULIA_VERSION}

# Run init.jl
ARG TEMP_DIR="/tmp"
ARG INIT_SCRIPT_PATH="${TEMP_DIR}/init.jl"
ARG PROJECT_FILE_PATH="${TEMP_DIR}/Project.toml"
ARG JULIA="${JULIAUP_DIR}/bin/julia"
COPY ["Project.toml", "${PROJECT_FILE_PATH}"]
COPY ["init.jl", "${INIT_SCRIPT_PATH}"]
RUN ${JULIA} ${INIT_SCRIPT_PATH} ${TEMP_DIR} ${PYTHON_VERSION} && \
    rm ${INIT_SCRIPT_PATH} && \
    rm ${PROJECT_FILE_PATH}

# Clean up apt
RUN apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/inulab_julia_devcontainer