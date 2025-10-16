# Jupyter-Boilerplate Project

This project provides configurations for using Jupyter and Julia together in VSCode.

## Getting Started

1. Install VSCode, Docker, and the VSCode.

2. Create a new repository from this boilerplate.

3. Clone the new repository.

3. Open VSCode and use the [Dev Copntainers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension to open the project.

Or you can use [GitHub Codespaces](https://marketplace.visualstudio.com/items?itemName=GitHub.codespaces).

## Update the Project

1. Update `JULIA_VERSION` argument in `.devcontainer/Dockerfile`.
    You can check the available versions on the [releases page](https://github.com/JuliaLang/julia/releases).

2. Update Python versions in `.devcontainer/Dockerfile`.

3. Update the Julia package versions in `.devcontainer/Project.toml`.

4. Update the VSCode extension versions in `.devcontainer/devcontainer.json`.
