# Justfile for Dolphin

# Configuration
set shell := ["bash", "-c"]

# Default recipe: list available recipes
default:
    @just --list

# Standard (GPU-oriented) installation
install:
    #!/usr/bin/env bash
    if [ "{{os()}}" = "linux" ] && [ "${IN_FHS_ENV:-0}" != "1" ]; then
        echo "Linux detected outside FHS. Running installation within FHS container..."
        exec dolphin-fhs-env -c "just install"
    fi
    
    echo "Creating virtual environment if it doesn't exist..."
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    
    echo "Upgrading pip, setuptools, wheel..."
    .venv/bin/pip install --upgrade pip setuptools wheel
    
    if [ "{{os()}}" = "macos" ]; then
        echo "macOS detected. Filtering out Linux-only and uncompiled Apple Silicon packages (triton, deepspeed, decord)..."
        grep -v -E "triton|deepspeed|decord" requirements.txt > .venv/temp-requirements.txt
        echo "Installing requirements..."
        .venv/bin/pip install -r .venv/temp-requirements.txt
        rm .venv/temp-requirements.txt
    else
        echo "Installing GPU requirements..."
        .venv/bin/pip install -r requirements.txt
    fi
    
    echo "GPU/Standard Installation complete!"

# CPU-only installation (lightweight, excludes Triton/DeepSpeed and installs CPU PyTorch on Linux)
install-cpu:
    #!/usr/bin/env bash
    if [ "{{os()}}" = "linux" ] && [ "${IN_FHS_ENV:-0}" != "1" ]; then
        echo "Linux detected outside FHS. Running installation within FHS container..."
        exec dolphin-fhs-env -c "just install-cpu"
    fi
    
    echo "Creating virtual environment if it doesn't exist..."
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    
    echo "Upgrading pip, setuptools, wheel..."
    .venv/bin/pip install --upgrade pip setuptools wheel
    
    echo "Installing CPU-only PyTorch and torchvision..."
    if [ "{{os()}}" = "linux" ]; then
        .venv/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    else
        .venv/bin/pip install torch==2.6.0 torchvision==0.21.0
    fi
    
    echo "Filtering out GPU-only and uncompiled Apple Silicon packages (triton, deepspeed, decord) for CPU installation..."
    grep -v -E "triton|deepspeed|decord|torch==|torchvision==" requirements.txt > .venv/temp-requirements.txt
    
    echo "Installing remaining requirements..."
    .venv/bin/pip install -r .venv/temp-requirements.txt
    rm .venv/temp-requirements.txt
    
    echo "CPU-only Installation complete!"

# Remove all packages (destroying the virtual environment)
clean:
    echo "Removing virtual environment (.venv)..."
    rm -rf .venv
    echo "All packages removed."

# Doctor script to investigate dependency issues
doctor:
    #!/usr/bin/env bash
    if [ "{{os()}}" = "linux" ] && [ "${IN_FHS_ENV:-0}" != "1" ]; then
        echo "Linux detected outside FHS. Running doctor inside FHS container..."
        exec dolphin-fhs-env -c "just doctor"
    fi

    echo "=========================================================="
    echo " DOLPHIN DEPENDENCY DOCTOR"
    echo "=========================================================="
    echo "Date: $(date)"
    echo "OS: $(uname -s) ($(uname -m))"
    echo "Nix-shell status: ${IN_NIX_SHELL:-not in nix-shell}"
    echo "FHS container status: ${IN_FHS_ENV:+Inside FHS container (IN_FHS_ENV=1)}"
    if [ -z "${IN_FHS_ENV:-}" ]; then
        echo "FHS container status: Outside FHS container"
    fi
    echo "----------------------------------------------------------"
    
    # 1. Check Python
    echo -n "Checking Python: "
    if command -v python3 >/dev/null 2>&1; then
        echo "FOUND ($(python3 --version))"
    else
        echo "MISSING"
    fi
    
    # 2. Check Virtualenv
    echo -n "Checking Virtual Environment (.venv): "
    if [ -d .venv ]; then
        echo "PRESENT"
        echo -n "  Checking pip in .venv: "
        if [ -x .venv/bin/pip ]; then
            echo "FOUND"
            echo "  Installed packages:"
            .venv/bin/pip list | sed 's/^/    /'
        else
            echo "MISSING OR NOT EXECUTABLE"
        fi
    else
        echo "MISSING (.venv directory does not exist)"
    fi
    
    # 3. Check System Libraries (Linux only)
    if [ "{{os()}}" = "linux" ]; then
        echo "--------------------------------------------------"
        echo "Checking required system libraries for OpenCV/ML..."
        libs=(
            "libGL.so.1"
            "libgthread-2.0.so.0"
            "libX11.so.6"
            "libXext.so.6"
            "libXrender.so.1"
            "libz.so.1"
            "libstdc++.so.6"
        )
        for lib in "${libs[@]}"; do
            echo -n "  $lib: "
            if python3 -c "import ctypes; ctypes.CDLL('$lib')" >/dev/null 2>&1; then
                echo "OK (loadable via ctypes)"
            else
                echo "FAIL (cannot be loaded)"
            fi
        done
    fi
    
    # 4. Check PyTorch/Cuda and critical imports
    echo "--------------------------------------------------"
    echo "Checking core Python package imports..."
    imports=(
        "torch"
        "torchvision"
        "transformers"
        "cv2"
        "fitz"
    )
    if [ -d .venv ]; then
        for pkg in "${imports[@]}"; do
            echo -n "  $pkg: "
            if .venv/bin/python -c "import $pkg" >/dev/null 2>&1; then
                echo "OK (import successful)"
            else
                echo "FAIL (import failed or package not installed)"
                echo -n "    Error detail: "
                .venv/bin/python -c "import $pkg" 2>&1 | tr '\n' ' '
                echo ""
            fi
        done
        
        # Optional decord check
        echo -n "  decord (optional): "
        if .venv/bin/python -c "import decord" >/dev/null 2>&1; then
            echo "OK"
        else
            echo "NOT INSTALLED (Unused by Dolphin on Apple Silicon)"
        fi
    else
        echo "  Skipped (no virtual environment to test imports in)"
    fi
    echo "=========================================================="
