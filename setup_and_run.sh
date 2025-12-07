#!/bin/bash

# FaceFusion Setup and Launch Script for M1 Mac
# This script creates a virtual environment, installs dependencies, and launches FaceFusion

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FaceFusion Setup and Launch Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python version and find best available
echo -e "${YELLOW}Checking Python version...${NC}"

# Check if pyenv local version is set (highest priority)
PYTHON_CMD=""
if [ -f ".python-version" ]; then
    PYENV_VERSION=$(cat .python-version 2>/dev/null | tr -d ' \n')
    if [ -n "$PYENV_VERSION" ]; then
        # Try python first (pyenv sets this up)
        if command -v python &> /dev/null; then
            PYTHON_CMD="python"
            PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
            echo -e "${GREEN}Using pyenv Python ${PYTHON_VERSION}${NC}"
        elif command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
            PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
            echo -e "${GREEN}Using pyenv Python ${PYTHON_VERSION}${NC}"
        fi
    fi
fi

# Try to find Python 3.11 or 3.12 (recommended for FaceFusion)
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        echo -e "${GREEN}Found Python 3.12 (recommended)${NC}"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo -e "${GREEN}Found Python 3.11 (recommended)${NC}"
    fi
fi

# Fallback to python3 or python
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Error: python3 or python is not installed${NC}"
        exit 1
    fi
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required. Found ${PYTHON_VERSION}${NC}"
    exit 1
fi

# Warn if using Python 3.14+ (may have compatibility issues)
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 14 ]; then
    echo -e "${YELLOW}Warning: Python ${PYTHON_VERSION} may have compatibility issues with some packages${NC}"
    echo -e "${YELLOW}Consider using Python 3.11 or 3.12 for better compatibility${NC}"
    if command -v pyenv &> /dev/null; then
        echo -e "${YELLOW}You can install Python 3.12 with: pyenv install 3.12.0${NC}"
    fi
    echo ""
fi

# Check for required system dependencies
echo -e "${YELLOW}Checking system dependencies...${NC}"

if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed${NC}"
    echo -e "${YELLOW}Install with: brew install ffmpeg${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ffmpeg found${NC}"

if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ curl found${NC}"

# Virtual environment setup
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python -m pip install --upgrade pip --quiet

# Check if requirements are already installed
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check if key packages are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    # numpy 2.3.4 has compatibility issues, use 1.x instead
    # Install packages individually with compatible versions
    python -m pip install gradio-rangeslider==0.0.8 --quiet
    python -m pip install gradio==5.44.1 --quiet
    python -m pip install "numpy>=1.24,<2.0" --quiet || python -m pip install numpy==1.26.4 --quiet
    python -m pip install onnx==1.19.1 --quiet
    python -m pip install opencv-python==4.12.0.88 --quiet
    python -m pip install psutil==7.1.2 --quiet
    python -m pip install tqdm==4.67.1 --quiet
    python -m pip install "scipy>=1.10,<2.0" --quiet || python -m pip install scipy==1.11.4 --quiet
    
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Install ONNX Runtime (default CPU version for M1 Mac)
echo -e "${YELLOW}Installing ONNX Runtime...${NC}"
if ! python -c "import onnxruntime" 2>/dev/null; then
    PYTHON_MAJOR=$(python --version | cut -d' ' -f2 | cut -d'.' -f1)
    PYTHON_MINOR=$(python --version | cut -d' ' -f2 | cut -d'.' -f2)
    
    # For Python 3.14+, try latest version without pinning
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 14 ]; then
        echo -e "${YELLOW}Python 3.14+ detected, installing latest compatible onnxruntime...${NC}"
        if python -m pip install onnxruntime --quiet 2>/dev/null; then
            echo -e "${GREEN}✓ ONNX Runtime installed (latest version)${NC}"
        else
            echo -e "${RED}Error: Could not install onnxruntime for Python 3.14${NC}"
            echo -e "${YELLOW}Please use Python 3.11 or 3.12 for better compatibility${NC}"
            echo -e "${YELLOW}Install with pyenv: pyenv install 3.12.0 && pyenv local 3.12.0${NC}"
            exit 1
        fi
    else
        # For Python 3.10-3.13, use pinned version
        python -m pip install onnxruntime==1.23.2 --quiet
        echo -e "${GREEN}✓ ONNX Runtime installed${NC}"
    fi
else
    echo -e "${GREEN}✓ ONNX Runtime already installed${NC}"
fi

# Check if models need to be downloaded
echo ""
echo -e "${YELLOW}Checking for models...${NC}"
echo -e "${BLUE}Note: Models will be downloaded automatically on first run if needed${NC}"
echo ""

# Launch FaceFusion
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Launching FaceFusion...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}The UI will open in your browser automatically.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop FaceFusion${NC}"
echo ""

# Run FaceFusion
python facefusion.py run

