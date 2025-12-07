# FaceFusion Quick Start Guide

## One-Command Setup and Launch

### Option 1: Using Python 3.11 (Recommended)

If you have Python 3.14 and encounter compatibility issues, use Python 3.11:

```bash
# First, set up Python 3.11 (if you have pyenv)
./setup_python311.sh

# Then run the setup
./setup_and_run.sh
```

### Option 2: Direct Setup

Simply run:

```bash
./setup_and_run.sh
```

**Note:** If you're using Python 3.14, the script will try to install the latest compatible onnxruntime. For best compatibility, use Python 3.11 or 3.12.

This script will:
1. ✅ Check Python version (3.10+)
2. ✅ Check system dependencies (ffmpeg, curl)
3. ✅ Create a virtual environment (if it doesn't exist)
4. ✅ Install all Python dependencies
5. ✅ Launch FaceFusion UI

The UI will automatically open in your browser.

## What the Script Does

- **Creates `venv/` directory** - Virtual environment for isolated dependencies
- **Installs requirements** - All packages from `requirements.txt`
- **Installs ONNX Runtime** - CPU version optimized for M1 Mac
- **Launches FaceFusion** - Opens the Gradio web interface

## Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install onnxruntime==1.23.2

# Run FaceFusion
python facefusion.py run
```

## Running After Initial Setup

Once set up, you can simply:

```bash
source venv/bin/activate
python facefusion.py run
```

Or use the script again (it will skip setup if already done).

## Troubleshooting

**If you get onnxruntime installation errors:**
- Python 3.14 may have compatibility issues
- **Solution:** Use Python 3.11 or 3.12 instead
  ```bash
  # If you have pyenv:
  ./setup_python311.sh
  ./setup_and_run.sh
  
  # Or manually:
  pyenv install 3.11.14
  pyenv local 3.11.14
  ./setup_and_run.sh
  ```

**Other issues:**
- Make sure you have Python 3.10+ installed
- Install ffmpeg: `brew install ffmpeg`
- Check that you're in the facefusion directory
- If venv exists with wrong Python version: `rm -rf venv` and run again

**To reset everything:**
```bash
rm -rf venv
./setup_and_run.sh
```

## Development Workflow

1. Make your code changes
2. Run `./setup_and_run.sh` to test
3. The virtual environment keeps dependencies isolated

