# FaceFusion Setup Guide for M1 Mac

## Prerequisites Check ✅

You already have:
- ✅ Python 3.14.0 (meets 3.10+ requirement)
- ✅ FFmpeg 8.0 installed
- ✅ curl installed

## Quick Setup Steps

### 1. Install Python Dependencies

Run the installer script to set up all dependencies:

```bash
python install.py --onnxruntime default --skip-conda
```

**Note:** The `--skip-conda` flag is used because you're not using conda. The `--onnxruntime default` uses the standard CPU version which works great on M1 Macs.

### 2. Verify Installation

Check that everything is set up correctly:

```bash
python facefusion.py --version
```

### 3. Run FaceFusion

**For UI mode (recommended for development):**
```bash
python facefusion.py run
```

This will:
- Start the Gradio web interface
- Open in your browser automatically
- Allow you to test the UI and features

**For CLI mode:**
```bash
python facefusion.py --help
```

## M1 Mac Specific Notes

### CoreML Support
FaceFusion has built-in CoreML support for macOS, which can leverage Apple's Neural Engine on M1/M2 chips for better performance. The code automatically detects and uses CoreML when available.

### Performance
- ONNX Runtime works well on M1 Macs with the default CPU provider
- CoreML can provide additional acceleration for certain operations
- Processing speed will be good but not as fast as dedicated GPU setups

### Troubleshooting

**If you encounter issues:**

1. **Python version conflicts:**
   - Make sure you're using Python 3.10-3.12 (you have 3.14 which should work, but 3.12 is recommended)
   - Consider using `pyenv` or `venv` for isolation

2. **ONNX Runtime issues:**
   - The default CPU version should work fine
   - If you need CoreML acceleration, it should be detected automatically

3. **FFmpeg issues:**
   - Your FFmpeg is installed via Homebrew, which is perfect
   - Make sure it's in your PATH (it is: `/opt/homebrew/bin/ffmpeg`)

4. **Dependency conflicts:**
   - Consider using a virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     python install.py --onnxruntime default --skip-conda
     ```

## Development Workflow

1. **Make your changes** to the codebase
2. **Test locally:**
   ```bash
   python facefusion.py run
   ```
3. **Run tests** (if you want):
   ```bash
   pip install pytest
   pytest
   ```

## Next Steps

Once set up, you can:
- Test the current functionality
- Start implementing the planned improvements
- Use the UI mockup in `PlanningDocs/ui-mockup.html` as a reference
- Develop and test new features locally

## Useful Commands

```bash
# Run UI
python facefusion.py run

# Force download models (first time setup)
python facefusion.py force-download

# Run in headless mode
python facefusion.py headless-run --source-image path/to/source.jpg --target-video path/to/video.mp4 --output-video path/to/output.mp4

# See all options
python facefusion.py run --help
```

