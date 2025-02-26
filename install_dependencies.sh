#!/bin/bash
# Comprehensive dependency installation for PDF to JSONL converter

# Update package lists
echo "Updating package lists..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
elif command -v yum &> /dev/null; then
    sudo yum update -y
elif command -v brew &> /dev/null; then
    brew update
fi

# Install system dependencies for Tesseract OCR
echo "Installing system dependencies for OCR..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y tesseract-ocr libtesseract-dev
elif command -v yum &> /dev/null; then
    sudo yum install -y tesseract tesseract-devel
elif command -v brew &> /dev/null; then
    brew install tesseract
else
    echo "WARNING: Could not install Tesseract OCR using package manager. Please install manually."
    echo "Windows users: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Create and activate virtual environment (optional)
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip

# Install PyMuPDF with multiple methods to ensure it works
echo "Installing PyMuPDF (provides the 'fitz' module)..."
pip install pymupdf
pip install PyMuPDF  # Alternative capitalization
python -c "import fitz; print(f'PyMuPDF version: {fitz.version}')" || pip install --force-reinstall pymupdf

# Install other dependencies
pip install pytesseract Pillow numpy
pip install sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu  # Use CUDA version if GPU available
pip install transformers

# Final verification of fitz module
echo "Verifying PyMuPDF installation..."
if python -c "import fitz; print(f'PyMuPDF successfully imported. Version: {fitz.version}')" ; then
    echo "✅ PyMuPDF installed correctly!"
else
    echo "❌ PyMuPDF installation issue detected."
    echo "Run the fix_pymupdf.py script to resolve the issue:"
    echo "python fix_pymupdf.py"
fi

echo "Installation complete!" 
