#!/bin/bash
# Setup script for French Tutor with PersonaPlex on RunPod

set -e

echo "=== French Tutor PersonaPlex Setup ==="

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You'll need it to download PersonaPlex."
    echo "Set it with: export HF_TOKEN='your-token'"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. You'll need it for the intelligence layer."
    echo "Set it with: export OPENAI_API_KEY='your-key'"
fi

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    git-lfs \
    2>/dev/null || true

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Clone PersonaPlex if not exists
if [ ! -d "personaplex" ]; then
    echo "Cloning PersonaPlex..."
    git lfs install
    if [ -n "$HF_TOKEN" ]; then
        git clone https://USER:${HF_TOKEN}@huggingface.co/nvidia/personaplex-7b-v1 personaplex || {
            echo "Failed to clone PersonaPlex. Make sure HF_TOKEN is set correctly."
            echo "You may need to accept the model license at https://huggingface.co/nvidia/personaplex-7b-v1"
        }
    else
        echo "ERROR: HF_TOKEN not set. Cannot clone PersonaPlex."
        exit 1
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Set your API keys if not already set:"
echo "   export OPENAI_API_KEY='your-key'"
echo "   export HF_TOKEN='your-token'"
echo ""
echo "2. Start the server:"
echo "   python server.py"
echo ""
echo "3. Access via RunPod's exposed port 7860"
