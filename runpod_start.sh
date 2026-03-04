#!/bin/bash
# RunPod startup script for French Tutor

echo "=== French Tutor - RunPod Startup ==="
echo ""

# Navigate to workspace
cd /workspace/french-tutor-personaplex 2>/dev/null || {
    echo "Project not found in /workspace"
    echo "Clone it first: git clone <repo> /workspace/french-tutor-personaplex"
    exit 1
}

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Set up environment
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Verify API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Set it in RunPod's environment variables or create a .env file"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not set - PersonaPlex download may fail"
fi

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

# Start the server
echo ""
echo "Starting French Tutor server on port 7860..."
echo "Access via your RunPod URL"
echo ""

python server.py
