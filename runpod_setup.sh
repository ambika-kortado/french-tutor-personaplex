#!/bin/bash
# RunPod Setup Script for French Tutor with PersonaPlex
# Run this after starting a RunPod instance with GPU

set -e

echo "=============================================="
echo "Setting up French Tutor with PersonaPlex"
echo "=============================================="

# Update system
apt-get update && apt-get install -y git ffmpeg libsndfile1

# Install Python dependencies
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn websockets openai python-dotenv soundfile numpy

# Clone Moshi/PersonaPlex
echo "Installing Moshi..."
pip install moshi

# Clone the app
echo "Cloning app..."
cd /workspace
git clone https://github.com/anthropics/french-tutor-personaplex.git app 2>/dev/null || true

# Or copy files if local
echo "App ready!"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your API key: export OPENAI_API_KEY='sk-...'"
echo "2. Run: cd /workspace/app && python dual_track_tutor.py"
echo "3. Access via RunPod's exposed port 7860"
echo "=============================================="
