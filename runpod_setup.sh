#!/bin/bash
# RunPod Setup Script for French Tutor with PersonaPlex
# Run this after starting a RunPod instance with GPU (24GB+ VRAM recommended)

set -e

echo "=============================================="
echo "Setting up French Tutor with PersonaPlex"
echo "=============================================="

# Update system
apt-get update && apt-get install -y git ffmpeg libsndfile1

# Install Python dependencies
pip install --upgrade pip

# Install Moshi from official repo
echo "Installing Moshi from kyutai-labs..."
pip install moshi

# Install other dependencies
pip install fastapi uvicorn websockets openai python-dotenv soundfile numpy

# Clone the app
echo "Cloning app..."
cd /workspace
git clone https://github.com/ambika-kortado/french-tutor-personaplex.git app
cd app

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your API key:"
echo "   export OPENAI_API_KEY='your-key-here'"
echo ""
echo "2. Run the server:"
echo "   python dual_track_tutor.py --port 7860"
echo ""
echo "3. Access via RunPod's proxy URL (Connect tab)"
echo "=============================================="
