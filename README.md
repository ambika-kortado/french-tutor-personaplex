# French Tutor - PersonaPlex Demo

Interactive French learning demo using NVIDIA PersonaPlex for full-duplex voice conversation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RunPod GPU Instance                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PersonaPlex (Voice Layer)                │   │
│  │  - Speech-to-Speech (7B parameters)                   │   │
│  │  - Full-duplex conversation                           │   │
│  │  - Real-time voice synthesis                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Intelligence Layer (GPT-4o)                 │   │
│  │  - French language teaching logic                     │   │
│  │  - Grammar correction                                 │   │
│  │  - Vocabulary suggestions                             │   │
│  │  - Conversation steering                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              French Tutor Persona                     │   │
│  │  - Patient, encouraging teacher                       │   │
│  │  - Adapts to learner level                           │   │
│  │  - Provides cultural context                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- RunPod instance with A100 80GB (recommended) or A100 40GB
- HuggingFace token (for PersonaPlex model access)
- OpenAI API key (for intelligence layer)

## Quick Start on RunPod

1. Create a RunPod instance with the NVIDIA PyTorch template
2. SSH into the instance
3. Clone this repo and run setup:

```bash
git clone <this-repo>
cd french-tutor-personaplex
chmod +x setup.sh
./setup.sh
```

4. Set your API keys:
```bash
export OPENAI_API_KEY="your-key"
export HF_TOKEN="your-huggingface-token"
```

5. Start the server:
```bash
python server.py
```

6. Access via RunPod's exposed port (default: 7860)

## Features

- Real-time French conversation practice
- Instant pronunciation feedback
- Grammar corrections spoken naturally
- Vocabulary building through context
- Cultural tips and explanations
