"""
French Tutor with NVIDIA PersonaPlex
Full-duplex voice conversation for interactive French learning
"""

import os
import asyncio
import json
from typing import Optional
import gradio as gr
from openai import OpenAI

# PersonaPlex will be loaded dynamically
personaplex_model = None

# OpenAI client for intelligence layer
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# French tutor system prompt
FRENCH_TUTOR_PROMPT = """You are Marie, a warm and patient French tutor from Paris.
You're having a real-time voice conversation with a student learning French.

Your teaching style:
- Speak primarily in French, but explain in English when the student struggles
- Correct pronunciation and grammar gently, without interrupting flow
- Use simple vocabulary first, then introduce new words in context
- Celebrate small wins enthusiastically ("Très bien!", "Excellent!")
- Share cultural tidbits naturally in conversation

Current conversation context:
- This is a voice conversation, keep responses concise (1-3 sentences)
- If the student makes a mistake, correct it naturally by rephrasing
- Adapt difficulty based on the student's responses
- Ask questions to keep the conversation flowing

Remember: You're speaking out loud, so avoid text-only elements like bullet points or lists.
Keep it conversational and encouraging!"""

# Conversation history for context
conversation_history = []


def get_teaching_response(student_speech: str) -> str:
    """
    Use GPT-4o to generate an intelligent teaching response.
    This provides the 'brain' while PersonaPlex provides the 'voice'.
    """
    global conversation_history

    # Add student message
    conversation_history.append({
        "role": "user",
        "content": f"[Student said]: {student_speech}"
    })

    # Keep last 10 exchanges for context
    recent_history = conversation_history[-20:]

    messages = [
        {"role": "system", "content": FRENCH_TUTOR_PROMPT},
        *recent_history
    ]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,  # Keep responses concise for voice
            temperature=0.8,  # Some creativity for natural conversation
        )

        tutor_response = response.choices[0].message.content

        # Add to history
        conversation_history.append({
            "role": "assistant",
            "content": tutor_response
        })

        return tutor_response

    except Exception as e:
        print(f"OpenAI error: {e}")
        return "Pardon, je n'ai pas compris. Pouvez-vous répéter?"


def load_personaplex():
    """Load the PersonaPlex model for voice synthesis"""
    global personaplex_model

    try:
        # Import PersonaPlex - adjust based on actual API
        from personaplex import PersonaPlex

        print("Loading PersonaPlex model...")
        personaplex_model = PersonaPlex.from_pretrained(
            "nvidia/PersonaPlex",
            device_map="auto",
            torch_dtype="auto"
        )
        print("PersonaPlex loaded successfully!")
        return True

    except ImportError:
        print("PersonaPlex not installed. Running in simulation mode.")
        return False
    except Exception as e:
        print(f"Error loading PersonaPlex: {e}")
        return False


def process_voice(audio_input):
    """
    Process incoming voice and generate response.
    In full implementation, PersonaPlex handles both STT and TTS.
    """
    if audio_input is None:
        return None, "Please speak into the microphone."

    try:
        # If PersonaPlex is loaded, use it for full-duplex
        if personaplex_model is not None:
            # PersonaPlex API (conceptual - adjust to actual API)
            transcription = personaplex_model.transcribe(audio_input)
            teaching_response = get_teaching_response(transcription)
            audio_response = personaplex_model.synthesize(
                teaching_response,
                voice="french_female",  # French-accented voice
                emotion="warm"
            )
            return audio_response, f"You: {transcription}\n\nMarie: {teaching_response}"

        else:
            # Simulation mode - just return the text
            # In real usage, you'd integrate with Whisper for STT
            return None, "PersonaPlex not loaded. Using simulation mode.\n\nSpeak in French and I'll help you learn!"

    except Exception as e:
        print(f"Processing error: {e}")
        return None, f"Error: {str(e)}"


def create_demo():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="French Tutor - PersonaPlex",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🇫🇷 French Tutor with PersonaPlex

        Practice conversational French with Marie, your AI tutor!

        **How to use:**
        1. Click the microphone button and speak (French or English)
        2. Marie will respond in French, correcting and teaching as you go
        3. Keep the conversation flowing naturally!

        **Tips:**
        - Start simple: "Bonjour, je m'appelle..."
        - Ask questions: "Comment dit-on 'hello' en français?"
        - Practice scenarios: "Je voudrais commander un café"
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎤 Speak here",
                )

                submit_btn = gr.Button(
                    "Send to Marie",
                    variant="primary",
                    size="lg"
                )

                # Quick phrases for beginners
                gr.Markdown("### Quick Phrases")
                quick_phrases = gr.Examples(
                    examples=[
                        ["Bonjour!"],
                        ["Comment allez-vous?"],
                        ["Je ne comprends pas"],
                        ["Pouvez-vous répéter?"],
                    ],
                    inputs=[],
                    label="Click to hear pronunciation"
                )

            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="🔊 Marie's Response",
                    autoplay=True
                )

                conversation_display = gr.Textbox(
                    label="Conversation",
                    lines=10,
                    interactive=False
                )

                # Reset button
                reset_btn = gr.Button("Start New Conversation", variant="secondary")

        # Status bar
        with gr.Row():
            status = gr.Markdown("*Ready to practice! Click the microphone to begin.*")

        # Wire up the interface
        submit_btn.click(
            fn=process_voice,
            inputs=[audio_input],
            outputs=[audio_output, conversation_display]
        )

        def reset_conversation():
            global conversation_history
            conversation_history = []
            return "Conversation reset. Bonjour! Je suis Marie, votre tutrice de français. Comment puis-je vous aider aujourd'hui?"

        reset_btn.click(
            fn=reset_conversation,
            outputs=[conversation_display]
        )

    return demo


def main():
    """Main entry point"""
    print("=" * 50)
    print("French Tutor with PersonaPlex")
    print("=" * 50)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Intelligence layer will fail.")

    # Try to load PersonaPlex
    personaplex_loaded = load_personaplex()

    if not personaplex_loaded:
        print("\n" + "=" * 50)
        print("Running in SIMULATION MODE")
        print("PersonaPlex not available - voice features limited")
        print("=" * 50 + "\n")

    # Create and launch the demo
    demo = create_demo()

    # Launch with settings for RunPod
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # RunPod default
        share=False,            # Don't create public link
        show_error=True
    )


if __name__ == "__main__":
    main()
