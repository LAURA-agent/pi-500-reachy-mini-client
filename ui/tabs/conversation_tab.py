"""
Conversation Tab - Main chat interface with text/audio input
"""
import gradio as gr
import requests

API_BASE_URL = "http://localhost:8888"


def send_text_message(text, chat_history):
    """Send text message to robot."""
    if not text.strip():
        return chat_history, ""

    # Add user message to history
    chat_history.append({"role": "user", "content": text})

    try:
        # Send to conversation API
        response = requests.post(
            f"{API_BASE_URL}/conversation/start",
            json={"text": text},
            timeout=5
        )

        if response.status_code == 200:
            # TODO: This will eventually get the actual response from MCP/Messages API
            chat_history.append({"role": "assistant", "content": "Processing your message..."})
        else:
            chat_history.append({"role": "assistant", "content": f"Error: {response.text}"})

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Connection error: {str(e)}"})

    return chat_history, ""


def process_audio(audio_file, chat_history):
    """Process audio input from microphone."""
    if audio_file is None:
        return chat_history

    chat_history.append({"role": "user", "content": "[Audio input received]"})
    chat_history.append({"role": "assistant", "content": "Audio transcription not yet implemented"})

    # TODO: Send audio to STT service
    # TODO: Then send transcribed text to conversation API

    return chat_history


def get_robot_state():
    """Get current robot state from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/state/current", timeout=2)
        if response.status_code == 200:
            data = response.json()
            state = data.get("state", "unknown")
            return f"🤖 State: **{state.upper()}**"
        else:
            return "🤖 State: **ERROR**"
    except:
        return "🤖 State: **DISCONNECTED**"


def create_conversation_tab():
    """Create the conversation tab interface."""

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 💬 Conversation")

            # Chat history
            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                height=500,
                avatar_images=None,
                show_copy_button=True
            )

            # Text input
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=4
                )
                send_button = gr.Button("Send", variant="primary", scale=1)

            # Audio input
            gr.Markdown("### 🎤 Voice Input")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Click to record"
            )

        # Sidebar with robot status
        with gr.Column(scale=1):
            gr.Markdown("## 🤖 Robot Status")

            state_display = gr.Markdown(value=get_robot_state())
            refresh_state_btn = gr.Button("🔄 Refresh State", variant="secondary")

            gr.Markdown("### Quick Actions")
            with gr.Column():
                wake_btn = gr.Button("☀️ Wake Up", variant="secondary", size="sm")
                sleep_btn = gr.Button("😴 Sleep Mode", variant="secondary", size="sm")
                pout_btn = gr.Button("😤 Pout Mode", variant="secondary", size="sm")
                stop_btn = gr.Button("⏹️ Stop All", variant="secondary", size="sm")

    # Event handlers
    send_button.click(
        fn=send_text_message,
        inputs=[text_input, chatbot],
        outputs=[chatbot, text_input]
    )

    text_input.submit(
        fn=send_text_message,
        inputs=[text_input, chatbot],
        outputs=[chatbot, text_input]
    )

    audio_input.change(
        fn=process_audio,
        inputs=[audio_input, chatbot],
        outputs=[chatbot]
    )

    refresh_state_btn.click(
        fn=get_robot_state,
        outputs=[state_display]
    )

    # TODO: Implement quick action handlers
    # wake_btn.click(...)
    # sleep_btn.click(...)
    # pout_btn.click(...)
    # stop_btn.click(...)
