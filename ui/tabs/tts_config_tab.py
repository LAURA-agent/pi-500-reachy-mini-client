"""
TTS Config Tab - Text-to-speech provider selection and voice settings
"""
import gradio as gr
import requests
import json

API_BASE_URL = "http://localhost:8888"


def load_tts_config():
    """Load current TTS config from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/config/tts", timeout=2)
        if response.status_code == 200:
            data = response.json()
            config = data.get("config", {})
            return (
                config.get("active_provider", "elevenlabs"),
                json.dumps(config, indent=2)
            )
    except:
        pass
    return ("elevenlabs", "{}")


def save_tts_config(provider, config_json):
    """Save TTS config to API."""
    try:
        config = json.loads(config_json)
        config["active_provider"] = provider

        response = requests.post(
            f"{API_BASE_URL}/config/tts",
            json={"config": config},
            timeout=5
        )

        if response.status_code == 200:
            return "✅ TTS config saved successfully"
        else:
            return f"❌ Error: {response.text}"

    except json.JSONDecodeError:
        return "❌ Invalid JSON format"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def create_tts_config_tab():
    """Create the TTS config tab interface."""
    gr.Markdown("## 🔊 Text-to-Speech Configuration")

    initial_provider, initial_config = load_tts_config()

    with gr.Row():
        with gr.Column(scale=2):
            provider_radio = gr.Radio(
                choices=["elevenlabs", "local_tts"],
                value=initial_provider,
                label="Active TTS Provider"
            )

            config_editor = gr.Code(
                value=initial_config,
                language="json",
                label="Configuration JSON",
                lines=15
            )

        with gr.Column(scale=1):
            gr.Markdown("### Actions")
            save_btn = gr.Button("💾 Save Config", variant="primary")
            reload_btn = gr.Button("🔄 Reload from Disk", variant="secondary")
            test_btn = gr.Button("🔊 Test TTS", variant="secondary")

            status_display = gr.Markdown("Status: Ready")

    save_btn.click(
        fn=save_tts_config,
        inputs=[provider_radio, config_editor],
        outputs=[status_display]
    )
