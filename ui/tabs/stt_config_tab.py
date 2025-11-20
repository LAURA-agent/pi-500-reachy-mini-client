"""
STT Config Tab - Speech-to-text provider selection and settings
"""
import gradio as gr
import requests
import json

API_BASE_URL = "http://localhost:8888"


def load_stt_config():
    """Load current STT config from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/config/stt", timeout=2)
        if response.status_code == 200:
            data = response.json()
            config = data.get("config", {})
            return (
                config.get("active_provider", "elevenlabs_scribe"),
                json.dumps(config, indent=2)
            )
    except:
        pass
    return ("elevenlabs_scribe", "{}")


def save_stt_config(provider, config_json):
    """Save STT config to API."""
    try:
        config = json.loads(config_json)
        config["active_provider"] = provider

        response = requests.post(
            f"{API_BASE_URL}/config/stt",
            json={"config": config},
            timeout=5
        )

        if response.status_code == 200:
            return "✅ STT config saved successfully"
        else:
            return f"❌ Error: {response.text}"

    except json.JSONDecodeError:
        return "❌ Invalid JSON format"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def reload_stt_config():
    """Reload STT config from disk."""
    try:
        response = requests.post(f"{API_BASE_URL}/config/reload/stt", timeout=2)
        if response.status_code == 200:
            provider, config_json = load_stt_config()
            return (provider, config_json, "✅ Config reloaded")
        else:
            return (None, None, f"❌ Error: {response.text}")
    except Exception as e:
        return (None, None, f"❌ Connection error: {str(e)}")


def create_stt_config_tab():
    """Create the STT config tab interface."""
    gr.Markdown("## 🎤 Speech-to-Text Configuration")

    # Load current config
    initial_provider, initial_config = load_stt_config()

    with gr.Row():
        with gr.Column(scale=2):
            provider_radio = gr.Radio(
                choices=["elevenlabs_scribe", "whisper_hf"],
                value=initial_provider,
                label="Active STT Provider"
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
            test_btn = gr.Button("🧪 Test STT", variant="secondary")

            status_display = gr.Markdown("Status: Ready")

            gr.Markdown("### Provider Info")
            gr.Markdown("""
**ElevenLabs Scribe:**
- Cloud API
- Fast, accurate
- Requires API key

**Whisper HF:**
- Hugging Face Space
- Free tier available
- Good multilingual support
            """)

    # Event handlers
    save_btn.click(
        fn=save_stt_config,
        inputs=[provider_radio, config_editor],
        outputs=[status_display]
    )

    reload_btn.click(
        fn=reload_stt_config,
        outputs=[provider_radio, config_editor, status_display]
    )
