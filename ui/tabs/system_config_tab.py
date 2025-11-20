"""
System Config Tab - Breathing, movement arbiter, network, daemon settings
"""
import gradio as gr
import requests
import json

API_BASE_URL = "http://localhost:8888"


def load_movement_config():
    """Load current movement config from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/config/movement", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return json.dumps(data.get("config", {}), indent=2)
    except:
        pass
    return "{}"


def save_movement_config(config_json):
    """Save movement config to API."""
    try:
        config = json.loads(config_json)

        response = requests.post(
            f"{API_BASE_URL}/config/movement",
            json={"config": config},
            timeout=5
        )

        if response.status_code == 200:
            return "✅ System config saved successfully"
        else:
            return f"❌ Error: {response.text}"

    except json.JSONDecodeError:
        return "❌ Invalid JSON format"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def create_system_config_tab():
    """Create the system config tab interface."""
    gr.Markdown("## ⚙️ System Configuration")

    initial_config = load_movement_config()

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Movement & Behavior Settings")

            config_editor = gr.Code(
                value=initial_config,
                language="json",
                label="Movement Configuration JSON",
                lines=20
            )

        with gr.Column(scale=1):
            gr.Markdown("### Actions")
            save_btn = gr.Button("💾 Save Config", variant="primary")
            reload_btn = gr.Button("🔄 Reload from Disk", variant="secondary")

            status_display = gr.Markdown("Status: Ready")

            gr.Markdown("### System Info")
            gr.Markdown("""
**Breathing:**
- Amplitude, frequency control
- Auto-pause during moves

**Face Tracking:**
- Smoothing, max offset
- Camera integration

**Speech Motion:**
- Antenna synchronization
- TTS coordination
            """)

    save_btn.click(
        fn=save_movement_config,
        inputs=[config_editor],
        outputs=[status_display]
    )
