"""
Inference Config Tab - MCP server vs Messages API configuration
"""
import gradio as gr
import requests
import json

API_BASE_URL = "http://localhost:8888"


def load_inference_config():
    """Load current inference config from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/config/inference", timeout=2)
        if response.status_code == 200:
            data = response.json()
            config = data.get("config", {})
            return (
                config.get("active_backend", "mcp_server"),
                json.dumps(config, indent=2)
            )
    except:
        pass
    return ("mcp_server", "{}")


def save_inference_config(backend, config_json):
    """Save inference config to API."""
    try:
        config = json.loads(config_json)
        config["active_backend"] = backend

        response = requests.post(
            f"{API_BASE_URL}/config/inference",
            json={"config": config},
            timeout=5
        )

        if response.status_code == 200:
            return "✅ Inference config saved successfully"
        else:
            return f"❌ Error: {response.text}"

    except json.JSONDecodeError:
        return "❌ Invalid JSON format"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def create_inference_config_tab():
    """Create the inference config tab interface."""
    gr.Markdown("## 🧠 Inference Backend Configuration")

    initial_backend, initial_config = load_inference_config()

    with gr.Row():
        with gr.Column(scale=2):
            backend_radio = gr.Radio(
                choices=["mcp_server", "messages_api"],
                value=initial_backend,
                label="Active Inference Backend"
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
            test_btn = gr.Button("🧪 Test Connection", variant="secondary")

            status_display = gr.Markdown("Status: Ready")

            gr.Markdown("### Backend Info")
            gr.Markdown("""
**MCP Server:**
- Connects to LAURA agent
- Session-based
- Full context management

**Messages API:**
- Direct Anthropic API
- Stateless requests
- Simpler, faster
            """)

    save_btn.click(
        fn=save_inference_config,
        inputs=[backend_radio, config_editor],
        outputs=[status_display]
    )
