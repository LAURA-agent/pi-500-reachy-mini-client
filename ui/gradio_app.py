"""
Main Gradio UI for Reachy Mini Client
Provides comprehensive control interface with 8 tabs styled like Chrome browser tabs.
"""
import gradio as gr
import requests
from pathlib import Path

# API server URL
API_BASE_URL = "http://localhost:8888"

# Chrome-style tab CSS using Pollen Robotics color palette
CHROME_TAB_CSS = """
/* Pollen Robotics Color Palette */
:root {
    --pink: #ff6170;
    --yellow-gold: #ffc261;
    --light-blue: #3bb0d1;
    --light-green: #3dde99;
    --navy-blue: #2B4C7E;
    --white: #FFFFFF;
    --navy-blue-dark: #1E3A5F;
    --light-blue-dark: #2a9dbd;
}

/* Chrome-style tabs */
.gradio-container {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* Tab container styling */
.tab-nav {
    background: linear-gradient(to bottom, var(--navy-blue) 0%, var(--navy-blue-dark) 100%) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 8px 0 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}

/* Individual tab buttons - Chrome style */
button[role="tab"] {
    background: linear-gradient(to bottom, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: rgba(255,255,255,0.7) !important;
    padding: 10px 20px !important;
    margin-right: 2px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

button[role="tab"]:hover {
    background: linear-gradient(to bottom, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%) !important;
    color: var(--white) !important;
}

/* Active tab */
button[role="tab"][aria-selected="true"] {
    background: var(--white) !important;
    color: var(--navy-blue) !important;
    border-color: var(--light-blue) !important;
    font-weight: 600 !important;
}

/* Tab content area */
.tabitem {
    background: var(--white) !important;
    border: 2px solid var(--navy-blue) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 20px !important;
}

/* Section headers */
.gr-form h2, .gr-box h2 {
    color: var(--navy-blue) !important;
    border-bottom: 3px solid var(--light-blue) !important;
    padding-bottom: 8px !important;
    margin-bottom: 15px !important;
    font-weight: 600 !important;
}

/* Buttons */
.gr-button {
    background: linear-gradient(135deg, var(--light-blue) 0%, var(--light-blue-dark) 100%) !important;
    border: none !important;
    color: var(--white) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(59, 176, 209, 0.4) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--navy-blue) 0%, var(--navy-blue-dark) 100%) !important;
}

.gr-button-secondary {
    background: linear-gradient(135deg, var(--light-green) 0%, #2bc985 100%) !important;
}

/* Input fields */
.gr-input, .gr-textbox, .gr-dropdown {
    border: 2px solid var(--light-blue) !important;
    border-radius: 6px !important;
}

.gr-input:focus, .gr-textbox:focus, .gr-dropdown:focus {
    border-color: var(--navy-blue) !important;
    box-shadow: 0 0 0 3px rgba(59, 176, 209, 0.2) !important;
}

/* Sliders */
.gr-slider input[type="range"]::-webkit-slider-thumb {
    background: var(--light-blue) !important;
}

.gr-slider input[type="range"]::-webkit-slider-track {
    background: linear-gradient(to right, var(--light-blue) 0%, var(--pink) 100%) !important;
}
"""


def create_ui():
    """Create the main Gradio interface with 8 tabs."""

    with gr.Blocks(css=CHROME_TAB_CSS, title="Reachy Mini Control Panel", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Reachy Mini Control Panel", elem_classes="header-title")

        with gr.Tabs(elem_id="main-tabs"):
            # Tab 1: Conversation
            with gr.Tab("💬 Conversation", id="tab-conversation"):
                from ui.tabs.conversation_tab import create_conversation_tab
                create_conversation_tab()

            # Tab 2: Manual Control
            with gr.Tab("🎮 Manual Control", id="tab-manual"):
                from ui.tabs.manual_control_tab import create_manual_control_tab
                create_manual_control_tab()

            # Tab 3: Choreography
            with gr.Tab("🎵 Choreography", id="tab-choreography"):
                from ui.tabs.choreography_tab import create_choreography_tab
                create_choreography_tab()

            # Tab 4: Robot Viewer
            with gr.Tab("📹 Robot Viewer", id="tab-viewer"):
                from ui.tabs.robot_viewer_tab import create_robot_viewer_tab
                create_robot_viewer_tab()

            # Tab 5: STT Config
            with gr.Tab("🎤 STT Config", id="tab-stt"):
                from ui.tabs.stt_config_tab import create_stt_config_tab
                create_stt_config_tab()

            # Tab 6: TTS Config
            with gr.Tab("🔊 TTS Config", id="tab-tts"):
                from ui.tabs.tts_config_tab import create_tts_config_tab
                create_tts_config_tab()

            # Tab 7: Inference Config
            with gr.Tab("🧠 Inference", id="tab-inference"):
                from ui.tabs.inference_config_tab import create_inference_config_tab
                create_inference_config_tab()

            # Tab 8: System Config
            with gr.Tab("⚙️ System", id="tab-system"):
                from ui.tabs.system_config_tab import create_system_config_tab
                create_system_config_tab()

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
