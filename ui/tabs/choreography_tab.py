"""
Choreography Tab - Routine builder, pre-recorded moves
"""
import gradio as gr


def create_choreography_tab():
    """Create the choreography tab interface."""
    gr.Markdown("## 🎵 Choreography Builder")
    gr.Markdown("_Coming soon: Pre-recorded move library, routine builder, and BPM control_")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Dance Moves")
            gr.Markdown("- side_to_side_sway\n- jackson_square\n- dizzy_spin\n- ...")

        with gr.Column():
            gr.Markdown("### Emotion Moves")
            gr.Markdown("- amazed1\n- confused1\n- happy1\n- ...")
