"""
Manual Control Tab - 6-DOF sliders, antenna control, goto positions
"""
import gradio as gr
import requests

API_BASE_URL = "http://localhost:8888"
DAEMON_URL = "http://localhost:8100"


def apply_position(x, y, z, yaw, pitch, roll, left_ant, right_ant, duration):
    """Apply manual position to robot."""
    try:
        # Send to daemon's goto endpoint
        response = requests.post(
            f"{DAEMON_URL}/api/joints/target",
            json={
                "position": {"x": x, "y": y, "z": z},
                "orientation": {"yaw": yaw, "pitch": pitch, "roll": roll},
                "antennas": {"left": left_ant, "right": right_ant},
                "duration": duration
            },
            timeout=5
        )

        if response.status_code == 200:
            return "✅ Position applied successfully"
        else:
            return f"❌ Error: {response.text}"

    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def reset_to_neutral():
    """Reset robot to neutral pose."""
    return (0, 0, 0, 0, 0, 0, 0, 0, "✅ Reset to neutral position")


def create_manual_control_tab():
    """Create the manual control tab interface."""

    gr.Markdown("## 🎮 Manual Position Control")

    with gr.Row():
        # Left column: Position control
        with gr.Column():
            gr.Markdown("### Position (X, Y, Z)")

            x_slider = gr.Slider(
                minimum=-3, maximum=3, value=0, step=0.1,
                label="X Position", info="Left (-) / Right (+)"
            )

            y_slider = gr.Slider(
                minimum=-3, maximum=3, value=0, step=0.1,
                label="Y Position", info="Back (-) / Forward (+)"
            )

            z_slider = gr.Slider(
                minimum=-3, maximum=3, value=0, step=0.1,
                label="Z Position", info="Down (-) / Up (+)"
            )

        # Middle column: Rotation control
        with gr.Column():
            gr.Markdown("### Rotation (Yaw, Pitch, Roll)")

            yaw_slider = gr.Slider(
                minimum=-180, maximum=180, value=0, step=5,
                label="Yaw", info="Left (-) / Right (+)"
            )

            pitch_slider = gr.Slider(
                minimum=-90, maximum=90, value=0, step=5,
                label="Pitch", info="Down (-) / Up (+)"
            )

            roll_slider = gr.Slider(
                minimum=-180, maximum=180, value=0, step=5,
                label="Roll", info="CCW (-) / CW (+)"
            )

        # Right column: Antenna control
        with gr.Column():
            gr.Markdown("### Antenna Control")

            left_ant_slider = gr.Slider(
                minimum=-3, maximum=3, value=0, step=0.1,
                label="Left Antenna", info="Safe: -3 to 1"
            )

            right_ant_slider = gr.Slider(
                minimum=-3, maximum=3, value=0, step=0.1,
                label="Right Antenna", info="Safe: -3 to 1"
            )

            duration_input = gr.Number(
                value=1.0, minimum=0.1, maximum=10, step=0.1,
                label="Duration (seconds)"
            )

    # Control buttons
    with gr.Row():
        apply_btn = gr.Button("✅ Apply Position", variant="primary", scale=2)
        reset_btn = gr.Button("🔄 Reset to Neutral", variant="secondary", scale=1)

    status_display = gr.Markdown("Status: Ready")

    # Quick look directions
    gr.Markdown("### 👀 Quick Look Directions")
    with gr.Row():
        look_left_btn = gr.Button("← Left")
        look_right_btn = gr.Button("Right →")
        look_up_btn = gr.Button("↑ Up")
        look_down_btn = gr.Button("↓ Down")
        look_front_btn = gr.Button("• Front")

    # Event handlers
    apply_btn.click(
        fn=apply_position,
        inputs=[x_slider, y_slider, z_slider, yaw_slider, pitch_slider, roll_slider,
                left_ant_slider, right_ant_slider, duration_input],
        outputs=[status_display]
    )

    reset_btn.click(
        fn=reset_to_neutral,
        outputs=[x_slider, y_slider, z_slider, yaw_slider, pitch_slider, roll_slider,
                 left_ant_slider, right_ant_slider, status_display]
    )

    # TODO: Implement quick look direction buttons
    # look_left_btn.click(...)
    # look_right_btn.click(...)
