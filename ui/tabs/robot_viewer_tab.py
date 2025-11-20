"""
Robot Viewer Tab - 3D simulator feed, camera stream, live monitoring
"""
import gradio as gr


def create_robot_viewer_tab():
    """Create the robot viewer tab interface."""
    gr.Markdown("## 📹 Robot Viewer")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 3D Simulator View")
            gr.Markdown("_Video stream: http://localhost:8100/api/camera/stream.mjpg_")
            # TODO: Add MJPEG stream viewer
            gr.Image(label="Simulator Feed", height=400)

        with gr.Column():
            gr.Markdown("### Live Pose Data")
            # TODO: Add real-time chart
            gr.Markdown("_Chart will show 8-axis pose data in real-time_")
