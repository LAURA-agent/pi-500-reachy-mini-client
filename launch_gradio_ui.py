#!/usr/bin/env python3
"""
Launch script for Reachy Mini Gradio UI
Run this to start the control panel interface.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.gradio_app import create_ui

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Reachy Mini Control Panel")
    print("=" * 60)
    print()
    print("Starting Gradio UI on http://localhost:7860")
    print()
    print("Make sure the API server is running on port 8888:")
    print("  python main.py")
    print()
    print("=" * 60)

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
