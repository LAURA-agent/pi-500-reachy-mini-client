#!/usr/bin/env python3

import asyncio
import base64
from pathlib import Path
from typing import Optional, Tuple
from system.client_system_manager import ClientSystemManager


class SystemCommandManager:
    """
    Manages system command detection and execution.
    
    Handles detection of system commands in transcripts, command execution,
    and coordination with TTS and audio systems for feedback.
    """
    
    def __init__(self, client_settings, save_client_settings_func, 
                 get_active_tts_provider_func, set_active_tts_provider_func):
        self.client_settings = client_settings
        self.save_client_settings = save_client_settings_func
        self.get_active_tts_provider = get_active_tts_provider_func
        self.set_active_tts_provider = set_active_tts_provider_func
        self.system_manager = ClientSystemManager()

    def detect_system_command(self, transcript: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Detect system commands in transcript"""
        t = transcript.lower()
        if "enable remote tts" in t or "api tts" in t:
            return True, "switch_tts_mode", "api"
        elif "enable local tts" in t or "local tts" in t:
            return True, "switch_tts_mode", "local"
        elif "text only mode" in t or "text only" in t:
            return True, "switch_tts_mode", "text"
        elif "switch tts provider to cartesia" in t:
            return True, "switch_api_tts_provider", "cartesia"
        elif "switch tts provider to elevenlabs" in t:
            return True, "switch_api_tts_provider", "elevenlabs"
        elif "switch tts provider to piper" in t:
            return True, "switch_api_tts_provider", "piper"
        elif "calibrate listening" in t or "run calibration" in t or "calibrate vad" in t:
            return True, "vad_calibration", None
        elif "load documents" in t or "scan documents" in t:
            return True, "load_documents", None
        elif "clear documents" in t or "remove documents" in t:
            return True, "clear_documents", None
        elif "document status" in t or "show documents" in t:
            return True, "document_status", None
        # Reminder acknowledgment commands
        elif any(phrase in t for phrase in ["i'm going to bed", "going to bed", "bedtime", "time for bed"]):
            return True, "clear_reminder", "bedtime"
        elif any(phrase in t for phrase in ["i exercised", "workout done", "finished exercising", "exercise complete"]):
            return True, "clear_reminder", "exercise"
        elif any(phrase in t for phrase in ["reminder done", "finished that", "task complete", "i did it"]):
            return True, "clear_reminder", "general"
        return False, None, None

    async def upload_document(self, file_path: str, mcp_session, session_id) -> dict | None:
        """Upload document to server"""
        if not mcp_session or not session_id: 
            return None
            
        try:
            with open(file_path, 'rb') as f: 
                content = base64.b64encode(f.read()).decode('utf-8')
                
            return await mcp_session.call_tool("upload_document", arguments={
                "session_id": session_id, 
                "filename": Path(file_path).name, 
                "content": content
            })
        except Exception as e:
            print(f"[ERROR] Document upload failed: {e}")
            return None

    async def handle_system_command(self, cmd_type, cmd_arg, mcp_session, session_id, tts_handler, audio_coordinator):
        """Handle system commands with audio feedback"""
        print(f"[INFO] System command: {cmd_type}('{cmd_arg}')")
        
        uploaded_count = 0
        pending_files = []
        processed_files = []
        
        if cmd_type == "clear_reminder":
            # Use ClientSystemManager for reminder clearing
            success = await self.system_manager.clear_reminder(cmd_arg, mcp_session, session_id)
            if success:
                conf_audio, conf_engine = await tts_handler.generate_audio(
                    f"{cmd_arg} reminder cleared.", persona_name="laura"
                )
                if conf_audio:
                    await audio_coordinator.handle_tts_playback(conf_audio, conf_engine)
            return
        elif cmd_type == "switch_tts_mode":
            self.client_settings["tts_mode"] = cmd_arg
        elif cmd_type == "switch_api_tts_provider":
            self.set_active_tts_provider(cmd_arg)
        elif cmd_type == "vad_calibration":
            print("[INFO] Starting VAD calibration...")
            # Run calibration subprocess
            process = await asyncio.create_subprocess_exec(
                'python3', 'vad_calib.py',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Reload VAD settings after calibration
                from client_config import load_client_settings
                load_client_settings()
                print("[INFO] VAD calibration complete and settings reloaded.")
            else:
                print(f"[ERROR] VAD calibration failed: {stderr.decode()}")
        elif cmd_type == "load_documents":
            print("[INFO] Manually scanning for documents...")
            query_files_path = Path(self.client_settings.get("QUERY_FILES_DIR", "/home/user/RP500-Client/query_files"))
            if query_files_path.exists():
                for file_to_upload in query_files_path.iterdir():
                    if file_to_upload.is_file():
                        upload_result = await self.upload_document(str(file_to_upload), mcp_session, session_id)
                        if upload_result:
                            uploaded_count += 1
                            # Move to offload directory
                            offload_path = Path(self.client_settings.get("QUERY_OFFLOAD_DIR", "/home/user/RP500-Client/query_offload"))
                            offload_path.mkdir(parents=True, exist_ok=True)
                            try:
                                file_to_upload.rename(offload_path / file_to_upload.name)
                                print(f"[INFO] Document {file_to_upload.name} uploaded and moved to offload")
                            except Exception as e:
                                print(f"[ERROR] Could not move {file_to_upload.name}: {e}")
            print(f"[INFO] Document scan complete. Uploaded {uploaded_count} files.")
        elif cmd_type == "clear_documents":
            print("[INFO] Clearing documents...")
            # Clear any cached documents on server side
            if mcp_session and session_id:
                try:
                    # We could add a clear_documents tool to the server, but for now just report
                    print("[INFO] Documents cleared from local cache.")
                except Exception as e:
                    print(f"[WARN] Could not clear server documents: {e}")
        elif cmd_type == "document_status":
            print("[INFO] Checking document status...")
            query_files_path = Path(self.client_settings.get("QUERY_FILES_DIR", "/home/user/RP500-Client/query_files"))
            offload_path = Path(self.client_settings.get("QUERY_OFFLOAD_DIR", "/home/user/RP500-Client/query_offload"))
            
            if query_files_path.exists():
                pending_files = [f.name for f in query_files_path.iterdir() if f.is_file()]
            
            if offload_path.exists():
                processed_files = [f.name for f in offload_path.iterdir() if f.is_file()]
            
            print(f"[INFO] Document Status: {len(pending_files)} pending, {len(processed_files)} processed")
            if pending_files:
                print(f"[INFO] Pending files: {', '.join(pending_files)}")
        
        self.save_client_settings()
        
        # Confirmation audio
        if cmd_type in ["switch_tts_mode", "switch_api_tts_provider"]:
            conf_audio, conf_engine = await tts_handler.generate_audio(
                f"TTS set to {cmd_arg}.", persona_name="laura"
            )
        elif cmd_type == "load_documents":
            conf_audio, conf_engine = await tts_handler.generate_audio(
                f"Document scan completed. {uploaded_count} files uploaded.", persona_name="laura"
            )
        elif cmd_type == "clear_documents":
            conf_audio, conf_engine = await tts_handler.generate_audio(
                "Documents cleared.", persona_name="laura"
            )
        elif cmd_type == "document_status":
            conf_audio, conf_engine = await tts_handler.generate_audio(
                f"Document status: {len(pending_files)} pending, {len(processed_files)} processed.", persona_name="laura"
            )
        else:
            conf_audio, conf_engine = await tts_handler.generate_audio(
                "Command completed.", persona_name="laura"
            )
        
        if conf_audio:
            await audio_coordinator.handle_tts_playback(conf_audio, conf_engine)

    async def check_and_upload_documents(self, mcp_session, session_id):
        """Check for and upload any pending documents"""
        query_files_path = Path(self.client_settings.get("QUERY_FILES_DIR", "/home/user/RP500-Client/query_files"))
        if query_files_path.exists():
            for file_to_upload in query_files_path.iterdir():
                if file_to_upload.is_file():
                    upload_result = await self.upload_document(str(file_to_upload), mcp_session, session_id)
                    
                    # Move to offload directory
                    offload_path = Path(self.client_settings.get("QUERY_OFFLOAD_DIR", "/home/user/RP500-Client/query_offload"))
                    offload_path.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        file_to_upload.rename(offload_path / file_to_upload.name)
                        print(f"[INFO] Document {file_to_upload.name} uploaded and moved to offload")
                    except Exception as e:
                        print(f"[ERROR] Could not move {file_to_upload.name}: {e}")