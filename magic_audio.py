# magic_audio.py
import sounddevice as sd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import requests
import queue # Keep for potential internal use in a refactored class, but not primary flow
from scipy.io import wavfile
import boto3
import io
import openai
import webrtcvad # Keep for VAD if still used by a segmenter locally
import re
from dotenv import load_dotenv
import os
import traceback
from datetime import timezone
import pytz

# This file's direct audio capture and transcription orchestration via threads
# is being replaced by client-side capture and WebSocket streaming to api_server.py.
# The core transcription logic (_transcribe_audio_segment_openai, filter_hallucinations, etc.)
# will be moved to a new `transcription_service.py`.
# This file might be deprecated or heavily refactored to only contain specific audio utilities
# if any are still needed by the backend (e.g., server-side audio file manipulation if that becomes a feature).

# For now, we will comment out most of its class structure to avoid conflicts
# and indicate that its primary role is superseded.
# Utility functions like timestamp formatting might be moved to transcription_service.py
# or a general utils module.

load_dotenv()
logger = logging.getLogger(__name__)


# class Frame:
#     def __init__(self, bytes, timestamp, duration):
#         self.bytes = bytes
#         self.timestamp = timestamp
#         self.duration = duration

# class MagicAudio:
    # ... (Most of the class content will be removed or refactored) ...

    # The following utility functions are good candidates to be moved to
    # transcription_service.py or a shared utility module if they are
    # generally useful for timestamping or text processing.

    # def get_current_time(self):
    #     """Get current time in UTC"""
    #     return datetime.now(timezone.utc)

    # def get_elapsed_time(self):
    #     """Get elapsed time since session start in seconds, accounting for pauses."""
    #     if not self.session_start_utc: return 0
    #     now = self.get_current_time()
    #     elapsed_since_start = (now - self.session_start_utc).total_seconds()
    #     if self.is_paused and self.pause_start_time:
    #          elapsed_since_start = (datetime.fromtimestamp(self.pause_start_time, timezone.utc) - self.session_start_utc).total_seconds()
    #     return max(0, elapsed_since_start)


    # def get_timestamp(self, elapsed_seconds):
    #     """Convert elapsed seconds to UTC timestamp"""
    #     return self.session_start_utc + timedelta(seconds=elapsed_seconds)

    # def format_time(self, seconds):
    #     """Format elapsed seconds as HH:MM:SS with timezone"""
    #     if seconds is None:
    #         return "00:00:00"
    #     timestamp = self.get_timestamp(seconds)
    #     local_time = timestamp.astimezone(self.local_tz)
    #     return local_time.strftime("%H:%M:%S")

    # def format_timestamp_range(self, start_seconds, end_seconds):
    #     """Format a time range with timezone"""
    #     start_time = self.format_time(start_seconds)
    #     end_time = self.format_time(end_seconds)
    #     local_time = self.session_start_utc.astimezone(self.local_tz)
    #     tz_name = local_time.strftime('%Z')
    #     return f"[{start_time} - {end_time} {tz_name}]"

    # def filter_hallucinations(self, text):
    #     ... (This logic will move to transcription_service.py) ...

    # def is_valid_transcription(self, text):
    #     ... (This logic will move to transcription_service.py) ...
        
    # def transcribe_audio(self, audio_filename, api_key):
    #     ... (This core OpenAI call logic will move to transcription_service.py) ...

logger.info("magic_audio.py loaded, but its primary recording/transcription role is being refactored.")
logger.info("Core transcription logic will move to transcription_service.py and be orchestrated by api_server.py WebSocket handler.")

if __name__ == '__main__':
    # This main execution block is no longer relevant for the new architecture
    # as api_server.py will be the entry point.
    print("magic_audio.py is not intended to be run directly in the new architecture.")
    print("Its functionality is being integrated into api_server.py and transcription_service.py")