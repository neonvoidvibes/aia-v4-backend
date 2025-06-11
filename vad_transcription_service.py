"""
VAD-Filtered Real-Time Transcription Service

This module implements the robust VAD-filtered transcription pipeline as outlined in the research document.
It provides real-time Voice Activity Detection to prevent Whisper hallucinations on silent audio.

Key Features:
- File-based WebM to WAV conversion using ffmpeg for stability
- WebRTC VAD for reliable voice activity detection
- Per-session isolation with thread-safe processing
- Comprehensive debug logging for pipeline monitoring
- Fallback strategies and error handling
"""

import os
import io
import logging
import json
import uuid
import time
import threading
import subprocess
import queue
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
import wave
import numpy as np

import webrtcvad
import openai
import requests

# Try to import audio processing libraries
try:
    from scipy import signal
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - advanced audio filtering disabled")

# Configure logging
logger = logging.getLogger(__name__)

class VADTranscriptionService:
    """
    Main service class for VAD-filtered real-time transcription.
    Handles the complete pipeline from WebM blobs to filtered transcripts.
    """
    
    def __init__(self, openai_api_key: str, vad_aggressiveness: int = 2):
        """
        Initialize the VAD transcription service.
        
        Args:
            openai_api_key: OpenAI API key for Whisper transcription
            vad_aggressiveness: VAD aggressiveness level (0-3, higher = more strict)
        """
        self.openai_api_key = openai_api_key
        self.vad_aggressiveness = vad_aggressiveness
        
        # Initialize WebRTC VAD
        try:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
            logger.info(f"VAD initialized with aggressiveness level {vad_aggressiveness}")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")
            raise RuntimeError(f"VAD initialization failed: {e}")
        
        # Audio processing configuration
        self.target_sample_rate = 16000  # 16 kHz for Whisper and VAD
        self.target_channels = 1  # Mono
        self.vad_frame_duration_ms = 30  # 30ms frames for VAD
        self.vad_frame_bytes = int(self.target_sample_rate * 2 * self.vad_frame_duration_ms / 1000)  # 960 bytes
        
        logger.info(f"VAD Service initialized - Sample Rate: {self.target_sample_rate}Hz, "
                   f"Frame Duration: {self.vad_frame_duration_ms}ms, "
                   f"Frame Size: {self.vad_frame_bytes} bytes")

    def process_webm_blob_to_wav(self, webm_blob_path: str, output_wav_path: str, 
                                session_id: str) -> Tuple[bool, float]:
        """
        Convert WebM blob to WAV using ffmpeg for stable processing.
        
        Args:
            webm_blob_path: Path to the WebM blob file
            output_wav_path: Path where WAV output should be saved
            session_id: Session ID for logging context
            
        Returns:
            Tuple of (success: bool, duration_seconds: float)
        """
        logger.debug(f"Session {session_id}: Starting WebM to WAV conversion")
        logger.debug(f"Session {session_id}: Input: {webm_blob_path}")
        logger.debug(f"Session {session_id}: Output: {output_wav_path}")
        
        if not os.path.exists(webm_blob_path):
            logger.error(f"Session {session_id}: WebM blob file not found: {webm_blob_path}")
            return False, 0.0
            
        try:
            # ffmpeg command for robust WebM to WAV conversion
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # overwrite output if exists
                "-i", webm_blob_path,  # input WebM file
                "-ac", str(self.target_channels),  # mono channel
                "-ar", str(self.target_sample_rate),  # 16 kHz sample rate
                "-acodec", "pcm_s16le",  # 16-bit PCM little endian
                "-f", "wav",  # WAV format
                output_wav_path
            ]
            
            logger.info(f"Session {session_id}: Executing ffmpeg: {' '.join(ffmpeg_cmd)}")
            
            start_time = time.time()
            result = subprocess.run(
                ffmpeg_cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            conversion_time = time.time() - start_time
            
            logger.info(f"Session {session_id}: ffmpeg conversion completed in {conversion_time:.3f}s")
            
            # Get actual duration using ffprobe
            duration = self._get_audio_duration(output_wav_path, session_id)
            
            if duration > 0:
                logger.info(f"Session {session_id}: WAV file created successfully - Duration: {duration:.2f}s")
                return True, duration
            else:
                logger.warning(f"Session {session_id}: WAV file created but duration detection failed")
                return True, 0.0
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Session {session_id}: ffmpeg conversion failed")
            logger.error(f"Session {session_id}: ffmpeg stderr: {e.stderr}")
            logger.error(f"Session {session_id}: ffmpeg stdout: {e.stdout}")
            return False, 0.0
        except Exception as e:
            logger.error(f"Session {session_id}: Unexpected error during conversion: {e}", exc_info=True)
            return False, 0.0

    def _get_audio_duration(self, wav_path: str, session_id: str) -> float:
        """
        Get audio duration using ffprobe with fallback to wave module.
        
        Args:
            wav_path: Path to WAV file
            session_id: Session ID for logging context
            
        Returns:
            Duration in seconds, 0.0 if detection fails
        """
        # Try ffprobe first (more reliable)
        try:
            ffprobe_cmd = [
                "ffprobe", "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                wav_path
            ]
            
            result = subprocess.run(
                ffprobe_cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            duration = float(result.stdout.strip())
            logger.debug(f"Session {session_id}: ffprobe detected duration: {duration:.2f}s")
            return duration
            
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"Session {session_id}: ffprobe duration detection failed: {e}")
        
        # Fallback to wave module
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                duration = frames / float(sample_rate)
                logger.debug(f"Session {session_id}: wave module detected duration: {duration:.2f}s")
                return duration
        except Exception as e:
            logger.warning(f"Session {session_id}: wave module duration detection failed: {e}")
        
        logger.error(f"Session {session_id}: All duration detection methods failed")
        return 0.0

    def detect_voice_activity(self, wav_path: str, session_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """
        Perform Voice Activity Detection on a WAV file and extract voiced frames.
        Enhanced with minimum duration filtering and voice continuity checks.
        
        Args:
            wav_path: Path to WAV file to analyze
            session_id: Session ID for logging context
            
        Returns:
            Tuple of (voiced_audio_bytes: bytes, vad_details: Dict[str, Any])
        """
        logger.debug(f"Session {session_id}: Starting enhanced VAD analysis and extraction on {wav_path}")
        
        voiced_audio_data = bytearray()
        
        # Enhanced VAD configuration
        MIN_SEGMENT_DURATION_SECONDS = float(os.getenv('VAD_MIN_SEGMENT_DURATION', '1.0'))
        MIN_CONTINUITY_FRAMES = int(os.getenv('VAD_CONTINUITY_THRESHOLD', '10'))
        
        vad_details = {
            "total_frames": 0,
            "voiced_frames": 0,
            "voiced_frame_ratio": 0.0,
            "processing_time_ms": 0,
            "frame_duration_ms": self.vad_frame_duration_ms,
            "aggressiveness": self.vad_aggressiveness,
            "min_duration_seconds": MIN_SEGMENT_DURATION_SECONDS,
            "min_continuity_frames": MIN_CONTINUITY_FRAMES,
            "longest_voice_run_frames": 0,
            "voice_fragmentation_score": 0.0,
            "minimum_duration_rejected": False,
            "continuity_rejected": False
        }
        
        try:
            start_time = time.time()
            
            with wave.open(wav_path, 'rb') as wf:
                sample_rate, channels, sampwidth = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
                logger.debug(f"Session {session_id}: WAV format - Rate: {sample_rate}Hz, Channels: {channels}, Width: {sampwidth} bytes")
                
                # Format checks
                if sample_rate != self.target_sample_rate: logger.warning(f"Session {session_id}: Sample rate mismatch - Expected: {self.target_sample_rate}Hz, Got: {sample_rate}Hz")
                if channels != self.target_channels: logger.warning(f"Session {session_id}: Channel mismatch - Expected: {self.target_channels}, Got: {channels}")
                if sampwidth != 2: logger.warning(f"Session {session_id}: Sample width mismatch - Expected: 2 bytes (16-bit), Got: {sampwidth} bytes")
                
                audio_data = wf.readframes(wf.getnframes())
            
            # First pass: analyze frame-by-frame voice activity
            voiced_frames, total_frames = 0, 0
            voice_decisions = []  # Track voice decisions for continuity analysis
            
            for i in range(0, len(audio_data) - self.vad_frame_bytes + 1, self.vad_frame_bytes):
                frame = audio_data[i:i + self.vad_frame_bytes]
                total_frames += 1
                
                try:
                    is_voice = self.vad.is_speech(frame, sample_rate)
                    voice_decisions.append(is_voice)
                    if is_voice:
                        voiced_frames += 1
                except Exception as frame_error:
                    logger.warning(f"Session {session_id}: VAD error on frame {total_frames}: {frame_error}")
                    voice_decisions.append(False)
                    continue
            
            processing_time = (time.time() - start_time) * 1000
            voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
            
            # Calculate total voiced duration
            voiced_duration_seconds = (voiced_frames * self.vad_frame_duration_ms) / 1000.0
            
            # Analyze voice continuity
            voice_runs = self._analyze_voice_continuity(voice_decisions)
            longest_run = max(voice_runs) if voice_runs else 0
            fragmentation_score = self._calculate_fragmentation_score(voice_decisions, voiced_frames)
            
            vad_details.update({
                "total_frames": total_frames,
                "voiced_frames": voiced_frames,
                "voiced_frame_ratio": voiced_ratio,
                "processing_time_ms": processing_time,
                "voiced_duration_seconds": voiced_duration_seconds,
                "longest_voice_run_frames": longest_run,
                "voice_fragmentation_score": fragmentation_score,
                "voice_runs": voice_runs
            })
            
            # Enhanced filtering checks
            
            # Check 1: Minimum duration filter
            if voiced_duration_seconds < MIN_SEGMENT_DURATION_SECONDS:
                vad_details["minimum_duration_rejected"] = True
                logger.info(f"Session {session_id}: VAD rejected - insufficient voiced duration {voiced_duration_seconds:.2f}s < {MIN_SEGMENT_DURATION_SECONDS}s")
                return b"", vad_details
            
            # Check 2: Voice continuity filter
            if longest_run < MIN_CONTINUITY_FRAMES:
                vad_details["continuity_rejected"] = True
                logger.info(f"Session {session_id}: VAD rejected - insufficient voice continuity. Longest run: {longest_run} < {MIN_CONTINUITY_FRAMES} frames")
                return b"", vad_details
            
            # Check 3: Fragmentation filter (too scattered voice)
            if fragmentation_score > 0.7:  # High fragmentation
                vad_details["fragmentation_rejected"] = True
                logger.info(f"Session {session_id}: VAD rejected - high voice fragmentation score: {fragmentation_score:.3f}")
                return b"", vad_details
            
            # All checks passed - extract voiced audio
            frame_index = 0
            for i in range(0, len(audio_data) - self.vad_frame_bytes + 1, self.vad_frame_bytes):
                if frame_index < len(voice_decisions) and voice_decisions[frame_index]:
                    frame = audio_data[i:i + self.vad_frame_bytes]
                    voiced_audio_data.extend(frame)
                frame_index += 1
            
            logger.info(f"Session {session_id}: Enhanced VAD Analysis Complete. "
                       f"Total: {total_frames}, Voiced: {voiced_frames}, Ratio: {voiced_ratio:.3f}, "
                       f"Duration: {voiced_duration_seconds:.2f}s, Longest Run: {longest_run}, "
                       f"Fragmentation: {fragmentation_score:.3f}. "
                       f"Extracted {len(voiced_audio_data)} voiced bytes.")
            
            return bytes(voiced_audio_data), vad_details
            
        except Exception as e:
            logger.error(f"Session {session_id}: VAD analysis failed: {e}", exc_info=True)
            vad_details["error"] = str(e)
            return b"", vad_details
    
    def _analyze_voice_continuity(self, voice_decisions: List[bool]) -> List[int]:
        """
        Analyze voice continuity by finding consecutive voice runs.
        
        Args:
            voice_decisions: List of boolean voice decisions for each frame
            
        Returns:
            List of voice run lengths (in frames)
        """
        runs = []
        current_run = 0
        
        for is_voice in voice_decisions:
            if is_voice:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        
        # Add final run if it was ongoing
        if current_run > 0:
            runs.append(current_run)
        
        return runs
    
    def _calculate_fragmentation_score(self, voice_decisions: List[bool], voiced_frames: int) -> float:
        """
        Calculate voice fragmentation score (0.0 = continuous, 1.0 = highly fragmented).
        
        Args:
            voice_decisions: List of boolean voice decisions
            voiced_frames: Total number of voiced frames
            
        Returns:
            Fragmentation score between 0.0 and 1.0
        """
        if voiced_frames == 0 or len(voice_decisions) == 0:
            return 1.0
        
        # Count voice transitions (voice -> silence or silence -> voice)
        transitions = 0
        for i in range(1, len(voice_decisions)):
            if voice_decisions[i] != voice_decisions[i-1]:
                transitions += 1
        
        # Normalize by the number of voiced frames
        # More transitions relative to voiced content = higher fragmentation
        if voiced_frames == 0:
            return 1.0
        
        fragmentation = transitions / (2 * voiced_frames)  # Divide by 2 since each voice segment has at most 2 transitions
        return min(fragmentation, 1.0)  # Cap at 1.0

    def apply_audio_enhancement_pipeline(self, wav_path: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Apply audio quality enhancement pipeline to WAV file.
        
        Args:
            wav_path: Path to input WAV file
            session_id: Session ID for logging context
            
        Returns:
            Tuple of (enhanced_wav_path, enhancement_details)
        """
        enhancement_details = {
            "original_file": wav_path,
            "noise_gate_applied": False,
            "spectral_filter_applied": False,
            "compression_applied": False,
            "enhancement_time_ms": 0,
            "rms_before": 0.0,
            "rms_after": 0.0,
            "dynamic_range_before": 0.0,
            "dynamic_range_after": 0.0,
            "scipy_available": SCIPY_AVAILABLE
        }
        
        if not SCIPY_AVAILABLE:
            logger.warning(f"Session {session_id}: Audio enhancement skipped - scipy not available")
            return wav_path, enhancement_details
        
        try:
            start_time = time.time()
            
            # Read original audio
            sample_rate, audio_data = wavfile.read(wav_path)
            
            if audio_data.dtype != np.int16:
                logger.warning(f"Session {session_id}: Converting audio from {audio_data.dtype} to int16")
                audio_data = audio_data.astype(np.int16)
            
            # Calculate original metrics
            enhancement_details["rms_before"] = self._calculate_rms(audio_data)
            enhancement_details["dynamic_range_before"] = self._calculate_dynamic_range(audio_data)
            
            logger.debug(f"Session {session_id}: Original audio - RMS: {enhancement_details['rms_before']:.3f}, "
                        f"Dynamic Range: {enhancement_details['dynamic_range_before']:.1f}dB")
            
            # Step 1: Apply noise gate
            audio_data, gate_applied = self._apply_noise_gate(audio_data, session_id)
            enhancement_details["noise_gate_applied"] = gate_applied
            
            # Step 2: Apply spectral filtering
            audio_data, filter_applied = self._apply_spectral_filter(audio_data, sample_rate, session_id)
            enhancement_details["spectral_filter_applied"] = filter_applied
            
            # Step 3: Apply dynamic range compression
            audio_data, compression_applied = self._apply_compression(audio_data, session_id)
            enhancement_details["compression_applied"] = compression_applied
            
            # Calculate enhanced metrics
            enhancement_details["rms_after"] = self._calculate_rms(audio_data)
            enhancement_details["dynamic_range_after"] = self._calculate_dynamic_range(audio_data)
            
            # Write enhanced audio to new file
            enhanced_path = wav_path.replace('.wav', '_enhanced.wav')
            wavfile.write(enhanced_path, sample_rate, audio_data)
            
            enhancement_details["enhancement_time_ms"] = (time.time() - start_time) * 1000
            enhancement_details["enhanced_file"] = enhanced_path
            
            logger.info(f"Session {session_id}: Audio enhancement complete in {enhancement_details['enhancement_time_ms']:.1f}ms. "
                       f"RMS: {enhancement_details['rms_before']:.3f} → {enhancement_details['rms_after']:.3f}, "
                       f"Dynamic Range: {enhancement_details['dynamic_range_before']:.1f}dB → {enhancement_details['dynamic_range_after']:.1f}dB")
            
            return enhanced_path, enhancement_details
            
        except Exception as e:
            logger.error(f"Session {session_id}: Audio enhancement failed: {e}", exc_info=True)
            enhancement_details["error"] = str(e)
            return wav_path, enhancement_details
    
    def _apply_noise_gate(self, audio_data: np.ndarray, session_id: str) -> Tuple[np.ndarray, bool]:
        """Apply noise gate to remove low-level background noise."""
        try:
            # Configuration
            gate_threshold = float(os.getenv('AUDIO_NOISE_GATE_THRESHOLD', '0.03'))
            fade_duration_samples = int(0.01 * 16000)  # 10ms fade
            
            # Calculate threshold in absolute terms
            max_amplitude = np.max(np.abs(audio_data))
            threshold = int(max_amplitude * gate_threshold)
            
            # Apply gate with smooth transitions
            gated_audio = audio_data.copy()
            below_threshold = np.abs(audio_data) < threshold
            
            # Find regions to gate
            gate_regions = []
            in_gate_region = False
            start_idx = 0
            
            for i, below in enumerate(below_threshold):
                if below and not in_gate_region:
                    start_idx = i
                    in_gate_region = True
                elif not below and in_gate_region:
                    gate_regions.append((start_idx, i))
                    in_gate_region = False
            
            # Add final region if needed
            if in_gate_region:
                gate_regions.append((start_idx, len(audio_data)))
            
            # Apply gating with fades
            gates_applied = 0
            for start, end in gate_regions:
                if end - start > fade_duration_samples * 2:  # Only gate if region is long enough
                    # Fade out
                    fade_out_end = min(start + fade_duration_samples, end)
                    fade_factor = np.linspace(1.0, 0.0, fade_out_end - start)
                    gated_audio[start:fade_out_end] = (gated_audio[start:fade_out_end] * fade_factor).astype(np.int16)
                    
                    # Gate middle section
                    if fade_out_end < end - fade_duration_samples:
                        gated_audio[fade_out_end:end - fade_duration_samples] = 0
                    
                    # Fade in
                    fade_in_start = max(end - fade_duration_samples, start)
                    if fade_in_start < end:
                        fade_factor = np.linspace(0.0, 1.0, end - fade_in_start)
                        gated_audio[fade_in_start:end] = (gated_audio[fade_in_start:end] * fade_factor).astype(np.int16)
                    
                    gates_applied += 1
            
            logger.debug(f"Session {session_id}: Noise gate applied {gates_applied} regions with threshold {gate_threshold}")
            return gated_audio, gates_applied > 0
            
        except Exception as e:
            logger.warning(f"Session {session_id}: Noise gate failed: {e}")
            return audio_data, False
    
    def _apply_spectral_filter(self, audio_data: np.ndarray, sample_rate: int, session_id: str) -> Tuple[np.ndarray, bool]:
        """Apply spectral filtering to remove noise outside speech range."""
        try:
            # Configuration
            highpass_freq = 80  # Remove rumble below 80Hz
            lowpass_freq = 8000  # Remove noise above 8kHz
            notch_freqs = [50, 60]  # Power line noise
            
            # Convert to float for processing
            audio_float = audio_data.astype(np.float32)
            nyquist = sample_rate / 2
            
            # High-pass filter (remove rumble)
            if highpass_freq < nyquist:
                sos_hp = signal.butter(4, highpass_freq / nyquist, btype='high', output='sos')
                audio_float = signal.sosfilt(sos_hp, audio_float)
            
            # Low-pass filter (remove high-frequency noise)
            if lowpass_freq < nyquist:
                sos_lp = signal.butter(4, lowpass_freq / nyquist, btype='low', output='sos')
                audio_float = signal.sosfilt(sos_lp, audio_float)
            
            # Notch filters for power line noise
            for notch_freq in notch_freqs:
                if notch_freq < nyquist:
                    # Create notch filter with Q factor of 30
                    sos_notch = signal.iirnotch(notch_freq / nyquist, Q=30, output='sos')
                    audio_float = signal.sosfilt(sos_notch, audio_float)
            
            # Convert back to int16
            filtered_audio = np.clip(audio_float, -32768, 32767).astype(np.int16)
            
            logger.debug(f"Session {session_id}: Spectral filtering applied - "
                        f"HP: {highpass_freq}Hz, LP: {lowpass_freq}Hz, Notch: {notch_freqs}")
            return filtered_audio, True
            
        except Exception as e:
            logger.warning(f"Session {session_id}: Spectral filtering failed: {e}")
            return audio_data, False
    
    def _apply_compression(self, audio_data: np.ndarray, session_id: str) -> Tuple[np.ndarray, bool]:
        """Apply dynamic range compression to normalize voice levels."""
        try:
            # Configuration
            ratio = 3.0  # 3:1 compression ratio
            threshold_db = -20.0  # Threshold in dB
            attack_samples = int(0.005 * 16000)  # 5ms attack
            release_samples = int(0.05 * 16000)  # 50ms release
            makeup_gain_db = 6.0  # Makeup gain
            
            # Convert to float and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Calculate envelope using peak detection
            envelope = np.abs(audio_float)
            
            # Smooth envelope with attack/release
            smoothed_envelope = envelope.copy()
            gain_reduction = np.ones_like(envelope)
            
            for i in range(1, len(envelope)):
                if envelope[i] > smoothed_envelope[i-1]:
                    # Attack
                    alpha = 1.0 - np.exp(-1.0 / attack_samples)
                else:
                    # Release
                    alpha = 1.0 - np.exp(-1.0 / release_samples)
                
                smoothed_envelope[i] = alpha * envelope[i] + (1.0 - alpha) * smoothed_envelope[i-1]
            
            # Apply compression
            threshold_linear = 10.0 ** (threshold_db / 20.0)
            
            for i, env_val in enumerate(smoothed_envelope):
                if env_val > threshold_linear:
                    # Calculate gain reduction
                    env_db = 20.0 * np.log10(max(env_val, 1e-10))
                    over_threshold_db = env_db - threshold_db
                    compressed_db = threshold_db + (over_threshold_db / ratio)
                    gain_reduction[i] = 10.0 ** ((compressed_db - env_db) / 20.0)
            
            # Apply gain reduction and makeup gain
            compressed_audio = audio_float * gain_reduction
            makeup_gain_linear = 10.0 ** (makeup_gain_db / 20.0)
            compressed_audio *= makeup_gain_linear
            
            # Convert back to int16 with clipping
            compressed_audio = np.clip(compressed_audio * 32768.0, -32768, 32767).astype(np.int16)
            
            logger.debug(f"Session {session_id}: Compression applied - "
                        f"Ratio: {ratio}:1, Threshold: {threshold_db}dB, Makeup: {makeup_gain_db}dB")
            return compressed_audio, True
            
        except Exception as e:
            logger.warning(f"Session {session_id}: Compression failed: {e}")
            return audio_data, False
    
    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio data."""
        try:
            return float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0)
        except:
            return 0.0
    
    def _calculate_dynamic_range(self, audio_data: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        try:
            audio_float = audio_data.astype(np.float32) / 32768.0
            max_val = np.max(np.abs(audio_float))
            min_val = np.percentile(np.abs(audio_float[np.abs(audio_float) > 0]), 5)  # 5th percentile as noise floor
            if min_val > 0 and max_val > 0:
                return 20.0 * np.log10(max_val / min_val)
            return 0.0
        except:
            return 0.0

    def transcribe_with_whisper(self, wav_path: str, session_id: str, 
                               language_setting: str = "any") -> Optional[Dict[str, Any]]:
        """
        Transcribe audio using Whisper API with enhanced error handling.
        
        Args:
            wav_path: Path to WAV file to transcribe
            session_id: Session ID for logging context
            language_setting: Language setting ('en', 'sv', 'any')
            
        Returns:
            Whisper transcription result or None if failed
        """
        logger.info(f"Session {session_id}: Starting Whisper transcription")
        logger.info(f"Session {session_id}: File: {wav_path}, Language: {language_setting}")
        
        if not os.path.exists(wav_path):
            logger.error(f"Session {session_id}: WAV file not found: {wav_path}")
            return None
        
        try:
            with open(wav_path, "rb") as audio_file:
                # Prepare API request
                url = "https://api.openai.com/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {self.openai_api_key}"}
                
                # Build request data
                data_payload = {
                    'model': 'whisper-1',
                    'response_format': 'verbose_json',
                    'temperature': 0.0,
                    'no_speech_threshold': 0.9,        # Strict threshold
                    'logprob_threshold': -0.5,         # Strict threshold  
                    'compression_ratio_threshold': 1.8  # Strict threshold
                }
                
                # Handle language setting
                if language_setting in ["en", "sv"]:
                    data_payload['language'] = language_setting
                    logger.info(f"Session {session_id}: Whisper language set to '{language_setting}'")
                else:
                    logger.info(f"Session {session_id}: Whisper language auto-detect (setting: '{language_setting}')")
                
                # Construct initial prompt for better accuracy
                initial_prompt_base = (
                    "Please focus on transcribing professional business discourse accurately. "
                    "Avoid common conversational filler phrases if they obscure meaning. "
                    "Do not transcribe generic social media phrases, common video outros like 'subscribe to my channel' "
                    "or 'thanks for watching', or phrases in languages clearly unrelated to a business meeting context "
                    "unless they are direct quotes or proper nouns. If unsure, prioritize clear speech over interpreting noise."
                )
                
                if language_setting == "en":
                    initial_prompt = "The primary language is English. " + initial_prompt_base
                elif language_setting == "sv":
                    initial_prompt = "The primary language is Swedish. " + initial_prompt_base
                else:
                    initial_prompt = "The primary languages expected are English and Swedish. " + initial_prompt_base
                
                data_payload['initial_prompt'] = initial_prompt
                
                # Prepare file for upload
                filename = os.path.basename(wav_path)
                files = {'file': (filename, audio_file, 'audio/wav')}
                
                # Make API call with retry logic
                max_retries = 3
                timeout = 900  # 15 minutes
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Session {session_id}: Whisper API call attempt {attempt + 1}/{max_retries}")
                        
                        start_time = time.time()
                        response = requests.post(
                            url, 
                            headers=headers, 
                            data=data_payload, 
                            files=files, 
                            timeout=timeout
                        )
                        api_time = time.time() - start_time
                        
                        response.raise_for_status()
                        transcription = response.json()
                        
                        logger.info(f"Session {session_id}: Whisper API succeeded in {api_time:.2f}s")
                        
                        # Validate response structure
                        if 'segments' in transcription and isinstance(transcription['segments'], list):
                            logger.info(f"Session {session_id}: Transcription contains {len(transcription['segments'])} segments")
                            return transcription
                        else:
                            logger.warning(f"Session {session_id}: Unexpected transcription format: {transcription}")
                            return None
                    
                    except requests.exceptions.Timeout:
                        logger.warning(f"Session {session_id}: Whisper API timeout on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            raise
                    
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Session {session_id}: Whisper API error on attempt {attempt + 1}: {e}")
                        if hasattr(e, 'response') and e.response:
                            try:
                                error_detail = e.response.json()
                                logger.error(f"Session {session_id}: API error detail: {error_detail}")
                            except json.JSONDecodeError:
                                logger.error(f"Session {session_id}: API error response (non-JSON): {e.response.text[:500]}")
                        if attempt == max_retries - 1:
                            raise
                    
                    except Exception as e:
                        logger.error(f"Session {session_id}: Unexpected Whisper API error on attempt {attempt + 1}: {e}")
                        if attempt == max_retries - 1:
                            raise
                    
                    # Wait before retry
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 8)
                        logger.info(f"Session {session_id}: Waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                
                return None
                
        except Exception as e:
            logger.error(f"Session {session_id}: Whisper transcription failed: {e}", exc_info=True)
            return None

    def process_audio_chunk(self, webm_blob_bytes: bytes, session_id: str, 
                           temp_dir: str, language_setting: str = "any") -> Dict[str, Any]:
        """
        Process a complete audio chunk through the VAD-filtered pipeline.
        
        Args:
            webm_blob_bytes: Raw WebM blob data
            session_id: Session ID for logging context
            temp_dir: Temporary directory for processing files
            language_setting: Language setting for transcription
            
        Returns:
            Processing result dictionary with transcription and metadata
        """
        chunk_id = uuid.uuid4().hex
        logger.info(f"Session {session_id}: Processing audio chunk {chunk_id} ({len(webm_blob_bytes)} bytes)")
        
        processing_start_time = time.time()
        result = {
            "chunk_id": chunk_id, "session_id": session_id, "processing_start_time": processing_start_time,
            "webm_blob_size_bytes": len(webm_blob_bytes), "has_voice": False, "transcription": None,
            "vad_details": {}, "processing_times": {}, "errors": [], "files_created": [], "files_cleaned": []
        }
        
        webm_path = os.path.join(temp_dir, f"chunk_{chunk_id}.webm")
        wav_path = os.path.join(temp_dir, f"chunk_{chunk_id}.wav")
        clean_wav_path = os.path.join(temp_dir, f"clean_chunk_{chunk_id}.wav")
        
        try:
            # Step 1: Write WebM blob
            step_start = time.time()
            with open(webm_path, 'wb') as f: f.write(webm_blob_bytes)
            result["files_created"].append(webm_path)
            result["processing_times"]["webm_write"] = (time.time() - step_start) * 1000
            
            # Step 2: Convert WebM to WAV
            step_start = time.time()
            conversion_success, duration = self.process_webm_blob_to_wav(webm_path, wav_path, session_id)
            result["processing_times"]["webm_to_wav"] = (time.time() - step_start) * 1000
            if not conversion_success:
                raise RuntimeError("WebM to WAV conversion failed")
            result["files_created"].append(wav_path)
            result["audio_duration_seconds"] = duration
            
            # Step 2.5: Apply audio enhancement pipeline
            step_start = time.time()
            enhanced_wav_path, enhancement_details = self.apply_audio_enhancement_pipeline(wav_path, session_id)
            result["processing_times"]["audio_enhancement"] = (time.time() - step_start) * 1000
            result["enhancement_details"] = enhancement_details
            if enhanced_wav_path != wav_path:
                result["files_created"].append(enhanced_wav_path)
            
            # Use enhanced audio for VAD analysis
            vad_input_path = enhanced_wav_path
            
            # Step 3: VAD analysis and extraction
            step_start = time.time()
            voiced_audio_bytes, vad_details = self.detect_voice_activity(vad_input_path, session_id)
            result["processing_times"]["vad_analysis"] = (time.time() - step_start) * 1000
            result["has_voice"] = len(voiced_audio_bytes) > 0
            result["vad_details"] = vad_details
            
            if not result["has_voice"]:
                logger.info(f"Session {session_id}: No voice detected - skipping Whisper transcription")
                result["skip_reason"] = "no_voice_detected"
            else:
                # Step 4: Write clean WAV and transcribe
                step_start = time.time()
                try:
                    with wave.open(clean_wav_path, 'wb') as wf:
                        wf.setnchannels(self.target_channels)
                        wf.setsampwidth(2) # 16-bit
                        wf.setframerate(self.target_sample_rate)
                        wf.writeframes(voiced_audio_bytes)
                    result["files_created"].append(clean_wav_path)
                    logger.info(f"Session {session_id}: Created clean WAV with voiced audio at {clean_wav_path}")
                except Exception as wave_error:
                    raise RuntimeError(f"Failed to write clean WAV file: {wave_error}") from wave_error

                transcription = self.transcribe_with_whisper(clean_wav_path, session_id, language_setting)
                result["processing_times"]["whisper_transcription"] = (time.time() - step_start) * 1000
                result["transcription"] = transcription
                
                if transcription:
                    logger.info(f"Session {session_id}: Whisper transcription completed in {result['processing_times']['whisper_transcription']:.1f}ms with {len(transcription.get('segments',[]))} segments.")
                else:
                    logger.warning(f"Session {session_id}: Whisper returned no results for voiced audio.")
                    result["errors"].append("Whisper transcription failed or returned no results")

        except Exception as e:
            error_msg = f"Unexpected error during chunk processing: {e}"
            logger.error(f"Session {session_id}: {error_msg}", exc_info=True)
            result["errors"].append(error_msg)
        
        finally:
            # Step 5: Cleanup temporary files (but keep clean_wav_path for callback)
            cleanup_start = time.time()
            cleanup_files = [webm_path, wav_path]
            
            # Add enhanced WAV file to cleanup if it was created
            enhanced_wav_path = result.get("enhancement_details", {}).get("enhanced_file")
            if enhanced_wav_path and enhanced_wav_path != wav_path:
                cleanup_files.append(enhanced_wav_path)
            
            # Don't clean up clean_wav_path yet - it's needed for the callback
            for file_path in cleanup_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        result["files_cleaned"].append(file_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Session {session_id}: Failed to clean up {file_path}: {cleanup_error}")
            result["processing_times"]["cleanup"] = (time.time() - cleanup_start) * 1000
        
        total_processing_time = (time.time() - processing_start_time) * 1000
        result["processing_times"]["total"] = total_processing_time
        result["processing_end_time"] = time.time()
        
        logger.info(f"Session {session_id}: Chunk {chunk_id} processing completed in {total_processing_time:.1f}ms. Voice: {result['has_voice']}, Transcription: {'Success' if result['transcription'] else 'None/Failed'}")
        
        return result


class SessionAudioProcessor:
    """
    Manages audio processing for a single VAD transcription session.
    Handles audio blob accumulation, segmentation, and VAD pipeline coordination.
    """
    
    def __init__(self, session_id: str, temp_dir: str, vad_service: VADTranscriptionService,
                 result_callback: Callable[[Dict[str, Any]], None],
                 language_setting: str = "any", segment_duration_target: float = 15.0):
        """
        Initialize session audio processor.
        
        Args:
            session_id: Unique session identifier
            temp_dir: Session temporary directory
            vad_service: VAD transcription service instance
            result_callback: Callback for transcription results
            language_setting: Language setting for transcription
            segment_duration_target: Target segment duration in seconds
        """
        self.session_id = session_id
        self.temp_dir = temp_dir
        self.vad_service = vad_service
        self.result_callback = result_callback
        self.language_setting = language_setting
        self.segment_duration_target = segment_duration_target
        
        # Create session temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Audio accumulation state
        self.webm_global_header = b""
        self.current_segment_bytes = bytearray()
        self.accumulated_duration = 0.0
        self.is_first_blob = True
        
        # Processing queue and threading
        self.audio_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.processing_thread: Optional[threading.Thread] = None
        self.session_lock = threading.RLock()
        self.is_active = False
        
        # Processing statistics
        self.processing_stats = {
            "chunks_received": 0,
            "chunks_processed": 0,
            "chunks_with_voice": 0,
            "chunks_transcribed": 0,
            "total_processing_time_ms": 0,
            "errors": [],
            "session_start_time": time.time()
        }
        
        logger.info(f"Session {self.session_id}: Audio processor initialized")
        logger.info(f"Session {self.session_id}: Temp dir: {self.temp_dir}")
        logger.info(f"Session {self.session_id}: Language: {self.language_setting}, Target duration: {self.segment_duration_target}s")

    def start(self):
        """Start the audio processing worker thread."""
        with self.session_lock:
            if self.is_active:
                logger.warning(f"Session {self.session_id}: Already started")
                return
            
            self.is_active = True
            self.processing_thread = threading.Thread(
                target=self._processing_worker,
                name=f"VAD-Worker-{self.session_id[:8]}"
            )
            self.processing_thread.start()
            
        logger.info(f"Session {self.session_id}: Audio processor started")

    def stop(self):
        """Stop the audio processor and clean up resources."""
        logger.info(f"Session {self.session_id}: Stopping audio processor")
        
        with self.session_lock:
            if not self.is_active:
                logger.info(f"Session {self.session_id}: Already stopped")
                return
            
            self.is_active = False
        
        # Process any remaining audio
        self._process_final_segment()
        
        # Signal worker thread to stop
        try:
            self.audio_queue.put(None, timeout=5.0)  # Sentinel value
        except queue.Full:
            logger.warning(f"Session {self.session_id}: Queue full when stopping")
        
        # Wait for worker thread to complete
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10.0)
            if self.processing_thread.is_alive():
                logger.error(f"Session {self.session_id}: Worker thread did not stop gracefully")
        
        # Clean up session
        self._cleanup_session()
        
        logger.info(f"Session {self.session_id}: Audio processor stopped")

    def add_audio_blob(self, webm_blob_bytes: bytes):
        """
        Add an audio blob for processing.
        
        Args:
            webm_blob_bytes: Raw WebM audio data
        """
        if not self.is_active:
            logger.warning(f"Session {self.session_id}: Received audio blob but processor not active")
            return
        
        logger.debug(f"Session {self.session_id}: Received audio blob ({len(webm_blob_bytes)} bytes)")
        
        with self.session_lock:
            self.processing_stats["chunks_received"] += 1
            
            # Handle first blob (contains WebM header)
            if self.is_first_blob:
                self.webm_global_header = bytes(webm_blob_bytes)
                self.is_first_blob = False
                logger.info(f"Session {self.session_id}: Captured WebM global header ({len(self.webm_global_header)} bytes)")
            
            # Accumulate blob data
            self.current_segment_bytes.extend(webm_blob_bytes)
            self.accumulated_duration += 3.0  # Assuming 3-second blobs from client
            
            logger.debug(f"Session {self.session_id}: Accumulated {self.accumulated_duration}s of audio data")
            
            # Check if we have enough for a segment
            if self.accumulated_duration >= self.segment_duration_target:
                self._queue_current_segment_for_processing()

    def _queue_current_segment_for_processing(self):
        """Queue the current accumulated segment for processing."""
        if not self.current_segment_bytes:
            logger.debug(f"Session {self.session_id}: No segment data to queue")
            return
        
        # Prepare segment for processing
        segment_bytes = bytes(self.current_segment_bytes)
        header_bytes = self.webm_global_header
        
        # Combine header and segment data if needed
        if header_bytes and not segment_bytes.startswith(header_bytes):
            combined_bytes = header_bytes + segment_bytes
        else:
            combined_bytes = segment_bytes
        
        # Queue for processing
        processing_item = {
            "segment_bytes": combined_bytes,
            "accumulated_duration": self.accumulated_duration,
            "timestamp": time.time()
        }
        
        try:
            self.audio_queue.put(processing_item, block=False)
            logger.info(f"Session {self.session_id}: Queued segment for processing - "
                       f"Size: {len(combined_bytes)} bytes, Duration: {self.accumulated_duration:.1f}s")
        except queue.Full:
            logger.warning(f"Session {self.session_id}: Audio queue full, dropping segment")
            self.processing_stats["errors"].append("Queue full - segment dropped")
        
        # Reset accumulation
        self.current_segment_bytes = bytearray()
        self.accumulated_duration = 0.0

    def _processing_worker(self):
        """Worker thread for processing audio segments."""
        logger.info(f"Session {self.session_id}: Processing worker thread started")
        
        while self.is_active:
            try:
                # Get next item from queue with timeout
                item = self.audio_queue.get(timeout=5.0)
                
                # Check for sentinel value (stop signal)
                if item is None:
                    logger.info(f"Session {self.session_id}: Received stop signal")
                    break
                
                # Process the segment
                self._process_segment(item)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Timeout - continue checking if still active
                continue
            except Exception as e:
                logger.error(f"Session {self.session_id}: Error in processing worker: {e}", exc_info=True)
                self.processing_stats["errors"].append(f"Worker error: {e}")
        
        logger.info(f"Session {self.session_id}: Processing worker thread stopped")

    def _process_segment(self, item: Dict[str, Any]):
        """Process a single audio segment with timeout protection."""
        segment_bytes = item["segment_bytes"]
        
        logger.info(f"Session {self.session_id}: Processing segment - "
                   f"Size: {len(segment_bytes)} bytes")
        
        # Create processing subdirectory
        processing_dir = os.path.join(self.temp_dir, "processing")
        os.makedirs(processing_dir, exist_ok=True)
        
        # Thread-based timeout protection
        result = None
        timeout_occurred = threading.Event()
        processing_completed = threading.Event()
        timeout_seconds = 60  # 60-second timeout
        
        def processing_task():
            """Task that performs the actual processing."""
            nonlocal result
            try:
                result = self.vad_service.process_audio_chunk(
                    webm_blob_bytes=segment_bytes,
                    session_id=self.session_id,
                    temp_dir=processing_dir,
                    language_setting=self.language_setting
                )
                processing_completed.set()
            except Exception as e:
                logger.error(f"Session {self.session_id}: Error in processing task: {e}", exc_info=True)
                with self.session_lock:
                    self.processing_stats["errors"].append(f"Processing task error: {e}")
                processing_completed.set()
        
        def timeout_handler():
            """Handler that sets timeout flag after delay."""
            timeout_occurred.set()
            logger.error(f"Session {self.session_id}: Segment processing timed out after {timeout_seconds}s")
        
        try:
            # Start processing task in a separate thread
            processing_thread = threading.Thread(target=processing_task, name=f"Process-{self.session_id[:8]}")
            processing_thread.start()
            
            # Start timeout timer
            timeout_timer = threading.Timer(timeout_seconds, timeout_handler)
            timeout_timer.start()
            
            # Wait for either processing completion or timeout
            processing_completed.wait(timeout=timeout_seconds + 5)  # Extra 5s grace period
            
            # Cancel timeout timer if processing completed
            timeout_timer.cancel()
            
            # Check results
            if timeout_occurred.is_set():
                logger.error(f"Session {self.session_id}: Segment processing timed out - worker thread may be blocked")
                with self.session_lock:
                    self.processing_stats["errors"].append(f"Segment timeout after {timeout_seconds}s")
                return
            
            if not processing_completed.is_set():
                logger.warning(f"Session {self.session_id}: Processing thread did not complete within grace period")
                with self.session_lock:
                    self.processing_stats["errors"].append("Processing thread hung - grace period exceeded")
                return
            
            # Wait for processing thread to finish (should be immediate at this point)
            processing_thread.join(timeout=2.0)
            if processing_thread.is_alive():
                logger.error(f"Session {self.session_id}: Processing thread still alive after completion signal")
            
            # Process results if we have them
            if result:
                # Update statistics
                with self.session_lock:
                    self.processing_stats["chunks_processed"] += 1
                    self.processing_stats["total_processing_time_ms"] += result.get("processing_times", {}).get("total", 0)
                    
                    if result.get("has_voice"):
                        self.processing_stats["chunks_with_voice"] += 1
                        
                    if result.get("transcription"):
                        self.processing_stats["chunks_transcribed"] += 1
                    
                    if result.get("errors"):
                        self.processing_stats["errors"].extend(result["errors"])
                
                # Use the callback to pass the result back to the bridge
                if result.get("transcription") and result.get("has_voice"):
                    # Add the path to the clean WAV file for the handler
                    result["clean_wav_path"] = os.path.join(processing_dir, f"clean_chunk_{result['chunk_id']}.wav")
                    self.result_callback(result)
                else:
                    logger.debug(f"Session {self.session_id}: Segment processed but no transcription - "
                               f"Voice: {result.get('has_voice')}, "
                               f"Skip reason: {result.get('skip_reason', 'unknown')}")
            else:
                logger.warning(f"Session {self.session_id}: Processing completed but no result returned")
                with self.session_lock:
                    self.processing_stats["errors"].append("Processing completed but no result")
        
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error in timeout-protected processing: {e}", exc_info=True)
            with self.session_lock:
                self.processing_stats["errors"].append(f"Timeout protection error: {e}")

    def _process_final_segment(self):
        """Process any remaining audio data when stopping."""
        with self.session_lock:
            if self.current_segment_bytes:
                logger.info(f"Session {self.session_id}: Processing final segment - "
                           f"Size: {len(self.current_segment_bytes)} bytes")
                self._queue_current_segment_for_processing()
        
        # Wait for queue to empty
        try:
            self.audio_queue.join()
            logger.info(f"Session {self.session_id}: All queued segments processed")
        except Exception as e:
            logger.warning(f"Session {self.session_id}: Error waiting for queue completion: {e}")

    def _cleanup_session(self):
        """Clean up session resources."""
        logger.info(f"Session {self.session_id}: Starting session cleanup")
        
        # Log final statistics
        stats = self.processing_stats
        logger.info(f"Session {self.session_id}: Final Statistics:")
        logger.info(f"  Chunks Received: {stats['chunks_received']}")
        logger.info(f"  Chunks Processed: {stats['chunks_processed']}")
        logger.info(f"  Chunks with Voice: {stats['chunks_with_voice']}")
        logger.info(f"  Chunks Transcribed: {stats['chunks_transcribed']}")
        logger.info(f"  Total Processing Time: {stats['total_processing_time_ms']:.1f}ms")
        logger.info(f"  Errors: {len(stats['errors'])}")
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Session {self.session_id}: Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Session {self.session_id}: Failed to clean up temp directory: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        with self.session_lock:
            return dict(self.processing_stats)


class VADTranscriptionManager:
    """
    Manages multiple VAD transcription sessions with global configuration.
    Provides session lifecycle management and resource coordination.
    """
    
    def __init__(self, openai_api_key: str, base_temp_dir: str = "tmp_vad_audio_sessions",
                 vad_aggressiveness: int = 2):
        """
        Initialize the VAD transcription manager.
        
        Args:
            openai_api_key: OpenAI API key for Whisper
            base_temp_dir: Base directory for session temporary files
            vad_aggressiveness: Default VAD aggressiveness level
        """
        self.openai_api_key = openai_api_key
        self.base_temp_dir = base_temp_dir
        self.vad_aggressiveness = vad_aggressiveness
        
        # Create base temp directory
        os.makedirs(self.base_temp_dir, exist_ok=True)
        
        # Initialize VAD service
        self.vad_service = VADTranscriptionService(
            openai_api_key=openai_api_key,
            vad_aggressiveness=vad_aggressiveness
        )
        
        # Active sessions
        self.active_sessions: Dict[str, "SessionAudioProcessor"] = {}
        self.sessions_lock = threading.RLock()
        
        logger.info(f"VAD Transcription Manager initialized")
        logger.info(f"Base temp dir: {self.base_temp_dir}")
        logger.info(f"VAD aggressiveness: {self.vad_aggressiveness}")

    def create_session(self, session_id: str, language_setting: str = "any",
                      segment_duration_target: float = 15.0,
                      result_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> "SessionAudioProcessor":
        """
        Create a new transcription session.
        
        Args:
            session_id: Unique session identifier
            language_setting: Language setting for transcription
            segment_duration_target: Target segment duration in seconds
            result_callback: Optional callback for transcription results.
            
        Returns:
            SessionAudioProcessor instance
        """
        with self.sessions_lock:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already exists")
                return self.active_sessions[session_id]
            
            # Create session temp directory
            session_temp_dir = os.path.join(self.base_temp_dir, session_id)
            
            # Define a default no-op callback if none is provided
            def no_op_callback(data):
                logger.debug(f"Session {session_id}: No-op callback for result: {data.get('chunk_id')}")

            # Create session processor
            processor = SessionAudioProcessor(
                session_id=session_id,
                temp_dir=session_temp_dir,
                vad_service=self.vad_service,
                result_callback=result_callback or no_op_callback,
                language_setting=language_setting,
                segment_duration_target=segment_duration_target
            )
            
            self.active_sessions[session_id] = processor
            
            logger.info(f"Created session {session_id} with language '{language_setting}' "
                       f"and {segment_duration_target}s segments")
            
            return processor

    def get_session(self, session_id: str) -> Optional["SessionAudioProcessor"]:
        """Get an existing session."""
        with self.sessions_lock:
            return self.active_sessions.get(session_id)

    def destroy_session(self, session_id: str):
        """Destroy a session and clean up resources."""
        with self.sessions_lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found for destruction")
                return
            
            processor = self.active_sessions[session_id]
            
            # Stop the processor
            processor.stop()
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Destroyed session {session_id}")

    def get_all_session_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active sessions."""
        with self.sessions_lock:
            return {
                session_id: processor.get_statistics()
                for session_id, processor in self.active_sessions.items()
            }

    def cleanup_all_sessions(self):
        """Clean up all active sessions."""
        with self.sessions_lock:
            session_ids = list(self.active_sessions.keys())
            
        for session_id in session_ids:
            self.destroy_session(session_id)
        
        logger.info("All sessions cleaned up")
