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
from typing import Dict, Any, Optional, List, Tuple
import wave

import webrtcvad
import openai
import requests

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
        
        Args:
            wav_path: Path to WAV file to analyze
            session_id: Session ID for logging context
            
        Returns:
            Tuple of (voiced_audio_bytes: bytes, vad_details: Dict[str, Any])
        """
        logger.debug(f"Session {session_id}: Starting VAD analysis and extraction on {wav_path}")
        
        voiced_audio_data = bytearray()
        
        vad_details = {
            "total_frames": 0,
            "voiced_frames": 0,
            "voiced_frame_ratio": 0.0,
            "processing_time_ms": 0,
            "frame_duration_ms": self.vad_frame_duration_ms,
            "aggressiveness": self.vad_aggressiveness
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
            
            voiced_frames, total_frames = 0, 0
            
            for i in range(0, len(audio_data) - self.vad_frame_bytes + 1, self.vad_frame_bytes):
                frame = audio_data[i:i + self.vad_frame_bytes]
                total_frames += 1
                
                try:
                    if self.vad.is_speech(frame, sample_rate):
                        voiced_frames += 1
                        voiced_audio_data.extend(frame)
                except Exception as frame_error:
                    logger.warning(f"Session {session_id}: VAD error on frame {total_frames}: {frame_error}")
                    continue
            
            processing_time = (time.time() - start_time) * 1000
            voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0
            
            vad_details.update({
                "total_frames": total_frames, "voiced_frames": voiced_frames,
                "voiced_frame_ratio": voiced_ratio, "processing_time_ms": processing_time
            })
            
            logger.info(f"Session {session_id}: VAD Analysis Complete. Total Frames: {total_frames}, Voiced: {voiced_frames}, Ratio: {voiced_ratio:.3f}. Extracted {len(voiced_audio_data)} voiced bytes.")
            
            return bytes(voiced_audio_data), vad_details
            
        except Exception as e:
            logger.error(f"Session {session_id}: VAD analysis failed: {e}", exc_info=True)
            vad_details["error"] = str(e)
            return b"", vad_details

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
            
            # Step 3: VAD analysis and extraction
            step_start = time.time()
            voiced_audio_bytes, vad_details = self.detect_voice_activity(wav_path, session_id)
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
            # Step 5: Cleanup temporary files
            cleanup_start = time.time()
            for file_path in [webm_path, wav_path, clean_wav_path]:
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
        """Process a single audio segment."""
        segment_bytes = item["segment_bytes"]
        
        logger.info(f"Session {self.session_id}: Processing segment - "
                   f"Size: {len(segment_bytes)} bytes")
        
        # Create processing subdirectory
        processing_dir = os.path.join(self.temp_dir, "processing")
        os.makedirs(processing_dir, exist_ok=True)
        
        try:
            # Process through VAD pipeline
            result = self.vad_service.process_audio_chunk(
                webm_blob_bytes=segment_bytes,
                session_id=self.session_id,
                temp_dir=processing_dir,
                language_setting=self.language_setting
            )
            
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
            
            # Handle transcription result
            if result.get("transcription") and result.get("has_voice"):
                self._handle_transcription_result(result)
            else:
                logger.debug(f"Session {self.session_id}: Segment processed but no transcription - "
                           f"Voice: {result.get('has_voice')}, "
                           f"Skip reason: {result.get('skip_reason', 'unknown')}")
        
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error processing segment: {e}", exc_info=True)
            with self.session_lock:
                self.processing_stats["errors"].append(f"Segment processing error: {e}")

    def _handle_transcription_result(self, result: Dict[str, Any]):
        """Handle successful transcription result."""
        transcription = result.get("transcription", {})
        segments = transcription.get("segments", [])
        
        if not segments:
            logger.debug(f"Session {self.session_id}: No segments in transcription result")
            return
        
        logger.info(f"Session {self.session_id}: Processing {len(segments)} transcription segments")
        
        # Apply existing filtering logic from original transcription service
        from transcription_service import (
            filter_by_duration_and_confidence,
            detect_cross_segment_repetition,
            analyze_silence_gaps,
            filter_hallucinations,
            is_valid_transcription
        )
        
        # Filter segments
        filtered_segments = [s for s in segments if filter_by_duration_and_confidence(s)]
        filtered_segments = detect_cross_segment_repetition(filtered_segments)
        filtered_segments = analyze_silence_gaps(filtered_segments)
        
        # Process filtered segments
        valid_segments = []
        for segment in filtered_segments:
            raw_text = segment.get('text', '').strip()
            filtered_text = filter_hallucinations(raw_text)
            
            if is_valid_transcription(filtered_text):
                segment['filtered_text'] = filtered_text
                valid_segments.append(segment)
        
        if valid_segments:
            logger.info(f"Session {self.session_id}: {len(valid_segments)} valid segments after filtering")
            
            # Store valid segments for bridge to process
            self._store_processed_segments(valid_segments, result)
        else:
            logger.debug(f"Session {self.session_id}: No valid segments after filtering")

    def _store_processed_segments(self, valid_segments: List[Dict[str, Any]], result: Dict[str, Any]):
        """Store processed segments for the integration bridge to handle."""
        # Create a result object that the bridge can process
        processed_result = {
            "session_id": self.session_id,
            "segments": valid_segments,
            "audio_duration_seconds": result.get("audio_duration_seconds", 0.0),
            "processing_times": result.get("processing_times", {}),
            "chunk_id": result.get("chunk_id"),
            "timestamp": time.time()
        }
        
        # Store in a queue or trigger callback for bridge processing
        # This will be handled by the integration bridge
        logger.debug(f"Session {self.session_id}: Stored {len(valid_segments)} processed segments for bridge integration")
        
        # For now, log the transcribed text
        for segment in valid_segments:
            logger.info(f"Session {self.session_id}: VAD Transcribed: '{segment.get('filtered_text', '')}'")

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
        self.active_sessions: Dict[str, SessionAudioProcessor] = {}
        self.sessions_lock = threading.RLock()
        
        logger.info(f"VAD Transcription Manager initialized")
        logger.info(f"Base temp dir: {self.base_temp_dir}")
        logger.info(f"VAD aggressiveness: {self.vad_aggressiveness}")

    def create_session(self, session_id: str, language_setting: str = "any",
                      segment_duration_target: float = 15.0) -> SessionAudioProcessor:
        """
        Create a new transcription session.
        
        Args:
            session_id: Unique session identifier
            language_setting: Language setting for transcription
            segment_duration_target: Target segment duration in seconds
            
        Returns:
            SessionAudioProcessor instance
        """
        with self.sessions_lock:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already exists")
                return self.active_sessions[session_id]
            
            # Create session temp directory
            session_temp_dir = os.path.join(self.base_temp_dir, session_id)
            
            # Create session processor
            processor = SessionAudioProcessor(
                session_id=session_id,
                temp_dir=session_temp_dir,
                vad_service=self.vad_service,
                language_setting=language_setting,
                segment_duration_target=segment_duration_target
            )
            
            self.active_sessions[session_id] = processor
            
            logger.info(f"Created session {session_id} with language '{language_setting}' "
                       f"and {segment_duration_target}s segments")
            
            return processor

    def get_session(self, session_id: str) -> Optional[SessionAudioProcessor]:
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