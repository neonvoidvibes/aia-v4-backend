import sounddevice as sd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import requests
import queue
from scipy.io import wavfile
import boto3
import io
import openai
import webrtcvad
import re
from dotenv import load_dotenv
import os
import traceback
from datetime import timezone
import pytz

load_dotenv()

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

class MagicAudio:
    def __init__(self, agent=None, event=None, language=None, user_id="0112", project_name="River"):
        self.user_id = user_id
        self.project_name = project_name
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.language = language
        self.agent = agent
        self.event = event

        # Timezone configuration
        self.local_tz = pytz.timezone('UTC')  # Default to UTC
        try:
            self.local_tz = pytz.timezone(time.tzname[0])  # Try to get system timezone
        except Exception as e:
            print(f"Could not determine local timezone, using UTC: {e}")

        # Session timing
        self.session_start_time = datetime.now(self.local_tz)
        self.session_start_utc = self.session_start_time.astimezone(timezone.utc)

        # Get credentials from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise RuntimeError('No OPENAI_API_KEY set in environment')
        openai.api_key = self.openai_api_key

        # AWS configuration from environment
        self.region_name = os.getenv('AWS_REGION')
        self.bucket_name = os.getenv('AWS_S3_BUCKET')
        if not all([self.region_name, self.bucket_name]):
            raise RuntimeError('AWS configuration missing in environment')

        self.s3_client = boto3.client(
            's3',
            region_name=self.region_name,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        # Audio settings
        self.fs = 16000
        self.buffer_duration = 30
        self.chunk_duration = 10
        self.buffer_size = int(self.buffer_duration * self.fs)
        self.audio_buffer = np.zeros(self.buffer_size, dtype='int16')

        # Threading and queue setup
        self.buffer_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.transcription_queue = queue.Queue(maxsize=50)  # Limit queue size to prevent memory issues
        self.processed_queue = queue.Queue(maxsize=50)
        self.SENTINEL = object()
        self.num_workers = 4      # Increased number of workers
        self.max_retries = 3
        self.transcription_timeout = 30  # Timeout for transcription attempts

        # Session and file management
        base_filename = f"transcript_D{self.session_start_time.strftime('%Y%m%d')}-T{self.session_start_time.strftime('%H%M%S')}_uID-{self.user_id}_oID-river_aID-{self.agent or 'none'}_eID-{self.event or 'none'}_sID-{self.session_id}.txt"
        if self.agent and self.event:
            self.transcript_filename = f"organizations/river/agents/{self.agent}/events/{self.event}/transcripts/{base_filename}"
        else:
            self.transcript_filename = f"_files/transcripts/{base_filename}"

        # State management
        self.transcribed_segments = []
        self.transcript_content = ""

        # Recording state
        self.is_paused = False
        self.pause_start_time = None
        self.last_silence_time = None
        self.silence_logged = False

        # Silence tracking
        self.consecutive_silence_count = 0
        self.last_active_timestamp = None
        self.silence_threshold = 3  # Number of consecutive silent chunks before considering a silence gap

        # Locks and processing settings
        self.lock = threading.Lock()
        self.vad = webrtcvad.Vad(2)
        self.threads = []
        self.energy_threshold = 1000

        # Initialize files
        self.write_header()

        # Initialize total paused time
        self.total_paused_time = 0

    def audio_callback(self, indata, frames, time_info, status):
        if self.stop_event.is_set():
            raise sd.CallbackStop()
        with self.buffer_lock:
            self.audio_buffer[:-frames] = self.audio_buffer[frames:]
            self.audio_buffer[-frames:] = indata[:, 0]

    def record_audio(self):
        try:
            with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.fs, dtype='int16'):
                while not self.stop_event.is_set():
                    if self.pause_event.is_set():
                        self.pause_event.wait()
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in record_audio: {e}")

    def is_silent(self, audio_segment):
        audio_bytes = audio_segment.tobytes()
        frames = self.frame_generator(30, audio_bytes)
        frames = list(frames)
        if not frames:
            self.consecutive_silence_count += 1
            self._check_silence_marker()
            return True

        # Calculate energy
        energy = np.sum(np.abs(audio_segment))
        if energy < self.energy_threshold:
            self.consecutive_silence_count += 1
            self._check_silence_marker()
            return True

        # If paused, treat as silent without checking VAD or energy
        if self.pause_event.is_set():
             # Don't increment consecutive silence count during pause
             # self.consecutive_silence_count += 1
             # self._check_silence_marker() # Don't log silence markers during pause
             return True

        # Use VAD to detect speech
        speech = any(self.vad.is_speech(frame.bytes, self.fs) for frame in frames)
        if not speech:
            self.consecutive_silence_count += 1
            self._check_silence_marker()
            return True

        # Reset silence counter and update last active timestamp when speech detected
        self.consecutive_silence_count = 0
        self.last_active_timestamp = self.get_elapsed_time()
        self.silence_logged = False
        return False

    def frame_generator(self, frame_duration_ms, audio):
        n = int(self.fs * (frame_duration_ms / 1000.0) * 2)  # 16-bit audio
        for i in range(0, len(audio), n):
            yield Frame(audio[i:i+n], i, len(audio))

    def segment_audio(self):
        try:
            buffer_filled = False
            segment_count = 0
            last_segment_time = 0

            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    # If paused, reset last_segment_time to ensure the next segment starts fresh
                    # This prevents processing old data immediately after resume.
                    # Also reset silence counter.
                    if last_segment_time != -1: # Only reset once per pause start
                        print("Segmenter paused, resetting segment timer and silence count.")
                        last_segment_time = -1 # Use -1 to indicate reset state
                        self.consecutive_silence_count = 0
                        self.silence_logged = False
                    time.sleep(0.1)
                    continue

                # If resuming from pause, reset last_segment_time to current time
                if last_segment_time == -1:
                    last_segment_time = self.get_elapsed_time()
                    print(f"Segmenter resumed, setting last segment time to {self.format_time(last_segment_time)}")

                current_time = self.get_elapsed_time()
                elapsed_since_last = current_time - last_segment_time

                if not buffer_filled and current_time >= self.chunk_duration:
                    buffer_filled = True

                if buffer_filled and elapsed_since_last >= self.chunk_duration:
                    with self.buffer_lock:
                        samples_to_extract = int(self.chunk_duration * self.fs)
                        audio_segment = np.copy(self.audio_buffer[-samples_to_extract:])

                    if not self.is_silent(audio_segment) and not self.pause_event.is_set():
                        segment_time = current_time - self.chunk_duration
                        audio_filename = f"audio_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        self.save_audio(audio_filename, audio_segment)

                        try:
                            print(f"Processing chunk at time {self.format_time(segment_time)}")
                            self.transcription_queue.put(([(audio_filename, segment_time)], segment_time))
                        except queue.Full:
                            print("Warning: Transcription queue full, skipping chunk")
                            self.cleanup_audio_file(audio_filename)

                    segment_count += 1
                    last_segment_time = current_time

                time.sleep(0.1)

            # Process final chunk if needed and not paused
            if buffer_filled and not self.pause_event.is_set():
                with self.buffer_lock:
                    samples_to_extract = int(self.chunk_duration * self.fs)
                    audio_segment = np.copy(self.audio_buffer[-samples_to_extract:])

                if not self.is_silent(audio_segment):
                    segment_time = self.get_elapsed_time() - self.chunk_duration
                    audio_filename = f"audio_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    self.save_audio(audio_filename, audio_segment)

                    try:
                        print(f"Processing final chunk at time {self.format_time(segment_time)}")
                        self.transcription_queue.put(([(audio_filename, segment_time)], segment_time))
                        self.transcription_queue.join()
                    except queue.Full:
                        print("Warning: Could not process final chunk")
                        self.cleanup_audio_file(audio_filename)

            # Add sentinel values after all chunks are processed
            print("Adding sentinel values to transcription queue")
            for _ in range(self.num_workers):
                self.transcription_queue.put(self.SENTINEL)

        except Exception as e:
            print(f"Error in segment_audio: {e}")
            traceback.print_exc()

    def save_audio(self, filename, audio_data):
        buffer = io.BytesIO()
        wavfile.write(buffer, self.fs, audio_data)
        buffer.seek(0)
        try:
            self.s3_client.upload_fileobj(buffer, self.bucket_name, filename)
            print(f"Audio segment uploaded to S3: {filename}")
        except Exception as e:
            print(f"Error uploading audio segment to S3: {e}")
        finally:
            buffer.close()

    def transcribe_audio_worker(self):
        while True:
            try:
                print(f"Worker waiting for next batch...")
                item = self.transcription_queue.get(timeout=5)  # Add timeout to queue get
                if item == self.SENTINEL:
                    print("Worker received SENTINEL, stopping")
                    self.transcription_queue.task_done()
                    break

                try:
                    batch_list, batch_start_time = item
                    print(f"Worker processing batch from time {self.format_time(batch_start_time)}")
                    batch_transcription = []
                    failed_files = []

                    for audio_filename, segment_time in batch_list:
                        retries = 0
                        success = False

                        while retries < self.max_retries and not success:
                            try:
                                print(f"Attempting to transcribe {audio_filename}")
                                transcription = self.transcribe_audio(audio_filename, self.openai_api_key)

                                if transcription:
                                    print(f"Successfully transcribed {audio_filename}")
                                    batch_transcription.append((transcription, segment_time))
                                    success = True
                                else:
                                    print(f"No transcription returned for {audio_filename}")
                                    retries += 1

                            except requests.exceptions.Timeout:
                                print(f"Timeout transcribing {audio_filename}")
                                retries += 1
                            except Exception as e:
                                print(f"Error transcribing {audio_filename}: {e}")
                                retries += 1

                            if not success:
                                if retries < self.max_retries:
                                    time.sleep(min(2 ** retries, 8))  # Exponential backoff
                                else:
                                    failed_files.append(audio_filename)
                                    print(f"Failed to transcribe {audio_filename} after {self.max_retries} attempts")

                        # Clean up the audio file
                        try:
                            self.cleanup_audio_file(audio_filename)
                        except Exception as e:
                            print(f"Error cleaning up {audio_filename}: {e}")

                    if batch_transcription:
                        print(f"Sending batch transcription to processed queue")
                        self.processed_queue.put((batch_transcription, batch_start_time))

                    if failed_files:
                        print(f"Failed to transcribe {len(failed_files)} files in batch")

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.transcription_queue.task_done()

            except queue.Empty:
                continue  # Keep worker alive if queue is empty
            except Exception as e:
                print(f"Critical error in worker: {e}")
                import traceback
                traceback.print_exc()

    def process_transcription_results(self):
        """Process transcription results in order"""
        try:
            pending_results = {}
            next_expected_time = 0
            next_process_time = 0
            last_reset_time = 0
            queue_timeout = 5  # Timeout for queue operations in seconds
            print("Starting result processor")

            while True:
                try:
                    current_time = self.get_elapsed_time()

                    # Check for extended silence and reset timing if needed
                    if self.consecutive_silence_count >= self.silence_threshold:
                        if current_time - last_reset_time > self.chunk_duration:
                            next_process_time = int(current_time / self.chunk_duration) * self.chunk_duration
                            next_expected_time = next_process_time + self.chunk_duration
                            last_reset_time = current_time
                            print(f"Resetting timing due to silence. Next process time: {self.format_timestamp_range(next_process_time, next_process_time + self.chunk_duration)}, expected: {self.format_timestamp_range(next_expected_time, next_expected_time + self.chunk_duration)}")

                    print(f"Waiting for next result (processing time: {self.format_timestamp_range(next_process_time, next_process_time + self.chunk_duration)}, expected: {self.format_timestamp_range(next_expected_time, next_expected_time + self.chunk_duration)})")

                    try:
                        batch_transcription, batch_start_time = self.processed_queue.get(timeout=queue_timeout)
                    except queue.Empty:
                        # If we're still recording
                        if not self.stop_event.is_set():
                            if self.consecutive_silence_count >= self.silence_threshold:
                                # During silence, just continue waiting
                                continue
                            else:
                                # During active recording, advance the expected time
                                expected_chunks = int(current_time / self.chunk_duration)
                                next_expected_time = expected_chunks * self.chunk_duration
                                print(f"No results yet, advancing expected time to: {self.format_timestamp_range(next_expected_time, next_expected_time + self.chunk_duration)}")
                        continue

                    # Check for sentinel
                    if batch_transcription is None and batch_start_time is None:
                        print("Received sentinel in processor, finishing up...")
                        # Process any remaining results in order
                        for time_key in sorted(pending_results.keys()):
                            self._process_batch_result(pending_results[time_key], time_key)
                        break

                    print(f"Got result for time {self.format_timestamp_range(batch_start_time, batch_start_time + self.chunk_duration)}")
                    pending_results[batch_start_time] = batch_transcription

                    # Process results that are ready
                    while pending_results:
                        earliest_time = min(pending_results.keys())

                        # If this is the next result we expect, or we're behind, process it
                        if earliest_time >= next_process_time:
                            print(f"Processing result for time {self.format_timestamp_range(earliest_time, earliest_time + self.chunk_duration)}")
                            transcriptions = pending_results.pop(earliest_time)
                            self._process_batch_result(transcriptions, earliest_time)
                            next_process_time = earliest_time + self.chunk_duration
                            next_expected_time = next_process_time + self.chunk_duration
                            print(f"Updated timing - Process: {self.format_timestamp_range(next_process_time, next_process_time + self.chunk_duration)}")
                        else:
                            # We have an earlier result than expected
                            print(f"Processing out-of-order result for time {self.format_timestamp_range(earliest_time, earliest_time + self.chunk_duration)}")
                            transcriptions = pending_results.pop(earliest_time)
                            self._process_batch_result(transcriptions, earliest_time)

                except Exception as e:
                    print(f"Error in process_transcription_results loop: {e}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Error in process_transcription_results: {e}")
            traceback.print_exc()

    def _process_batch_result(self, transcriptions, start_time):
        """Helper method to process a batch of transcription results"""
        for transcription, segment_time in transcriptions:
            segments = transcription.get('segments', [])
            full_text = ""
            last_adjusted_end = segment_time

            for segment in segments:
                adjusted_start = segment['start'] + segment_time
                adjusted_end = segment['end'] + segment_time
                raw_transcribed_text = segment['text'].strip() # Get raw text first

                # Apply post-processing filter for hallucinations
                transcribed_text = self.filter_hallucinations(raw_transcribed_text)

                # Check validity *after* filtering
                if self.is_valid_transcription(transcribed_text):
                    timestamp_range = self.format_timestamp_range(adjusted_start, adjusted_end)
                    full_text += f"{timestamp_range} {transcribed_text}\n"
                    self.transcribed_segments.append({
                        'start': adjusted_start,
                        # Ensure end time doesn't exceed the expected chunk boundary for text format
                        'end': min(adjusted_end, segment_time + self.chunk_duration),
                        'text': transcribed_text, # Use filtered text
                    })
                    # Use the calculated adjusted_end, even if capped above, for tracking last segment end
                    last_adjusted_end = min(adjusted_end, segment_time + self.chunk_duration)
                else:
                     print(f"Filtered out segment: '{raw_transcribed_text}'") # Log filtered segments

            if full_text: # Check if anything remains after filtering
                print(f"Adding transcription to content: {full_text[:100]}...")
                with self.lock:
                    self.transcript_content += full_text
                    self.upload_transcript()

    def is_valid_transcription(self, text):
        # Check for non-sensical content
        if not text:
            return False
        # Disallow lines that are only emojis or non-alphanumerics
        if re.fullmatch(r'[^\w\s]+', text):
            return False
        # Optionally, set length constraints
        if len(text) < 2:  # Too short to be meaningful
            return False
        return True

    def transcribe_audio(self, audio_filename, api_key):
        buffer = io.BytesIO()
        try:
            self.s3_client.download_fileobj(self.bucket_name, audio_filename, buffer)
            buffer.seek(0)
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {api_key}"}

            # Prepare form data fields
            data = {
                'model': 'whisper-1',
                'response_format': 'text',
                'temperature': 0.0, # Set temperature to reduce randomness
            }

            # Add language parameter OR initial_prompt to data, but not both
            if self.language:
                data['language'] = self.language
                # When language is set, DO NOT send initial_prompt
            else:
                # Only send initial_prompt when language is NOT set (auto-detect)
                data['initial_prompt'] = '''===System Prompt===
You are a highly precise speech transcriber. Transcribe the following audio in the correct language. Follow these guidelines strictly:

1. Only transcribe actual speech that is present in the audio
   - Ignore out-of-context greetings, conclusions, or conversational markers used (examples: "Thank you.", "Thanks for watching!", "Thank you for watching!", "Good night.", "Very good.", etc.)
   - Ignore out-of-context languages used (examples: "おやすみなさい。", etc.)

2. For silent segments:
   - Do not generate filler text or common phrases
   - Do not insert greetings, conclusions, or conversational markers
   - Do not insert other languages
   - If no speech is detected for a whole chunk, return "<<Extended silence>>"

3. For actual speech:
   - Transcribe exactly what is said without adding extra repetitions
   - If a phrase seems to repeat unnaturally, transcribe it only once
   - Stop transcription if the same word/phrase appears to loop

4. Ignore background noise, breaths, and non-speech sounds
5. Maintain accuracy over completion - it's better to transcribe nothing than to hallucinate content

Remember: The goal is exact transcription of real speech only, not generating plausible conversation.'''

            # Add parameters to reduce hallucinations/repetition
            data['logprob_threshold'] = -0.7 # Make slightly stricter
            data['compression_ratio_threshold'] = 2.2
            data['no_speech_threshold'] = 0.7 # Increase threshold for detecting speech

            # Prepare the file part separately
            files_param = {
                'file': (audio_filename, buffer, 'audio/wav')
            }

            # Make the request with separate 'data' and 'files'
            print(f"Sending request to Whisper API with data: {data}") # Debug: Print parameters being sent
            response = requests.post(url, headers=headers, data=data, files=files_param, timeout=self.transcription_timeout)
            response.raise_for_status()

            # Handle text response format (using the 'data' dict to check format)
            if data['response_format'] == 'text':
                transcribed_text = response.text.strip()
                if transcribed_text and transcribed_text.lower() != "<<extended silence>>": # Avoid adding silence markers as transcriptions
                    # Create a simplified structure similar to verbose_json format
                    transcription = {
                        'segments': [{
                            'start': 0,  # We don't get timing info in text format
                            'end': self.chunk_duration, # Use chunk duration
                            'text': transcribed_text
                        }]
                    }
                    print(f"Transcription successful (text format) for {audio_filename}.")
                    return transcription
                else:
                    print(f"Transcription returned empty or silence marker for {audio_filename}.")
                    return None # Treat silence markers or empty results as no transcription

            # Handle verbose_json response format (or others like json, srt, vtt)
            else:
                # Assuming JSON-based format like verbose_json or json
                try:
                    transcription = response.json()
                    print(f"Transcription successful (json format) for {audio_filename}.")
                    # Basic validation: check if it has 'text' or 'segments'
                    if 'text' in transcription or 'segments' in transcription:
                         return transcription
                    else:
                         print(f"Unexpected JSON structure for {audio_filename}: {transcription}")
                         return None
                except requests.exceptions.JSONDecodeError:
                     # Handle non-JSON text formats like srt, vtt
                     transcribed_text = response.text.strip()
                     if transcribed_text:
                         # For simplicity, wrap non-JSON text in basic segment structure
                         transcription = {
                             'segments': [{
                                 'start': 0,
                                 'end': self.chunk_duration,
                                 'text': transcribed_text
                             }]
                         }
                         print(f"Transcription successful (non-JSON text format) for {audio_filename}.")
                         return transcription
                     else:
                         print(f"Transcription returned empty text for {audio_filename}.")
                         return None

        except Exception as e:
            print(f"Transcription error for {audio_filename}: {e}")
            # Add more detail for debugging network/API errors
            if isinstance(e, requests.exceptions.RequestException):
                 print(f"Response status: {getattr(e.response, 'status_code', 'N/A')}, Response text: {getattr(e.response, 'text', 'N/A')}")
            return None
        finally:
            buffer.close()

    def filter_hallucinations(self, text):
        """Remove known hallucination patterns from transcribed text."""
        # Patterns to filter (case-insensitive)
        # Using raw strings (r"...") and word boundaries (\b) where appropriate
        patterns = [
            r"^\s*Över\.?\s*$",                             # Line is just "Över."
            r"Översättning av.*",                          # Subtitling credits
            r"www\.sdimedia\.com",                         # Subtitling credits URL
            r"^\s*Svensktextning\.nu\s*$",                  # Subtitling credits
            r"^\s*Tack (för|till).*(tittade|hjälpte).*",    # Broader "Thanks for watching/helping" variants
            r"^\s*Trio(,\s*Trio)*\.?\s*$",                  # Repetitive "Trio"
            r"^\s*(Ja|Nej)(,\s*(Ja|Nej))*\.?\s*$",         # Repetitive "Ja" or "Nej"
            # Add more specific patterns here if needed
        ]

        # Original text for logging comparison
        original_text_repr = repr(text)

        # Normalize text for comparison (lowercase, remove extra whitespace)
        normalized_text = ' '.join(text.lower().split())

        for pattern in patterns:
            # Use re.IGNORECASE for case-insensitivity
            match = re.search(pattern, normalized_text, re.IGNORECASE)
            if match:
                print(f"Filter: Matched pattern '{pattern}' on text {original_text_repr}. Filtering out.")
                return "" # Return empty string if a pattern matches

        # If no patterns matched
        # print(f"Filter: No pattern matched for text {original_text_repr}. Keeping.") # Optional: Log kept text
        return text # Return original text if no patterns match

    def upload_transcript(self):
        try:
            buffer = io.BytesIO(self.transcript_content.encode('utf-8'))
            self.s3_client.put_object(Bucket=self.bucket_name, Key=self.transcript_filename, Body=buffer.getvalue())
            print(f"Transcript uploaded to S3: {self.transcript_filename}")
        except Exception as e:
            print(f"Error uploading transcript to S3: {e}")
        finally:
            buffer.close()

    def write_header(self):
        header = f"# Transcript - Session {self.session_id}\n\n"
        self.transcript_content += header
        self.upload_transcript()

    def cleanup_audio_file(self, filename):
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
            print(f"Audio file deleted from S3: {filename}")
        except Exception as e:
            print(f"Error deleting file {filename} from S3: {e}")

    def cleanup_remaining_files(self):
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=f'audio_{self.session_id}')
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.wav'):
                            self.cleanup_audio_file(key)
        except Exception as e:
            print(f"Error cleaning up remaining files in S3 for session {self.session_id}: {e}")

    def get_current_time(self):
        """Get current time in UTC"""
        return datetime.now(timezone.utc)

    def get_elapsed_time(self):
        """Get elapsed time since session start in seconds, accounting for pauses."""
        if not self.session_start_utc: return 0
        now = self.get_current_time()
        elapsed_since_start = (now - self.session_start_utc).total_seconds()
        # If paused, calculate time up to the pause start
        if self.is_paused and self.pause_start_time:
            # This requires pause_start_time to be an absolute time (like time.time())
            # Or, if session_start_utc is adjusted on resume, this works:
             elapsed_since_start = (datetime.fromtimestamp(self.pause_start_time, timezone.utc) - self.session_start_utc).total_seconds()
        return max(0, elapsed_since_start)


    def get_timestamp(self, elapsed_seconds):
        """Convert elapsed seconds to UTC timestamp"""
        return self.session_start_utc + timedelta(seconds=elapsed_seconds)

    def format_time(self, seconds):
        """Format elapsed seconds as HH:MM:SS with timezone"""
        if seconds is None:
            return "00:00:00"

        timestamp = self.get_timestamp(seconds)
        local_time = timestamp.astimezone(self.local_tz)
        return local_time.strftime("%H:%M:%S")

    def format_timestamp_range(self, start_seconds, end_seconds):
        """Format a time range with timezone"""
        start_time = self.format_time(start_seconds)
        end_time = self.format_time(end_seconds)
        local_time = self.session_start_utc.astimezone(self.local_tz)
        tz_name = local_time.strftime('%Z')
        return f"[{start_time} - {end_time} {tz_name}]"

    def pause(self):
        """Pause recording and audio processing."""
        if not self.is_paused:
            self.pause_event.set() # Signal threads to pause processing/segmenting
            self.is_paused = True
            self.pause_start_time = time.time() # Record when pause started for duration calculation
            self._add_pause_marker("<<Recording paused>>") # Add marker to transcript
            print("MagicAudio: Paused.")
        else:
            print("MagicAudio: Already paused.")

    def resume(self):
        """Resume recording and audio processing."""
        if self.is_paused:
            # Calculate pause duration and adjust start time for accurate elapsed time
            if self.pause_start_time and self.session_start_utc:
                 pause_duration = time.time() - self.pause_start_time
                 print(f"MagicAudio: Resuming after {pause_duration:.2f}s pause.")
                 # Adjust session start time forward by the pause duration
                 # This ensures get_elapsed_time calculates correctly after resume
                 self.session_start_utc += timedelta(seconds=pause_duration)
                 print(f"MagicAudio: Adjusted session start time to {self.session_start_utc}")
            else:
                 print("MagicAudio: Resuming, but couldn't calculate exact pause duration.")

            self.is_paused = False
            self.pause_start_time = None # Clear pause start time
            self.pause_event.clear() # Allow threads to continue
            self.silence_logged = False # Reset silence log flag
            self._add_pause_marker("<<Recording resumed>>") # Add marker to transcript
            # Optional: Log current time marker immediately after resume
            # current_time = self.get_elapsed_time()
            # timestamp = self.format_timestamp_range(current_time, current_time + self.chunk_duration)
            # marker_text = f"{timestamp} Current time: {self.format_time(current_time)}\n"
            # with self.lock:
            #     self.transcript_content += marker_text
            #     self.upload_transcript()
            print("MagicAudio: Resumed.")
        else:
             print("MagicAudio: Not currently paused.")

    def _add_pause_marker(self, message):
        """Add a pause marker to the transcript"""
        current_time = self.get_elapsed_time()
        timestamp = self.format_timestamp_range(current_time, current_time + self.chunk_duration)
        marker_text = f"{timestamp} {message}\n"
        with self.lock:
            self.transcript_content += marker_text
            self.upload_transcript()

    def _check_silence_marker(self):
        """Check if we should add a silence marker"""
        if self.consecutive_silence_count == self.silence_threshold and not self.silence_logged:
            self._add_pause_marker("<<Extended silence>>")
            self.silence_logged = True

    def start(self):
        self.stop_event.clear()
        self.pause_event.clear()
        # Start recording thread
        record_thread = threading.Thread(target=self.record_audio, name='recording')
        record_thread.start()
        self.threads.append(record_thread)

        # Start segmentation thread
        segment_thread = threading.Thread(target=self.segment_audio, name='segmentation')
        segment_thread.start()
        self.threads.append(segment_thread)

        # Start multiple transcription worker threads
        for i in range(self.num_workers):
            worker_thread = threading.Thread(target=self.transcribe_audio_worker, name=f'worker-{i+1}')
            worker_thread.daemon = True
            worker_thread.start()
            self.threads.append(worker_thread)

        # Start result processor thread
        processor_thread = threading.Thread(target=self.process_transcription_results, name='processor')
        processor_thread.daemon = True
        processor_thread.start()
        self.threads.append(processor_thread)

    def stop(self):
        try:
            print("Stop requested, waiting for processing to complete...")
            self.stop_event.set()

            # Wait for audio segmentation to finish
            for thread in self.threads:
                if thread is not None and thread.is_alive() and thread.name == 'segmentation':
                    print("Waiting for segmentation to complete...")
                    thread.join(timeout=10)

            # Wait for all queued items to be processed
            print("Waiting for transcription queue to empty...")
            try:
                self.transcription_queue.join()
            except Exception as e:
                print(f"Error waiting for transcription queue: {e}")

            print("Waiting for processed queue to empty...")
            try:
                # Wait for processor to finish current batch
                time.sleep(2)
                # Add a sentinel to the processed queue to ensure the processor exits
                self.processed_queue.put((None, None))
            except Exception as e:
                print(f"Error handling processed queue: {e}")

            # Wait for threads to complete with timeout
            for thread in self.threads:
                if thread is not None and thread.is_alive():
                    thread_name = getattr(thread, 'name', 'unknown')
                    print(f"Waiting for thread {thread_name} to complete...")
                    thread.join(timeout=30)

            print("Cleanup remaining files...")
            self.cleanup_remaining_files()

            self.threads = []
            print("Stop completed")

        except Exception as e:
            print(f"Error in stop: {e}")
            traceback.print_exc()