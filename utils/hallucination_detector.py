"""
Hallucination Detection System for VAD-Filtered Transcription

This module implements a multi-layered approach to detect and prevent Whisper hallucinations
during silence or low-quality audio segments. It maintains transcript history and performs
similarity analysis to identify repeated content patterns.
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from difflib import SequenceMatcher
import unicodedata

logger = logging.getLogger(__name__)

class TranscriptHistoryManager:
    """
    Manages a circular buffer of recent transcriptions for hallucination detection.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize the transcript history manager.
        
        Args:
            max_history: Maximum number of transcripts to keep in history
        """
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.session_start_time = time.time()
        self.initial_transcript: Optional[Dict[str, Any]] = None
        
        logger.info(f"TranscriptHistoryManager initialized with max_history={max_history}")
    
    def add_transcript(self, text: str, timestamp: float = None) -> None:
        """
        Add a new transcript to the history.
        
        Args:
            text: Transcribed text content
            timestamp: Unix timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Normalize text for consistent processing
        normalized_text = self._normalize_text(text)
        
        transcript_entry = {
            'text': text,
            'normalized_text': normalized_text,
            'timestamp': timestamp,
            'relative_time': timestamp - self.session_start_time
        }
        
        # Store the first transcript separately for long-term checks
        if not self.initial_transcript:
            # Limit the initial transcript to the first 7 words for focused checking
            words = normalized_text.split()
            short_text = " ".join(words[:7])
            
            self.initial_transcript = {
                'text': " ".join(text.split()[:7]),
                'normalized_text': short_text,
                'timestamp': timestamp
            }
            logger.info(f"Stored initial transcript phrase: '{short_text}'")

        self.history.append(transcript_entry)
        logger.debug(f"Added transcript to history: '{text[:50]}...' at {timestamp}")
    
    def get_recent_transcripts(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get the most recent n transcripts.
        
        Args:
            n: Number of recent transcripts to return
            
        Returns:
            List of transcript dictionaries
        """
        return list(self.history)[-n:] if len(self.history) >= n else list(self.history)
    
    def get_recent_phrases(self, n: int = 3) -> List[str]:
        """
        Get normalized text from recent transcripts.
        
        Args:
            n: Number of recent transcripts to get
            
        Returns:
            List of normalized text strings
        """
        recent = self.get_recent_transcripts(n)
        return [entry['normalized_text'] for entry in recent]
    
    def clear_history(self) -> None:
        """Clear all transcript history."""
        self.history.clear()
        self.initial_transcript = None
        logger.info("Transcript history cleared")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove punctuation and extra whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common redaction patterns
        normalized = re.sub(r'\[person_redacted\]', '', normalized)
        normalized = re.sub(r'\[.*?\]', '', normalized)  # Remove any bracketed content
        
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', normalized)
        
        return normalized.strip()


class HallucinationDetector:
    """
    Detects hallucinated content by analyzing similarity patterns and repetition.
    """
    
    def __init__(self, similarity_threshold: float = 0.5, min_transcript_length: int = 2):
        """
        Initialize the hallucination detector - AGGRESSIVE ROBUSTNESS MODE.
        
        Args:
            similarity_threshold: Threshold for considering text as similar (0.0-1.0) - AGGRESSIVE: lowered to 0.5 for maximum robustness
            min_transcript_length: Minimum number of words to consider for detection - AGGRESSIVE: lowered to 2 words
        """
        self.similarity_threshold = similarity_threshold
        self.min_transcript_length = min_transcript_length
        
        # Common hallucination patterns (regex patterns)
        self.hallucination_patterns = [
            r'\b(hello|hey|hi)\s+(hello|hey|hi)\b',  # Repeated greetings
            r'\b(\w+)\s+\1\s+\1\b',  # Same word repeated 3+ times
            r'\b(test|testing)\s+(test|testing)\b',  # Test phrases
            r'\b(hej|hallå)\s+(hej|hallå|hejsan)\b',  # Swedish greetings
            r'\b(hejsan)\s+(hejsan)\b',  # Swedish repeated greetings
            r'\bthanks\s+for\s+watching\b',  # YouTube outros
            r'\bsubscribe\s+to\s+my\s+channel\b',  # Social media phrases
            r'\blike\s+and\s+subscribe\b',  # More social media
            r'\bthis\s+video\s+is\s+sponsored\b',  # Sponsored content
        ]
        
        logger.info(f"HallucinationDetector initialized with threshold={similarity_threshold}")
    
    def is_hallucination(self, new_transcript: str, history_manager: TranscriptHistoryManager) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if a new transcript is likely a hallucination.
        
        Args:
            new_transcript: New transcribed text to evaluate
            history_manager: TranscriptHistoryManager instance with history
            
        Returns:
            Tuple of (is_hallucination: bool, reason: str, corrected_transcript: Optional[str])
        """
        if not new_transcript or len(new_transcript.strip()) == 0:
            return True, "empty_transcript", None
        
        # Normalize the new transcript
        normalized_new = history_manager._normalize_text(new_transcript)
        
        # Check minimum length
        word_count = len(normalized_new.split())
        if word_count < self.min_transcript_length:
            logger.debug(f"Transcript too short ({word_count} words): '{new_transcript}'")
            return True, f"too_short_{word_count}_words", None
        
        # Check against known hallucination patterns
        pattern_match = self._check_hallucination_patterns(normalized_new)
        if pattern_match:
            logger.warning(f"Hallucination pattern detected: '{pattern_match}' in '{new_transcript}'")
            return True, f"pattern_match_{pattern_match}", None

        # Check for repetition of the initial phrase of the session dynamically
        if history_manager.initial_transcript and len(history_manager.history) > 0:
            initial_phrase_full_norm = history_manager.initial_transcript['normalized_text']
            initial_phrase_words = initial_phrase_full_norm.split()
            
            # Dynamically check for prefixes of decreasing length (from 7 down to 2)
            for phrase_len in range(min(len(initial_phrase_words), 7), 1, -1):
                phrase_to_check = " ".join(initial_phrase_words[:phrase_len])
                
                if normalized_new.startswith(phrase_to_check):
                    # Found a match. Correct it by stripping the corresponding number of words from the original.
                    num_words_to_strip = len(phrase_to_check.split())
                    original_words = new_transcript.split()
                    
                    if len(original_words) > num_words_to_strip:
                        corrected_words = original_words[num_words_to_strip:]
                        corrected_transcript = " ".join(corrected_words)
                        
                        if len(corrected_words) >= self.min_transcript_length:
                            logger.warning(f"Corrected dynamic initial phrase repetition. Phrase: '{phrase_to_check}', Original: '{new_transcript}', Corrected: '{corrected_transcript}'")
                            return True, "initial_phrase_repetition_corrected", corrected_transcript
                        else:
                            logger.warning(f"Initial phrase repetition detected, but correction is too short. Phrase: '{phrase_to_check}', Original: '{new_transcript}'")
                            return True, "initial_phrase_repetition_uncorrectable", None
                    
                    # If a match was found and handled, break the loop
                    break

        # Check for repetition against the previous transcript. This handles both full and partial repeats.
        is_repetition, corrected_transcript = self._check_for_repeated_prefix(new_transcript, history_manager)
        if is_repetition:
            if corrected_transcript is not None:
                # This was a correctable overlap.
                return True, "prefix_repetition_corrected", corrected_transcript
            else:
                # This was an uncorrectable full match.
                return True, "full_repetition", None
        
        # Check similarity against recent history (this is for full-sentence repetition)
        recent_phrases = history_manager.get_recent_phrases(3)
        if recent_phrases:
            similarity_result = self._check_similarity(normalized_new, recent_phrases)
            if similarity_result[0]:
                logger.warning(f"Similarity hallucination detected: '{new_transcript}' similar to recent transcript")
                return True, f"similarity_{similarity_result[1]:.3f}", None
        
        # Check for internal repetition within the transcript
        internal_repetition = self._check_internal_repetition(normalized_new)
        if internal_repetition:
            logger.warning(f"Internal repetition detected: '{internal_repetition}' in '{new_transcript}'")
            return True, f"internal_repetition_{internal_repetition}", None
        
        return False, "valid", None

    def _check_for_repeated_prefix(self, new_text: str, history_manager: TranscriptHistoryManager) -> Tuple[bool, Optional[str]]:
        """
        Checks for repetition against the most recent transcript.
        
        Returns a tuple of (bool, Optional[str]):
        - (True, str): An overlap was found and corrected. The string is the corrected text.
        - (True, None): A full, exact repetition was found. This is uncorrectable.
        - (False, None): No repetition was found.
        """
        if not history_manager.history:
            return False, None

        original_new_words = new_text.split()
        normalized_new_text = history_manager._normalize_text(new_text)

        # Check only against the most recent transcript in history
        previous_entry = list(history_manager.history)[-1]
        previous_text_normalized = previous_entry['normalized_text']

        # Case 1: Exact match with the previous transcript (full repetition hallucination)
        if normalized_new_text == previous_text_normalized:
            logger.warning(f"Uncorrectable full repetition detected. Text: '{new_text}'")
            # Return True for hallucination, but None for corrected_transcript to signal it should be dropped.
            return True, None

        # Case 2: Partial overlap (suffix of old is prefix of new)
        for len_to_compare in [4, 3, 2]:
            if len(original_new_words) > len_to_compare and ' ' in previous_text_normalized:
                previous_words_normalized = previous_text_normalized.split()
                if len(previous_words_normalized) < len_to_compare:
                    continue

                # Take the suffix from the *previous* transcript
                suffix_to_check = " ".join(previous_words_normalized[-len_to_compare:])
                
                # Check if the *new* transcript starts with this suffix
                if normalized_new_text.startswith(suffix_to_check):
                    # The overlap matches. Strip it from the original new text.
                    corrected_transcript = " ".join(original_new_words[len_to_compare:])
                    
                    if corrected_transcript.strip():
                        logger.warning(
                            f"Corrected overlap repetition. "
                            f"Overlap: '{suffix_to_check}', Original: '{new_text}', Corrected: '{corrected_transcript}'"
                        )
                        return True, corrected_transcript
        
        return False, None
    
    def _check_hallucination_patterns(self, normalized_text: str) -> Optional[str]:
        """
        Check text against known hallucination patterns.
        
        Args:
            normalized_text: Normalized text to check
            
        Returns:
            Matched pattern description or None
        """
        for i, pattern in enumerate(self.hallucination_patterns):
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return f"pattern_{i}"
        return None
    
    def _check_similarity(self, new_text: str, recent_phrases: List[str]) -> Tuple[bool, float]:
        """
        Check similarity against recent transcripts.
        
        Args:
            new_text: New normalized text
            recent_phrases: List of recent normalized phrases
            
        Returns:
            Tuple of (is_similar: bool, max_similarity: float)
        """
        max_similarity = 0.0
        
        for phrase in recent_phrases:
            if not phrase:
                continue
            
            # Calculate similarity using different methods
            ratio = SequenceMatcher(None, new_text, phrase).ratio()
            
            # Also check for substring matches (shorter text fully contained in longer)
            if len(new_text) < len(phrase):
                if new_text in phrase:
                    ratio = max(ratio, 0.9)  # High similarity for substring matches
            elif len(phrase) < len(new_text):
                if phrase in new_text:
                    ratio = max(ratio, 0.9)
            
            # Check n-gram similarity for partial matches
            ngram_sim = self._calculate_ngram_similarity(new_text, phrase)
            ratio = max(ratio, ngram_sim)
            
            max_similarity = max(max_similarity, ratio)
            
            if ratio >= self.similarity_threshold:
                return True, ratio
        
        return False, max_similarity
    
    def _calculate_ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """
        Calculate n-gram similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            n: N-gram size
            
        Returns:
            Similarity score (0.0-1.0)
        """
        def get_ngrams(text: str, n: int) -> set:
            words = text.split()
            if len(words) < n:
                return {text}
            return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_internal_repetition(self, text: str) -> Optional[str]:
        """
        Check for repetitive patterns within a single transcript.
        
        Args:
            text: Normalized text to check
            
        Returns:
            Description of repetition found, or None
        """
        words = text.split()
        
        # Check for immediate word repetition (word word) - but be more lenient
        # Only flag if it's an obvious hallucination pattern, not natural speech disfluency
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 4:  # Only flag longer words (5+ chars)
                # Skip common disfluency patterns that are natural in speech
                disfluency_whitelist = ['there', 'would', 'could', 'should', 'really', 'maybe', 'probably',
                                      'like', 'well', 'so', 'and', 'but', 'the', 'this', 'that', 'what', 'why', 'how']
                if words[i].lower() not in disfluency_whitelist:
                    return f"word_repeat_{words[i]}"
        
        # Check for phrase repetition
        for phrase_len in range(2, min(5, len(words) // 2 + 1)):
            for i in range(len(words) - phrase_len * 2 + 1):
                phrase1 = ' '.join(words[i:i + phrase_len])
                phrase2 = ' '.join(words[i + phrase_len:i + phrase_len * 2])
                if phrase1 == phrase2:
                    return f"phrase_repeat_{phrase1}"
        
        return None


class HallucinationManager:
    """
    Main manager class that coordinates transcript history and hallucination detection.
    """
    
    def __init__(self, session_id: str, similarity_threshold: float = 0.75,
                 max_history: int = 5, min_transcript_length: int = 2):
        """
        Initialize the hallucination manager for a session - BALANCED MODE.

        Args:
            session_id: Unique session identifier
            similarity_threshold: BALANCED: Default 0.75 for balanced robustness vs natural speech
            max_history: Maximum number of transcripts to keep in history
            min_transcript_length: Minimum transcript length in words
        """
        self.session_id = session_id
        self.history_manager = TranscriptHistoryManager(max_history)
        self.detector = HallucinationDetector(similarity_threshold, min_transcript_length)
        
        # Statistics
        self.stats = {
            'total_transcripts': 0,
            'hallucinations_detected': 0,
            'hallucination_reasons': {},
            'valid_transcripts': 0
        }
        
        logger.info(f"HallucinationManager initialized for session {session_id}")
    
    def process_transcript(self, transcript_text: str, timestamp: float = None) -> Tuple[bool, str, Optional[str]]:
        """
        Process a new transcript, check for hallucinations, and return a corrected version if applicable.
        
        Args:
            transcript_text: New transcript text
            timestamp: Unix timestamp (defaults to current time)
            
        Returns:
            Tuple of (is_valid: bool, reason: str, corrected_transcript: Optional[str])
        """
        self.stats['total_transcripts'] += 1
        
        is_hallucination, reason, corrected_transcript = self.detector.is_hallucination(
            transcript_text, 
            self.history_manager
        )
        
        if is_hallucination:
            self.stats['hallucinations_detected'] += 1
            self.stats['hallucination_reasons'][reason] = self.stats['hallucination_reasons'].get(reason, 0) + 1
            
            # If we have a corrected transcript, the "hallucination" is that we've cleaned it.
            # It's still a "valid" transcript in its corrected form.
            if corrected_transcript:
                logger.warning(f"Session {self.session_id}: Corrected hallucination. Reason: {reason}. Original: '{transcript_text}', Corrected: '{corrected_transcript}'")
                # Add the corrected version to history
                self.history_manager.add_transcript(corrected_transcript, timestamp)
                self.stats['valid_transcripts'] += 1
                # Return the corrected transcript as valid.
                return True, reason, corrected_transcript
            else:
                # This is an uncorrectable hallucination (e.g., similarity match, pattern match)
                logger.warning(f"Session {self.session_id}: Uncorrectable hallucination detected. Reason: {reason}. Text: '{transcript_text}'")
                return False, reason, None
        else:
            # Add to history only if it's valid
            self.history_manager.add_transcript(transcript_text, timestamp)
            self.stats['valid_transcripts'] += 1
            
            logger.debug(f"Session {self.session_id}: Valid transcript added - '{transcript_text[:50]}...'")
            return True, reason, transcript_text
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics for this session.
        
        Returns:
            Dictionary containing statistics
        """
        total = self.stats['total_transcripts']
        return {
            'session_id': self.session_id,
            'total_transcripts': total,
            'valid_transcripts': self.stats['valid_transcripts'],
            'hallucinations_detected': self.stats['hallucinations_detected'],
            'hallucination_rate': self.stats['hallucinations_detected'] / total if total > 0 else 0.0,
            'hallucination_reasons': dict(self.stats['hallucination_reasons']),
            'history_size': len(self.history_manager.history)
        }
    
    def clear_session(self) -> None:
        """Clear all session data."""
        self.history_manager.clear_history()
        self.stats = {
            'total_transcripts': 0,
            'hallucinations_detected': 0,
            'hallucination_reasons': {},
            'valid_transcripts': 0
        }
        logger.info(f"Session {self.session_id}: All data cleared")


# Global manager for tracking all active sessions
_session_managers: Dict[str, HallucinationManager] = {}

def get_hallucination_manager(session_id: str, **kwargs) -> HallucinationManager:
    """
    Get or create a hallucination manager for a session.
    
    Args:
        session_id: Unique session identifier
        **kwargs: Additional arguments for HallucinationManager constructor
        
    Returns:
        HallucinationManager instance
    """
    if session_id not in _session_managers:
        _session_managers[session_id] = HallucinationManager(session_id, **kwargs)
        logger.info(f"Created new HallucinationManager for session {session_id}")
    
    return _session_managers[session_id]

def cleanup_session_manager(session_id: str) -> None:
    """
    Clean up and remove a session manager.
    
    Args:
        session_id: Session ID to clean up
    """
    if session_id in _session_managers:
        del _session_managers[session_id]
        logger.info(f"Cleaned up HallucinationManager for session {session_id}")

def get_global_statistics() -> Dict[str, Any]:
    """
    Get statistics for all active sessions.
    
    Returns:
        Dictionary with global statistics
    """
    total_sessions = len(_session_managers)
    all_stats = [manager.get_statistics() for manager in _session_managers.values()]
    
    if not all_stats:
        return {
            'active_sessions': 0,
            'total_transcripts': 0,
            'total_hallucinations': 0,
            'global_hallucination_rate': 0.0
        }
    
    total_transcripts = sum(stats['total_transcripts'] for stats in all_stats)
    total_hallucinations = sum(stats['hallucinations_detected'] for stats in all_stats)
    
    return {
        'active_sessions': total_sessions,
        'total_transcripts': total_transcripts,
        'total_hallucinations': total_hallucinations,
        'global_hallucination_rate': total_hallucinations / total_transcripts if total_transcripts > 0 else 0.0,
        'session_stats': all_stats
    }
