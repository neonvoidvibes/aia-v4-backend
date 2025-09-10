import re
from collections import Counter
from typing import List, Dict, Any, Set


class RepetitionDetector:
    """
    Detects repeated phrases across transcript segments that are likely Whisper AI hallucinations.
    Provides exclusion lists for subsequent agents to ignore these patterns.
    """
    
    def __init__(self, min_phrase_length: int = 2, max_phrase_length: int = 8, 
                 min_repetitions: int = 3, min_segment_span: int = 2):
        """
        Args:
            min_phrase_length: Minimum number of words in a phrase to consider
            max_phrase_length: Maximum number of words in a phrase to consider
            min_repetitions: Minimum times a phrase must appear to be considered repetitive
            min_segment_span: Minimum number of different segments phrase must span
        """
        self.min_phrase_length = min_phrase_length
        self.max_phrase_length = max_phrase_length
        self.min_repetitions = min_repetitions
        self.min_segment_span = min_segment_span
    
    def detect_repetitions(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze segments and identify repeated phrases likely to be transcription artifacts.
        
        Returns:
            Dictionary containing:
            - repeated_phrases: List of phrases to exclude
            - phrase_counts: Count of each repeated phrase
            - affected_segments: Which segments contain repetitions
            - exclusion_instructions: Text for agents to reference
        """
        if not segments:
            return {
                "repeated_phrases": [],
                "phrase_counts": {},
                "affected_segments": [],
                "exclusion_instructions": "No repetitive phrases detected."
            }
        
        # Extract all text content from segments
        all_texts = []
        segment_texts = {}
        
        for i, seg in enumerate(segments):
            # Combine all text fields from segments
            text_parts = []
            
            # Check for common text fields in segments
            for field in ['text', 'summary', 'content']:
                if field in seg and seg[field]:
                    text_parts.append(seg[field])
            
            # Also check for quotes if they exist
            if 'quotes' in seg and isinstance(seg['quotes'], list):
                for quote in seg['quotes']:
                    if isinstance(quote, dict) and 'text' in quote:
                        text_parts.append(quote['text'])
            
            # Join and clean text
            segment_text = ' '.join(text_parts).strip()
            if segment_text:
                all_texts.append(segment_text)
                segment_texts[i] = segment_text
        
        if not all_texts:
            return {
                "repeated_phrases": [],
                "phrase_counts": {},
                "affected_segments": [],
                "exclusion_instructions": "No text content found for repetition analysis."
            }
        
        # Find repeated phrases
        repeated_phrases = self._find_repeated_phrases(all_texts, segment_texts)
        
        # Generate exclusion instructions
        exclusion_instructions = self._generate_exclusion_instructions(repeated_phrases)
        
        # Find affected segments
        affected_segments = self._find_affected_segments(repeated_phrases, segment_texts)
        
        return {
            "repeated_phrases": list(repeated_phrases.keys()),
            "phrase_counts": repeated_phrases,
            "affected_segments": affected_segments,
            "exclusion_instructions": exclusion_instructions
        }
    
    def _find_repeated_phrases(self, all_texts: List[str], segment_texts: Dict[int, str]) -> Dict[str, Dict]:
        """Find phrases that repeat across multiple segments."""
        phrase_data = {}
        
        # Generate all possible phrases of different lengths
        for text_idx, text in enumerate(all_texts):
            words = self._tokenize_text(text)
            
            for length in range(self.min_phrase_length, min(self.max_phrase_length + 1, len(words) + 1)):
                for start_idx in range(len(words) - length + 1):
                    phrase_words = words[start_idx:start_idx + length]
                    phrase = ' '.join(phrase_words).strip()
                    
                    # Skip very short or very long phrases
                    if len(phrase) < 10 or len(phrase) > 200:
                        continue
                    
                    # Normalize phrase for comparison
                    normalized_phrase = self._normalize_phrase(phrase)
                    
                    if normalized_phrase not in phrase_data:
                        phrase_data[normalized_phrase] = {
                            'original_phrase': phrase,
                            'count': 0,
                            'segments': set()
                        }
                    
                    phrase_data[normalized_phrase]['count'] += 1
                    phrase_data[normalized_phrase]['segments'].add(text_idx)
        
        # Filter to only repetitive phrases that span multiple segments
        repetitive_phrases = {}
        for normalized_phrase, data in phrase_data.items():
            if (data['count'] >= self.min_repetitions and 
                len(data['segments']) >= self.min_segment_span):
                repetitive_phrases[data['original_phrase']] = {
                    'count': data['count'],
                    'segments': list(data['segments'])
                }
        
        return repetitive_phrases
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization, preserving meaningful words."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on word boundaries but preserve common phrases
        words = re.findall(r'\b\w+(?:\s+\w+)*\b', text.lower())
        return [w.strip() for w in words if w.strip()]
    
    def _normalize_phrase(self, phrase: str) -> str:
        """Normalize phrase for comparison (lowercase, remove extra spaces)."""
        return re.sub(r'\s+', ' ', phrase.lower().strip())
    
    def _generate_exclusion_instructions(self, repeated_phrases: Dict[str, Dict]) -> str:
        """Generate instructions for agents about what phrases to exclude."""
        if not repeated_phrases:
            return "No repetitive phrases detected that need to be excluded."
        
        instructions = "CRITICAL: The following phrases appear to be Whisper AI transcription artifacts and must be completely ignored when analyzing patterns or content:\n\n"
        
        for i, (phrase, data) in enumerate(repeated_phrases.items(), 1):
            # Truncate very long phrases for readability
            display_phrase = phrase[:100] + "..." if len(phrase) > 100 else phrase
            instructions += f"{i}. \"{display_phrase}\" (appears {data['count']} times across {len(data['segments'])} segments)\n"
        
        instructions += "\nDo NOT base any organizational patterns, strategic insights, or business analysis on these repeated phrases. They are transcription errors, not actual conversation patterns."
        
        return instructions
    
    def _find_affected_segments(self, repeated_phrases: Dict[str, Dict], segment_texts: Dict[int, str]) -> List[int]:
        """Find which segments contain repetitive phrases."""
        affected_segments = set()
        
        for phrase, data in repeated_phrases.items():
            affected_segments.update(data['segments'])
        
        return sorted(list(affected_segments))