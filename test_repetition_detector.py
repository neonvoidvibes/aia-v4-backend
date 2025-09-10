#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.multi_agent_summarizer.repetition_detector import RepetitionDetector

def test_repetition_detection():
    """Test repetition detection on the actual segments."""
    
    # Load the segments file from the test output
    segments_path = "/Users/neonvoid/Library/Mobile Documents/com~apple~CloudDocs/Downloads/_test/transcript_D20250812-T080755_uID-315c7f7d-f83a-4926-9353-8dce893c7f28_oID-river_aID-_test_eID-0000_sID-0b7e88084a0a45efadd6056cad600cd8__segments.json"
    
    print(f"Loading segments from: {segments_path}")
    
    try:
        with open(segments_path, 'r', encoding='utf-8') as f:
            segments_data = json.load(f)
            
        # Handle the fact that the file contains a list directly
        if isinstance(segments_data, list):
            segments = segments_data
        else:
            segments = segments_data.get("segments", [])
            
        print(f"Loaded {len(segments)} segments")
        
        # Show a sample of the first segment structure
        if segments:
            print(f"First segment keys: {list(segments[0].keys())}")
            print(f"First segment text preview: {segments[0].get('text', '')[:100]}...")
        
        # Test repetition detection
        detector = RepetitionDetector(
            min_phrase_length=2, 
            max_phrase_length=8,
            min_repetitions=2,  # Lower threshold for testing
            min_segment_span=2
        )
        
        analysis = detector.detect_repetitions(segments)
        
        print("\n=== REPETITION ANALYSIS ===")
        print(f"Found {len(analysis['repeated_phrases'])} repeated phrases:")
        
        for phrase in analysis['repeated_phrases']:
            count = analysis['phrase_counts'][phrase]['count']
            segments = analysis['phrase_counts'][phrase]['segments']
            print(f"  - \"{phrase}\" appears {count} times in segments {segments}")
        
        print(f"\nAffected segments: {analysis['affected_segments']}")
        
        print(f"\nExclusion instructions:")
        print(analysis['exclusion_instructions'])
        
        return analysis
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_repetition_detection()