#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.multi_agent_summarizer.context_agent import ContextAgent

def debug_context_agent():
    """Debug the context agent to see why it's returning no content."""
    
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
        
        if segments:
            print(f"Sample segment content: {segments[0].get('text', '')[:200]}...")
        
        # Test context agent
        agent = ContextAgent()
        
        # Create mock repetition analysis
        repetition_analysis = {
            "repeated_phrases": ["test phrase"],
            "exclusion_instructions": "Test exclusion instructions"
        }
        
        print("\n=== TESTING CONTEXT AGENT ===")
        result = agent.run(segments, repetition_analysis)
        
        print(f"Result length: {len(result)}")
        print(f"Result content:\n{result}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_context_agent()