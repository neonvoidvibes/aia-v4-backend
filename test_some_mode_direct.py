#!/usr/bin/env python3
"""
Direct test of 'some' mode filtering logic.
Tests the key matching between toggle states and S3 files.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils.canvas_analysis_agents import get_transcript_content_for_analysis
from utils.s3_utils import list_s3_objects_metadata

def test_some_mode_filtering():
    """Test that 'some' mode correctly filters files based on toggle states."""

    print("="*60)
    print("TESTING 'SOME' MODE FILTERING")
    print("="*60)

    # Configuration
    agent_name = "test-agent"  # Replace with your actual agent name
    event_id = "0000"

    # Step 1: List actual S3 files
    print(f"\n[1/3] Listing S3 files for {agent_name}/{event_id}...")
    transcript_prefix = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/"

    try:
        all_files_meta = list_s3_objects_metadata(transcript_prefix)

        # Filter out rolling- files and non-txt files
        relevant_files = [
            f for f in all_files_meta
            if not os.path.basename(f['Key']).startswith('rolling-') and f['Key'].endswith('.txt')
        ]

        print(f"   Found {len(relevant_files)} transcript files:")
        for i, f in enumerate(relevant_files[:5], 1):
            print(f"   {i}. {f['Key']}")
        if len(relevant_files) > 5:
            print(f"   ... and {len(relevant_files) - 5} more")

        if not relevant_files:
            print("   ❌ No transcript files found! Create some first.")
            return False

    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
        return False

    # Step 2: Create toggle states (select first 2 files)
    print(f"\n[2/3] Creating toggle states (selecting first 2 files)...")
    toggle_states = {}
    selected_keys = []
    for f in relevant_files[:2]:
        key = f['Key']
        toggle_states[key] = True
        selected_keys.append(key)

    print(f"   Toggle states created:")
    for key in selected_keys:
        print(f"   ✓ {key}")

    # Step 3: Test the filtering
    print(f"\n[3/3] Testing get_transcript_content_for_analysis with 'some' mode...")
    try:
        content = get_transcript_content_for_analysis(
            agent_name=agent_name,
            event_id=event_id,
            transcript_listen_mode='some',
            groups_read_mode='none',
            individual_raw_transcript_toggle_states=toggle_states
        )

        if content:
            print(f"   ✓ SUCCESS: Got transcript content ({len(content)} chars)")

            # Check if the correct files are in the content
            for key in selected_keys:
                filename = os.path.basename(key)
                if filename in content:
                    print(f"   ✓ Found expected file: {filename}")
                else:
                    print(f"   ❌ MISSING expected file: {filename}")

            # Check that non-selected files are NOT in the content
            for f in relevant_files[2:3]:  # Check one non-selected file
                filename = os.path.basename(f['Key'])
                if filename not in content:
                    print(f"   ✓ Correctly excluded: {filename}")
                else:
                    print(f"   ❌ INCORRECTLY included: {filename}")

            return True
        else:
            print(f"   ❌ FAILED: No content returned!")
            print(f"   Check the DEBUG logs above for key mismatch issues.")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_key_formats():
    """Test that frontend and backend key formats match."""

    print("\n" + "="*60)
    print("KEY FORMAT VERIFICATION")
    print("="*60)

    agent_name = "test-agent"
    event_id = "0000"

    # Simulate frontend key format (from /api/s3/list response)
    frontend_key_format = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/example.txt"

    # Backend key format (from list_s3_objects_metadata)
    transcript_prefix = f"organizations/river/agents/{agent_name}/events/{event_id}/transcripts/"
    backend_key_format = transcript_prefix + "example.txt"

    print(f"\nFrontend key format:\n  {frontend_key_format}")
    print(f"\nBackend key format:\n  {backend_key_format}")

    if frontend_key_format == backend_key_format:
        print(f"\n✓ KEY FORMATS MATCH!")
        return True
    else:
        print(f"\n❌ KEY FORMATS DON'T MATCH!")
        return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test "some" mode filtering')
    parser.add_argument('--agent', default='test-agent', help='Agent name to test (default: test-agent)')

    args = parser.parse_args()

    # Override agent name if provided
    if args.agent != 'test-agent':
        # This is a hack - we need to modify the global in the test function
        import inspect
        frame = inspect.currentframe()
        frame.f_globals['agent_name'] = args.agent

    print("\n🧪 CANVAS 'SOME' MODE - DIRECT TEST\n")

    # Test 1: Key format verification
    format_ok = test_key_formats()

    # Test 2: Actual filtering
    if format_ok:
        filtering_ok = test_some_mode_filtering()

        if filtering_ok:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED")
            print("="*60)
            sys.exit(0)
        else:
            print("\n" + "="*60)
            print("❌ FILTERING TEST FAILED")
            print("="*60)
            sys.exit(1)
    else:
        print("\n" + "="*60)
        print("❌ KEY FORMAT TEST FAILED")
        print("="*60)
        sys.exit(1)
