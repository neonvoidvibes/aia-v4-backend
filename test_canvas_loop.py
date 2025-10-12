#!/usr/bin/env python3
"""
Looping test script for canvas transcript modes.
Runs until all modes are confirmed working.
"""

import os
import sys
import time
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress debug logs from dependencies
logging.getLogger('utils.transcript_utils').setLevel(logging.WARNING)
logging.getLogger('utils.s3_utils').setLevel(logging.WARNING)

from utils.canvas_analysis_agents import get_transcript_content_for_analysis
from utils.supabase_client import get_supabase_client


def check_supabase_mode(agent_name):
    """Check what mode is stored in Supabase for this agent."""
    print(f"\n{'='*60}")
    print(f"CHECKING SUPABASE FOR AGENT: {agent_name}")
    print(f"{'='*60}")

    client = get_supabase_client()
    if not client:
        print("❌ Could not connect to Supabase")
        return None, None

    try:
        result = client.table("agents").select("transcript_listen_mode, groups_read_mode").eq("name", agent_name).limit(1).execute()

        if result.data and len(result.data) > 0:
            listen_mode = result.data[0].get("transcript_listen_mode", "latest")
            groups_mode = result.data[0].get("groups_read_mode", "none")
            print(f"✓ Found in Supabase:")
            print(f"  transcript_listen_mode: {listen_mode}")
            print(f"  groups_read_mode: {groups_mode}")
            return listen_mode, groups_mode
        else:
            print(f"❌ Agent '{agent_name}' not found in Supabase agents table")
            return None, None

    except Exception as e:
        print(f"❌ Error querying Supabase: {e}")
        return None, None


def update_supabase_mode(agent_name, mode):
    """Update the transcript_listen_mode in Supabase."""
    print(f"\n[UPDATE] Setting transcript_listen_mode to '{mode}' in Supabase...")

    client = get_supabase_client()
    if not client:
        print("❌ Could not connect to Supabase")
        return False

    try:
        result = client.table("agents").update({
            "transcript_listen_mode": mode
        }).eq("name", agent_name).execute()

        print(f"✓ Updated Supabase to mode: {mode}")
        return True

    except Exception as e:
        print(f"❌ Error updating Supabase: {e}")
        return False


def test_mode(agent_name, mode, toggle_states=None):
    """Test a specific transcript mode."""
    print(f"\n{'='*60}")
    print(f"TEST: Mode={mode}")
    if toggle_states:
        print(f"      Toggle states: {len(toggle_states)} files")
    print(f"{'='*60}")

    # Step 1: Update Supabase to this mode
    if not update_supabase_mode(agent_name, mode):
        return False

    # Wait for update to propagate
    time.sleep(0.5)

    # Step 2: Call the analysis function
    print(f"\n[TEST] Calling get_transcript_content_for_analysis...")
    print(f"       agent={agent_name}, mode={mode}")
    if toggle_states:
        print(f"       toggle_states keys: {list(toggle_states.keys())[:2]}...")

    try:
        content = get_transcript_content_for_analysis(
            agent_name=agent_name,
            event_id='0000',
            transcript_listen_mode=mode,
            groups_read_mode='none',
            individual_raw_transcript_toggle_states=toggle_states
        )

        # Step 3: Verify result
        if mode == 'none':
            if content is None or len(content) < 100:
                print(f"✅ PASS: Mode 'none' returned minimal/no content")
                return True
            else:
                print(f"❌ FAIL: Mode 'none' returned {len(content)} chars (expected minimal)")
                return False

        elif mode == 'latest':
            if content and len(content) > 100:
                # Check that content mentions only ONE transcript file
                transcript_count = content.count("--- START Transcript Source:")
                if transcript_count == 1:
                    print(f"✅ PASS: Mode 'latest' returned 1 transcript ({len(content)} chars)")
                    return True
                else:
                    print(f"❌ FAIL: Mode 'latest' returned {transcript_count} transcripts (expected 1)")
                    return False
            else:
                print(f"❌ FAIL: Mode 'latest' returned no content")
                return False

        elif mode == 'some':
            if not toggle_states:
                print(f"⚠️  SKIP: Mode 'some' requires toggle_states")
                return True

            if content and len(content) > 100:
                # Check that content has exactly the toggled files
                expected_count = len([k for k, v in toggle_states.items() if v])
                transcript_count = content.count("--- START Transcript Source:")

                if transcript_count == expected_count:
                    print(f"✅ PASS: Mode 'some' returned {transcript_count} transcripts ({len(content)} chars)")
                    return True
                else:
                    print(f"❌ FAIL: Mode 'some' returned {transcript_count} transcripts (expected {expected_count})")
                    print(f"   Content preview: {content[:200]}...")
                    return False
            else:
                print(f"❌ FAIL: Mode 'some' returned no content")
                return False

        elif mode == 'all':
            if content and len(content) > 100:
                transcript_count = content.count("--- START Transcript Source:")
                if transcript_count > 1:
                    print(f"✅ PASS: Mode 'all' returned {transcript_count} transcripts ({len(content)} chars)")
                    return True
                else:
                    print(f"❌ FAIL: Mode 'all' returned only {transcript_count} transcript (expected multiple)")
                    return False
            else:
                print(f"❌ FAIL: Mode 'all' returned no content")
                return False

        return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_available_transcripts(agent_name):
    """Get list of available transcript files for creating toggle states."""
    from utils.s3_utils import list_s3_objects_metadata
    import os

    print(f"\n[INFO] Finding available transcripts for {agent_name}...")

    transcript_prefix = f"organizations/river/agents/{agent_name}/events/0000/transcripts/"
    try:
        all_files_meta = list_s3_objects_metadata(transcript_prefix)

        # Filter out rolling- files and non-txt files
        relevant_files = [
            f for f in all_files_meta
            if not os.path.basename(f['Key']).startswith('rolling-') and f['Key'].endswith('.txt')
        ]

        print(f"✓ Found {len(relevant_files)} transcript files")
        return relevant_files

    except Exception as e:
        print(f"❌ Error listing transcripts: {e}")
        return []


def run_full_test_suite(agent_name):
    """Run all transcript mode tests."""
    print(f"\n{'#'*60}")
    print(f"# CANVAS TRANSCRIPT MODES - AUTOMATED TEST")
    print(f"# Agent: {agent_name}")
    print(f"{'#'*60}")

    # Check current Supabase state
    check_supabase_mode(agent_name)

    # Get available transcripts for 'some' mode testing
    available_transcripts = get_available_transcripts(agent_name)

    # Create toggle states for 'some' mode (select first 2 files)
    toggle_states = {}
    if len(available_transcripts) >= 2:
        for f in available_transcripts[:2]:
            toggle_states[f['Key']] = True
        print(f"\n[INFO] Created toggle states for 'some' mode:")
        for key in toggle_states.keys():
            print(f"  ✓ {key}")
    else:
        print(f"\n⚠️  Only {len(available_transcripts)} transcripts available, 'some' mode test may be limited")

    # Run tests
    results = {}

    print(f"\n{'='*60}")
    print(f"RUNNING TESTS")
    print(f"{'='*60}")

    # Test 1: none
    results['none'] = test_mode(agent_name, 'none')

    # Test 2: latest
    results['latest'] = test_mode(agent_name, 'latest')

    # Test 3: some (with toggle states)
    if toggle_states:
        results['some'] = test_mode(agent_name, 'some', toggle_states)
    else:
        results['some'] = None
        print(f"\n⚠️  Skipping 'some' mode test (no transcripts available)")

    # Test 4: all
    results['all'] = test_mode(agent_name, 'all')

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# TEST RESULTS SUMMARY")
    print(f"{'#'*60}")

    passed = 0
    failed = 0
    skipped = 0

    for mode, result in results.items():
        if result is None:
            status = "⏭️  SKIP"
            skipped += 1
        elif result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"{status} - Mode '{mode}'")

    print(f"\n{'='*60}")
    print(f"TOTAL: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    return failed == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test canvas transcript modes (looping)')
    parser.add_argument('--agent', required=True, help='Agent name to test')
    parser.add_argument('--loop', action='store_true', help='Keep looping until all tests pass')
    parser.add_argument('--delay', type=int, default=5, help='Delay between loop iterations (seconds)')

    args = parser.parse_args()

    if args.loop:
        print(f"\n🔁 LOOP MODE - Will retry until all tests pass")
        print(f"   Delay between iterations: {args.delay}s")

        iteration = 1
        while True:
            print(f"\n\n{'*'*60}")
            print(f"* ITERATION {iteration}")
            print(f"{'*'*60}")

            success = run_full_test_suite(args.agent)

            if success:
                print(f"\n🎉 SUCCESS! All tests passed on iteration {iteration}")
                sys.exit(0)
            else:
                print(f"\n⚠️  Some tests failed. Retrying in {args.delay} seconds...")
                time.sleep(args.delay)
                iteration += 1
    else:
        success = run_full_test_suite(args.agent)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
