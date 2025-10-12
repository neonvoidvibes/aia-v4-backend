#!/usr/bin/env python3
"""
Automated test for canvas transcript modes.
Tests all combinations of Listen and Groups modes.
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Optional

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5001')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

class CanvasTranscriptTester:
    def __init__(self, agent_name: str, auth_token: str):
        self.agent_name = agent_name
        self.auth_token = auth_token
        self.results = []

    def test_listen_mode(self, mode: str, toggle_states: Optional[Dict[str, bool]] = None) -> bool:
        """Test a specific listen mode."""
        print(f"\n{'='*60}")
        print(f"TEST: Listen mode = {mode}")
        if toggle_states:
            print(f"      Toggle states = {len(toggle_states)} files")
        print(f"{'='*60}")

        # First, update the mode in Supabase
        print(f"[1/3] Setting transcript_listen_mode to '{mode}' in Supabase...")
        response = requests.post(
            f"{FRONTEND_URL}/api/agents/memory-prefs",
            json={
                'agent': self.agent_name,
                'transcript_listen_mode': mode
            },
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )

        if not response.ok:
            print(f"❌ FAIL: Could not set mode in Supabase: {response.status_code}")
            return False
        print(f"✓ Mode set in Supabase")

        # Wait a moment for the update to propagate
        time.sleep(0.5)

        # Trigger canvas analysis refresh
        print(f"[2/3] Triggering canvas analysis refresh...")
        payload = {
            'agent': self.agent_name,
            'clearPrevious': True
        }
        if toggle_states:
            payload['individualRawTranscriptToggleStates'] = toggle_states

        response = requests.post(
            f"{BACKEND_URL}/api/canvas/analysis/refresh",
            json=payload,
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )

        if not response.ok:
            print(f"❌ FAIL: Analysis refresh failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

        data = response.json()
        print(f"✓ Analysis refresh completed")
        print(f"   Results: {data.get('results', {})}")

        # Verify the analysis was generated
        print(f"[3/3] Verifying analysis content...")
        success = True
        for mode_name in ['mirror', 'lens', 'portal']:
            result = data.get('results', {}).get(mode_name, {})
            if result.get('success'):
                length = result.get('length', 0)
                print(f"✓ {mode_name}: {length} chars")
                if length < 100:
                    print(f"   ⚠️  Warning: Analysis seems too short ({length} chars)")
                    success = False
            else:
                print(f"❌ {mode_name}: Failed")
                success = False

        return success

    def test_groups_mode(self, groups_mode: str, listen_mode: str = 'latest') -> bool:
        """Test a specific groups mode."""
        print(f"\n{'='*60}")
        print(f"TEST: Groups mode = {groups_mode} (Listen = {listen_mode})")
        print(f"{'='*60}")

        # Set both modes in Supabase
        print(f"[1/3] Setting modes in Supabase...")
        response = requests.post(
            f"{FRONTEND_URL}/api/agents/memory-prefs",
            json={
                'agent': self.agent_name,
                'transcript_listen_mode': listen_mode,
                'groups_read_mode': groups_mode
            },
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )

        if not response.ok:
            print(f"❌ FAIL: Could not set modes: {response.status_code}")
            return False
        print(f"✓ Modes set")

        time.sleep(0.5)

        # Trigger refresh
        print(f"[2/3] Triggering analysis refresh...")
        response = requests.post(
            f"{BACKEND_URL}/api/canvas/analysis/refresh",
            json={
                'agent': self.agent_name,
                'clearPrevious': True
            },
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )

        if not response.ok:
            print(f"❌ FAIL: Analysis refresh failed: {response.status_code}")
            return False

        data = response.json()
        print(f"✓ Analysis refresh completed")

        # Verify
        print(f"[3/3] Verifying...")
        success = True
        for mode_name in ['mirror']:  # Just check one mode
            result = data.get('results', {}).get(mode_name, {})
            if result.get('success'):
                print(f"✓ {mode_name}: {result.get('length', 0)} chars")
            else:
                print(f"❌ {mode_name}: Failed")
                success = False

        return success

    def get_available_transcripts(self) -> List[Dict]:
        """Fetch available transcript files for this agent."""
        print(f"\n[INFO] Fetching available transcripts...")

        # This would need to call an API endpoint to list S3 files
        # For now, we'll construct some test data
        # In a real implementation, you'd call the backend to list files

        print(f"⚠️  Note: Using mock transcript data for testing")
        return []

    def run_all_tests(self):
        """Run comprehensive test suite."""
        print(f"\n{'#'*60}")
        print(f"# CANVAS TRANSCRIPT MODES - AUTOMATED TEST SUITE")
        print(f"# Agent: {self.agent_name}")
        print(f"{'#'*60}")

        # Test Listen modes
        print(f"\n## LISTEN MODE TESTS ##")

        tests = [
            ('none', None),
            ('latest', None),
            ('all', None),
        ]

        for mode, toggle_states in tests:
            success = self.test_listen_mode(mode, toggle_states)
            self.results.append({
                'test': f'Listen={mode}',
                'success': success
            })
            time.sleep(1)  # Pause between tests

        # Test Groups modes
        print(f"\n## GROUPS MODE TESTS ##")

        groups_tests = [
            ('none', 'latest'),
            ('latest', 'latest'),
            ('all', 'latest'),
        ]

        for groups_mode, listen_mode in groups_tests:
            success = self.test_groups_mode(groups_mode, listen_mode)
            self.results.append({
                'test': f'Groups={groups_mode}',
                'success': success
            })
            time.sleep(1)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        print(f"\n{'#'*60}")
        print(f"# TEST RESULTS SUMMARY")
        print(f"{'#'*60}")

        passed = sum(1 for r in self.results if r['success'])
        total = len(self.results)

        for result in self.results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} - {result['test']}")

        print(f"\n{'='*60}")
        print(f"TOTAL: {passed}/{total} tests passed")
        print(f"{'='*60}")

        return passed == total


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Test canvas transcript modes')
    parser.add_argument('--agent', required=True, help='Agent name to test')
    parser.add_argument('--token', required=True, help='Auth token (get from browser devtools)')
    parser.add_argument('--loop', action='store_true', help='Keep running until all tests pass')

    args = parser.parse_args()

    tester = CanvasTranscriptTester(args.agent, args.token)

    if args.loop:
        print("Running in LOOP mode - will retry until all tests pass")
        iteration = 1
        while True:
            print(f"\n\n{'*'*60}")
            print(f"* ITERATION {iteration}")
            print(f"{'*'*60}")

            tester.results = []  # Reset results
            tester.run_all_tests()

            if all(r['success'] for r in tester.results):
                print(f"\n🎉 SUCCESS! All tests passed on iteration {iteration}")
                break
            else:
                print(f"\n⚠️  Some tests failed. Retrying in 5 seconds...")
                time.sleep(5)
                iteration += 1
    else:
        tester.run_all_tests()
        sys.exit(0 if all(r['success'] for r in tester.results) else 1)


if __name__ == '__main__':
    main()
