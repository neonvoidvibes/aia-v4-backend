#!/usr/bin/env python3
"""
Quick diagnostic script to check what's happening with 'some' mode.
Run this while testing in the UI.
"""

import os
import sys
import time
import re
from datetime import datetime, timedelta

# Path to backend logs
LOG_DIR = "/Users/neonvoid/Documents/AI_apps/aia-v4/aia-v4-backend/logs"

def tail_logs(log_file, num_lines=50):
    """Get last N lines from log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-num_lines:]
    except FileNotFoundError:
        return []

def find_latest_log():
    """Find the most recent log file."""
    if not os.path.exists(LOG_DIR):
        print(f"⚠️  Log directory not found: {LOG_DIR}")
        print("Using current directory...")
        return None

    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    if not log_files:
        return None

    latest = max(log_files, key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)))
    return os.path.join(LOG_DIR, latest)

def analyze_canvas_logs(lines):
    """Analyze canvas-related log lines."""
    print("\n" + "="*60)
    print("CANVAS ANALYSIS LOG ANALYSIS")
    print("="*60)

    # Look for key patterns
    patterns = {
        'mode_detection': r"listen_mode=(\w+)",
        'toggle_states_received': r"received (\d+) toggle state entries",
        'toggle_keys': r"Toggle state keys: (.*)",
        'total_files': r"Found (\d+) total files in S3",
        's3_keys': r"S3 file keys: (.*)",
        'matched': r"Matched (\d+) toggled transcripts",
        'matched_keys': r"Matched keys: (.*)",
        'no_matches': r"NO MATCHES",
    }

    results = {k: [] for k in patterns}

    for line in lines:
        if 'canvas' not in line.lower() and 'transcript' not in line.lower():
            continue

        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                results[key].append((line.strip(), match.groups() if match.groups() else None))

    # Print results
    print("\n📊 FINDINGS:\n")

    if results['mode_detection']:
        print("✓ Detected modes:")
        for line, groups in results['mode_detection'][-3:]:  # Last 3
            print(f"  - {groups[0]}")

    if results['toggle_states_received']:
        print(f"\n✓ Toggle states received:")
        for line, groups in results['toggle_states_received'][-1:]:
            print(f"  - {groups[0]} entries")

    if results['toggle_keys']:
        print(f"\n✓ Toggle state keys (sample):")
        for line, groups in results['toggle_keys'][-1:]:
            print(f"  {groups[0]}")

    if results['s3_keys']:
        print(f"\n✓ S3 file keys (sample):")
        for line, groups in results['s3_keys'][-1:]:
            print(f"  {groups[0]}")

    if results['matched']:
        print(f"\n✓ Match results:")
        for line, groups in results['matched'][-1:]:
            print(f"  - Matched {groups[0]} files")

    if results['matched_keys']:
        print(f"\n✓ Matched keys:")
        for line, groups in results['matched_keys'][-1:]:
            print(f"  {groups[0]}")

    if results['no_matches']:
        print(f"\n❌ WARNING: NO MATCHES FOUND!")
        print("   This means toggle state keys don't match S3 keys.")
        print("   Check the key formats above to debug.")

    # Show full relevant logs at the end
    print("\n" + "-"*60)
    print("FULL RELEVANT LOGS (last 10):")
    print("-"*60)
    relevant_lines = [l for l in lines if any(k in l.lower() for k in ['canvas', 'transcript', 'toggle', 'some'])]
    for line in relevant_lines[-10:]:
        print(line.strip())

def watch_mode():
    """Watch logs in real-time."""
    print("👁️  WATCHING LOGS (press Ctrl+C to stop)...")
    print("Now trigger a canvas refresh in the UI with 'some' mode selected.\n")

    log_file = find_latest_log()
    if not log_file:
        print("❌ Could not find log file. Make sure backend is running and logging.")
        return

    print(f"Watching: {log_file}\n")

    try:
        # Get current position
        with open(log_file, 'r') as f:
            f.seek(0, 2)  # Go to end
            while True:
                line = f.readline()
                if line:
                    if any(k in line.lower() for k in ['canvas', 'transcript', 'debug', 'some']):
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {line.strip()}")
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped watching.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose canvas "some" mode issues')
    parser.add_argument('--watch', action='store_true', help='Watch logs in real-time')
    parser.add_argument('--lines', type=int, default=100, help='Number of log lines to analyze (default: 100)')

    args = parser.parse_args()

    if args.watch:
        watch_mode()
    else:
        log_file = find_latest_log()
        if not log_file:
            print("❌ Could not find log file.")
            print("Expected location:", LOG_DIR)
            sys.exit(1)

        print(f"📄 Analyzing: {log_file}")
        lines = tail_logs(log_file, args.lines)

        if not lines:
            print("❌ No log lines found.")
            sys.exit(1)

        analyze_canvas_logs(lines)

        print("\n" + "="*60)
        print("💡 NEXT STEPS:")
        print("="*60)
        print("1. Go to UI → Settings → Memory → Transcripts")
        print("2. Toggle specific files (enter 'some' mode)")
        print("3. Go to Canvas view (should auto-refresh)")
        print("4. Run this script again to see new logs")
        print("\nOr run with --watch to monitor in real-time")

if __name__ == '__main__':
    main()
