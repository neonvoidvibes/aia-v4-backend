#!/bin/bash
#
# Watch canvas analysis logs in real-time
# Run this while testing in the UI
#

LOG_FILE="logs/claude_chat.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Make sure the backend is running"
    exit 1
fi

echo "📊 WATCHING CANVAS ANALYSIS LOGS"
echo "================================"
echo ""
echo "Now test in the UI:"
echo "1. Go to Settings > Memory > Transcripts"
echo "2. Change Listen mode to 'some'"
echo "3. Toggle ON 2-3 specific transcript files"
echo "4. Switch to Canvas view"
echo "5. Watch for analysis refresh"
echo ""
echo "Looking for:"
echo "  ✓ transcript_listen_mode value"
echo "  ✓ Toggle states received"
echo "  ✓ S3 files found"
echo "  ✓ Matches found"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "================================"
echo ""

# Tail the log file and filter for relevant lines
tail -f "$LOG_FILE" | grep --line-buffered -E "Canvas:|transcript_listen_mode|toggle|DEBUG|some mode|Matched"
