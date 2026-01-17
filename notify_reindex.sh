#!/bin/bash
PID=765046
while kill -0 $PID 2>/dev/null; do
    sleep 30
done
notify-send --urgency=critical "Re-indexing Complete "Fashion retrieval re-indexed with tags. Ready to test accuracy again."
echo "✅ Re-indexing complete at $(date)"
