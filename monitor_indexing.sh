#!/bin/bash
# Monitor indexing progress in real-time

echo "======================================"
echo "INDEXING PROGRESS MONITOR"
echo "======================================"
echo ""

# Check if process is still running
if ps -p 540658 > /dev/null 2>&1; then
    echo "✅ Status: RUNNING (PID: 540658)"
else
    echo "⚠️  Status: COMPLETED or STOPPED"
fi

echo ""
echo "Latest Progress:"
echo "--------------------------------------"
tail -3 indexing_full_3200.log | grep -E "Indexing:|INFO"

echo ""
echo "To monitor continuously, run:"
echo "  watch -n 5 ./monitor_indexing.sh"
echo ""
echo "Or tail the log:"
echo "  tail -f indexing_full_3200.log"
