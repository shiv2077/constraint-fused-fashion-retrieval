#!/bin/bash
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval

# Clean up old artifacts
echo "Removing old artifacts directory..."
rm -rf artifacts

# Start indexing
echo "Starting indexing at $(date)"
nohup python -m src.indexer.build_index \
    --img_dir val_test2020/test \
    --out_dir artifacts \
    --max_images 3200 \
    > indexing_with_tags_final.log 2>&1 &

PID=$!
echo "Indexing started with PID: $PID"
echo $PID > indexing.pid

# Wait a few seconds to check it started
sleep 5

if ps -p $PID > /dev/null; then
    echo "✅ Process is running"
    echo "Log file: indexing_with_tags_final.log"
    echo "Monitor with: tail -f indexing_with_tags_final.log"
    echo "Check progress: grep 'Indexing:' indexing_with_tags_final.log | tail -1"
else
    echo "❌ Process failed to start"
    tail -20 indexing_with_tags_final.log
fi
