#!/bin/bash

PID=540658
LOG_FILE="indexing_full_3200.log"
CHECK_INTERVAL=30

echo "Monitoring indexing process (PID: $PID)..."
echo "Will send notification when complete."
echo ""

while kill -0 $PID 2>/dev/null; do
    PROGRESS=$(grep "Indexing:" $LOG_FILE | tail -1 | grep -oP '\d+/3200' | head -1)
    if [ -n "$PROGRESS" ]; then
        CURRENT=$(echo $PROGRESS | cut -d'/' -f1)
        PERCENT=$((CURRENT * 100 / 3200))
        echo -ne "\r[$(date +%H:%M:%S)] Progress: $PROGRESS ($PERCENT%)"
    fi
    sleep $CHECK_INTERVAL
done

echo -e "\n\n🎉 Indexing Complete! 🎉"

NUM_IMAGES=$(python3 -c "import json; print(json.load(open('artifacts/manifest.json'))['num_images'])" 2>/dev/null || echo "Unknown")

MESSAGE="✅ Fashion retrieval indexing complete!\n\n📊 Indexed: $NUM_IMAGES images\n⏱️ Completed at: $(date +%H:%M:%S)\n\n🧪 Ready for testing!"

notify-send --urgency=critical --icon=dialog-information "Indexing Complete" "$MESSAGE"

echo ""
echo "=========================================="
echo "         INDEXING COMPLETE!"
echo "=========================================="
echo "Indexed: $NUM_IMAGES images"
echo "Completed at: $(date +%H:%M:%S)"
echo ""
echo "Next steps:"
echo "  1. Run edge case tests: python test_edge_cases.py"
echo "  2. Run full evaluation: python -m src.evaluation.run_prompts --index_dir artifacts --out_dir evaluation_final"
echo "  3. Or run automated suite: ./wait_and_test.sh"
echo "=========================================="

zenity --info --title="Indexing Complete" --text="$MESSAGE" --width=400 2>/dev/null || true

wall "🎉 Fashion retrieval indexing is COMPLETE! Ready for testing." 2>/dev/null || true
