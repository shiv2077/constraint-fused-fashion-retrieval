#!/bin/bash

PID=540658
INDEX_DIR="artifacts"
LOG_FILE="indexing_full_3200.log"

echo "=========================================="
echo "Waiting for indexing to complete..."
echo "=========================================="
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo ""

while kill -0 $PID 2>/dev/null; do
    PROGRESS=$(grep "Indexing:" $LOG_FILE | tail -1 | grep -oP '\d+/3200' | head -1)
    if [ -n "$PROGRESS" ]; then
        CURRENT=$(echo $PROGRESS | cut -d'/' -f1)
        PERCENT=$((CURRENT * 100 / 3200))
        echo -ne "\rProgress: $PROGRESS ($PERCENT%) - Still running..."
    fi
    sleep 30
done

echo -e "\n\n=========================================="
echo "Indexing complete!"
echo "=========================================="

tail -20 $LOG_FILE | grep -E "Saved|INFO"

echo ""
echo "=========================================="
echo "Verifying index integrity..."
echo "=========================================="

if [ -f "$INDEX_DIR/manifest.json" ]; then
    NUM_IMAGES=$(python3 -c "import json; print(json.load(open('$INDEX_DIR/manifest.json'))['num_images'])")
    echo "✓ Indexed $NUM_IMAGES images"
else
    echo "✗ Manifest not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Running edge case tests..."
echo "=========================================="
python test_edge_cases.py

echo ""
echo "=========================================="
echo "Running full evaluation..."
echo "=========================================="
python -m src.evaluation.run_prompts --index_dir artifacts --out_dir evaluation_final

echo ""
echo "=========================================="
echo "Generating metrics report..."
echo "=========================================="

python -c "
import json
from pathlib import Path

results_file = Path('evaluation_final/results.json')
if results_file.exists():
    with open(results_file) as f:
        data = json.load(f)
    
    print('\\n' + '='*60)
    print('FINAL EVALUATION METRICS')
    print('='*60)
    
    for query_name, metrics in data.items():
        print(f'\\n{query_name}:')
        print(f'  P@5: {metrics.get(\"precision_at_5\", 0):.3f}')
        print(f'  R@5: {metrics.get(\"recall_at_5\", 0):.3f}')
        print(f'  Relevant results: {metrics.get(\"num_relevant\", 0)}/5')
    
    avg_p5 = sum(m.get('precision_at_5', 0) for m in data.values()) / len(data)
    avg_r5 = sum(m.get('recall_at_5', 0) for m in data.values()) / len(data)
    
    print(f'\\n' + '='*60)
    print(f'AVERAGE METRICS')
    print(f'='*60)
    print(f'Avg P@5: {avg_p5:.3f}')
    print(f'Avg R@5: {avg_r5:.3f}')
else:
    print('✗ Results file not found')
"

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
