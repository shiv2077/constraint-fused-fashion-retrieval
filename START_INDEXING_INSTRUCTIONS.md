# HOW TO START INDEXING AND MONITOR PROGRESS

## Step 1: Start Indexing

Open a terminal and run:

```bash
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval
rm -rf artifacts
nohup python -m src.indexer.build_index --img_dir val_test2020/test --out_dir artifacts --max_images 3200 > indexing_with_tags_final.log 2>&1 &
echo $! > indexing.pid
echo "Indexing started with PID: $(cat indexing.pid)"
```

## Step 2: Check Progress

Run this command anytime to see progress:

```bash
python monitor_indexing_progress.py
```

Or manually check the log:

```bash
grep "Indexing:" indexing_with_tags_final.log | tail -1
```

## Step 3: Monitor in Real-Time

```bash
tail -f indexing_with_tags_final.log
```

(Press Ctrl+C to stop watching)

## Expected Timeline

- **Total time**: ~45-50 minutes for 3200 images
- **Speed**: ~1.0-1.2 images/second
- **Progress updates**: Every few seconds in the log

## When Complete

The log will show:
```
2026-01-17 XX:XX:XX - root - INFO - Successfully processed 3200 images
2026-01-17 XX:XX:XX - root - INFO - Saved FAISS index to artifacts/vectors.faiss
2026-01-17 XX:XX:XX - root - INFO - Saved metadata to artifacts/metadata.json
2026-01-17 XX:XX:XX - root - INFO - Index building complete!
```

## After Completion

Run evaluation:
```bash
python -m src.evaluation.run_prompts --index_dir artifacts --out_dir evaluation_with_tags
```

## If Process Dies

If your laptop goes to sleep and the process stops:
1. Check last progress: `python monitor_indexing_progress.py`
2. Restart from scratch (no resume capability)
3. Keep laptop awake: `caffeinate` (on Mac) or disable sleep mode

## Verify Tags Are Working

After indexing completes, check tags:
```bash
python -c "import json; m = json.load(open('artifacts/metadata.json')); print('Sample tags:', m[0]['tags'])"
```

You should see NON-EMPTY tags like:
```python
Sample tags: {'colors': ['blue', 'white'], 'garments': ['dress', 'shoes'], 'contexts': ['runway']}
```

If all tags are empty `[]`, the indexing didn't work correctly.
