# 🚀 EMERGENCY INDEXING GUIDE

**PROBLEM**: Terminal is stuck and won't execute commands

**SOLUTION**: Use Python to run the indexing directly

## Option 1: Simple Direct Execution

Open VS Code terminal (new clean terminal) and paste:

```bash
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval
rm -rf artifacts
python3 -c "
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, '-m', 'src.indexer.build_index',
     '--img_dir', 'val_test2020/test',
     '--out_dir', 'artifacts', 
     '--max_images', '3200'],
    stdout=open('indexing_with_tags_final.log', 'w'),
    stderr=subprocess.STDOUT
)
print(f'Started with PID: {proc.pid}')
"
```

## Option 2: Use the provided script

```bash
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval
nohup python3 auto_index_notify.py > auto_index.log 2>&1 &
tail -f auto_index.log
```

## Option 3: Kill stuck terminal and start fresh

```bash
pkill -f "build_index"  # Kill any running indexing
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval
rm -rf artifacts
nohup python -m src.indexer.build_index --img_dir val_test2020/test --out_dir artifacts --max_images 3200 > indexing_with_tags_final.log 2>&1 &
```

## Check if it's working

```bash
# Check if process is running
ps aux | grep "build_index" | grep -v grep

# Check log file exists
ls -lh indexing_with_tags_final.log

# Monitor progress
tail -f indexing_with_tags_final.log
```

## Expected Output

After ~45 minutes you should see in the log:
```
Successfully processed 3200 images
Saved FAISS index to artifacts/vectors.faiss
Saved metadata to artifacts/metadata.json
Saved manifest to artifacts/manifest.json
Index building complete!
```

Then verify tags are populated:
```bash
python3 -c "import json; m=json.load(open('artifacts/metadata.json')); print(m[0]['tags'])"
```

Should show something like:
```python
{'colors': ['red', 'blue'], 'garments': ['dress', 'jacket'], 'contexts': ['runway']}
```

NOT empty lists.

## Once Done

Run evaluation:
```bash
python -m src.evaluation.run_prompts --index_dir artifacts --out_dir evaluation_with_tags
grep "P@5" evaluation_with_tags/evaluation_metrics.json
```

**IMPORTANT**: Keep your laptop AWAKE during indexing. If it sleeps, the process will stop.
