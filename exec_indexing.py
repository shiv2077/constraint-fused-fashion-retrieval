import subprocess
import sys
import os
from pathlib import Path

os.chdir("/home/shiv2077/dev/constraint-fused-fashion-retrieval")

# Remove old artifacts
import shutil
if Path("artifacts").exists():
    shutil.rmtree("artifacts")

# Start the indexing process
proc = subprocess.Popen(
    [sys.executable, "-m", "src.indexer.build_index",
     "--img_dir", "val_test2020/test",
     "--out_dir", "artifacts",
     "--max_images", "3200"],
    stdout=open("indexing_with_tags_final.log", "w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setsid
)

print(f"Indexing started with PID: {proc.pid}")
with open("indexing.pid", "w") as f:
    f.write(str(proc.pid))

# Keep checking if it's running
import time
time.sleep(3)

if proc.poll() is None:
    print("✅ Process is running successfully")
else:
    print("❌ Process failed")
    with open("indexing_with_tags_final.log") as f:
        print(f.read()[-500:])
