#!/usr/bin/env python3
"""Direct indexing launcher"""
import subprocess
import sys
from pathlib import Path

work_dir = Path("/home/shiv2077/dev/constraint-fused-fashion-retrieval")

# Clean artifacts
artifacts = work_dir / "artifacts"
if artifacts.exists():
    import shutil
    shutil.rmtree(artifacts)
    print(f"Cleaned {artifacts}")

# Start indexing
print("Starting indexing...")
log_file = work_dir / "indexing_with_tags_final.log"

proc = subprocess.Popen(
    [sys.executable, "-m", "src.indexer.build_index",
     "--img_dir", "val_test2020/test",
     "--out_dir", "artifacts",
     "--max_images", "3200"],
    cwd=str(work_dir),
    stdout=open(log_file, 'w'),
    stderr=subprocess.STDOUT,
    start_new_session=True  # Detach from parent
)

pid = proc.pid
print(f"✅ Indexing started with PID: {pid}")
print(f"📊 Log file: {log_file}")
print(f"⏱️ Expected time: ~45 minutes")
print(f"\n📈 Monitor progress with:")
print(f"   tail -f {log_file}")
print(f"   or")
print(f"   python monitor_indexing_progress.py")
