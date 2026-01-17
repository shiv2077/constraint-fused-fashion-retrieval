#!/usr/bin/env python3
"""
Automated indexing with email notification on completion
Run this as: nohup python3 auto_index_notify.py &
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import json
import socket

def send_notification(status, message):
    """Send desktop notification"""
    try:
        subprocess.run(['notify-send', status, message], check=False)
    except:
        pass

def main():
    os.chdir("/home/shiv2077/dev/constraint-fused-fashion-retrieval")
    
    # Clean up
    import shutil
    artifacts = Path("artifacts")
    if artifacts.exists():
        shutil.rmtree(artifacts)
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting indexing...")
    send_notification("Indexing", "Fashion retrieval indexing started")
    
    # Start process
    with open("indexing_with_tags_final.log", "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.indexer.build_index",
             "--img_dir", "val_test2020/test",
             "--out_dir", "artifacts",
             "--max_images", "3200"],
            stdout=log,
            stderr=subprocess.STDOUT
        )
    
    pid = proc.pid
    with open("indexing.pid", "w") as f:
        f.write(str(pid))
    
    print(f"[{time.strftime('%H:%M:%S')}] Process PID: {pid}")
    
    # Wait for completion
    return_code = proc.wait()
    
    print(f"[{time.strftime('%H:%M:%S')}] Indexing completed with code: {return_code}")
    
    # Check results
    if artifacts.exists() and (artifacts / "metadata.json").exists():
        with open(artifacts / "metadata.json") as f:
            meta = json.load(f)
        
        num_images = len(meta)
        sample_tags = meta[0].get('tags', {})
        
        msg = f"Indexed {num_images} images. Tags: {sample_tags}"
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")
        send_notification("✅ Indexing Complete", msg)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Artifacts not created properly")
        send_notification("❌ Indexing Failed", "Check logs")

if __name__ == "__main__":
    main()
