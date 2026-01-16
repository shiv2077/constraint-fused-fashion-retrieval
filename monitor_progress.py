#!/usr/bin/env python3
"""Monitor indexing progress."""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

def monitor_progress():
    """Monitor indexing progress."""
    artifacts_dir = Path("artifacts")
    metadata_file = artifacts_dir / "metadata.json"
    
    target_images = 3200
    start_time = time.time()
    
    print("Monitoring indexing progress...")
    print(f"Target: {target_images} images\n")
    
    last_count = 0
    
    while True:
        try:
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                current_count = len(metadata)
                
                if current_count != last_count:
                    elapsed = time.time() - start_time
                    rate = current_count / elapsed if elapsed > 0 else 0
                    
                    remaining = target_images - current_count
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta = timedelta(seconds=int(eta_seconds))
                    
                    progress_pct = (current_count / target_images) * 100
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Progress: {current_count}/{target_images} ({progress_pct:.1f}%) "
                          f"| Rate: {rate:.2f} img/s "
                          f"| ETA: {eta}")
                    
                    last_count = current_count
                    
                    if current_count >= target_images:
                        print("\n✓ Indexing complete!")
                        break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for indexing to start...")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_progress()
