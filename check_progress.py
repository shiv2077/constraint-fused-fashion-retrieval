#!/usr/bin/env python3

import sys
import time
import re
from pathlib import Path

LOG_FILE = "indexing_full_3200.log"
TOTAL = 3200

def get_progress():
    if not Path(LOG_FILE).exists():
        return None, "Log file not found"
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        match = re.search(r'Indexing:\s+(\d+)%.*?\|\s*(\d+)/(\d+)', line)
        if match:
            percent = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            
            match_time = re.search(r'\[(\d+:\d+)<(\d+:\d+),\s*([\d.]+)it/s\]', line)
            if match_time:
                elapsed = match_time.group(1)
                remaining = match_time.group(2)
                speed = float(match_time.group(3))
                return {
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'elapsed': elapsed,
                    'remaining': remaining,
                    'speed': speed
                }, None
    
    return None, "No progress found in log"

def format_time(seconds):
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"

if __name__ == "__main__":
    progress, error = get_progress()
    
    if error:
        print(f"✗ {error}")
        sys.exit(1)
    
    print("=" * 60)
    print("INDEXING PROGRESS")
    print("=" * 60)
    print(f"Status: {'COMPLETE ✓' if progress['current'] >= TOTAL else 'IN PROGRESS ⏳'}")
    print(f"Progress: {progress['current']}/{progress['total']} ({progress['percent']}%)")
    print(f"Speed: {progress['speed']:.2f} images/sec")
    print(f"Elapsed: {progress['elapsed']}")
    print(f"Remaining: {progress['remaining']}")
    print("=" * 60)
    
    bar_width = 50
    filled = int(bar_width * progress['current'] / progress['total'])
    bar = '█' * filled + '░' * (bar_width - filled)
    print(f"[{bar}] {progress['percent']}%")
    print("=" * 60)
    
    if progress['current'] >= TOTAL:
        print("\n✓ Indexing complete! You can now run tests.")
        sys.exit(0)
    else:
        print(f"\n⏳ Still indexing... {TOTAL - progress['current']} images remaining")
        sys.exit(2)
