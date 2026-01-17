#!/usr/bin/env python3
"""
Monitor indexing progress
Usage: python monitor_indexing_progress.py
"""
import subprocess
import time
import re
from pathlib import Path

log_file = Path("indexing_with_tags_final.log")
pid_file = Path("indexing.pid")

def get_latest_progress():
    if not log_file.exists():
        return None, "Log file not found"
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        match = re.search(r'Indexing:\s+(\d+)%.*?\|\s*(\d+)/(\d+).*?\[([\d:]+)<([\d:]+),\s*([\d.]+)it/s\]', line)
        if match:
            percent = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            elapsed = match.group(4)
            remaining = match.group(5)
            speed = float(match.group(6))
            return {
                'percent': percent,
                'current': current,
                'total': total,
                'elapsed': elapsed,
                'remaining': remaining,
                'speed': speed
            }, None
    
    return None, "No progress found in log"

def check_process():
    if pid_file.exists():
        with open(pid_file) as f:
            pid = f.read().strip()
        try:
            result = subprocess.run(['ps', '-p', pid], capture_output=True)
            return result.returncode == 0, pid
        except:
            return False, pid
    return False, None

if __name__ == "__main__":
    running, pid = check_process()
    
    print("="*60)
    print("INDEXING STATUS")
    print("="*60)
    
    if pid:
        print(f"Process PID: {pid}")
        print(f"Status: {'🟢 RUNNING' if running else '🔴 STOPPED'}")
    else:
        print("Status: No PID file found")
    
    print()
    
    progress, error = get_latest_progress()
    
    if progress:
        print(f"Progress: {progress['current']}/{progress['total']} ({progress['percent']}%)")
        print(f"Speed: {progress['speed']:.2f} images/sec")
        print(f"Elapsed: {progress['elapsed']}")
        print(f"Remaining: {progress['remaining']}")
        print()
        
        bar_width = 50
        filled = int(bar_width * progress['current'] / progress['total'])
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"[{bar}] {progress['percent']}%")
        
        if progress['current'] >= progress['total']:
            print("\n✅ Indexing complete!")
        elif not running:
            print("\n⚠️ Process stopped before completion")
    else:
        print(f"No progress data: {error}")
        if log_file.exists():
            print(f"\nLast 10 lines of log:")
            with open(log_file) as f:
                lines = f.readlines()
            for line in lines[-10:]:
                print(f"  {line.rstrip()}")
    
    print("="*60)
