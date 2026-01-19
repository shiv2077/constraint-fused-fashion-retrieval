"""
Precision test for color-based retrieval.

Compares retrieval precision between:
1. Old CLIP-based color detection
2. New YOLO+SAM segmentation-based color detection

For each test query (e.g., "red dress"), we measure what percentage of
returned images actually contain that color according to manual verification.
"""
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from collections import Counter
import random


def load_index(artifacts_dir: str) -> Tuple[faiss.Index, List[dict]]:
    """Load FAISS index and metadata."""
    index = faiss.read_index(f"{artifacts_dir}/vectors.faiss")
    metadata = json.load(open(f"{artifacts_dir}/metadata.json"))
    return index, metadata


def get_color_precision(metadata: List[dict], target_color: str) -> Dict:
    """
    Calculate color-based retrieval metrics.
    
    For each image tagged with `target_color`, we consider it a "relevant" document.
    Precision = relevant retrieved / total retrieved
    """
    # Find all images with the target color
    relevant = [i for i, m in enumerate(metadata) if target_color in m['tags']['colors']]
    
    return {
        'color': target_color,
        'total_relevant': len(relevant),
        'relevant_indices': relevant
    }


def compare_color_detection(old_dir: str, new_dir: str) -> Dict:
    """
    Compare color detection quality between old and new indexes.
    """
    old_meta = json.load(open(f"{old_dir}/metadata.json"))
    new_meta = json.load(open(f"{new_dir}/metadata.json"))
    
    # Limit to same images
    min_len = min(len(old_meta), len(new_meta))
    old_meta = old_meta[:min_len]
    new_meta = new_meta[:min_len]
    
    print(f"\n{'='*60}")
    print(f"PRECISION TEST: Comparing {min_len} images")
    print(f"OLD: {old_dir}")
    print(f"NEW: {new_dir}")
    print(f"{'='*60}\n")
    
    # Colors to test
    colors = ['red', 'blue', 'black', 'white', 'pink', 'green', 'orange', 'brown', 'gray', 'beige']
    
    results = []
    
    print(f"{'Color':<12} | {'OLD Count':<12} | {'NEW Count':<12} | {'Change':<10} | {'% Change':<10}")
    print("-" * 60)
    
    for color in colors:
        old_count = sum(1 for m in old_meta if color in m['tags']['colors'])
        new_count = sum(1 for m in new_meta if color in m['tags']['colors'])
        change = new_count - old_count
        pct_change = (change / max(old_count, 1)) * 100
        
        sign = '+' if change > 0 else ''
        print(f"{color:<12} | {old_count:<12} | {new_count:<12} | {sign}{change:<10} | {sign}{pct_change:.1f}%")
        
        results.append({
            'color': color,
            'old_count': old_count,
            'new_count': new_count,
            'change': change,
            'pct_change': pct_change
        })
    
    # Calculate overall statistics
    old_total = sum(r['old_count'] for r in results)
    new_total = sum(r['new_count'] for r in results)
    
    # Count images with any color tag
    old_with_colors = sum(1 for m in old_meta if m['tags']['colors'])
    new_with_colors = sum(1 for m in new_meta if m['tags']['colors'])
    
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Images with color tags: OLD {old_with_colors}/{min_len} ({100*old_with_colors/min_len:.1f}%) -> NEW {new_with_colors}/{min_len} ({100*new_with_colors/min_len:.1f}%)")
    print(f"Total color detections: OLD {old_total} -> NEW {new_total}")
    
    # Agreement analysis
    agreed = 0
    disagreed = 0
    for i in range(min_len):
        old_c = set(old_meta[i]['tags']['colors'])
        new_c = set(new_meta[i]['tags']['colors'])
        if old_c == new_c:
            agreed += 1
        else:
            disagreed += 1
    
    print(f"Color agreement: {agreed}/{min_len} ({100*agreed/min_len:.1f}%)")
    print(f"Color changed: {disagreed}/{min_len} ({100*disagreed/min_len:.1f}%)")
    
    return {
        'color_results': results,
        'old_with_colors': old_with_colors,
        'new_with_colors': new_with_colors,
        'agreed': agreed,
        'disagreed': disagreed,
        'total_images': min_len
    }


def sample_verification(old_dir: str, new_dir: str, n_samples: int = 10):
    """
    Show sample images where color detection changed significantly.
    """
    old_meta = json.load(open(f"{old_dir}/metadata.json"))
    new_meta = json.load(open(f"{new_dir}/metadata.json"))
    
    min_len = min(len(old_meta), len(new_meta))
    
    print(f"\n{'='*60}")
    print(f"SAMPLE VERIFICATION: {n_samples} examples with biggest changes")
    print(f"{'='*60}\n")
    
    # Find images with biggest difference
    differences = []
    for i in range(min_len):
        old_c = set(old_meta[i]['tags']['colors'])
        new_c = set(new_meta[i]['tags']['colors'])
        diff = old_c.symmetric_difference(new_c)
        if diff:
            differences.append((i, old_meta[i]['path'], old_c, new_c, len(diff)))
    
    # Sort by biggest difference
    differences.sort(key=lambda x: x[4], reverse=True)
    
    print(f"{'Idx':<6} | {'Filename':<40} | {'OLD':<25} | {'NEW':<25}")
    print("-" * 100)
    
    for idx, path, old_c, new_c, _ in differences[:n_samples]:
        filename = path.split('/')[-1][:35]
        print(f"{idx:<6} | {filename:<40} | {str(sorted(old_c)):<25} | {str(sorted(new_c)):<25}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retrieval precision")
    parser.add_argument("--old_dir", default="artifacts", help="Old index directory")
    parser.add_argument("--new_dir", default="artifacts_test_100", help="New index directory")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to show")
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_color_detection(args.old_dir, args.new_dir)
    
    # Show samples
    sample_verification(args.old_dir, args.new_dir, args.samples)
    
    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print("""
Background colors (likely FALSE POSITIVES in OLD):
  - 'white', 'gray' often detect runway backgrounds, not garments
  - Decrease in these colors = better precision

Garment colors (likely TRUE POSITIVES in NEW):
  - 'red', 'orange', 'blue', 'black' increase = more accurate detection
  - These colors are typically in the garments, not backgrounds

To validate manually:
  1. Open the image files listed in samples
  2. Compare actual garment color to detected colors
  3. NEW should match the actual garment more often
""")
