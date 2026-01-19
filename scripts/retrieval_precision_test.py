"""
Retrieval Precision Test for Fashion Retrieval System.

Tests retrieval precision by:
1. Querying for specific color+garment combinations
2. Checking if returned results actually have those attributes
3. Comparing old vs new index performance

Precision = (# correct results) / (# total results)
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
from typing import List, Dict, Tuple

# Test queries - combinations of color and garment
TEST_QUERIES = [
    ("red dress", ["red", "crimson", "scarlet", "burgundy", "maroon"], ["dress", "gown"]),
    ("blue shirt", ["blue", "navy", "teal", "azure"], ["shirt", "blouse", "top"]),
    ("black pants", ["black", "charcoal"], ["pants", "trousers", "jeans"]),
    ("white dress", ["white", "ivory", "cream"], ["dress", "gown"]),
    ("pink top", ["pink", "coral", "salmon", "rose"], ["top", "shirt", "blouse"]),
    ("black dress", ["black", "charcoal"], ["dress", "gown"]),
    ("gray coat", ["gray", "grey", "charcoal"], ["coat", "jacket", "blazer"]),
    ("orange dress", ["orange", "coral", "peach", "tangerine"], ["dress", "gown"]),
    ("brown jacket", ["brown", "tan", "khaki", "bronze"], ["jacket", "coat", "blazer"]),
    ("green shirt", ["green", "olive", "teal", "emerald"], ["shirt", "top", "blouse"]),
]


def load_index(artifacts_dir: str) -> Tuple[faiss.Index, List[dict]]:
    """Load FAISS index and metadata."""
    index = faiss.read_index(f"{artifacts_dir}/vectors.faiss")
    with open(f"{artifacts_dir}/metadata.json") as f:
        metadata = json.load(f)
    return index, metadata


def check_match(item_colors: List[str], item_garments: List[str], 
                expected_colors: List[str], expected_garments: List[str]) -> Tuple[bool, bool, bool]:
    """
    Check if item matches expected colors and garments.
    
    Returns: (color_match, garment_match, both_match)
    """
    # Normalize colors
    item_colors_lower = [c.lower() for c in item_colors]
    expected_colors_lower = [c.lower() for c in expected_colors]
    
    # Normalize garments
    item_garments_lower = [g.lower() for g in item_garments]
    expected_garments_lower = [g.lower() for g in expected_garments]
    
    # Check color match
    color_match = any(c in item_colors_lower for c in expected_colors_lower)
    
    # Check garment match
    garment_match = any(g in item_garments_lower for g in expected_garments_lower)
    
    return color_match, garment_match, (color_match and garment_match)


def run_retrieval_test(artifacts_dir: str, top_k: int = 10, use_hard_filter: bool = False) -> Dict:
    """
    Run retrieval precision test.
    """
    from pathlib import Path
    from src.retriever.search import search_and_rerank
    from src.common.config import SearchConfig
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL PRECISION TEST")
    print(f"Index: {artifacts_dir}")
    print(f"Top-K: {top_k}")
    print(f"Hard Filter: {use_hard_filter}")
    print(f"{'='*70}\n")
    
    results = []
    config = SearchConfig(topn=top_k)
    index_dir = Path(artifacts_dir)
    # Use same directory as img_root for simplicity
    img_root = Path("val_test2020/test")
    
    header = f"{'Query':<20} | {'Color P':<10} | {'Garment P':<10} | {'Both P':<10} | {'Retrieved':<10}"
    print(header)
    print("-" * len(header))
    
    for query_text, expected_colors, expected_garments in TEST_QUERIES:
        # Parse query constraints
        constraints = {
            'colors': [expected_colors[0]],  # Primary color
            'garments': [expected_garments[0]],  # Primary garment
        }
        
        # Run retrieval
        try:
            matches = search_and_rerank(
                query=query_text,
                index_dir=index_dir,
                img_root=img_root,
                config=config,
                hard_filter=use_hard_filter,
                constraints=constraints
            )
        except Exception as e:
            print(f"Error on '{query_text}': {e}")
            continue
        
        # Calculate precision
        color_correct = 0
        garment_correct = 0
        both_correct = 0
        
        for match in matches:
            item_colors = match.get('tags', {}).get('colors', [])
            item_garments = match.get('tags', {}).get('garments', [])
            
            c_match, g_match, both_match = check_match(
                item_colors, item_garments,
                expected_colors, expected_garments
            )
            
            if c_match:
                color_correct += 1
            if g_match:
                garment_correct += 1
            if both_match:
                both_correct += 1
        
        n = len(matches)
        if n == 0:
            color_p = garment_p = both_p = 0.0
        else:
            color_p = color_correct / n
            garment_p = garment_correct / n
            both_p = both_correct / n
        
        print(f"{query_text:<20} | {color_p*100:>7.1f}%  | {garment_p*100:>7.1f}%   | {both_p*100:>7.1f}%   | {n:<10}")
        
        results.append({
            'query': query_text,
            'color_precision': color_p,
            'garment_precision': garment_p,
            'combined_precision': both_p,
            'retrieved': n
        })
    
    # Calculate averages
    avg_color = np.mean([r['color_precision'] for r in results])
    avg_garment = np.mean([r['garment_precision'] for r in results])
    avg_combined = np.mean([r['combined_precision'] for r in results])
    
    print("-" * len(header))
    print(f"{'AVERAGE':<20} | {avg_color*100:>7.1f}%  | {avg_garment*100:>7.1f}%   | {avg_combined*100:>7.1f}%   |")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Average Color Precision:    {avg_color*100:.1f}%")
    print(f"Average Garment Precision:  {avg_garment*100:.1f}%")
    print(f"Average Combined Precision: {avg_combined*100:.1f}%")
    print(f"{'='*70}\n")
    
    return {
        'per_query': results,
        'avg_color_precision': avg_color,
        'avg_garment_precision': avg_garment,
        'avg_combined_precision': avg_combined
    }


def compare_indexes(old_dir: str, new_dir: str, top_k: int = 10):
    """
    Compare retrieval precision between two indexes.
    """
    print("\n" + "="*80)
    print("COMPARING TWO INDEXES")
    print("="*80)
    
    # Test old index (no hard filter)
    print("\n>>> OLD INDEX (No Hard Filter) <<<")
    old_results = run_retrieval_test(old_dir, top_k, use_hard_filter=False)
    
    # Test new index with hard filter
    print("\n>>> NEW INDEX (With Hard Filter) <<<")
    new_results = run_retrieval_test(new_dir, top_k, use_hard_filter=True)
    
    # Summary comparison
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    improvement_color = new_results['avg_color_precision'] - old_results['avg_color_precision']
    improvement_garment = new_results['avg_garment_precision'] - old_results['avg_garment_precision']
    improvement_combined = new_results['avg_combined_precision'] - old_results['avg_combined_precision']
    
    print(f"Color Precision:    {old_results['avg_color_precision']*100:.1f}% -> {new_results['avg_color_precision']*100:.1f}% ({improvement_color*100:+.1f}%)")
    print(f"Garment Precision:  {old_results['avg_garment_precision']*100:.1f}% -> {new_results['avg_garment_precision']*100:.1f}% ({improvement_garment*100:+.1f}%)")
    print(f"Combined Precision: {old_results['avg_combined_precision']*100:.1f}% -> {new_results['avg_combined_precision']*100:.1f}% ({improvement_combined*100:+.1f}%)")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retrieval precision")
    parser.add_argument("--artifacts_dir", default="artifacts_v2", help="Artifacts directory to test")
    parser.add_argument("--compare", action="store_true", help="Compare old vs new index")
    parser.add_argument("--old_dir", default="artifacts_segmented", help="Old index directory")
    parser.add_argument("--new_dir", default="artifacts_v2", help="New index directory")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--hard_filter", action="store_true", help="Use hard filtering")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_indexes(args.old_dir, args.new_dir, args.top_k)
    else:
        run_retrieval_test(args.artifacts_dir, args.top_k, args.hard_filter)
