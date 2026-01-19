"""Baseline: Vector-only retrieval without any constraint handling."""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.common.config import SearchConfig
from src.retriever.search import search_and_rerank


def evaluate_baseline(
    queries: List[str],
    index_dir: Path,
    img_root: Path,
    out_dir: Path,
    config: SearchConfig = None,
) -> Dict[str, Any]:
    """Evaluate baseline (vector-only retrieval) on all queries."""
    
    if config is None:
        config = SearchConfig()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    per_query_results = {}
    all_precisions = []
    all_recalls = []
    
    for i, query in enumerate(queries, 1):
        logging.info(f"[{i}/{len(queries)}] Baseline on: {query}")
        
        try:
            # Use search_and_rerank with baseline=True (vector-only)
            results = search_and_rerank(query, index_dir, img_root, config, baseline=True)
            
            # Compute REAL precision/recall using constraint matching (same as full system)
            from src.indexer.attribute_parser import parse_query_constraints
            query_constraints = parse_query_constraints(query)
            
            # Count items that satisfy all constraints
            relevant_in_topk = 0
            for item in results[:config.topk]:
                try:
                    item_tags = {
                        'colors': set(item.get('tags', {}).get('colors', [])),
                        'garments': set(item.get('tags', {}).get('garments', [])),
                        'contexts': set(item.get('tags', {}).get('contexts', [])),
                    }
                    
                    colors_match = all(c in item_tags['colors'] for c in query_constraints.get('colors', set()))
                    garments_match = all(g in item_tags['garments'] for g in query_constraints.get('garments', set()))
                    contexts_match = all(ctx in item_tags['contexts'] for ctx in query_constraints.get('contexts', set()))
                    
                    if colors_match and garments_match and contexts_match:
                        relevant_in_topk += 1
                except (KeyError, TypeError):
                    continue
            
            # Proper metric computation
            precision = relevant_in_topk / config.topk if config.topk > 0 else 0.0
            recall = relevant_in_topk / len(results) if len(results) > 0 else 0.0
            
            per_query_results[query] = {
                'precision': float(precision),
                'recall': float(recall),
                'num_results': len(results),
            }
            
            all_precisions.append(precision)
            all_recalls.append(recall)
        except Exception as e:
            logging.error(f"Error on query '{query}': {e}")
            per_query_results[query] = {
                'error': str(e),
                'precision': 0.0,
                'recall': 0.0,
            }
            all_precisions.append(0.0)
            all_recalls.append(0.0)
    
    # Compute statistics
    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    std_precision = np.std(all_precisions) if all_precisions else 0.0
    
    se_precision = std_precision / np.sqrt(len(all_precisions)) if len(all_precisions) > 0 else 0.0
    ci_lower = mean_precision - 1.96 * se_precision
    ci_upper = mean_precision + 1.96 * se_precision
    
    results_summary = {
        'baseline_type': 'Vector-Only (No Constraints)',
        'evaluation_date': datetime.now().isoformat(),
        'num_queries': len(queries),
        'topk': config.topk,
        'metrics': {
            'precision': {
                'mean': float(mean_precision),
                'std': float(std_precision),
                'min': float(np.min(all_precisions)) if all_precisions else 0.0,
                'max': float(np.max(all_precisions)) if all_precisions else 0.0,
                'ci_lower_95': float(ci_lower),
                'ci_upper_95': float(ci_upper),
            },
            'recall': {
                'mean': float(np.mean(all_recalls)) if all_recalls else 0.0,
                'std': float(np.std(all_recalls)) if all_recalls else 0.0,
            }
        },
        'per_query': per_query_results,
    }
    
    return results_summary


def print_baseline_results(results: Dict[str, Any]) -> None:
    """Print baseline results."""
    
    print("\n" + "="*70)
    print(f"BASELINE RESULTS - {results['baseline_type']}")
    print("="*70)
    print(f"\nNumber of queries: {results['num_queries']}")
    print(f"Evaluation date: {results['evaluation_date']}")
    
    metrics = results['metrics']
    p = metrics['precision']
    
    print("\nPRECISION:")
    print(f"  Mean:       {p['mean']:.4f}")
    print(f"  Std Dev:    {p['std']:.4f}")
    print(f"  Min/Max:    {p['min']:.4f} / {p['max']:.4f}")
    print(f"  95% CI:     [{p['ci_lower_95']:.4f}, {p['ci_upper_95']:.4f}]")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline vector-only evaluation")
    parser.add_argument('--index_dir', type=Path, default=Path('artifacts_no_tags'))
    parser.add_argument('--img_root', type=Path, default=Path('val_test2020/test'))
    parser.add_argument('--out_dir', type=Path, default=Path('baseline_results'))
    parser.add_argument('--num_queries', type=int, default=50)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Import queries
    from src.evaluation.queries_50 import ALL_QUERIES
    
    config = SearchConfig()
    queries = ALL_QUERIES[:args.num_queries]
    
    logging.info(f"Running baseline on {len(queries)} queries...")
    results = evaluate_baseline(queries, args.index_dir, args.img_root, args.out_dir, config)
    
    # Save results
    out_file = args.out_dir / "baseline_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_baseline_results(results)
    logging.info(f"Baseline results saved to {out_file}")


if __name__ == '__main__':
    main()
