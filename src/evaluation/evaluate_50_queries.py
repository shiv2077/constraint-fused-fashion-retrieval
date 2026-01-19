"""Run evaluation on 50 diverse queries with proper statistics."""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.common.config import SearchConfig
from src.retriever.search import search_and_rerank
from src.evaluation.queries_50 import ALL_QUERIES
from src.indexer.attribute_parser import parse_query_constraints


def compute_precision_recall(
    retrieved_items: List[Dict],
    query_constraints: Dict[str, set],
    topk: int = 5
) -> Dict[str, float]:
    """Compute precision and recall for retrieved items based on constraint matching."""
    
    if topk > len(retrieved_items):
        topk = len(retrieved_items)
    
    # Count items that satisfy all constraints
    relevant_in_topk = 0
    for item in retrieved_items[:topk]:
        try:
            item_tags = {
                'colors': set(item.get('tags', {}).get('colors', [])),
                'garments': set(item.get('tags', {}).get('garments', [])),
                'contexts': set(item.get('tags', {}).get('contexts', [])),
            }
            
            # Check if all query constraints are satisfied
            colors_match = all(c in item_tags['colors'] for c in query_constraints.get('colors', set()))
            garments_match = all(g in item_tags['garments'] for g in query_constraints.get('garments', set()))
            contexts_match = all(ctx in item_tags['contexts'] for ctx in query_constraints.get('contexts', set()))
            
            if colors_match and garments_match and contexts_match:
                relevant_in_topk += 1
        except (KeyError, TypeError):
            continue
    
    precision = relevant_in_topk / topk if topk > 0 else 0.0
    recall = relevant_in_topk / len(retrieved_items) if len(retrieved_items) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'relevant_in_topk': relevant_in_topk,
        'total_topk': topk,
    }


def evaluate_queries(
    queries: List[str],
    index_dir: Path,
    img_root: Path,
    out_dir: Path,
    config: SearchConfig,
) -> Dict[str, Any]:
    """Evaluate system on all queries."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Will store results for each query
    per_query_metrics = {}
    all_precisions = []
    all_recalls = []
    
    for i, query in enumerate(queries, 1):
        logging.info(f"[{i}/{len(queries)}] Evaluating: {query}")
        
        try:
            # Retrieve results
            results = search_and_rerank(query, index_dir, img_root, config)
            
            # Parse query constraints for evaluation
            query_constraints = parse_query_constraints(query)
            
            # Compute metrics using constraint-based relevance
            metrics = compute_precision_recall(results, query_constraints, config.topk)
            
            per_query_metrics[query] = {
                'precision@5': metrics['precision'],
                'recall@5': metrics['recall'],
                'relevant_in_topk': metrics['relevant_in_topk'],
                'total_topk': metrics['total_topk'],
                'num_results': len(results),
                'constraints': {
                    'colors': list(query_constraints.get('colors', set())),
                    'garments': list(query_constraints.get('garments', set())),
                    'contexts': list(query_constraints.get('contexts', set())),
                }
            }
            
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
        except Exception as e:
            logging.error(f"Error evaluating query '{query}': {e}")
            per_query_metrics[query] = {
                'error': str(e),
                'precision@5': 0.0,
                'recall@5': 0.0,
            }
            all_precisions.append(0.0)
            all_recalls.append(0.0)
    
    # Compute aggregate statistics
    mean_precision = np.mean(all_precisions)
    std_precision = np.std(all_precisions)
    median_precision = np.median(all_precisions)
    
    mean_recall = np.mean(all_recalls)
    std_recall = np.std(all_recalls)
    
    # 95% confidence interval
    se_precision = std_precision / np.sqrt(len(all_precisions))
    ci_lower = mean_precision - 1.96 * se_precision
    ci_upper = mean_precision + 1.96 * se_precision
    
    results_summary = {
        'evaluation_date': datetime.now().isoformat(),
        'num_queries': len(queries),
        'topk': config.topk,
        'metrics': {
            'precision@5': {
                'mean': float(mean_precision),
                'std': float(std_precision),
                'median': float(median_precision),
                'min': float(np.min(all_precisions)),
                'max': float(np.max(all_precisions)),
                'ci_lower_95': float(ci_lower),
                'ci_upper_95': float(ci_upper),
            },
            'recall@5': {
                'mean': float(mean_recall),
                'std': float(std_recall),
                'median': float(np.median(all_recalls)),
                'min': float(np.min(all_recalls)),
                'max': float(np.max(all_recalls)),
            }
        },
        'per_query': per_query_metrics,
    }
    
    return results_summary


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results nicely."""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS - 50 DIVERSE QUERIES")
    print("="*70)
    print(f"\nNumber of queries: {results['num_queries']}")
    print(f"Evaluation date: {results['evaluation_date']}")
    
    p5 = results['metrics']['precision@5']
    r5 = results['metrics']['recall@5']
    
    print("\nPRECISION @ 5:")
    print(f"  Mean:       {p5['mean']:.4f}")
    print(f"  Std Dev:    {p5['std']:.4f}")
    print(f"  Median:     {p5['median']:.4f}")
    print(f"  Min/Max:    {p5['min']:.4f} / {p5['max']:.4f}")
    print(f"  95% CI:     [{p5['ci_lower_95']:.4f}, {p5['ci_upper_95']:.4f}]")
    
    print("\nRECALL @ 5:")
    print(f"  Mean:       {r5['mean']:.4f}")
    print(f"  Std Dev:    {r5['std']:.4f}")
    print(f"  Median:     {r5['median']:.4f}")
    print(f"  Min/Max:    {r5['min']:.4f} / {r5['max']:.4f}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on 50 diverse queries")
    parser.add_argument('--index_dir', type=Path, default=Path('artifacts_no_tags'))
    parser.add_argument('--img_root', type=Path, default=Path('val_test2020/test'))
    parser.add_argument('--out_dir', type=Path, default=Path('evaluation_50queries'))
    
    args = parser.parse_args()
    
    config = SearchConfig()
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Evaluating on {len(ALL_QUERIES)} diverse queries...")
    results = evaluate_queries(ALL_QUERIES, args.index_dir, args.img_root, args.out_dir, config)
    
    # Save results
    out_file = args.out_dir / "evaluation_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_results(results)
    logging.info(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()
