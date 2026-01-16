"""
Compute retrieval evaluation metrics (Precision@K, Recall@K, NDCG, MAP).
"""

import json
import numpy as np
from typing import Dict, List
from pathlib import Path


def precision_at_k(relevant: List[bool], k: int) -> float:
    """Precision@K: fraction of top-K that are relevant."""
    if k == 0:
        return 0.0
    return sum(relevant[:k]) / k


def recall_at_k(relevant: List[bool], k: int, total_relevant: int) -> float:
    """Recall@K: fraction of all relevant items found in top-K."""
    if total_relevant == 0:
        return 0.0
    return sum(relevant[:k]) / total_relevant


def average_precision(relevant: List[bool]) -> float:
    """
    Average Precision: mean of precision values at each relevant position.
    Used to compute MAP (Mean Average Precision).
    """
    if not any(relevant):
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for i, is_relevant in enumerate(relevant, 1):
        if is_relevant:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    return sum(precisions) / len(precisions) if precisions else 0.0


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at K.
    relevances: list of relevance scores (0 or 1 for binary, or graded)
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Normalized DCG at K.
    Normalized by ideal DCG (sorting relevances in descending order).
    """
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_query_metrics(retrieved: List[str], ground_truth: Dict[str, bool], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
    """
    Compute metrics for a single query.
    
    Args:
        retrieved: List of retrieved filenames in rank order
        ground_truth: Dict mapping filename -> is_relevant
        k_values: K values to compute metrics at
    
    Returns:
        Dict of metrics
    """
    # Create relevance list in retrieved order
    relevant = [ground_truth.get(fname, False) for fname in retrieved]
    total_relevant = sum(ground_truth.values())
    
    metrics = {}
    
    for k in k_values:
        metrics[f'P@{k}'] = precision_at_k(relevant, k)
        metrics[f'R@{k}'] = recall_at_k(relevant, k, total_relevant)
        metrics[f'NDCG@{k}'] = ndcg_at_k([float(r) for r in relevant], k)
    
    metrics['AP'] = average_precision(relevant)
    metrics['total_relevant'] = total_relevant
    metrics['total_retrieved'] = len(retrieved)
    
    return metrics


def evaluate_with_ground_truth(results_file: str, ground_truth_file: str) -> Dict:
    """
    Evaluate retrieval results against ground truth annotations.
    
    Returns:
        Dict with per-query metrics and aggregated metrics
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    all_metrics = []
    per_query_metrics = {}
    
    # Handle both 'queries' and 'prompts' keys
    queries_data = results.get('queries', results.get('prompts', []))
    
    for query_data in queries_data:
        query = query_data['query']
        
        if query not in ground_truth:
            print(f"Warning: No ground truth for query: {query}")
            continue
        
        # Get results from either 'results' or 'main_results' key
        retrieved = query_data.get('results', query_data.get('main_results', []))
        retrieved_files = [r['filename'] for r in retrieved]
        gt = ground_truth[query]
        
        metrics = compute_query_metrics(retrieved_files, gt)
        per_query_metrics[query] = metrics
        all_metrics.append(metrics)
    
    # Compute macro-averaged metrics (average across queries)
    aggregated = {}
    if all_metrics:
        metric_keys = [k for k in all_metrics[0].keys() if k not in ['total_relevant', 'total_retrieved']]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            aggregated[key] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
    
    # Compute MAP (Mean Average Precision)
    if 'AP' in aggregated:
        aggregated['MAP'] = aggregated['AP']
    
    return {
        'per_query': per_query_metrics,
        'aggregated': aggregated,
        'num_queries': len(all_metrics)
    }


def automatic_evaluation(results_file: str) -> Dict:
    """
    Automatic evaluation using caption/tag matching as proxy for relevance.
    NOT as accurate as manual annotation but gives rough estimate.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    from src.indexer.attribute_parser import parse_query_constraints, extract_tags
    
    all_metrics = []
    per_query_metrics = {}
    
    # Handle both 'queries' and 'prompts' keys
    queries_data = results.get('queries', results.get('prompts', []))
    
    for query_data in queries_data:
        query = query_data['query']
        constraints = parse_query_constraints(query)
        
        # Get results from either 'results' or 'main_results' key
        retrieved = query_data.get('results', query_data.get('main_results', []))
        
        # Create pseudo ground truth based on constraint matching
        pseudo_gt = {}
        for result in retrieved:
            # Extract tags from caption if not already present
            if 'tags' not in result and 'caption' in result:
                result_tags = extract_tags(result['caption'])
            else:
                result_tags = result.get('tags', {'colors': [], 'garments': [], 'contexts': []})
            
            # Check if result matches query constraints
            score = 0.0
            total_constraints = 0
            
            # Check colors
            if constraints['colors']:
                total_constraints += 1
                if any(c in result_tags.get('colors', []) for c in constraints['colors']):
                    score += 1
            
            # Check garments
            if constraints['garments']:
                total_constraints += 1
                if any(g in result_tags.get('garments', []) for g in constraints['garments']):
                    score += 1
            
            # Check contexts
            if constraints['contexts']:
                total_constraints += 1
                if any(c in result_tags.get('contexts', []) for c in constraints['contexts']):
                    score += 1
            
            # Consider relevant if matches >= 50% of constraints
            # If no constraints, use constraint_score from results
            if total_constraints > 0:
                is_relevant = (score / total_constraints >= 0.5)
            else:
                # No explicit constraints - use ITM score as relevance indicator
                is_relevant = result.get('itm_score', 0.0) > 0.5
            
            pseudo_gt[result['filename']] = is_relevant
        
        retrieved_files = [r['filename'] for r in retrieved]
        metrics = compute_query_metrics(retrieved_files, pseudo_gt)
        per_query_metrics[query] = metrics
        all_metrics.append(metrics)
    
    # Aggregate
    aggregated = {}
    if all_metrics:
        metric_keys = [k for k in all_metrics[0].keys() if k not in ['total_relevant', 'total_retrieved']]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            aggregated[key] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
    
    if 'AP' in aggregated:
        aggregated['MAP'] = aggregated['AP']
    
    return {
        'per_query': per_query_metrics,
        'aggregated': aggregated,
        'num_queries': len(all_metrics),
        'note': 'Automatic evaluation using constraint matching. Less accurate than manual annotation.'
    }


def print_metrics_report(metrics: Dict):
    """Pretty print evaluation metrics."""
    print("\n" + "="*80)
    print("EVALUATION METRICS REPORT")
    print("="*80)
    
    agg = metrics['aggregated']
    
    print(f"\nNumber of queries evaluated: {metrics['num_queries']}")
    
    if 'note' in metrics:
        print(f"\n⚠️  {metrics['note']}")
    
    print("\n" + "-"*80)
    print("AGGREGATED METRICS (averaged across all queries)")
    print("-"*80)
    
    # Precision metrics
    print("\n📊 Precision @ K:")
    for k in [1, 3, 5, 10]:
        key = f'P@{k}'
        if key in agg:
            std_key = f'{key}_std'
            std = agg.get(std_key, 0)
            print(f"  P@{k:2d} = {agg[key]:.3f} ± {std:.3f}")
    
    # Recall metrics
    print("\n📈 Recall @ K:")
    for k in [1, 3, 5, 10]:
        key = f'R@{k}'
        if key in agg:
            std_key = f'{key}_std'
            std = agg.get(std_key, 0)
            print(f"  R@{k:2d} = {agg[key]:.3f} ± {std:.3f}")
    
    # NDCG metrics
    print("\n🎯 NDCG @ K:")
    for k in [1, 3, 5, 10]:
        key = f'NDCG@{k}'
        if key in agg:
            std_key = f'{key}_std'
            std = agg.get(std_key, 0)
            print(f"  NDCG@{k:2d} = {agg[key]:.3f} ± {std:.3f}")
    
    # MAP
    if 'MAP' in agg:
        map_std = agg.get('MAP_std', 0)
        print(f"\n⭐ Mean Average Precision (MAP) = {agg['MAP']:.3f} ± {map_std:.3f}")
    
    # Per-query breakdown
    print("\n" + "-"*80)
    print("PER-QUERY BREAKDOWN")
    print("-"*80)
    
    for query, qmetrics in metrics['per_query'].items():
        print(f"\n{query[:75]}")
        print(f"  Relevant: {qmetrics['total_relevant']:2d} | " +
              f"P@5={qmetrics.get('P@5', 0):.3f} | " +
              f"R@5={qmetrics.get('R@5', 0):.3f} | " +
              f"NDCG@5={qmetrics.get('NDCG@5', 0):.3f}")


def main():
    """Run evaluation metrics computation."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Compute retrieval evaluation metrics")
    parser.add_argument("--results_file", default="outputs/results.json", help="Results JSON from retrieval")
    parser.add_argument("--ground_truth", default=None, help="Ground truth annotations JSON (if available)")
    parser.add_argument("--output", default="outputs/evaluation_metrics.json", help="Where to save metrics")
    parser.add_argument("--mode", choices=['manual', 'auto'], default='auto',
                        help="manual: use ground_truth file, auto: automatic constraint-based eval")
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        print("Run evaluation first: python -m src.evaluation.run_prompts")
        sys.exit(1)
    
    print(f"Computing evaluation metrics...")
    print(f"Results file: {args.results_file}")
    
    if args.mode == 'manual':
        if not args.ground_truth:
            print("Error: --ground_truth required for manual mode")
            sys.exit(1)
        if not Path(args.ground_truth).exists():
            print(f"Error: Ground truth file not found: {args.ground_truth}")
            print("Run annotation first: python -m src.evaluation.annotate_results")
            sys.exit(1)
        
        print(f"Using manual ground truth: {args.ground_truth}")
        metrics = evaluate_with_ground_truth(args.results_file, args.ground_truth)
    else:
        print("Using automatic constraint-based evaluation")
        metrics = automatic_evaluation(args.results_file)
    
    # Save metrics
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to {args.output}")
    
    # Print report
    print_metrics_report(metrics)


if __name__ == "__main__":
    main()
