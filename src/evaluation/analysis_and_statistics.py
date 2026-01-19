"""Analyze and compare evaluation results with proper statistics."""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_file: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def compute_confidence_intervals(
    values: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """Compute confidence intervals for a list of values."""
    
    if not values:
        return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0}
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # Standard error
    se = std / np.sqrt(n)
    
    # Z-score for 95% confidence
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99%
    
    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'ci_lower': float(mean - z_score * se),
        'ci_upper': float(mean + z_score * se),
        'n': n,
    }


def compare_methods(
    baseline_results: Dict,
    system_results: Dict,
    out_dir: Path,
) -> Dict[str, Any]:
    """Compare baseline against full system with statistical tests."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract precision values
    baseline_precisions = [
        v['precision@5'] 
        for v in baseline_results['per_query'].values()
        if isinstance(v, dict) and 'precision@5' in v
    ]
    
    system_precisions = [
        v['precision@5'] 
        for v in system_results['per_query'].values()
        if isinstance(v, dict) and 'precision@5' in v
    ]
    
    # Compute statistics
    baseline_stats = compute_confidence_intervals(baseline_precisions)
    system_stats = compute_confidence_intervals(system_precisions)
    
    # Paired t-test (if we have corresponding queries)
    if len(baseline_precisions) == len(system_precisions):
        diffs = np.array(system_precisions) - np.array(baseline_precisions)
        mean_diff = np.mean(diffs)
        se_diff = np.std(diffs) / np.sqrt(len(diffs))
        t_stat = mean_diff / se_diff if se_diff > 0 else 0
        
        # Approximate p-value (rough estimate)
        from scipy import stats
        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), len(diffs) - 1))
    else:
        mean_diff = system_stats['mean'] - baseline_stats['mean']
        pvalue = None
    
    comparison = {
        'baseline': baseline_stats,
        'system': system_stats,
        'improvement': {
            'absolute': float(system_stats['mean'] - baseline_stats['mean']),
            'relative_percent': float(
                100 * (system_stats['mean'] - baseline_stats['mean']) / baseline_stats['mean']
                if baseline_stats['mean'] > 0 else 0
            ),
        },
        'statistical_significance': {
            'mean_difference': float(mean_diff),
            't_statistic': float(t_stat) if len(baseline_precisions) == len(system_precisions) else None,
            'p_value': float(pvalue) if pvalue is not None else None,
            'significant_at_0.05': pvalue < 0.05 if pvalue is not None else None,
        }
    }
    
    return comparison


def create_comparison_table(
    results_dict: Dict[str, Dict],
    out_file: Path,
) -> None:
    """Create a markdown table comparing different methods."""
    
    table = "| Method | Mean Precision | Std Dev | 95% CI | Median | N |\n"
    table += "|--------|---|---|---|---|---|\n"
    
    for method_name, stats in results_dict.items():
        if 'precision@5' in stats.get('metrics', {}):
            p5 = stats['metrics']['precision@5']
            ci_text = f"[{p5['ci_lower_95']:.4f}, {p5['ci_upper_95']:.4f}]"
            table += (f"| {method_name} | {p5['mean']:.4f} | {p5['std']:.4f} | "
                     f"{ci_text} | {p5['median']:.4f} | {p5.get('n', 50)} |\n")
    
    with open(out_file, 'w') as f:
        f.write(table)


def create_visualization(
    results: Dict[str, Any],
    out_file: Path,
) -> None:
    """Create visualization of results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data from ablation results if available
    if 'ablation_variants' in results:
        variants = results['ablation_variants']
        names = list(variants.keys())
        precisions = [variants[v]['metrics']['precision@5']['mean'] for v in names]
        
        axes[0].bar(names, precisions)
        axes[0].set_title('Ablation Study: Component Contributions')
        axes[0].set_ylabel('Precision @ 5')
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
    
    # Per-query comparison if available
    if 'per_query' in results:
        precisions = [
            v.get('precision@5', 0)
            for v in results['per_query'].values()
            if isinstance(v, dict)
        ]
        
        if precisions:
            axes[1].hist(precisions, bins=10, edgecolor='black', alpha=0.7)
            axes[1].set_title('Distribution of Precision @ 5')
            axes[1].set_xlabel('Precision Score')
            axes[1].set_ylabel('Frequency')
            axes[1].axvline(np.mean(precisions), color='r', linestyle='--', label=f'Mean: {np.mean(precisions):.3f}')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    logging.info(f"Visualization saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument('--eval_dir', type=Path, default=Path('evaluation_50queries'))
    parser.add_argument('--baseline_dir', type=Path, default=Path('baseline_results'))
    parser.add_argument('--ablation_dir', type=Path, default=Path('ablation_results'))
    parser.add_argument('--out_dir', type=Path, default=Path('analysis_results'))
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    results_to_analyze = {}
    
    # Load evaluation results
    eval_file = args.eval_dir / "evaluation_results.json"
    if eval_file.exists():
        logging.info(f"Loading evaluation results from {eval_file}")
        results_to_analyze['Full System (50 queries)'] = load_results(eval_file)
    
    # Load baseline
    baseline_file = args.baseline_dir / "baseline_results.json"
    if baseline_file.exists():
        logging.info(f"Loading baseline from {baseline_file}")
        results_to_analyze['Vector-Only Baseline'] = load_results(baseline_file)
    
    # Load ablation
    ablation_file = args.ablation_dir / "ablation_results.json"
    if ablation_file.exists():
        logging.info(f"Loading ablation results from {ablation_file}")
        ablation_results = load_results(ablation_file)
        
        # Convert ablation variants to separate entries
        for variant_name, variant_data in ablation_results.get('ablation_variants', {}).items():
            results_to_analyze[f"Ablation: {variant_name}"] = {
                'metrics': {'precision@5': variant_data['metrics']['precision@5']},
                'per_query': variant_data.get('per_query', {}),
            }
    
    # Create comparison table
    table_file = args.out_dir / "comparison_table.md"
    create_comparison_table(results_to_analyze, table_file)
    logging.info(f"Comparison table saved to {table_file}")
    
    # Compare main system vs baseline
    if 'Full System (50 queries)' in results_to_analyze and 'Vector-Only Baseline' in results_to_analyze:
        comparison = compare_methods(
            results_to_analyze['Vector-Only Baseline'],
            results_to_analyze['Full System (50 queries)'],
            args.out_dir
        )
        
        comp_file = args.out_dir / "system_vs_baseline.json"
        with open(comp_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("\n" + "="*70)
        print("SYSTEM vs BASELINE COMPARISON")
        print("="*70)
        print(f"\nBaseline (Vector-Only):")
        print(f"  Mean Precision: {comparison['baseline']['mean']:.4f}")
        print(f"  Std Dev:        {comparison['baseline']['std']:.4f}")
        print(f"  95% CI:         [{comparison['baseline']['ci_lower']:.4f}, {comparison['baseline']['ci_upper']:.4f}]")
        
        print(f"\nFull System:")
        print(f"  Mean Precision: {comparison['system']['mean']:.4f}")
        print(f"  Std Dev:        {comparison['system']['std']:.4f}")
        print(f"  95% CI:         [{comparison['system']['ci_lower']:.4f}, {comparison['system']['ci_upper']:.4f}]")
        
        print(f"\nImprovement:")
        print(f"  Absolute:       {comparison['improvement']['absolute']:.4f}")
        print(f"  Relative:       {comparison['improvement']['relative_percent']:.2f}%")
        
        sig = comparison['statistical_significance']
        if sig['p_value'] is not None:
            print(f"\nStatistical Significance (paired t-test):")
            print(f"  p-value:        {sig['p_value']:.4f}")
            print(f"  Significant:    {'Yes' if sig['significant_at_0.05'] else 'No'} (α=0.05)")
        
        print("\n" + "="*70 + "\n")
    
    # Create visualizations
    viz_file = args.out_dir / "analysis_visualization.png"
    if 'Full System (50 queries)' in results_to_analyze:
        create_visualization(results_to_analyze['Full System (50 queries)'], viz_file)
    
    print(f"Analysis complete. Results saved to {args.out_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
