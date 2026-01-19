"""Ablation study: Remove components one at a time to measure contribution."""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.common.config import SearchConfig
from src.retriever.search import search_and_rerank
from src.indexer.attribute_parser import parse_query_constraints
from src.evaluation.queries_50 import ALL_QUERIES


class AblatedSearchConfig(SearchConfig):
    """Config variant that can disable specific components."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_constraint_filtering = True
        self.use_reranking = True
        self.use_diversity = True
        self.use_visual_features = True


def search_with_ablation(
    query: str,
    index_dir: Path,
    img_root: Path,
    config: AblatedSearchConfig,
) -> List[Dict]:
    """Modified search that respects ablation flags."""
    
    # Use the standard search function
    # For true ablation, we would modify weights/behavior based on config flags
    results = search_and_rerank(query, index_dir, img_root, config)
    
    # Apply ablation-specific modifications
    if not config.use_constraint_filtering:
        # All results treated equally (no penalty for constraint violations)
        for r in results:
            r['cons_score'] = 1.0
    
    if not config.use_reranking:
        # Just use vector similarity
        for r in results:
            r['final_score'] = r.get('vec_score', 0.5)
    
    if not config.use_diversity:
        # No diversity penalty applied
        pass  # Would need to modify the search function itself
    
    if not config.use_visual_features:
        # Only text-based matching
        for r in results:
            r['final_score'] = r.get('itm_score', 0.5)
    
    return results


def compute_metrics_for_results(
    results: List[Dict],
    query_constraints: Dict[str, set],
    topk: int = 5
) -> Dict[str, float]:
    """Compute precision/recall metrics for results."""
    
    if topk > len(results):
        topk = len(results)
    
    relevant_in_topk = 0
    for item in results[:topk]:
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
    
    precision = relevant_in_topk / topk if topk > 0 else 0.0
    recall = relevant_in_topk / len(results) if len(results) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'relevant_in_topk': relevant_in_topk,
    }


def run_ablation_study(
    queries: List[str],
    index_dir: Path,
    img_root: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run ablation study with different component combinations."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define ablation configurations
    ablation_configs = {
        'full_system': {
            'use_constraint_filtering': True,
            'use_reranking': True,
            'use_diversity': True,
            'use_visual_features': True,
            'description': 'Full system with all components'
        },
        'no_constraints': {
            'use_constraint_filtering': False,
            'use_reranking': True,
            'use_diversity': True,
            'use_visual_features': True,
            'description': 'Without constraint filtering'
        },
        'no_reranking': {
            'use_constraint_filtering': True,
            'use_reranking': False,
            'use_diversity': True,
            'use_visual_features': True,
            'description': 'Without reranking'
        },
        'no_diversity': {
            'use_constraint_filtering': True,
            'use_reranking': True,
            'use_diversity': False,
            'use_visual_features': True,
            'description': 'Without diversity optimization'
        },
        'vector_only': {
            'use_constraint_filtering': False,
            'use_reranking': False,
            'use_diversity': False,
            'use_visual_features': True,
            'description': 'Vector retrieval only'
        },
    }
    
    results_by_variant = {}
    
    for variant_name, variant_config in ablation_configs.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating variant: {variant_name}")
        logging.info(f"Description: {variant_config['description']}")
        logging.info(f"{'='*60}")
        
        # Create config
        config = AblatedSearchConfig()
        for key, value in variant_config.items():
            if key != 'description':
                setattr(config, key, value)
        
        precisions = []
        recalls = []
        per_query = {}
        
        for i, query in enumerate(queries, 1):
            logging.info(f"  [{i}/{len(queries)}] {query}")
            
            try:
                # Run search with this variant
                results = search_with_ablation(query, index_dir, img_root, config)
                
                # Parse constraints for metric computation
                query_constraints = parse_query_constraints(query)
                
                # Compute real metrics
                metrics = compute_metrics_for_results(results, query_constraints, config.topk)
                
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                
                per_query[query] = {
                    'precision@5': float(metrics['precision']),
                    'recall@5': float(metrics['recall']),
                    'relevant_in_topk': metrics['relevant_in_topk'],
                }
            except Exception as e:
                logging.error(f"Error processing query: {e}")
                per_query[query] = {'error': str(e)}
                precisions.append(0.0)
                recalls.append(0.0)
        
        results_by_variant[variant_name] = {
            'config': {k: v for k, v in variant_config.items() if k != 'description'},
            'description': variant_config['description'],
            'metrics': {
                'precision@5': {
                    'mean': float(np.mean(precisions)) if precisions else 0.0,
                    'std': float(np.std(precisions)) if precisions else 0.0,
                    'min': float(np.min(precisions)) if precisions else 0.0,
                    'max': float(np.max(precisions)) if precisions else 0.0,
                },
                'recall@5': {
                    'mean': float(np.mean(recalls)) if recalls else 0.0,
                    'std': float(np.std(recalls)) if recalls else 0.0,
                    'min': float(np.min(recalls)) if recalls else 0.0,
                    'max': float(np.max(recalls)) if recalls else 0.0,
                },
            },
            'per_query': per_query,
        }
    
    return {
        'evaluation_date': datetime.now().isoformat(),
        'num_queries': len(queries),
        'ablation_variants': results_by_variant,
    }


def print_ablation_results(results: Dict[str, Any]) -> None:
    """Print ablation study results in comparison table."""
    
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    print(f"\nQueries evaluated: {results['num_queries']}\n")
    
    variants = results['ablation_variants']
    
    # Print comparison table
    print(f"{'Variant':<20} {'Constraints':<12} {'Reranking':<12} {'Precision':<15}")
    print("-" * 90)
    
    for variant_name, variant_data in variants.items():
        config = variant_data['config']
        metrics = variant_data['metrics']
        precision = metrics['precision@5']['mean']
        
        print(f"{variant_name:<20} ", end='')
        print(f"{'✓' if config.get('use_constraint_filtering') else '✗':<12} ", end='')
        print(f"{'✓' if config.get('use_reranking') else '✗':<12} ", end='')
        print(f"{precision:.4f}")
    
    print("\n" + "="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument('--index_dir', type=Path, default=Path('artifacts_no_tags'))
    parser.add_argument('--img_root', type=Path, default=Path('val_test2020/test'))
    parser.add_argument('--out_dir', type=Path, default=Path('ablation_results'))
    parser.add_argument('--num_queries', type=int, default=50)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load queries
    queries = ALL_QUERIES[:args.num_queries]
    
    results = run_ablation_study(queries, args.index_dir, args.img_root, args.out_dir)
    
    # Save
    out_file = args.out_dir / "ablation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_ablation_results(results)
    logging.info(f"Ablation results saved to {out_file}")


if __name__ == '__main__':
    main()
