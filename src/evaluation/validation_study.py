"""Validation study: Manual annotation of results to verify methodology."""

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


class AnnotationInterface:
    """Simple text-based interface for manual annotation."""
    
    def __init__(self, results_per_query: int = 5):
        self.results_per_query = results_per_query
        self.annotations = {}
    
    def display_result(self, query: str, results: List[Dict], idx: int) -> str:
        """Display a result and get user annotation."""
        
        result = results[idx]
        print(f"\n  [{idx+1}/{self.results_per_query}] {result['filename']}")
        print(f"       Similarity: {result.get('final_score', 0):.4f}")
        print(f"       Relevant? [y/n/?: ", end='', flush=True)
        
        annotation = input().strip().lower()
        if annotation not in ['y', 'n', '?']:
            annotation = '?'
        
        return annotation
    
    def annotate_results(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """Get manual annotations for query results."""
        
        annotations = {'query': query, 'results': []}
        
        for idx, result in enumerate(results[:self.results_per_query]):
            relevance = self.display_result(query, results, idx)
            annotations['results'].append({
                'filename': result['filename'],
                'relevance': relevance,
            })
        
        return annotations


def compute_inter_annotator_agreement(
    manual_annotations: List[Dict],
    auto_annotations: List[Dict],
) -> float:
    """Compute agreement between manual and automatic annotations."""
    
    if not manual_annotations or not auto_annotations:
        return 0.0
    
    agreement_count = 0
    for manual, auto in zip(manual_annotations, auto_annotations):
        manual_rel = manual.get('relevance') == 'y'
        auto_rel = auto.get('is_relevant', False)
        
        if manual_rel == auto_rel:
            agreement_count += 1
    
    return agreement_count / len(manual_annotations) if manual_annotations else 0.0


def run_validation_study(
    queries: List[str],
    index_dir: Path,
    img_root: Path,
    out_dir: Path,
    config: SearchConfig,
    interactive: bool = False,
) -> Dict[str, Any]:
    """Run validation study with optional manual annotation."""
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    validation_results = {
        'evaluation_date': datetime.now().isoformat(),
        'interactive': interactive,
        'num_queries': len(queries),
        'manual_annotations': [],
        'inter_annotator_agreement': [],
    }
    
    # Process all queries
    for i, query in enumerate(queries, 1):
        logging.info(f"[{i}/{len(queries)}] Validating: {query}")
        
        try:
            # Retrieve results
            results = search_and_rerank(query, index_dir, img_root, config)
            
            # Automated annotation based on constraints
            query_constraints = parse_query_constraints(query)
            auto_annotations = []
            
            for result in results[:config.topk]:
                try:
                    item_tags = {
                        'colors': set(result.get('tags', {}).get('colors', [])),
                        'garments': set(result.get('tags', {}).get('garments', [])),
                        'contexts': set(result.get('tags', {}).get('contexts', [])),
                    }
                    
                    colors_match = all(c in item_tags['colors'] for c in query_constraints.get('colors', set()))
                    garments_match = all(g in item_tags['garments'] for g in query_constraints.get('garments', set()))
                    contexts_match = all(ctx in item_tags['contexts'] for ctx in query_constraints.get('contexts', set()))
                    
                    is_relevant = colors_match and garments_match and contexts_match
                except (KeyError, TypeError):
                    is_relevant = False
                
                auto_annotations.append({
                    'filename': result['filename'],
                    'is_relevant': is_relevant,
                    'final_score': result.get('final_score', 0.0),
                })
            
            result_entry = {
                'query': query,
                'auto_annotations': auto_annotations,
                'manual_annotations': None,
                'agreement': None,
            }
            
            # Optional: Get manual annotations
            if interactive:
                print(f"\n{'='*60}")
                print(f"Manual Annotation for: {query}")
                print(f"{'='*60}")
                
                annotator = AnnotationInterface()
                manual_annot = annotator.annotate_results(query, results)
                result_entry['manual_annotations'] = manual_annot['results']
                
                # Compute agreement
                agreement = compute_inter_annotator_agreement(
                    manual_annot['results'],
                    auto_annotations
                )
                result_entry['agreement'] = float(agreement)
                validation_results['inter_annotator_agreement'].append(agreement)
            
            validation_results['manual_annotations'].append(result_entry)
        except Exception as e:
            logging.error(f"Error validating query '{query}': {e}")
            validation_results['manual_annotations'].append({
                'query': query,
                'error': str(e),
            })
    
    # Compute average agreement if available
    if validation_results['inter_annotator_agreement']:
        validation_results['avg_agreement'] = float(
            np.mean(validation_results['inter_annotator_agreement'])
        )
    
    return validation_results


def print_validation_results(results: Dict[str, Any]) -> None:
    """Print validation study results."""
    
    print("\n" + "="*70)
    print("VALIDATION STUDY RESULTS")
    print("="*70)
    print(f"\nQueries annotated: {results['num_queries']}")
    print(f"Interactive: {results['interactive']}")
    
    if 'avg_agreement' in results:
        print(f"\nInter-Annotator Agreement (Manual vs Auto):")
        print(f"  Mean: {results['avg_agreement']:.4f}")
        print(f"  (Higher is better - indicates methodology is sound)")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validation study with manual annotation")
    parser.add_argument('--index_dir', type=Path, default=Path('artifacts_no_tags'))
    parser.add_argument('--img_root', type=Path, default=Path('val_test2020/test'))
    parser.add_argument('--out_dir', type=Path, default=Path('validation_results'))
    parser.add_argument('--interactive', action='store_true', help='Enable manual annotation')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = SearchConfig()
    queries = ALL_QUERIES
    
    results = run_validation_study(
        queries,
        args.index_dir,
        args.img_root,
        args.out_dir,
        config,
        interactive=args.interactive
    )
    
    # Save
    out_file = args.out_dir / "validation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_validation_results(results)
    logging.info(f"Validation results saved to {out_file}")


if __name__ == '__main__':
    main()
