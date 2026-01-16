"""Run evaluation prompts and save results."""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.common.config import SearchConfig
from src.retriever.search import search_and_rerank
from src.evaluation.contact_sheet import create_contact_sheet


# Default evaluation prompts - from assignment requirements
DEFAULT_PROMPTS = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def run_evaluation(
    prompts: List[str],
    index_dir: Path,
    img_root: Path,
    out_dir: Path,
    config: SearchConfig,
    create_sheets: bool = True
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'prompts': []
    }
    
    for i, query in enumerate(prompts, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing prompt {i}/{len(prompts)}: {query}")
        logging.info(f"{'='*80}")
        
        results = search_and_rerank(
            query=query,
            index_dir=index_dir,
            img_root=img_root,
            config=config,
            baseline=False
        )
        
        baseline_results = search_and_rerank(
            query=query,
            index_dir=index_dir,
            img_root=img_root,
            config=config,
            baseline=True
        )
        
        prompt_results = {
            'query': query,
            'main_results': [
                {
                    'rank': j + 1,
                    'path': r['path'],
                    'filename': r['filename'],
                    'caption': r['caption'],
                    'vec_score': r['vec_score'],
                    'itm_score': r['itm_score'],
                    'cons_score': r['cons_score'],
                    'final_score': r['final_score'],
                    'penalty_applied': r.get('penalty_applied', False),
                    'dominant_color': r.get('dominant_color', 'unknown'),
                    'color_match': r.get('color_match', 'none'),
                    'matched_probes': r.get('matched_probes', []),
                    'probe_scores': r.get('probe_scores', []),
                }
                for j, r in enumerate(results)
            ],
            'baseline_results': [
                {
                    'rank': j + 1,
                    'path': r['path'],
                    'filename': r['filename'],
                    'caption': r['caption'],
                    'vec_score': r['vec_score'],
                    'final_score': r['final_score'],
                }
                for j, r in enumerate(baseline_results)
            ]
        }
        
        all_results['prompts'].append(prompt_results)
        
        if create_sheets:
            sheet_path = out_dir / f"prompt_{i:02d}_main.jpg"
            create_contact_sheet(results, sheet_path, add_labels=True)
            
            baseline_sheet_path = out_dir / f"prompt_{i:02d}_baseline.jpg"
            create_contact_sheet(baseline_results, baseline_sheet_path, add_labels=True)
    
    results_path = out_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"\nSaved evaluation results to {results_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation prompts and save results"
    )
    parser.add_argument(
        '--index_dir',
        type=Path,
        default=Path('artifacts'),
        help='Directory containing index artifacts'
    )
    parser.add_argument(
        '--img_root',
        type=Path,
        default=Path('/home/shiv2077/dev/constraint-fused-fashion-retrieval/val_test2020/test'),
        help='Root directory for image paths'
    )
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=Path('outputs'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='Number of results per prompt'
    )
    parser.add_argument(
        '--prompts_file',
        type=Path,
        default=None,
        help='JSON file with custom prompts (list of strings)'
    )
    parser.add_argument(
        '--no_sheets',
        action='store_true',
        help='Skip creating contact sheets'
    )
    
    args = parser.parse_args()
    
    if args.prompts_file and args.prompts_file.exists():
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
        logging.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS
        logging.info(f"Using {len(prompts)} default prompts")
    
    config = SearchConfig(topk=args.topk)
    
    run_evaluation(
        prompts=prompts,
        index_dir=args.index_dir,
        img_root=args.img_root,
        out_dir=args.out_dir,
        config=config,
        create_sheets=not args.no_sheets
    )
    
    logging.info("\nEvaluation complete!")
    
    print("\n" + "="*80)
    print("COMPUTING EVALUATION METRICS")
    print("="*80)
    
    try:
        from src.evaluation.compute_metrics import automatic_evaluation, print_metrics_report
        
        results_file = str(args.out_dir / "results.json")
        metrics = automatic_evaluation(results_file)
        
        metrics_file = args.out_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Metrics saved to {metrics_file}")
        print_metrics_report(metrics)
    except Exception as e:
        print(f"⚠️  Could not compute metrics: {e}")
        print("You can compute them manually later with:")
        print(f"  python -m src.evaluation.compute_metrics --results_file {args.out_dir}/results.json")


if __name__ == '__main__':
    main()
