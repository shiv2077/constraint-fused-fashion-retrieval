"""Search and rerank images based on multimodal signals."""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from tqdm import tqdm

from src.common.config import SearchConfig
from src.common.utils import get_device, load_image, set_seed
from src.models.siglip_embedder import SigLIPEmbedder
from src.models.blip_itm import BLIPITM
from src.indexer.attribute_parser import parse_query_constraints, compute_constraint_score


def load_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    faiss_path = index_dir / "vectors.faiss"
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    index = faiss.read_index(str(faiss_path))
    
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    logging.info(f"Loaded index with {len(metadata)} items")
    return index, metadata, manifest


def search_and_rerank(
    query: str,
    index_dir: Path,
    img_root: Path,
    config: SearchConfig,
    baseline: bool = False
) -> List[Dict[str, Any]]:
    set_seed(config.seed)
    device = get_device()
    
    index, metadata, manifest = load_index(index_dir)
    embedder = SigLIPEmbedder(model_name=config.model_name, device=device)
    query_embedding = embedder.encode_text(query, normalize=True)[0:1]
    
    query_constraints = parse_query_constraints(query)
    has_constraints = any(len(v) > 0 for v in query_constraints.values())
    
    logging.info(f"Query: {query}")
    if has_constraints:
        logging.info(f"Detected constraints: {query_constraints}")
    
    topn = config.topn
    vec_scores, indices = index.search(query_embedding.astype(np.float32), topn)
    vec_scores = vec_scores[0]
    indices = indices[0]
    
    candidates = []
    for idx, vec_score in zip(indices, vec_scores):
        if idx < len(metadata):
            item = metadata[int(idx)].copy()
            item['vec_score'] = float(vec_score)
            candidates.append(item)
    
    logging.info(f"Retrieved {len(candidates)} candidates from FAISS")
    
    if baseline:
        results = []
        for item in candidates:
            item['final_score'] = item['vec_score']
            item['itm_score'] = 0.0
            item['cons_score'] = 1.0
            results.append(item)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:config.topk]
    
    itm_model = BLIPITM(model_name=config.itm_model, device=device)
    
    logging.info("Computing ITM scores...")
    images = []
    valid_candidates = []
    
    for item in tqdm(candidates, desc="Loading images"):
        img_path = Path(item['path'])
        if not img_path.exists():
            # Try relative to img_root
            img_path = img_root / item['filename']
        
        img = load_image(img_path)
        if img is not None:
            images.append(img)
            valid_candidates.append(item)
        else:
            logging.warning(f"Failed to load image: {img_path}")
    
    if not images:
        logging.error("No valid images to rerank")
        return []
    
    batch_size = 8
    itm_scores = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_scores = itm_model.score(batch_images, query)
        itm_scores.append(batch_scores)
    itm_scores = np.concatenate(itm_scores)
    
    results = []
    for item, itm_score in zip(valid_candidates, itm_scores):
        item['itm_score'] = float(itm_score)
        
        item_tags = {
            'colors': set(item['tags']['colors']),
            'garments': set(item['tags']['garments']),
            'contexts': set(item['tags']['contexts']),
        }
        cons_score = compute_constraint_score(query_constraints, item_tags)
        item['cons_score'] = float(cons_score)
        
        vec_component = config.w_vec * item['vec_score']
        itm_component = config.w_itm * item['itm_score']
        cons_component = config.w_cons * item['cons_score']
        
        if has_constraints and cons_score < config.cons_penalty_threshold:
            penalty = config.cons_penalty_factor
        else:
            penalty = 1.0
        
        final_score = penalty * (vec_component + itm_component + cons_component)
        item['final_score'] = float(final_score)
        item['penalty_applied'] = penalty < 1.0
        results.append(item)
    
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:config.topk]


def print_results(results: List[Dict[str, Any]], query: str) -> None:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    for i, item in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  File: {item['path']}")
        print(f"  Caption: {item['caption']}")
        print(f"  Scores:")
        print(f"    Vector: {item['vec_score']:.4f}")
        print(f"    ITM: {item['itm_score']:.4f}")
        print(f"    Constraint: {item['cons_score']:.4f}")
        print(f"    Final: {item['final_score']:.4f}")
        if item.get('penalty_applied'):
            print(f"    [Penalty applied for low constraint satisfaction]")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Search fashion images with multimodal reranking"
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
        '--query',
        type=str,
        required=True,
        help='Natural language query'
    )
    parser.add_argument(
        '--topn',
        type=int,
        default=50,
        help='Number of candidates to retrieve'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='Number of final results to return'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Use baseline vector-only retrieval'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = SearchConfig(
        topn=args.topn,
        topk=args.topk,
    )
    
    # Search and rerank
    results = search_and_rerank(
        args.query,
        args.index_dir,
        args.img_root,
        config,
        baseline=args.baseline
    )
    
    # Print results
    print_results(results, args.query)


if __name__ == '__main__':
    main()
