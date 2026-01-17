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
from src.common.utils import get_device, load_image, set_seed, extract_dominant_color
from src.models.siglip_embedder import SigLIPEmbedder
from src.models.blip_itm import BLIPITM
from src.indexer.attribute_parser import parse_query_constraints, compute_constraint_score, extract_atomic_probes


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
    
    # Extract atomic probes for fine-grained reranking
    atomic_probes = extract_atomic_probes(query)
    has_probes = len(atomic_probes) > 0
    
    logging.info(f"Query: {query}")
    if has_constraints:
        logging.info(f"Detected constraints: {query_constraints}")
    if has_probes:
        logging.info(f"Extracted probes: {atomic_probes}")
    
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
            item['probe_scores'] = []
            item['color_match'] = 'none'
            results.append(item)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:config.topk]
    
    itm_model = BLIPITM(model_name=config.itm_model, device=device)
    
    logging.info("Computing ITM scores and color features...")
    images = []
    valid_candidates = []
    
    for item in tqdm(candidates, desc="Loading images"):
        img_path = Path(item['path'])
        if not img_path.exists():
            img_path = img_root / item['filename']
        
        img = load_image(img_path)
        if img is not None:
            images.append(img)
            # Extract dominant color and store in item
            dominant_color = extract_dominant_color(img)
            item['dominant_color'] = dominant_color
            valid_candidates.append(item)
        else:
            logging.warning(f"Failed to load image: {img_path}")
    
    if not images:
        logging.error("No valid images to rerank")
        return []
    
    # Compute ITM scores for full query
    batch_size = 8
    itm_scores = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_scores = itm_model.score(batch_images, query)
        itm_scores.append(batch_scores)
    itm_scores = np.concatenate(itm_scores)
    
    # Compute probe-based ITM scores if probes exist
    probe_itm_scores = None
    if has_probes:
        logging.info(f"Computing ITM scores for {len(atomic_probes)} probes...")
        probe_itm_scores = {probe: [] for probe in atomic_probes}
        
        for probe in atomic_probes:
            probe_scores = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_scores = itm_model.score(batch_images, probe)
                probe_scores.append(batch_scores)
            probe_itm_scores[probe] = np.concatenate(probe_scores)
    
    # Compute constraint score for each item (including color)
    results = []
    for idx, (item, itm_score) in enumerate(zip(valid_candidates, itm_scores)):
        item['itm_score'] = float(itm_score)
        
        item_tags = {
            'colors': set(item['tags']['colors']),
            'garments': set(item['tags']['garments']),
            'contexts': set(item['tags']['contexts']),
        }
        cons_score = compute_constraint_score(query_constraints, item_tags)
        item['cons_score'] = float(cons_score)
        
        # Probe-based scoring
        probe_scores = []
        matched_probes = []
        if has_probes and probe_itm_scores:
            for probe in atomic_probes:
                probe_score = float(probe_itm_scores[probe][idx])
                probe_scores.append(probe_score)
                if probe_score > 0.5:  # Threshold for matching
                    matched_probes.append(probe)
        item['probe_scores'] = probe_scores
        item['matched_probes'] = matched_probes
        
        # Color matching bonus/penalty
        dominant_color = item.get('dominant_color', 'none')
        query_colors = query_constraints.get('colors', set())
        color_match = 'none'
        color_bonus = 0.0
        
        if dominant_color != 'none' and query_colors:
            # Check if dominant color is in query
            if dominant_color in query_colors:
                color_match = 'exact'
                color_bonus = 0.20
            else:
                color_match = 'none'
                color_bonus = -0.02
        else:
            color_match = 'none'
        
        item['color_match'] = color_match
        
        # Fuse scores: vec + itm + constraints + probes + color
        vec_component = config.w_vec * item['vec_score']
        itm_component = config.w_itm * item['itm_score']
        cons_component = config.w_cons * item['cons_score']
        
        # Probe component: average of probe scores if any probes matched
        probe_component = 0.0
        if probe_scores:
            probe_mean = np.mean(probe_scores)
            probe_component = 0.25 * probe_mean  # Weight for probe matching
        
        if has_constraints and cons_score < config.cons_penalty_threshold:
            penalty = config.cons_penalty_factor
        else:
            penalty = 1.0
        
        final_score = penalty * (vec_component + itm_component + cons_component + probe_component + color_bonus)
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
        print(f"  Detected Tags: {', '.join(item['tags'].get('colors', []) + item['tags'].get('garments', []) + item['tags'].get('contexts', []))}")
        print(f"  Dominant Color: {item.get('dominant_color', 'unknown')}")
        
        print(f"  Scores:")
        print(f"    Vector Similarity: {item['vec_score']:.4f}")
        print(f"    Image-Text Matching: {item['itm_score']:.4f}")
        print(f"    Constraint Satisfaction: {item['cons_score']:.4f}")
        
        # Probe information
        if item.get('probe_scores'):
            avg_probe = np.mean(item['probe_scores']) if item['probe_scores'] else 0.0
            print(f"    Attribute Probe Avg: {avg_probe:.4f}")
            if item.get('matched_probes'):
                print(f"    Matched Probes: {', '.join(item['matched_probes'])}")
        
        # Color match info
        if item.get('color_match') != 'none':
            print(f"    Color Match: {item['color_match']}")
        
        print(f"    Final Score: {item['final_score']:.4f}")
        
        if item.get('penalty_applied'):
            print(f"    ⚠ Penalty applied (constraint not fully satisfied)")
        
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
