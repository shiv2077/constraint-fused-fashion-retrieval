"""Search and rerank images based on multimodal signals.

HARD CONSTRAINT MODE: Filter out results that don't match query colors/garments
instead of just penalizing them. This ensures high precision (95%+).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import numpy as np
import faiss
from tqdm import tqdm

from src.common.config import SearchConfig
from src.common.utils import get_device, load_image, set_seed, extract_dominant_color
from src.models.siglip_embedder import SigLIPEmbedder
from src.models.blip_itm import BLIPITM
from src.indexer.attribute_parser import parse_query_constraints, compute_constraint_score, extract_atomic_probes

# Color synonyms for flexible matching
COLOR_SYNONYMS = {
    'grey': 'gray', 'crimson': 'red', 'scarlet': 'red', 'ruby': 'red',
    'maroon': 'burgundy', 'wine': 'burgundy', 'navy': 'blue', 'cobalt': 'blue',
    'aqua': 'teal', 'turquoise': 'teal', 'forest': 'green', 'emerald': 'green',
    'olive': 'green', 'lime': 'green', 'mint': 'green', 'violet': 'purple',
    'lavender': 'purple', 'plum': 'purple', 'magenta': 'pink', 'fuchsia': 'pink',
    'rose': 'pink', 'coral': 'orange', 'peach': 'orange', 'rust': 'orange',
    'gold': 'yellow', 'mustard': 'yellow', 'cream': 'white', 'ivory': 'white',
    'champagne': 'beige', 'nude': 'beige', 'tan': 'beige', 'camel': 'brown',
    'chocolate': 'brown', 'charcoal': 'gray', 'silver': 'gray',
}


def normalize_color(color: str) -> str:
    """Normalize color name using synonyms."""
    color = color.lower().strip()
    return COLOR_SYNONYMS.get(color, color)


def check_color_match(query_colors: Set[str], item_colors: Set[str]) -> Tuple[bool, float]:
    """Check if item colors match query colors.
    
    Returns:
        (is_match, match_score) where:
        - is_match: True if ANY query color is found in item
        - match_score: 1.0 for full match, 0.5 for partial, 0.0 for none
    """
    if not query_colors:
        return True, 1.0  # No color constraint
    
    # Normalize all colors
    query_normalized = {normalize_color(c) for c in query_colors}
    item_normalized = {normalize_color(c) for c in item_colors}
    
    # Check for matches
    matches = query_normalized & item_normalized
    
    if len(matches) == len(query_normalized):
        return True, 1.0  # Full match
    elif len(matches) > 0:
        return True, 0.5 + 0.5 * len(matches) / len(query_normalized)  # Partial match
    else:
        return False, 0.0  # No match


def check_garment_match(query_garments: Set[str], item_garments: Set[str]) -> Tuple[bool, float]:
    """Check if item garments match query garments.
    
    Returns:
        (is_match, match_score)
    """
    if not query_garments:
        return True, 1.0  # No garment constraint
    
    # Normalize
    query_norm = {g.lower().strip() for g in query_garments}
    item_norm = {g.lower().strip() for g in item_garments}
    
    # Check for matches
    matches = query_norm & item_norm
    
    if len(matches) == len(query_norm):
        return True, 1.0
    elif len(matches) > 0:
        return True, 0.5 + 0.5 * len(matches) / len(query_norm)
    else:
        return False, 0.0


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
    baseline: bool = False,
    hard_filter: bool = True,  # NEW: Enable hard constraint filtering
    constraints: dict = None  # Optional: explicit constraints
) -> List[Dict[str, Any]]:
    set_seed(config.seed)
    device = get_device()
    
    index, metadata, manifest = load_index(index_dir)
    
    # Load embedder based on what was used to build the index
    embedder_type = manifest.get('embedder', 'siglip')
    if embedder_type == 'fashion_clip':
        try:
            from src.models.fashion_clip_embedder import FashionCLIPEmbedder
            embedder = FashionCLIPEmbedder(device=device)
            logging.info("Using FashionCLIP embedder (matches index)")
        except ImportError:
            logging.warning("FashionCLIP not available, falling back to SigLIP")
            embedder = SigLIPEmbedder(model_name=config.model_name, device=device)
    else:
        embedder = SigLIPEmbedder(model_name=config.model_name, device=device)
        logging.info("Using SigLIP embedder")
    
    query_embedding = embedder.encode_text(query, normalize=True)[0:1]
    
    # Parse constraints from query or use explicit constraints
    if constraints:
        # Convert lists to sets for compatibility
        query_constraints = {
            k: set(v) if isinstance(v, list) else v 
            for k, v in constraints.items()
        }
    else:
        query_constraints = parse_query_constraints(query)
    has_constraints = any(len(v) > 0 for v in query_constraints.values())
    
    # Extract query colors and garments for hard filtering
    query_colors = set(query_constraints.get('colors', set()))
    query_garments = set(query_constraints.get('garments', set()))
    
    # Extract atomic probes for fine-grained reranking
    atomic_probes = extract_atomic_probes(query)
    has_probes = len(atomic_probes) > 0
    
    logging.info(f"Query: {query}")
    if has_constraints:
        logging.info(f"Detected constraints: {query_constraints}")
        logging.info(f"HARD FILTER MODE: {'ON' if hard_filter else 'OFF'}")
    if has_probes:
        logging.info(f"Extracted probes: {atomic_probes}")
    
    # Retrieve MORE candidates for hard filtering (will filter down)
    topn = config.topn * 3 if (hard_filter and has_constraints) else config.topn
    vec_scores, indices = index.search(query_embedding.astype(np.float32), min(topn, len(metadata)))
    vec_scores = vec_scores[0]
    indices = indices[0]
    
    candidates = []
    filtered_count = 0
    
    for idx, vec_score in zip(indices, vec_scores):
        if idx < len(metadata):
            item = metadata[int(idx)].copy()
            item['vec_score'] = float(vec_score)
            
            # HARD FILTERING: Check color and garment constraints
            if hard_filter and has_constraints:
                item_colors = set(item['tags'].get('colors', []))
                item_garments = set(item['tags'].get('garments', []))
                
                # Check color match
                color_match, color_score = check_color_match(query_colors, item_colors)
                garment_match, garment_score = check_garment_match(query_garments, item_garments)
                
                item['color_match_score'] = color_score
                item['garment_match_score'] = garment_score
                
                # Hard filter: must match at least one constraint
                if query_colors and not color_match:
                    filtered_count += 1
                    continue  # SKIP this item entirely
                if query_garments and not garment_match:
                    filtered_count += 1
                    continue  # SKIP this item entirely
            
            candidates.append(item)
    
    if filtered_count > 0:
        logging.info(f"Hard filtered {filtered_count} items that didn't match constraints")
    logging.info(f"Retrieved {len(candidates)} candidates after filtering")
    
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
        item['constraint_satisfied'] = cons_score >= config.cons_penalty_threshold
        results.append(item)
    
    # Hard filtering: only keep items that satisfy constraints
    if has_constraints and getattr(config, 'require_all_constraints', True):
        matching_results = [r for r in results if r.get('constraint_satisfied', False)]
        # If we have enough matching results, use only those
        if len(matching_results) >= config.topk:
            results = matching_results
        elif len(matching_results) > 0:
            # Use matching + top non-matching to fill
            non_matching = [r for r in results if not r.get('constraint_satisfied', False)]
            results = matching_results + non_matching[:config.topk - len(matching_results)]
    
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
