"""Build index from a folder of images."""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from tqdm import tqdm

from src.common.config import IndexConfig, save_config
from src.common.utils import get_device, get_image_files, load_image, set_seed
from src.models.siglip_embedder import SigLIPEmbedder
from src.models.blip_captioner import BLIPCaptioner
from src.indexer.attribute_parser import extract_tags


def build_index(
    img_dir: Path,
    out_dir: Path,
    config: IndexConfig
) -> None:
    set_seed(config.seed)
    device = get_device()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    embedder = SigLIPEmbedder(model_name=config.model_name, device=device)
    captioner = BLIPCaptioner(model_name=config.caption_model, device=device)
    
    # Get image files
    image_files = get_image_files(img_dir, max_images=config.max_images)
    
    if not image_files:
        logging.error(f"No images found in {img_dir}")
        return
    
    # Process images
    embeddings_list = []
    metadata_list = []
    
    logging.info("Processing images...")
    for idx, img_path in enumerate(tqdm(image_files, desc="Indexing")):
        try:
            # Load image
            img = load_image(img_path)
            if img is None:
                logging.warning(f"Skipping {img_path}: failed to load")
                continue
            
            # Generate embedding
            embedding = embedder.encode_image(img, normalize=True)[0]
            
            # Generate caption
            caption = captioner.caption(img)
            
            # Extract tags
            tags = extract_tags(caption)
            
            # Store
            embeddings_list.append(embedding)
            metadata_list.append({
                'id': len(metadata_list),
                'filename': img_path.name,
                'path': str(img_path.absolute()),
                'caption': caption,
                'tags': {
                    'colors': list(tags['colors']),
                    'garments': list(tags['garments']),
                    'contexts': list(tags['contexts']),
                }
            })
            
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            continue
    
    if not embeddings_list:
        logging.error("No images were successfully processed")
        return
    
    logging.info(f"Successfully processed {len(embeddings_list)} images")
    
    # Build FAISS index
    embeddings = np.array(embeddings_list, dtype=np.float32)
    dimension = embeddings.shape[1]
    
    logging.info(f"Building FAISS index with dimension {dimension}")
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index.add(embeddings)
    
    # Save FAISS index
    faiss_path = out_dir / "vectors.faiss"
    faiss.write_index(index, str(faiss_path))
    logging.info(f"Saved FAISS index to {faiss_path}")
    
    # Save metadata
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    logging.info(f"Saved metadata to {metadata_path}")
    
    # Save manifest
    manifest = {
        'num_images': len(metadata_list),
        'embedding_dim': dimension,
        'config': config.to_dict(),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Saved manifest to {manifest_path}")
    
    logging.info("Index building complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build fashion retrieval index from images"
    )
    parser.add_argument(
        '--img_dir',
        type=Path,
        default=Path('/home/shiv2077/dev/constraint-fused-fashion-retrieval/val_test2020/test'),
        help='Directory containing images'
    )
    parser.add_argument(
        '--out_dir',
        type=Path,
        default=Path('artifacts'),
        help='Output directory for index artifacts'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = IndexConfig(
        max_images=args.max_images,
        batch_size=args.batch_size,
    )
    
    # Build index
    build_index(args.img_dir, args.out_dir, config)


if __name__ == '__main__':
    main()
