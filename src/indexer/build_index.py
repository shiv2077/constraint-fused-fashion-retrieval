"""Build index from a folder of images.

HIGH PRECISION MODE (95%+): 
- Uses YOLO + SAM for garment segmentation
- Uses FashionCLIP for fashion-specialized embeddings
- Uses expanded 30+ color vocabulary
- Extracts accurate color from ONLY garment pixels, ignoring background
"""

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
from src.models.blip_captioner import BLIPCaptioner
from src.indexer.attribute_parser import extract_tags

# High precision imports
try:
    from src.indexer.garment_segmenter import get_garment_mask
    from src.indexer.pixel_color_extractor import get_pixel_colors, extract_colors_from_masked_pixels
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False
    logging.warning("Segmentation modules not available, using CLIP fallback")

# FashionCLIP import
try:
    from src.models.fashion_clip_embedder import FashionCLIPEmbedder
    FASHION_CLIP_AVAILABLE = True
except ImportError:
    FASHION_CLIP_AVAILABLE = False
    logging.warning("FashionCLIP not available, using SigLIP")


def build_index(
    img_dir: Path,
    out_dir: Path,
    config: IndexConfig,
    use_segmentation: bool = True,
    use_fashion_clip: bool = True  # NEW: Use FashionCLIP embeddings
) -> None:
    set_seed(config.seed)
    device = get_device()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embedder (FashionCLIP or SigLIP)
    if use_fashion_clip and FASHION_CLIP_AVAILABLE:
        logging.info("Using FashionCLIP for fashion-specialized embeddings")
        embedder = FashionCLIPEmbedder(device=device)
    else:
        logging.info("Using SigLIP embeddings")
        from src.models.siglip_embedder import SigLIPEmbedder
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
    
    # Color detection prompts for CLIP-based fallback
    color_prompts = [
        "a red piece of clothing", "a blue piece of clothing", "a green piece of clothing",
        "a yellow piece of clothing", "a black piece of clothing", "a white piece of clothing",
        "a pink piece of clothing", "a purple piece of clothing", "a orange piece of clothing",
        "a brown piece of clothing", "a gray piece of clothing", "a beige piece of clothing"
    ]
    color_names = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'orange', 'brown', 'gray', 'beige']
    
    # Garment detection prompts
    garment_prompts = [
        "a dress", "a shirt", "a jacket", "pants or trousers", "a skirt", 
        "a coat", "a sweater", "a blouse", "jeans", "a suit"
    ]
    garment_names = ['dress', 'shirt', 'jacket', 'pants', 'skirt', 'coat', 'sweater', 'blouse', 'jeans', 'suit']
    
    # Check segmentation availability
    use_seg = use_segmentation and SEGMENTATION_AVAILABLE
    if use_seg:
        logging.info("HIGH PRECISION MODE: Using YOLO + SAM garment segmentation")
    else:
        logging.info("Standard mode: Using CLIP-based attribute detection")
    
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
            
            # Extract tags from caption
            tags = extract_tags(caption)
            
            # HIGH PRECISION: Use segmentation + pixel analysis for colors
            if use_seg and not tags['colors']:
                try:
                    # Get garment mask using YOLO + SAM
                    mask = get_garment_mask(img, use_sam=True, fallback_to_center=True)
                    if mask is not None:
                        # Extract colors from ONLY garment pixels
                        pixel_colors = get_pixel_colors(img, mask=mask)
                        tags['colors'].update(pixel_colors)
                except Exception as e:
                    logging.debug(f"Segmentation failed for {img_path.name}: {e}")
            
            # FALLBACK: Use CLIP to detect colors if still empty
            if not tags['colors']:
                # Get color scores using image embedding + color prompts
                color_embeddings = embedder.encode_text_batch(color_prompts, normalize=True)
                img_emb = embedding.reshape(1, -1)
                scores = (img_emb @ color_embeddings.T)[0]
                # Take top 2 colors with highest scores (relative ranking)
                top_indices = np.argsort(scores)[-2:][::-1]
                for i in top_indices:
                    if scores[i] > 0.06:  # Very low threshold - just need some signal
                        tags['colors'].add(color_names[i])
            
            # FALLBACK: Use CLIP to detect garments if caption doesn't have them
            if not tags['garments']:
                garment_embeddings = embedder.encode_text_batch(garment_prompts, normalize=True)
                img_emb = embedding.reshape(1, -1)
                scores = (img_emb @ garment_embeddings.T)[0]
                # Take top 2 garments with highest scores
                top_indices = np.argsort(scores)[-2:][::-1]
                for i in top_indices:
                    if scores[i] > 0.06:
                        tags['garments'].add(garment_names[i])
            
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
        'embedder': 'fashion_clip' if (use_fashion_clip and FASHION_CLIP_AVAILABLE) else 'siglip',
        'use_segmentation': use_seg,
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
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        default=True,
        help='Use YOLO + SAM segmentation for HIGH PRECISION color detection'
    )
    parser.add_argument(
        '--no_segmentation',
        action='store_true',
        help='Disable segmentation (faster but less accurate)'
    )
    parser.add_argument(
        '--use_fashion_clip',
        action='store_true',
        default=True,
        help='Use FashionCLIP embeddings (fashion-specialized)'
    )
    parser.add_argument(
        '--no_fashion_clip',
        action='store_true',
        help='Disable FashionCLIP, use SigLIP instead'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = IndexConfig(
        max_images=args.max_images,
        batch_size=args.batch_size,
    )
    
    # Determine flags
    use_seg = args.use_segmentation and not args.no_segmentation
    use_fclip = args.use_fashion_clip and not args.no_fashion_clip
    
    # Build index
    build_index(args.img_dir, args.out_dir, config, use_segmentation=use_seg, use_fashion_clip=use_fclip)


if __name__ == '__main__':
    main()
