"""Common utilities for the fashion retrieval system."""

import logging
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available, using CPU")
    return device


def load_image(image_path: Path, convert_rgb: bool = True) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path)
        if convert_rgb and img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return None


def get_image_files(img_dir: Path, max_images: Optional[int] = None) -> List[Path]:
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    image_files = []
    for ext in supported_extensions:
        image_files.extend(img_dir.glob(f"*{ext}"))
        image_files.extend(img_dir.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    logging.info(f"Found {len(image_files)} image files in {img_dir}")
    return image_files


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
