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


def extract_dominant_color(img: Image.Image, num_colors: int = 5) -> str:
    """
    Extract dominant color from image using HSV-based clustering.
    
    Args:
        img: PIL Image
        num_colors: Number of dominant colors to consider
        
    Returns:
        String name of dominant color (red/blue/green/yellow/white/black/etc.)
    """
    from collections import Counter
    
    img_array = np.array(img.convert('RGB'))
    pixels = img_array.reshape(-1, 3)
    
    hsv_img = np.array(img.convert('HSV'))
    h_pixels = hsv_img.reshape(-1, 3)[:, 0]
    
    color_ranges = {
        'red': [(0, 15), (345, 360)],
        'orange': [(15, 45)],
        'yellow': [(45, 65)],
        'green': [(65, 155)],
        'cyan': [(85, 100)],
        'blue': [(100, 265)],
        'purple': [(265, 290)],
        'magenta': [(290, 330)],
        'pink': [(330, 345)],
    }
    
    color_counts = Counter()
    
    for h in h_pixels:
        for color_name, ranges in color_ranges.items():
            for start, end in ranges:
                if start <= h <= end:
                    color_counts[color_name] += 1
                    break
    
    if color_counts:
        dominant_color = color_counts.most_common(1)[0][0]
    else:
        dominant_color = 'neutral'
    
    brightness = np.mean(np.max(pixels, axis=1))
    if brightness > 200:
        if dominant_color == 'neutral':
            return 'white'
    elif brightness < 50:
        if dominant_color == 'neutral':
            return 'black'
    
    return dominant_color
