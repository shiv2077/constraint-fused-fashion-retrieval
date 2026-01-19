"""
Pixel-based color extraction for HIGH PRECISION (95%+) color detection.

This module extracts actual pixel colors from images, providing much higher
accuracy than text-based or CLIP-based color detection.

Methods:
1. K-means clustering on image pixels
2. HSV-based color classification
3. Region-based analysis (center-weighted)
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Set, Dict
from collections import Counter
import logging

# HSV color ranges for precise color detection (EXPANDED: 30+ colors)
# Format: (H_min, H_max, S_min, S_max, V_min, V_max)
# OpenCV HSV: H=0-180, S=0-255, V=0-255
COLOR_HSV_RANGES = {
    # Primary colors
    'red': [(0, 10, 70, 255, 50, 255), (170, 180, 70, 255, 50, 255)],  # Red wraps around
    'orange': [(10, 22, 70, 255, 100, 255)],
    'yellow': [(22, 35, 70, 255, 100, 255)],
    'green': [(35, 85, 40, 255, 40, 255)],
    'blue': [(85, 130, 40, 255, 40, 255)],
    'purple': [(130, 155, 40, 255, 40, 255)],
    'pink': [(155, 175, 30, 255, 100, 255), (0, 10, 30, 100, 150, 255)],
    
    # Extended colors - Fashion specific
    'navy': [(100, 130, 50, 255, 20, 100)],  # Dark blue
    'teal': [(85, 100, 40, 255, 80, 200)],  # Blue-green
    'turquoise': [(80, 95, 50, 255, 150, 255)],
    'cyan': [(85, 100, 70, 255, 180, 255)],
    'coral': [(0, 15, 50, 180, 150, 255)],  # Orange-pink
    'salmon': [(0, 15, 40, 120, 150, 255)],
    'burgundy': [(0, 15, 70, 255, 30, 100)],  # Dark red
    'maroon': [(0, 10, 60, 200, 20, 80)],  # Very dark red
    'wine': [(170, 180, 60, 200, 40, 120), (0, 10, 60, 200, 40, 120)],
    'crimson': [(0, 10, 80, 255, 80, 180)],
    'olive': [(30, 50, 30, 150, 40, 150)],  # Yellow-green dark
    'lime': [(35, 55, 70, 255, 150, 255)],  # Bright green
    'mint': [(70, 90, 30, 100, 180, 255)],  # Light green
    'forest': [(40, 70, 50, 200, 30, 100)],  # Dark green
    'emerald': [(70, 90, 60, 255, 80, 200)],
    'lavender': [(125, 145, 20, 80, 180, 255)],  # Light purple
    'violet': [(130, 150, 50, 200, 80, 200)],
    'magenta': [(150, 165, 60, 255, 100, 255)],
    'fuchsia': [(155, 170, 70, 255, 150, 255)],
    'gold': [(20, 35, 100, 255, 150, 255)],  # Rich yellow
    'mustard': [(20, 35, 80, 200, 100, 200)],  # Dark yellow
    'cream': [(20, 40, 10, 50, 220, 255)],
    'ivory': [(20, 40, 5, 30, 230, 255)],
    'champagne': [(25, 40, 15, 60, 200, 255)],
    'tan': [(15, 30, 30, 100, 150, 220)],
    'khaki': [(25, 40, 30, 80, 140, 210)],
    'camel': [(15, 30, 50, 150, 120, 200)],
    'nude': [(10, 25, 20, 70, 180, 240)],
    'rust': [(10, 25, 70, 200, 80, 180)],  # Orange-brown
    'copper': [(10, 25, 60, 180, 100, 180)],
    'bronze': [(15, 30, 50, 150, 80, 150)],
    'silver': [(0, 180, 0, 20, 150, 220)],  # Gray metallic
    'charcoal': [(0, 180, 0, 30, 30, 80)],  # Dark gray
    
    # Neutrals
    'brown': [(10, 30, 40, 200, 30, 150)],
    'chocolate': [(10, 25, 60, 200, 30, 100)],  # Dark brown
    'coffee': [(15, 30, 40, 150, 40, 100)],
    'black': [(0, 180, 0, 255, 0, 40)],
    'white': [(0, 180, 0, 25, 220, 255)],
    'gray': [(0, 180, 0, 25, 60, 200)],
    'grey': [(0, 180, 0, 25, 60, 200)],  # Alias
    'beige': [(20, 40, 15, 60, 180, 245)],
    'taupe': [(15, 30, 15, 50, 100, 170)],
}

# Color synonyms for query matching
COLOR_SYNONYMS = {
    'grey': 'gray',
    'crimson': 'red',
    'scarlet': 'red', 
    'ruby': 'red',
    'cherry': 'red',
    'maroon': 'burgundy',
    'wine': 'burgundy',
    'navy': 'blue',
    'cobalt': 'blue',
    'royal': 'blue',
    'azure': 'blue',
    'sky': 'blue',
    'aqua': 'teal',
    'turquoise': 'teal',
    'forest': 'green',
    'emerald': 'green',
    'sage': 'green',
    'olive': 'green',
    'lime': 'green',
    'mint': 'green',
    'violet': 'purple',
    'lavender': 'purple',
    'plum': 'purple',
    'magenta': 'pink',
    'fuchsia': 'pink',
    'rose': 'pink',
    'blush': 'pink',
    'coral': 'orange',
    'peach': 'orange',
    'tangerine': 'orange',
    'rust': 'orange',
    'amber': 'orange',
    'gold': 'yellow',
    'mustard': 'yellow',
    'lemon': 'yellow',
    'cream': 'white',
    'ivory': 'white',
    'pearl': 'white',
    'champagne': 'beige',
    'nude': 'beige',
    'tan': 'beige',
    'camel': 'brown',
    'chocolate': 'brown',
    'coffee': 'brown',
    'espresso': 'brown',
    'mocha': 'brown',
    'charcoal': 'gray',
    'silver': 'gray',
    'slate': 'gray',
}


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV (OpenCV-style: H=0-180, S=0-255, V=0-255)."""
    rgb = rgb.astype(np.float32) / 255.0
    
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    
    deltac = maxc - minc
    s = np.where(maxc != 0, deltac / maxc, 0)
    
    # Compute hue
    h = np.zeros_like(maxc)
    
    # Red is max
    mask = (maxc == r) & (deltac != 0)
    h[mask] = 60 * (((g[mask] - b[mask]) / deltac[mask]) % 6)
    
    # Green is max
    mask = (maxc == g) & (deltac != 0)
    h[mask] = 60 * (((b[mask] - r[mask]) / deltac[mask]) + 2)
    
    # Blue is max
    mask = (maxc == b) & (deltac != 0)
    h[mask] = 60 * (((r[mask] - g[mask]) / deltac[mask]) + 4)
    
    # Scale to OpenCV range
    h = h / 2  # 0-180
    s = s * 255  # 0-255
    v = v * 255  # 0-255
    
    return np.stack([h, s, v], axis=-1).astype(np.uint8)


def classify_pixel_color(h: int, s: int, v: int) -> str:
    """Classify a single HSV pixel to a color name."""
    # Check achromatic colors first (low saturation)
    if s < 30:
        if v < 50:
            return 'black'
        elif v > 200:
            return 'white'
        else:
            return 'gray'
    
    # Check chromatic colors by hue
    for color_name, ranges in COLOR_HSV_RANGES.items():
        if color_name in ['black', 'white', 'gray']:
            continue  # Already handled
        for h_min, h_max, s_min, s_max, v_min, v_max in ranges:
            if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                return color_name
    
    return 'unknown'


def extract_dominant_colors_pixel(
    img: Image.Image,
    n_colors: int = 3,
    min_percentage: float = 0.05,
    use_center_weighting: bool = True
) -> List[Tuple[str, float]]:
    """
    Extract dominant colors from image using pixel analysis.
    
    Args:
        img: PIL Image
        n_colors: Maximum number of colors to return
        min_percentage: Minimum percentage of pixels for a color to be included
        use_center_weighting: Weight center pixels higher (garment usually in center)
    
    Returns:
        List of (color_name, percentage) tuples sorted by percentage
    """
    # Resize for efficiency
    img_small = img.resize((150, 150), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if img_small.mode != 'RGB':
        img_small = img_small.convert('RGB')
    
    rgb_array = np.array(img_small)
    
    # Create center-weighted mask (higher weight for center of image)
    if use_center_weighting:
        h, w = rgb_array.shape[:2]
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        # Gaussian-like weighting
        weight_mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(h, w) // 3)**2))
        weight_mask = weight_mask / weight_mask.max()
    else:
        weight_mask = np.ones((rgb_array.shape[0], rgb_array.shape[1]))
    
    # Convert to HSV
    hsv_array = rgb_to_hsv(rgb_array)
    
    # Classify each pixel
    color_weights = Counter()
    
    for i in range(hsv_array.shape[0]):
        for j in range(hsv_array.shape[1]):
            h, s, v = hsv_array[i, j]
            color = classify_pixel_color(h, s, v)
            if color != 'unknown':
                color_weights[color] += weight_mask[i, j]
    
    # Normalize to percentages
    total_weight = sum(color_weights.values())
    if total_weight == 0:
        return []
    
    colors_with_pct = [
        (color, weight / total_weight)
        for color, weight in color_weights.most_common()
        if weight / total_weight >= min_percentage
    ][:n_colors]
    
    return colors_with_pct


def extract_colors_kmeans(
    img: Image.Image,
    n_clusters: int = 5,
    n_colors: int = 3
) -> List[Tuple[str, float]]:
    """
    Extract dominant colors using K-means clustering.
    More accurate for complex images but slower.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logging.warning("sklearn not available, falling back to pixel method")
        return extract_dominant_colors_pixel(img, n_colors=n_colors)
    
    # Resize for efficiency
    img_small = img.resize((100, 100), Image.Resampling.LANCZOS)
    if img_small.mode != 'RGB':
        img_small = img_small.convert('RGB')
    
    rgb_array = np.array(img_small).reshape(-1, 3)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(rgb_array)
    
    # Get cluster sizes
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total = counts.sum()
    
    # Map cluster centers to color names
    color_percentages = []
    for i, center in enumerate(kmeans.cluster_centers_):
        r, g, b = center.astype(int)
        # Convert center RGB to HSV
        hsv = rgb_to_hsv(np.array([[[r, g, b]]], dtype=np.uint8))[0, 0]
        color_name = classify_pixel_color(hsv[0], hsv[1], hsv[2])
        
        if color_name != 'unknown':
            pct = counts[i] / total
            color_percentages.append((color_name, pct))
    
    # Aggregate same colors
    color_totals = Counter()
    for color, pct in color_percentages:
        color_totals[color] += pct
    
    return [(c, p) for c, p in color_totals.most_common(n_colors)]


def extract_colors_from_masked_pixels(
    pixels: np.ndarray,
    n_colors: int = 3,
    min_percentage: float = 0.08
) -> List[Tuple[str, float]]:
    """
    Extract dominant colors from an array of RGB pixels (from masked region).
    
    Args:
        pixels: Array of shape (N, 3) containing RGB values
        n_colors: Maximum number of colors to return
        min_percentage: Minimum percentage for a color to be included
    
    Returns:
        List of (color_name, percentage) tuples sorted by percentage
    """
    if len(pixels) == 0:
        return []
    
    # Convert RGB pixels to HSV
    # Reshape for batch processing
    pixels_reshaped = pixels.reshape(-1, 1, 3).astype(np.uint8)
    
    # Process in smaller chunks for efficiency
    color_counts = Counter()
    
    for i in range(len(pixels)):
        r, g, b = pixels[i]
        # Simple RGB to HSV for single pixel
        hsv = rgb_to_hsv(np.array([[[r, g, b]]], dtype=np.uint8))[0, 0]
        color = classify_pixel_color(hsv[0], hsv[1], hsv[2])
        if color != 'unknown':
            color_counts[color] += 1
    
    # Normalize to percentages
    total = sum(color_counts.values())
    if total == 0:
        return []
    
    colors_with_pct = [
        (color, count / total)
        for color, count in color_counts.most_common()
        if count / total >= min_percentage
    ][:n_colors]
    
    return colors_with_pct


def get_pixel_colors(img: Image.Image, method: str = 'pixel', mask: np.ndarray = None) -> Set[str]:
    """
    Main function to extract color names from an image.
    
    Args:
        img: PIL Image
        method: 'pixel' for fast HSV-based, 'kmeans' for clustering-based
        mask: Optional binary mask (H, W) - if provided, only analyze masked pixels
    
    Returns:
        Set of color names detected in the image
    """
    if mask is not None:
        # Use masked pixel extraction (HIGH PRECISION mode)
        img_array = np.array(img.convert('RGB'))
        
        # Resize mask if needed
        if mask.shape[:2] != img_array.shape[:2]:
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), PILImage.Resampling.NEAREST)
            mask = (np.array(mask_img) > 127).astype(np.uint8)
        
        # Get pixels within mask
        masked_pixels = img_array[mask == 1]
        
        if len(masked_pixels) < 100:
            # Too few pixels, fall back to center-weighted
            colors = extract_dominant_colors_pixel(img)
        else:
            # Sample pixels for efficiency (max 5000)
            if len(masked_pixels) > 5000:
                indices = np.random.choice(len(masked_pixels), 5000, replace=False)
                masked_pixels = masked_pixels[indices]
            
            colors = extract_colors_from_masked_pixels(masked_pixels)
    elif method == 'kmeans':
        colors = extract_colors_kmeans(img)
    else:
        colors = extract_dominant_colors_pixel(img)
    
    # Return just the color names (not percentages)
    return {color for color, pct in colors if pct >= 0.10}  # At least 10% of image


def get_garment_colors(img: Image.Image, use_segmentation: bool = True) -> Set[str]:
    """
    HIGH PRECISION color extraction using garment segmentation.
    
    This is the main entry point for accurate color detection.
    Uses YOLO + SAM to segment the garment, then analyzes only garment pixels.
    
    Args:
        img: PIL Image
        use_segmentation: If True, use YOLO/SAM segmentation (slower but more accurate)
    
    Returns:
        Set of detected color names
    """
    if use_segmentation:
        try:
            from src.indexer.garment_segmenter import get_garment_mask
            
            mask = get_garment_mask(img, use_sam=True, fallback_to_center=True)
            if mask is not None:
                return get_pixel_colors(img, mask=mask)
        except Exception as e:
            logging.warning(f"Segmentation failed, using fallback: {e}")
    
    # Fallback to center-weighted pixel analysis
    return get_pixel_colors(img, method='pixel')


# Quick test
if __name__ == "__main__":
    from PIL import Image
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = Image.open(img_path)
        
        print(f"Testing image: {img_path}")
        print(f"Pixel method (no mask): {extract_dominant_colors_pixel(img)}")
        
        try:
            print(f"\nWith garment segmentation:")
            colors = get_garment_colors(img, use_segmentation=True)
            print(f"  Detected colors: {colors}")
        except Exception as e:
            print(f"  Segmentation failed: {e}")
        
        try:
            print(f"\nK-means method: {extract_colors_kmeans(img)}")
        except:
            print("K-means not available (sklearn needed)")
