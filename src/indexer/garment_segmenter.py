"""
Garment Segmentation using YOLO + SAM + GroundingDINO for HIGH PRECISION color/attribute detection.

This module segments garments from the background to enable accurate
color extraction from ONLY the clothing, not the runway/background.

Segmentation Methods:
1. YOLO + SAM (default): Detect person, segment body, exclude head/feet
2. GroundingDINO + SAM (fashion-specific): Segment specific garment types by text

Pipeline:
1. YOLO/GroundingDINO detects target region
2. SAM segments the detected region
3. Heuristics refine garment area
4. Return masked image for color analysis
"""

import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict
import logging

# Lazy loading for heavy models
_yolo_model = None
_sam_model = None
_grounding_dino_model = None

# Garment types for GroundingDINO detection
GARMENT_PROMPTS = [
    "dress", "shirt", "blouse", "jacket", "coat", "sweater",
    "pants", "trousers", "jeans", "skirt", "top", "suit",
    "gown", "outfit", "clothing", "garment"
]


def get_yolo_model():
    """Lazy load YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logging.info("Loading YOLO model for person detection...")
        _yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
    return _yolo_model


def get_sam_model():
    """Lazy load SAM model."""
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        logging.info("Loading SAM model for segmentation...")
        _sam_model = SAM('sam2_b.pt')  # Base SAM2 model
    return _sam_model


def get_grounding_dino_model():
    """Lazy load GroundingDINO model for text-based object detection."""
    global _grounding_dino_model
    if _grounding_dino_model is None:
        try:
            from groundingdino.util.inference import Model
            import groundingdino
            import os
            
            # Get model paths
            config_path = os.path.join(
                os.path.dirname(groundingdino.__file__),
                "config", "GroundingDINO_SwinT_OGC.py"
            )
            
            # Check if weights exist, if not use a fallback approach
            weights_path = "groundingdino_swint_ogc.pth"
            
            if os.path.exists(weights_path) and os.path.exists(config_path):
                logging.info("Loading GroundingDINO model...")
                _grounding_dino_model = Model(config_path, weights_path)
            else:
                logging.warning("GroundingDINO weights not found, will use YOLO fallback")
                _grounding_dino_model = "FALLBACK"
        except Exception as e:
            logging.warning(f"Failed to load GroundingDINO: {e}")
            _grounding_dino_model = "FALLBACK"
    
    return _grounding_dino_model


def detect_garment_with_grounding_dino(
    img: Image.Image,
    garment_type: str = "clothing"
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detect specific garment type using GroundingDINO.
    
    Args:
        img: PIL Image
        garment_type: Text prompt for detection (e.g., "dress", "jacket")
        
    Returns:
        List of (x1, y1, x2, y2) bounding boxes or None
    """
    model = get_grounding_dino_model()
    
    if model == "FALLBACK":
        return None
    
    try:
        import numpy as np
        import cv2
        
        # Convert PIL to numpy
        img_np = np.array(img)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Run detection
        detections = model.predict_with_classes(
            image=img_np,
            classes=[garment_type],
            box_threshold=0.3,
            text_threshold=0.25
        )
        
        if len(detections.xyxy) == 0:
            return None
        
        boxes = []
        for box in detections.xyxy:
            boxes.append(tuple(map(int, box)))
        
        return boxes
        
    except Exception as e:
        logging.debug(f"GroundingDINO detection failed: {e}")
        return None


def detect_person_bbox(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the main person in the image using YOLO.
    
    Returns:
        (x1, y1, x2, y2) bounding box or None if no person detected
    """
    model = get_yolo_model()
    
    # Run detection
    results = model(img, verbose=False, classes=[0])  # class 0 = person
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    
    # Get the largest person detection (most prominent)
    boxes = results[0].boxes
    areas = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Move to CPU first
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        areas.append(area)
    
    largest_idx = np.argmax(areas)
    
    box = boxes[largest_idx].xyxy[0].cpu().numpy()
    return tuple(map(int, box))


def segment_person(img: Image.Image, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Segment the person using SAM with the bounding box prompt.
    
    Returns:
        Binary mask (H, W) where 1 = person, 0 = background
    """
    model = get_sam_model()
    
    # Run SAM with bounding box prompt
    results = model(img, bboxes=[list(bbox)], verbose=False)
    
    if len(results) == 0 or results[0].masks is None:
        return None
    
    # Get the mask
    mask = results[0].masks.data[0].cpu().numpy()
    return mask.astype(np.uint8)


def extract_garment_region(
    person_mask: np.ndarray,
    exclude_head_ratio: float = 0.15,
    exclude_feet_ratio: float = 0.10
) -> np.ndarray:
    """
    Extract garment region from person mask by excluding head and feet.
    
    This focuses on the torso/clothing area for better color detection.
    """
    h, w = person_mask.shape
    
    # Find vertical extent of person
    rows_with_person = np.where(person_mask.any(axis=1))[0]
    if len(rows_with_person) == 0:
        return person_mask
    
    top_row = rows_with_person[0]
    bottom_row = rows_with_person[-1]
    person_height = bottom_row - top_row
    
    # Create garment mask (exclude head and feet)
    garment_mask = person_mask.copy()
    
    # Exclude head region
    head_cutoff = int(top_row + person_height * exclude_head_ratio)
    garment_mask[:head_cutoff, :] = 0
    
    # Exclude feet region  
    feet_cutoff = int(bottom_row - person_height * exclude_feet_ratio)
    garment_mask[feet_cutoff:, :] = 0
    
    return garment_mask


def get_garment_mask(
    img: Image.Image,
    use_sam: bool = True,
    fallback_to_center: bool = True
) -> Optional[np.ndarray]:
    """
    Main function: Get a mask of the garment region in the image.
    
    Args:
        img: PIL Image
        use_sam: If True, use SAM for precise segmentation. If False, use bbox only.
        fallback_to_center: If detection fails, use center region as fallback
    
    Returns:
        Binary mask (H, W) where 1 = garment area, 0 = background
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Step 1: Detect person
    bbox = detect_person_bbox(img)
    
    if bbox is None:
        if fallback_to_center:
            # Fallback: use center 60% of image
            logging.debug("No person detected, using center fallback")
            mask = np.zeros((h, w), dtype=np.uint8)
            y1, y2 = int(h * 0.2), int(h * 0.8)
            x1, x2 = int(w * 0.2), int(w * 0.8)
            mask[y1:y2, x1:x2] = 1
            return mask
        return None
    
    if use_sam:
        # Step 2: Segment person with SAM
        try:
            person_mask = segment_person(img, bbox)
            if person_mask is not None:
                # Step 3: Extract garment region (exclude head/feet)
                garment_mask = extract_garment_region(person_mask)
                
                # Resize mask to match image if needed
                if garment_mask.shape != (h, w):
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray((garment_mask * 255).astype(np.uint8))
                    mask_img = mask_img.resize((w, h), PILImage.Resampling.NEAREST)
                    garment_mask = (np.array(mask_img) > 127).astype(np.uint8)
                
                return garment_mask
        except Exception as e:
            logging.warning(f"SAM segmentation failed: {e}")
    
    # Fallback: Use bounding box with head/feet exclusion
    x1, y1, x2, y2 = bbox
    box_height = y2 - y1
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Exclude head (top 15%) and feet (bottom 10%)
    garment_y1 = int(y1 + box_height * 0.15)
    garment_y2 = int(y2 - box_height * 0.10)
    
    # Narrow the box horizontally a bit (exclude arms at edges)
    box_width = x2 - x1
    garment_x1 = int(x1 + box_width * 0.1)
    garment_x2 = int(x2 - box_width * 0.1)
    
    mask[garment_y1:garment_y2, garment_x1:garment_x2] = 1
    
    return mask


def apply_mask_to_image(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply mask to image, setting non-garment pixels to white.
    Useful for visualization.
    """
    img_array = np.array(img)
    
    # Expand mask to 3 channels
    mask_3d = np.stack([mask] * 3, axis=-1)
    
    # Apply mask (white background for masked areas)
    result = np.where(mask_3d, img_array, 255)
    
    return Image.fromarray(result.astype(np.uint8))


def get_masked_pixels(img: Image.Image, mask: np.ndarray) -> np.ndarray:
    """
    Get only the pixels within the mask region.
    
    Returns:
        Array of shape (N, 3) containing RGB values of garment pixels
    """
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Get pixels where mask is 1
    garment_pixels = img_array[mask == 1]
    
    return garment_pixels


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = Image.open(img_path).convert('RGB')
        
        print(f"Testing garment segmentation on: {img_path}")
        
        # Get mask
        mask = get_garment_mask(img, use_sam=True)
        
        if mask is not None:
            print(f"Mask shape: {mask.shape}")
            print(f"Garment pixels: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.1f}%)")
            
            # Save visualization
            masked_img = apply_mask_to_image(img, mask)
            out_path = img_path.replace('.jpg', '_masked.jpg')
            masked_img.save(out_path)
            print(f"Saved masked image to: {out_path}")
        else:
            print("Failed to segment garment")
    else:
        print("Usage: python garment_segmenter.py <image_path>")
