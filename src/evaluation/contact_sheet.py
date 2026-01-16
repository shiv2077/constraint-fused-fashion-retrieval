"""Create contact sheet (image grid) from search results."""

import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import math


def create_contact_sheet(
    results: List[Dict[str, Any]],
    output_path: Path,
    img_size: int = 256,
    cols: int = 5,
    add_labels: bool = True
) -> None:
    if not results:
        logging.warning("No results to create contact sheet")
        return
    
    n_images = len(results)
    rows = math.ceil(n_images / cols)
    
    label_height = 30 if add_labels else 0
    
    canvas_width = cols * img_size
    canvas_height = rows * (img_size + label_height)
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for idx, result in enumerate(results):
        row = idx // cols
        col = idx % cols
        
        img_path = Path(result['path'])
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
            
            square_img = Image.new('RGB', (img_size, img_size), color='gray')
            offset_x = (img_size - img.width) // 2
            offset_y = (img_size - img.height) // 2
            square_img.paste(img, (offset_x, offset_y))
            
            x = col * img_size
            y = row * (img_size + label_height)
            canvas.paste(square_img, (x, y))
            
            if add_labels:
                draw = ImageDraw.Draw(canvas)
                label = f"#{idx + 1} ({result.get('final_score', 0):.3f})"
                text_y = y + img_size + 5
                draw.text((x + 5, text_y), label, fill='black', font=font)
            
        except Exception as e:
            logging.error(f"Failed to add image {img_path} to contact sheet: {e}")
            continue
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    logging.info(f"Saved contact sheet to {output_path}")
