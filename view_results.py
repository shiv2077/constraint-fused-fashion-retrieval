#!/usr/bin/env python3
"""
Simple image viewer for query results.
Shows retrieved images in a grid with scores.
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


def create_query_result_sheet(
    results,
    query,
    output_path,
    img_size=256,
    cols=5
):
    """Create a visual contact sheet showing retrieved images."""
    if not results:
        print("No results to display")
        return None
    
    n_images = len(results)
    rows = math.ceil(n_images / cols)
    
    # Title area
    title_height = 80
    label_height = 60
    
    # Canvas
    canvas_width = cols * img_size
    canvas_height = title_height + rows * (img_size + label_height)
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Load font
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = label_font = small_font = ImageFont.load_default()
    
    # Draw title
    title_text = f"Query: {query[:80]}"
    draw.text((10, 10), title_text, fill='black', font=title_font)
    draw.text((10, 45), f"Retrieved {n_images} images", fill='gray', font=small_font)
    
    # Place images
    for idx, result in enumerate(results):
        row = idx // cols
        col = idx % cols
        
        img_path = Path(result['path'])
        try:
            # Load and resize
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
            
            # Center in square
            square_img = Image.new('RGB', (img_size, img_size), color='lightgray')
            offset_x = (img_size - img.width) // 2
            offset_y = (img_size - img.height) // 2
            square_img.paste(img, (offset_x, offset_y))
            
            # Position
            x = col * img_size
            y = title_height + row * (img_size + label_height)
            canvas.paste(square_img, (x, y))
            
            # Labels
            draw = ImageDraw.Draw(canvas)
            
            # Rank and score
            rank_text = f"#{idx + 1}"
            score_text = f"Score: {result.get('final_score', 0):.3f}"
            
            text_y = y + img_size + 5
            draw.text((x + 5, text_y), rank_text, fill='blue', font=label_font)
            draw.text((x + 5, text_y + 20), score_text, fill='black', font=small_font)
            
            # Caption (truncated)
            caption = result.get('caption', '')[:30] + '...'
            draw.text((x + 5, text_y + 38), caption, fill='gray', font=small_font)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    # Save
    canvas.save(output_path, quality=95)
    print(f"✅ Saved visualization to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="View retrieved images from query")
    parser.add_argument("--results", default="outputs/results.json", help="Results JSON file")
    parser.add_argument("--query_idx", type=int, help="Query index to visualize (0-based)")
    parser.add_argument("--query", help="Query text to search for")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Display image after creating")
    args = parser.parse_args()
    
    # Load results
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        print("Run evaluation first: python -m src.evaluation.run_prompts")
        sys.exit(1)
    
    with open(args.results) as f:
        data = json.load(f)
    
    # Handle both 'queries' and 'prompts' keys
    queries_data = data.get('queries', data.get('prompts', []))
    
    if not queries_data:
        print("No queries found in results")
        sys.exit(1)
    
    # Find query
    if args.query_idx is not None:
        if args.query_idx >= len(queries_data):
            print(f"Error: Query index {args.query_idx} out of range (0-{len(queries_data)-1})")
            sys.exit(1)
        query_data = queries_data[args.query_idx]
    elif args.query:
        # Search for matching query
        query_data = None
        for qd in queries_data:
            if args.query.lower() in qd['query'].lower():
                query_data = qd
                break
        if not query_data:
            print(f"Error: Query not found: {args.query}")
            print(f"Available queries:")
            for i, qd in enumerate(queries_data):
                print(f"  {i}: {qd['query']}")
            sys.exit(1)
    else:
        # Show all queries
        print("Available queries:")
        for i, qd in enumerate(queries_data):
            print(f"  {i}: {qd['query']}")
        print("\nUsage:")
        print(f"  {sys.argv[0]} --query_idx 0")
        print(f"  {sys.argv[0]} --query 'red dress'")
        sys.exit(0)
    
    # Get results
    query_text = query_data['query']
    results = query_data.get('results', query_data.get('main_results', []))
    
    if not results:
        print(f"No results for query: {query_text}")
        sys.exit(1)
    
    # Create output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        safe_query = "".join(c if c.isalnum() else "_" for c in query_text[:30])
        output_path = output_dir / f"query_{safe_query}.jpg"
    
    # Create visualization
    print(f"\nQuery: {query_text}")
    print(f"Results: {len(results)} images")
    print(f"Creating visualization...")
    
    result_path = create_query_result_sheet(
        results,
        query_text,
        output_path
    )
    
    if result_path and args.show:
        # Try to open with default viewer
        import subprocess
        import platform
        
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["xdg-open", str(result_path)], check=False)
            elif system == "Darwin":
                subprocess.run(["open", str(result_path)], check=False)
            elif system == "Windows":
                subprocess.run(["start", str(result_path)], check=False, shell=True)
            print(f"✅ Opened {result_path} in default viewer")
        except:
            print(f"⚠️  Could not open automatically. View manually: {result_path}")


if __name__ == "__main__":
    main()
