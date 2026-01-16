# How Image Retrieval Works - Complete Guide

## Yes, Images ARE Being Retrieved! 

The system **retrieves actual image files** from your database. Here's how it works:

---

## What Happens When You Query

### Input:
```bash
./query.sh "A red tie and a white shirt in a formal setting"
```

### System Process:

1. **Encodes your query** into a 1152-dimensional vector using SigLIP
2. **Searches** the FAISS index containing 25 (or 3200) image vectors
3. **Retrieves** the top-20 candidate image files based on similarity
4. **Reranks** them using BLIP-ITM and constraint matching
5. **Returns** the top-5 best matching images with their file paths

---

## Output: Retrieved Images

### 1. Terminal Output (Text Paths)
```
Query: "A red tie and a white shirt in a formal setting"

Result 1:
  File: /path/to/004e9e21cd1aca568a8ffc77a54638ce.jpg  ← ACTUAL IMAGE FILE
  Caption: a model in a red skirt and white shirt
  Scores: Vec=0.0486, ITM=0.0000, Constraint=0.60
  
Result 2:
  File: /path/to/016247873d36ee0ea830a3827107d52c.jpg  ← ACTUAL IMAGE FILE
  Caption: a woman wearing a black leather jacket, white t shirt
  ...
```

These are **real image files** from your dataset at:
`/home/shiv2077/dev/constraint-fused-fashion-retrieval/val_test2020/test/`

### 2. Visual Output (Contact Sheets)

The system ALSO creates visual grids showing the retrieved images:

**Files created**:
- `outputs/prompt_01_main.jpg` - Grid showing top-5 images for query 1
- `outputs/prompt_02_main.jpg` - Grid showing top-5 images for query 2
- `outputs/prompt_03_main.jpg` - Grid showing top-5 images for query 3
- etc.

**Example contact sheet contents**:
```
┌─────────────────────────────────────────────────┐
│ Query: A red tie and a white shirt in a formal │
│                                                 │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │Image │ │Image │ │Image │ │Image │ │Image │  │
│ │  #1  │ │  #2  │ │  #3  │ │  #4  │ │  #5  │  │
│ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
│ Score:   Score:   Score:   Score:   Score:    │
│ 0.109    0.013    0.012    0.009    0.004     │
└─────────────────────────────────────────────────┘
```

---

## How to View Retrieved Images

### Method 1: Open Contact Sheets (Easiest)

The evaluation already created visual grids for you!

```bash
# View the contact sheets
xdg-open outputs/prompt_01_main.jpg  # Query 1 results
xdg-open outputs/prompt_02_main.jpg  # Query 2 results
xdg-open outputs/prompt_03_main.jpg  # Query 3 results
xdg-open outputs/prompt_04_main.jpg  # Query 4 results
xdg-open outputs/prompt_05_main.jpg  # Query 5 results

# Or open all at once
xdg-open outputs/*.jpg
```

**What you'll see**: A grid showing the 5 retrieved images side-by-side with their ranks and scores.

### Method 2: View Individual Images

```bash
# Open the actual retrieved image files
xdg-open /home/shiv2077/dev/constraint-fused-fashion-retrieval/val_test2020/test/004e9e21cd1aca568a8ffc77a54638ce.jpg
```

### Method 3: Use the New Viewer Script

```bash
# View results for query 0
python3 view_results.py --query_idx 0 --show

# View results for a specific query text
python3 view_results.py --query "red tie" --show

# Just create visualization without opening
python3 view_results.py --query_idx 0 --output my_results.jpg

# See available queries
python3 view_results.py
```

### Method 4: Browse in File Manager

```bash
# Open file manager to browse images
nautilus outputs/  # For GNOME
dolphin outputs/   # For KDE
thunar outputs/    # For XFCE

# Or browse the original dataset
nautilus val_test2020/test/
```

---

## Real Example: What Was Retrieved

### Query: "A red tie and a white shirt in a formal setting"

**Retrieved Images** (5 actual JPG files):

1. **004e9e21cd1aca568a8ffc77a54638ce.jpg**
   - Caption: "a model in a red skirt and white shirt"
   - Matched: red ✓, white ✓, shirt ✓ (60% match)
   - Score: 0.1094 (highest)
   
2. **016247873d36ee0ea830a3827107d52c.jpg**
   - Caption: "a woman wearing a black leather jacket, white t shirt and black tights"
   - Matched: white ✓ (40% match)
   - Score: 0.0129
   
3. **005b37fce3c0f641d327d95dd832f51b.jpg**
   - Caption: "a woman in a white shirt and blue pants"
   - Matched: white ✓, shirt ✓ (40% match)
   - Score: 0.0121

4-5. Additional images with lower scores...

**These are REAL image files** retrieved from the database!

---

## Understanding the Retrieval Process

### What the System Does:

```
Your Query
    ↓
[Convert to 1152-D vector]
    ↓
[Search FAISS index with 25 image vectors]
    ↓
[Find top-20 most similar images]  ← IMAGE RETRIEVAL HAPPENS HERE
    ↓
[Load the actual image files]
    ↓
[Rerank with BLIP-ITM]
    ↓
[Apply constraint penalties]
    ↓
[Return top-5 best images]  ← FINAL RETRIEVED IMAGES
    ↓
Output:
1. Text list with file paths
2. Visual contact sheet (grid of images)
```

### Key Point:

The system doesn't just return "pointers" or "references" - it:
1. **Identifies** which images match your query (retrieval)
2. **Returns** the actual file paths
3. **Creates** visual grids showing the images
4. **Saves** them to `outputs/` directory

---

## Where Are The Images?

### Dataset Location:
```
/home/shiv2077/dev/constraint-fused-fashion-retrieval/val_test2020/test/
├── 003d41dd20f271d27219fe7ee6de727d.jpg  ← Your images
├── 004e9e21cd1aca568a8ffc77a54638ce.jpg
├── 005b37fce3c0f641d327d95dd832f51b.jpg
├── ...
└── (3200 total images)
```

### Retrieved Results Visualizations:
```
/home/shiv2077/dev/constraint-fused-fashion-retrieval/outputs/
├── prompt_01_main.jpg       ← Visual grid of query 1 results (5 images)
├── prompt_02_main.jpg       ← Visual grid of query 2 results (5 images)
├── prompt_03_main.jpg       ← Visual grid of query 3 results (5 images)
├── prompt_04_main.jpg       ← Visual grid of query 4 results (5 images)
├── prompt_05_main.jpg       ← Visual grid of query 5 results (5 images)
├── prompt_XX_baseline.jpg   ← Baseline (vector-only) results
└── results.json             ← Full results data
```

---

## How to Use for Demo/Submission

### For Your PDF Submission:

1. **Run evaluation** (if not done):
   ```bash
   conda run -n ml python -m src.evaluation.run_prompts
   ```

2. **Open the contact sheets**:
   ```bash
   xdg-open outputs/prompt_01_main.jpg
   xdg-open outputs/prompt_02_main.jpg
   # etc.
   ```

3. **Take screenshots** of the contact sheets showing:
   - Query text
   - Retrieved images in a grid
   - Rank and score for each image

4. **Include in your PDF**:
   ```
   Query: "A red tie and a white shirt in a formal setting"
   
   Retrieved Images:
   [INSERT: outputs/prompt_05_main.jpg screenshot]
   
   Analysis:
   - Top result shows red and white garments (60% constraint match)
   - System correctly identifies color and garment attributes
   - Penalty applied to results with <50% match
   ```

### For Live Demo:

```bash
# Run a custom query
./query.sh "casual summer dress"

# View the results
python3 view_results.py --query "summer" --show

# Or just open the output images
xdg-open outputs/*.jpg
```

---

## Comparison: What You See vs. What CLIP Would Show

### Vanilla CLIP Output:
```
Result 1: /path/to/image_12345.jpg (similarity: 0.73)
Result 2: /path/to/image_67890.jpg (similarity: 0.71)
...
```
Just file paths, no understanding of "red vs blue shirt" order.

### Your System Output:
```
Result 1: red_skirt_white_shirt.jpg
  Caption: "model in red skirt and white shirt"
  Constraint match: 60% (red✓ white✓ shirt✓)
  Final score: 0.1094 (boosted by good match)
  
Result 2: black_jacket_white_shirt.jpg
  Caption: "woman in black jacket, white t-shirt"
  Constraint match: 40% (only white✓)
  Final score: 0.0129 (penalty applied)
  [Penalty applied for low constraint satisfaction]
```
Understands composition, enforces constraints, explains ranking.

---

## Quick Commands Summary

```bash
# View all retrieval results
xdg-open outputs/*.jpg

# View specific query results
python3 view_results.py --query_idx 0 --show

# View evaluation results
cat outputs/evaluation_metrics.json

# Browse original images
nautilus val_test2020/test/

# Run new query
./query.sh "your query here"

# Check what was retrieved
cat outputs/results.json | grep -A5 "query"
```

---

## The Bottom Line

**YES, images are being retrieved!**

- ✅ System searches 25 (or 3200) actual image files
- ✅ Returns top-5 best matching images with file paths
- ✅ Creates visual grids showing the retrieved images
- ✅ Saves to `outputs/prompt_XX_main.jpg`
- ✅ You can open them with `xdg-open outputs/*.jpg`

**The retrieval is working - you just need to view the output images!**

---

## Visual Proof

Run this to see the retrieved images right now:

```bash
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval

# Show what images were retrieved
echo "Retrieved images from last evaluation:"
ls -lh outputs/*.jpg

# Open them all
xdg-open outputs/prompt_01_main.jpg &
xdg-open outputs/prompt_02_main.jpg &
xdg-open outputs/prompt_03_main.jpg &

echo ""
echo "✅ These contact sheets show the actual retrieved images!"
echo "   Each grid displays the top-5 images for that query."
```

**Your image retrieval system IS working - you just need to look at the visual outputs!** 🎉
