# Setup and Running Guide

## System Status

**GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (5.7 GB VRAM)  
**CUDA**: 12.8  
**Python**: 3.12.11 (conda environment: ml)  
**Dataset**: 3200 images in `val_test2020/test/`

## Dependencies Installed

All required packages are installed in the `ml` conda environment:
- PyTorch 2.7.1 + CUDA 12.8 ✓
- transformers 4.57.3 ✓
- Pillow 11.0.0 ✓
- faiss-gpu 1.13.2 ✓
- numpy 2.4.1 ✓
- tqdm ✓
- sentencepiece 0.2.1 ✓

## Running the Project

### 1. Build Index (Currently Running)

The indexing is currently running in the background:
```bash
# Process ID: 117948
# Check progress:
tail -f indexing_full.log

# Or check if still running:
ps aux | grep "src.indexer.build_index"
```

**Expected time**: ~18-20 minutes for 3200 images at ~2.8 images/second

### 2. Once Indexing Completes

Check the artifacts:
```bash
ls -lh artifacts/
# Should contain:
# - vectors.faiss (FAISS index)
# - metadata.json (captions & tags)
# - manifest.json (configuration)
```

### 3. Run Queries

Single query:
```bash
conda run -n ml python -m src.retriever.search \
  --query "red dress for summer party" \
  --topk 5 \
  --topn 20
```

**Note**: `--topn 20` keeps GPU memory under 6GB. Adjust based on query complexity.

### 4. Run Evaluation

Run all test prompts:
```bash
conda run -n ml python -m src.evaluation.run_prompts \
  --topk 5
```

This will:
- Run 5 evaluation prompts
- Save `outputs/results.json` with all results
- Create contact sheets for each prompt (main + baseline)

## Performance Notes

### GPU Memory Management

- RTX 3060 has 5.7 GB VRAM
- Each model loads ~2GB
- ITM reranking processes candidates in batches of 8
- Default `topn=20` to avoid OOM errors
- For higher `topn`, reduce batch size in search.py

### Speed Benchmarks

**Indexing** (with SigLIP + BLIP caption):
- ~2.8-3.0 images/second
- 3200 images: ~18-20 minutes

**Search + Rerank** (with ITM):
- Vector search: <1 second
- ITM reranking (20 candidates): ~3-5 seconds
- Total per query: ~5-8 seconds

## Monitoring Current Indexing

```bash
# Check progress
tail -f indexing_full.log

# Check artifacts size (increases as indexing progresses)
watch -n 5 'ls -lh artifacts/'

# Check GPU usage
nvidia-smi

# Estimated completion
# Started: ~00:14
# Duration: ~18-20 minutes
# Expected finish: ~00:32-00:34
```

## Quick Commands

```bash
# Activate environment
conda activate ml

# Search (after indexing completes)
python -m src.retriever.search --query "blue jeans casual"

# Evaluation
python -m src.evaluation.run_prompts

# Rebuild index (if needed)
python -m src.indexer.build_index --max_images 3200
```

## Troubleshooting

### Out of Memory
Reduce `--topn` or batch size:
```bash
python -m src.retriever.search --query "..." --topn 10
```

### Slow Performance
Verify GPU is being used:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Downloads
First run downloads models (~2GB total) to HuggingFace cache:
- `~/.cache/huggingface/hub/`
