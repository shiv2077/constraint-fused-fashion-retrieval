#!/bin/bash
# Run full indexing with all 3200 images

cd /home/shiv2077/dev/constraint-fused-fashion-retrieval

echo "Starting indexing of 3200 images..."
echo "This will take approximately 18-20 minutes"
echo "Log will be saved to indexing_full.log"
echo ""

# Run with conda
conda run -n ml python -m src.indexer.build_index --max_images 3200 2>&1 | tee indexing_full.log

echo ""
echo "Indexing complete! Check artifacts/ directory"
