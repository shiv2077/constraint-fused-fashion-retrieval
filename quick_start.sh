#!/bin/bash
cd /home/shiv2077/dev/constraint-fused-fashion-retrieval
rm -rf artifacts
python -m src.indexer.build_index --img_dir val_test2020/test --out_dir artifacts --max_images 3200 > indexing_with_tags_final.log 2>&1 &
echo "PID: $!" > indexing.pid
cat indexing.pid
