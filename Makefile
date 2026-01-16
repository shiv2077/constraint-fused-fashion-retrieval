.PHONY: help index query eval clean install

help:
	@echo "Constraint-Aware Fashion Retrieval - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install dependencies"
	@echo "  make index           - Build index from images"
	@echo "  make query QUERY='...' - Run a single query"
	@echo "  make eval            - Run evaluation prompts"
	@echo "  make clean           - Remove artifacts and outputs"
	@echo ""

install:
	pip install -r requirements.txt

# Default image directory
IMG_DIR ?= /home/shiv2077/dev/constraint-fused-fashion-retrieval/val_test2020/test
MAX_IMAGES ?= 700

index:
	python -m src.indexer.build_index \
		--img_dir $(IMG_DIR) \
		--out_dir artifacts \
		--max_images $(MAX_IMAGES)

# Usage: make query QUERY="red dress for summer party"
QUERY ?= "red dress suitable for a summer party"
TOPK ?= 5

query:
	python -m src.retriever.search \
		--index_dir artifacts \
		--img_root $(IMG_DIR) \
		--query $(QUERY) \
		--topk $(TOPK)

eval:
	python -m src.evaluation.run_prompts \
		--index_dir artifacts \
		--img_root $(IMG_DIR) \
		--out_dir outputs \
		--topk 5

# Compute metrics only (after running eval)
.PHONY: metrics
metrics:
	$(CONDA_RUN) python -m src.evaluation.compute_metrics --results_file outputs/results.json

clean:
	rm -rf artifacts/ outputs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.DEFAULT_GOAL := help
