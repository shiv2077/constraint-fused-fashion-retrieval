#!/bin/bash
# Quick query script

if [ -z "$1" ]; then
    echo "Usage: ./query.sh \"your query here\" [topk]"
    echo ""
    echo "Examples:"
    echo "  ./query.sh \"red dress for summer party\""
    echo "  ./query.sh \"blue jeans casual\" 10"
    exit 1
fi

QUERY="$1"
TOPK="${2:-5}"

cd /home/shiv2077/dev/constraint-fused-fashion-retrieval

if [ ! -f "artifacts/vectors.faiss" ]; then
    echo "Error: Index not found. Run ./run_indexing.sh first!"
    exit 1
fi

# Run search
echo "🔍 Searching for: ${QUERY}"
echo ""

conda run -n ml python -m src.retriever.search \
    --query "${QUERY}" \
    --topk ${TOPK}

# Show retrieved images
echo ""
echo "📸 Retrieved Images:"
echo "   The actual image files that were found are listed above."
echo ""
echo "💡 To see a visual grid of the results, you can:"
echo "   1. Check the evaluation outputs: ls outputs/*.jpg"
echo "   2. Open them: xdg-open outputs/prompt_01_main.jpg"
echo "   3. Or view the images at the paths shown above"
echo ""
    --topn 20
