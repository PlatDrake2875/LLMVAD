#!/bin/bash
# Run all LLM handler tests sequentially to avoid GPU memory conflicts

set -e  # Exit on error

echo "=========================================="
echo "Running LLM Handler Tests Sequentially"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run HuggingFace tests
echo -e "${BLUE}[1/3] Running HuggingFace tests...${NC}"
if uv run pytest tests/test_llm_handlers.py -m huggingface -v; then
    echo -e "${GREEN}✓ HuggingFace tests passed${NC}"
else
    echo -e "${RED}✗ HuggingFace tests failed${NC}"
    exit 1
fi
echo ""

# Wait a bit for GPU memory to clear
echo "Waiting 5 seconds for GPU memory to clear..."
sleep 5
echo ""

# Run vLLM tests
echo -e "${BLUE}[2/3] Running vLLM tests...${NC}"
if uv run pytest tests/test_llm_handlers.py -m vllm -v; then
    echo -e "${GREEN}✓ vLLM tests passed${NC}"
else
    echo -e "${RED}✗ vLLM tests failed${NC}"
    exit 1
fi
echo ""

# Run factory tests (lightweight, no GPU needed)
echo -e "${BLUE}[3/3] Running Factory tests...${NC}"
if uv run pytest tests/test_llm_handlers.py -m factory -v; then
    echo -e "${GREEN}✓ Factory tests passed${NC}"
else
    echo -e "${RED}✗ Factory tests failed${NC}"
    exit 1
fi
echo ""

echo "=========================================="
echo -e "${GREEN}All tests passed! ✓${NC}"
echo "=========================================="
