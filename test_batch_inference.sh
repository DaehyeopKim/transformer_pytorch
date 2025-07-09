#!/bin/bash

echo "=== Batch Inference Test ==="
echo "Testing inference.py with multiple context-query pairs"
echo

# Multiple pairs test
echo "Multiple pairs test (batch processing):"
python3 inference.py \
    --context ./sample/sample_context.txt ./sample/sample_context2.txt ./sample/sample_context3.txt \
    --query ./sample/sample_query.txt ./sample/sample_query2.txt ./sample/sample_query3.txt \
    --model_path test_model.pth \
    --log info \
    --infermode greedy \
    --max_length 50

echo
echo "Test completed!"
