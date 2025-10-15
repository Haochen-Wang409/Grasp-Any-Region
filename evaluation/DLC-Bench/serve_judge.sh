MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct

vllm serve $MODEL_PATH \
    --served-model-name llama3.1-8b \
    --api-key sk-abc123 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --port 8007 \
    --host localhost
