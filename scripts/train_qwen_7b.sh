export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=31
export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-conf=timeout=36000 \
    main.py \
    --project OpenReasonerZero \
    --experiment_name qwen2.5-7b \
    --model_name Qwen/Qwen2.5-7B \
    --train_data_path data/orz.json \
    --test_data_path data/olympiadbench.json \
    --max_prompt_length 2048 \
    --max_response_length 2048 \
    --reward_fn_path rewards/math.py \
    --micro_batch_size_per_device 4