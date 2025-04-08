export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py \
    --project OpenReasonerZero \
    --experiment_name qwen2.5_7b_reinforce \
    --model_name Qwen/Qwen2.5-7B \
    --tp_size 2 \
    --train_data_path data/orz.json \
    --test_data_path data/olympiadbench.json \
    --max_prompt_length 2048 \
    --max_response_length 6144 \
    --reward_fn_path rewards/math.py \
    --batch_size 16 \
    --rollout_per_prompt 8 \
    --max_length_per_device 8192 \
    --test_freq 8 \
    --save_freq 32 \
    --disable_wandb \
    --ring_attn_size 2 \
    --ring_head_stride 2

    # --critic_model_name Qwen/Qwen2.5-7B \