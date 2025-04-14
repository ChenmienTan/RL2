# export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py \
    data.batch_size=16 \
    actor.rollout.rollout_per_prompt=8