# export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py \
    adv.estimator=gae \
    actor.sp_size=2 \
    critic.sp_size=2