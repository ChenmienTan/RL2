# export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py \
    actor.kl.coef=1e-2 \
    actor.kl.type=reward \
    actor.kl.level=token \
    actor.kl.estimator=k1 \
    adv.estimator=gae \
    +trainer.disable_wandb=true