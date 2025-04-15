# export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py \
    actor.kl.coef=1e-3 \
    actor.kl.type=loss \
    actor.kl.level=sequence \
    actor.kl.estimator=k3 \
    adv.estimator=gae \
    trainer.disable_wandb=true