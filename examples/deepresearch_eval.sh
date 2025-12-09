# BrowseComp data needs to be post-processed by your own

torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    rollout.server_args.model_path=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    rollout.server_args.tp_size=4 \
    rollout.train.path=data/browsecomp.jsonl \
    rollout.test.path=data/browsecomp.jsonl \
    rollout.test.prompts_per_rollout=32 \
    rollout.test.sampling_params.temperature=0.7 \
    rollout.test.sampling_params.max_new_tokens=2048 \
    rollout.env_path=envs/deep_research.py \
    trainer.project=DeepResearch \
    trainer.experiment_name=Tongyi-DeepResearch-30B-A3B \
    trainer.eval_only=true \
    trainer.use_wandb=false