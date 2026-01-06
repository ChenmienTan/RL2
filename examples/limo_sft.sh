torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.sft \
    data.train.path=Chenmien/LIMO \
    data.train.max_length=16384 \
    data.train.batch_size=32 \
    actor.model_name=Qwen/Qwen2.5-7B-Instruct \
    actor.cp_size=4 \
    actor.max_length_per_device=4096 \
    trainer.project=LIMO \
    trainer.experiment_name=qwen2.5-7b-inst \
    trainer.n_epochs=15