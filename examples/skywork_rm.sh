torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.rm \
    data.train.path=Chenmien/SkyworkRM \
    data.train.max_length=2048 \
    critic.model_name=meta-llama/Llama-3.1-8B-Instruct \
    critic.max_length_per_device=8192 \
    trainer.project=SkyworkRM \
    trainer.experiment_name=llama-3.1-8b-inst