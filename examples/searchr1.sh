# You should firstly follow https://github.com/PeterGriffinJin/Search-R1 
# to download the index and corpus.

vllm serve intfloat/e5-base-v2 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.1 &

python envs/local_search_service.py \
    --model_name intfloat/e5-base-v2 \
    --index_path path/to/your/index \
    --corpus_path path/to/your/corpus \
    --top_k 3 &

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health) -ne 200 ]; do
    sleep 1
done

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:10000/health) -ne 200 ]; do
    sleep 1
done

torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=train@Chenmien/SearchR1 \
    train_data.responses_per_prompt=5 \
    train_data.max_turns=2 \
    train_data.sampling_params.max_new_tokens=512 \
    test_data.path=test@Chenmien/SearchR1 \
    actor.model_name=Qwen/Qwen2.5-3B \
    actor.max_length_per_device=8192 \
    rollout.train_prompts_per_rollout=256 \
    "rollout.train_sampling_params.stop=['</search>','</answer>']" \
    rollout.env_path=envs/searchr1.py \
    trainer.project=SearchR1 \
    trainer.experiment_name=qwen2.5-3b_reinforce \
    trainer.total_steps=256 \
    trainer.test_freq=8