# You should firstly follow https://github.com/PeterGriffinJin/Search-R1 
# to download the index and corpus.

python -m sglang.launch_server \
    --model-path intfloat/e5-base-v2 \
    --is-embedding \
    --tp 4 \
    --mem-fraction-static 0.1 \
    --log-level warning &

python envs/local_search_service.py \
    --model_name intfloat/e5-base-v2 \
    --index_path path/to/your/index \
    --corpus_path path/to/your/corpus \
    --top_k 3 &

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:30000/health) -ne 200 ]; do
    sleep 1
done

while [ $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health) -ne 200 ]; do
    sleep 1
done

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    data.train_data_path=train@Chenmien/SearchR1 \
    data.test_data_path=test@Chenmien/SearchR1 \
    data.prompts_per_rollout=256 \
    data.responses_per_prompt=8 \
    actor.model_name=Qwen/Qwen2.5-3B-Instruct \
    actor.max_length_per_device=8192 \
    rollout.max_turns=4 \
    rollout.train_sampling_params.max_new_tokens=512 \
    rollout.env_path=envs/searchr1.py \
    adv.estimator=reinforce \
    trainer.project=SearchR1 \
    trainer.experiment_name=qwen2.5-3b-inst_reinforce \
    trainer.test_freq=8