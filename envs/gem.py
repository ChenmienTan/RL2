from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig
import asyncio
from collections import deque
from transformers import AutoTokenizer
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns
from RL2.datasets import (
    Sample,
    initialize_state_dict,
    add_llm_response,
    add_env_response
)
from RL2.utils.communication import async_request

NUM_ENVS = 16
ENV_ID = "rg:letter_counting"
WRAPPERS = ""

ENV_POOL = []
for idx in range(NUM_ENVS):
    env = gem.make(env_id=ENV_ID, seed=233 + idx)
    wrappers = get_wrapper_fns(WRAPPERS, tokenizer=None)
    for wrapper in wrappers:
        env = wrapper(env)
    ENV_POOL.append(env)

AVAILABLE_ENVS = deque(range(NUM_ENVS))
SEMAPHORE = asyncio.Semaphore(NUM_ENVS)
LOCK = asyncio.Lock()

async def env_step(
    tokenizer: AutoTokenizer,
    sample: Sample
) -> Dict[str, Any]:

    (
        prompt,
        reward,
        terminated,
        truncated,
        _
    ) = ENV_POOL[sample.sample["env_idx"]].step(sample.action_text)
    prompt = f"Question: {prompt}\nPlease reason step by step, "\
        "and put your final answer within \\boxed{}."
    next_state = tokenizer.apply_chat_template(
        {"role": "user", "content": prompt},
        add_generation_prompt=True,
        tokenize=False
    )
    done = terminated or truncated
    
    if done:
        async with LOCK:
            AVAILABLE_ENVS.append(sample.sample["env_idx"])
        SEMAPHORE.release()

    return {
        "next_state": next_state,
        "done": done,
        "reward": reward
    }

async def generate(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample
):
    sampling_params = OmegaConf.to_container(config.sampling_params)

    match sample.status:

        case Sample.Status.RUNNING:

            await SEMAPHORE.acquire()
            async with LOCK:
                env_idx = AVAILABLE_ENVS.popleft()
            prompt, _ = ENV_POOL[env_idx].reset()
            prompt = f"Question: {prompt}\nPlease reason step by step, "\
                "and put your final answer within \\boxed{}."

            sample.state_text = tokenizer.apply_chat_template(
                {"role": "user", "content": prompt},
                add_generation_prompt=True,
                tokenize=False
            )
            sample.sample["env_idx"] = env_idx
            sample.state_dict = initialize_state_dict(
                tokenizer, sample.state_text
            )

        case Sample.Status.ABORTED:
            sample.status = Sample.Status.RUNNING

        case Sample.Status.DONE:
            return

    while True:

        response = await async_request(
            config.router_url,
            "generate",
            json={
                "input_ids": sample.state_dict["states"],
                "sampling_params": {
                    **sampling_params,
                    "max_new_tokens": sampling_params["max_new_tokens"] - sample.previous_response_length,
                    "no_stop_trim": True
                },
                "return_logprob": True
            }
        )
        add_llm_response(sample, response)
        if sample.status == Sample.Status.ABORTED:
            return
        
        response = await env_step(tokenizer, sample)
        add_env_response(tokenizer, sample, response)
        if sample.status == Sample.Status.DONE:
            return