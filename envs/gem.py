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
PROMPT_TEMPLATE = "qwen3_general"

def apply_no_template(observation):
    return observation

def apply_qwen3_general_template(observation):
    return (
        f"<|im_start|>user\nQuestion: {observation}\nPlease reason step by step,"
        " and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>"
        "assistant\n"
    )

def apply_qwen3_game_template(observation):
    return (
        "<|im_start|>user\nYou are playing language games. Make valid actions to win."
        f"\nObservation: {observation}\nPlease reason step by step, and put your final"
        " answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_code_template(observation):
    return (
        "You are an expert Python programmer. You will be given a question (problem"
        " specification) and will generate a correct Python program that matches the"
        f" specification and passes all tests.\nQuestion: {observation}"
        "\nPlease reason step by step, and write your code in markdown format, e.g.,"
        " ```python\n# YOUR CODE HERE\n```."
    )

TEMPLATE_FACTORY = {
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "qwen3_game": apply_qwen3_game_template,
    "code": apply_code_template,
}

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

async def env_step(env_idx: int, action_text: str):

    (
        next_state,
        reward,
        terminated,
        truncated,
        _
    ) = ENV_POOL[env_idx].step(action_text)
    next_state = TEMPLATE_FACTORY[PROMPT_TEMPLATE](next_state)
    done = terminated or truncated
    
    if done:
        async with LOCK:
            AVAILABLE_ENVS.append(env_idx)
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
            state_text, _ = ENV_POOL[env_idx].reset()

            sample.state_text = TEMPLATE_FACTORY[PROMPT_TEMPLATE](state_text)
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
        
        response = await env_step(sample.sample["env_idx"], sample.action_text)
        add_env_response(tokenizer, sample, response)
        if sample.status == Sample.Status.DONE:
            return