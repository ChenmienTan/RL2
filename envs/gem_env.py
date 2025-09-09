import random
import time
import logging
import asyncio
from typing import Dict, Any, List
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns

NUM_ENVS = 16
GAME = "rg:letter_counting"
WRAPPERS = ""
PROMPT_TEMPLATE = "qwen3_general"
ENV_POOL = []
ENV_LOCKS = []
ENV_IN_USE = []

def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_no_template(observation: str) -> str:
    return observation

def apply_qwen3_general_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_code_template(observation: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {observation}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )

TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}

def _initialize_environments():
    global ENV_POOL, ENV_LOCKS, ENV_IN_USE
    
    logging.info(f"Initializing {NUM_ENVS} GEM environments...")
    
    for i in range(NUM_ENVS):
        try:
            env = gem.make(
                env_id=GAME,
                seed=233 + i,
            )
            wrappers = get_wrapper_fns(WRAPPERS if WRAPPERS else "", tokenizer=None)
            for wrapper in wrappers:
                env = wrapper(env)
            ENV_POOL.append(env)
            ENV_LOCKS.append(asyncio.Lock())
            ENV_IN_USE.append(False)
            logging.info(f"Initialized environment {i}/{NUM_ENVS}")
        except Exception as e:
            logging.error(f"Failed to initialize environment {i}: {e}")
            raise
    
    logging.info(f"Successfully initialized {len(ENV_POOL)} GEM environments")

_initialize_environments()

async def acquire_env_lock(extra_info: Dict[str, Any]) -> int:
    env_idx = extra_info['idx'] % NUM_ENVS
    await ENV_LOCKS[env_idx].acquire()
    ENV_IN_USE[env_idx] = True
    return env_idx

def release_env_lock(env_idx: int):
    ENV_IN_USE[env_idx] = False
    ENV_LOCKS[env_idx].release()

async def reset(extra_info: Dict[str, Any], **kwargs) -> str:
    env_idx = await acquire_env_lock(extra_info)
    
    try:
        observation, _ = ENV_POOL[env_idx].reset()
        
        if isinstance(observation, list):
            observation = observation[0]

        formatted_observation = TEMPLATE_FACTORY[PROMPT_TEMPLATE](observation)
        
        return formatted_observation
    except Exception as e:
        release_env_lock(env_idx)
        logging.error(f"Error resetting environment {env_idx}: {e}")
        raise

async def step(state: str, action: str, extra_info: Dict[str, Any]) -> Dict[str, Any]:
    request_idx = extra_info['idx']
    env_idx = request_idx % NUM_ENVS
    
    if not ENV_IN_USE[env_idx]:
        raise RuntimeError(f"Environment {env_idx} not locked - reset() must be called first")
    
    try:
        next_obs, reward, terminated, truncated, info = ENV_POOL[env_idx].step([action])
        
        if isinstance(next_obs, list):
            next_obs = next_obs[0]
        if isinstance(reward, list):
            reward = reward[0]
        if isinstance(terminated, list):
            terminated = terminated[0]
        if isinstance(truncated, list):
            truncated = truncated[0]
        if isinstance(info, list):
            info = info[0] if info else {}
        
        done = terminated or truncated
        
        if done:
            release_env_lock(env_idx)

        formatted_next_obs = TEMPLATE_FACTORY[PROMPT_TEMPLATE](next_obs)
        return {
            "next_state": formatted_next_obs,
            "reward": float(reward),
            "score": float(reward),
            "done": done,
            "extra_info": info
        }
    except Exception as e:
        release_env_lock(env_idx)
        logging.error(f"Error stepping environment {env_idx}: {e}")
        raise