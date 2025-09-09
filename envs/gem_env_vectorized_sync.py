import random
import time
import logging
from typing import Dict, Any, List, Optional
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns
import asyncio

NUM_ENVS = 16
GAME = "game:GuessTheNumber-v0"
WRAPPERS = "concat"
PROMPT_TEMPLATE = "qwen3_general"

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

class VectorizedGemEnv:
    
    def __init__(self):
        self.num_envs = NUM_ENVS
        self.active_episodes = {}
        self.pending_actions = {}
        self.episode_states = {}
        self.action_ready_count = 0
        self.batch_results = {}
        
        logging.info(f"Initializing {NUM_ENVS} vectorized GEM environments...")
        
        self.vec_env = gem.make_vec(
            [GAME] * NUM_ENVS,
            vec_kwargs=[{"seed": 233 + i} for i in range(NUM_ENVS)],
            wrappers=get_wrapper_fns(WRAPPERS if WRAPPERS else "", tokenizer=None),
            async_mode=True,
        )
        
        logging.info(f"Successfully initialized vectorized GEM environment with {NUM_ENVS} parallel environments")
    
    def reset_episode(self, episode_id: str, extra_info: Dict[str, Any]) -> str:
        available_env_idx = None
        for i in range(self.num_envs):
            if i not in self.active_episodes.values():
                available_env_idx = i
                break
        
        if available_env_idx is None:
            raise RuntimeError("No available environments for new episode")
        
        observations, _ = self.vec_env.reset()
        observation = observations[available_env_idx]
        
        self.active_episodes[episode_id] = available_env_idx
        self.episode_states[episode_id] = {
            "env_idx": available_env_idx,
            "extra_info": extra_info
        }
        
        formatted_observation = TEMPLATE_FACTORY[PROMPT_TEMPLATE](observation)
        return formatted_observation
    
    def add_action(self, episode_id: str, action: str) -> bool:
        if episode_id not in self.active_episodes:
            raise RuntimeError(f"Episode {episode_id} not active")
        
        self.pending_actions[episode_id] = action
        self.action_ready_count += 1
        
        return self.action_ready_count == len(self.active_episodes)
    
    def execute_batch_step(self) -> Dict[str, Dict[str, Any]]:
        if len(self.pending_actions) != len(self.active_episodes):
            raise RuntimeError(f"Not all actions ready: {len(self.pending_actions)}/{len(self.active_episodes)}")
        
        action_array = [None] * self.num_envs
        episode_to_idx = {}
        
        for episode_id, action in self.pending_actions.items():
            env_idx = self.active_episodes[episode_id]
            action_array[env_idx] = action
            episode_to_idx[episode_id] = env_idx
        
        for i in range(self.num_envs):
            if action_array[i] is None:
                action_array[i] = ""
        
        results = {}
        
        try:
            next_obs_list, reward_list, terminated_list, truncated_list, info_list = self.vec_env.step(action_array)
            
            for episode_id, env_idx in episode_to_idx.items():
                extra_info = self.episode_states[episode_id]["extra_info"]
                
                next_obs = next_obs_list[env_idx]
                reward = reward_list[env_idx]
                terminated = terminated_list[env_idx]
                truncated = truncated_list[env_idx]
                info = info_list[env_idx] if info_list else {}
                
                done = terminated or truncated
                
                updated_extra_info = {
                    **extra_info,
                    **info
                }
                
                if done:
                    del self.active_episodes[episode_id]
                    del self.episode_states[episode_id]
                else:
                    self.episode_states[episode_id]["extra_info"] = updated_extra_info
                
                formatted_next_obs = TEMPLATE_FACTORY[PROMPT_TEMPLATE](next_obs)
                results[episode_id] = {
                    "next_state": formatted_next_obs,
                    "reward": float(reward),
                    "score": float(reward),
                    "done": done,
                    "extra_info": updated_extra_info
                }
                
        except Exception as e:
            logging.error(f"Error in batch step execution: {e}")
            for episode_id in list(self.active_episodes.keys()):
                results[episode_id] = {"error": str(e)}
            self.active_episodes.clear()
            self.episode_states.clear()
        
        self.pending_actions.clear()
        self.action_ready_count = 0
        
        return results

VECTORIZED_ENV = VectorizedGemEnv()

EPISODE_RESULTS = {}
WAITING_EPISODES = set()

async def reset(extra_info: Dict[str, Any], **kwargs) -> str:
    episode_id = str(extra_info.get('idx', random.randint(0, 999999)))
    
    try:
        observation = VECTORIZED_ENV.reset_episode(episode_id, extra_info)
        return observation
    except Exception as e:
        logging.error(f"Error resetting episode {episode_id}: {e}")
        raise

async def step(state: str, action: str, extra_info: Dict[str, Any]) -> Dict[str, Any]:
    episode_id = str(extra_info.get('idx', 0))
    
    try:
        all_ready = VECTORIZED_ENV.add_action(episode_id, action)
        WAITING_EPISODES.add(episode_id)
        
        if not all_ready:
            while not all_ready:
                await asyncio.sleep(0.001)
                all_ready = len(VECTORIZED_ENV.pending_actions) == len(VECTORIZED_ENV.active_episodes)
        
        if all_ready and episode_id in WAITING_EPISODES:
            batch_results = VECTORIZED_ENV.execute_batch_step()
            
            for waiting_episode_id in WAITING_EPISODES:
                if waiting_episode_id in batch_results:
                    EPISODE_RESULTS[waiting_episode_id] = batch_results[waiting_episode_id]
            
            WAITING_EPISODES.clear()
        
        while episode_id not in EPISODE_RESULTS:
            await asyncio.sleep(0.001)
        
        result = EPISODE_RESULTS.pop(episode_id)
        
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
            
    except Exception as e:
        WAITING_EPISODES.discard(episode_id)
        EPISODE_RESULTS.pop(episode_id, None)
        logging.error(f"Error stepping episode {episode_id}: {e}")
        raise

def get_active_episodes() -> List[str]:
    """Return list of currently active episode IDs."""
    return list(VECTORIZED_ENV.active_episodes.keys())

def get_pending_actions() -> Dict[str, str]:
    """Return dictionary of pending actions by episode ID."""
    return dict(VECTORIZED_ENV.pending_actions)