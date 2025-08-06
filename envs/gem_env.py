"""
GEM Environment Integration Module for RL2

This module provides optional integration with GEM environments for RL2 training.
It contains all GEM-specific logic including prompt templates, action extraction,
and environment management.
"""

import re
import logging
from typing import List, Literal, Optional, Sequence, Tuple
from dataclasses import dataclass

# Optional GEM imports - graceful handling if not installed
try:
    import gem
    from gem.utils.parsing import extract_last_boxed_answer
    from gem.wrappers.wrapper_factory import get_wrapper_fns
    GEM_AVAILABLE = True
except ImportError:
    GEM_AVAILABLE = False
    gem = None


# Invalid action to be sent to the env to trigger format error penalty
INVALID_ACTION = "<｜INVALID_ACTION｜>"


def apply_qwen3_game_template(observation: str) -> str:
    """Apply Qwen3 game-specific template to observation."""
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_no_template(observation: str) -> str:
    """Apply no template - return observation as-is."""
    return observation


def apply_qwen3_general_template(question: str) -> str:
    """Apply Qwen3 general template to question."""
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_code_template(question: str) -> str:
    """Apply code-specific template to question."""
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {question}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )


TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}


@dataclass
class GEMTransition:
    """Data structure for GEM environment transitions."""
    obs: str
    action: str
    reward: float
    done: bool
    
    prompt: str
    prompt_ids: list
    response: str
    response_ids: list
    
    response_is_truncated: bool
    action_is_formatted: bool
    
    def format(self):
        """Format transition for logging/debugging."""
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.reward,
            "done": int(self.done),
            "prompt": self.prompt,
            "response": self.response,
        }


class GEMEnvironmentManager:
    """Manages GEM environment integration for RL2."""
    
    def __init__(self, config, tokenizer):
        """Initialize GEM environment manager."""
        if not GEM_AVAILABLE:
            raise ImportError(
                "GEM is not installed. Please install GEM to use GEM environments:\n"
                "pip install gem-rl"
            )
        
        self.config = config
        self.tokenizer = tokenizer
        self.gem_config = config.gem_env
        
        # Get environment wrappers
        wrappers = get_wrapper_fns(self.gem_config.wrappers, tokenizer=tokenizer)
        
        # Instantiate vectorized environment
        self.env = gem.make_vec(
            [self.gem_config.env_id] * self.gem_config.num_env,
            vec_kwargs=[
                {"seed": self.gem_config.get("seed", 233) + j} 
                for j in range(self.gem_config.num_env)
            ],
            wrappers=wrappers,
            async_mode=self.gem_config.get("async_env", False),
        )
        
        logging.info(f"Initialized GEM environment: {self.gem_config.env_id}")
    
    def extract_action(self, text: str, prompt_template: str, model_path: str = "") -> str:
        """
        Extract and format the actual action from the model's output.
        
        This method handles different template formats and ensures the action
        is properly formatted for the environment.
        """
        if not text:
            return ""
        
        try:
            formatted_action = None
            if prompt_template in ["qwen3_game", "qwen3_general"] or (
                prompt_template == "no" and "qwen" in model_path.lower()
            ):
                formatted_action = extract_last_boxed_answer(text)
                if formatted_action is None:
                    formatted_action = text.strip()
            elif prompt_template == "code":
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
                if not code_blocks:
                    formatted_action = None
                else:
                    formatted_action = code_blocks[-1].strip()
            else:
                # Default: use text as-is
                formatted_action = text.strip()
            
            if formatted_action is None:
                formatted_action = INVALID_ACTION
            
            return formatted_action
            
        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            return INVALID_ACTION
    
    def format_observation(self, observation: str, prompt_template: str, apply_chat_template: bool) -> str:
        """Format observation using the specified template."""
        formatted_obs = TEMPLATE_FACTORY.get(prompt_template, apply_no_template)(observation)
        
        if apply_chat_template:
            formatted_obs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": formatted_obs}],
                tokenize=False,
                add_generation_prompt=True,
            )
        
        return formatted_obs
    
    def collect_episodes(self, min_steps: int, max_model_len: int, 
                        prompt_template: str, apply_chat_template: bool,
                        keep_generation_failed: bool) -> Tuple[List[List[GEMTransition]], dict]:
        """
        Collect episodes from GEM environment.
        
        Returns:
            Tuple of (finished_episodes, collection_info)
        """
        obs, _ = self.env.reset()
        episodes = [[] for _ in range(self.env.num_envs)]
        finished_episodes = []
        finished_episodes_tool_uses = []
        finished_episodes_tool_success = []
        num_generation_failed = 0
        
        while True:
            # Format observations
            formatted_observations = []
            for observation in obs:
                formatted_obs = self.format_observation(observation, prompt_template, apply_chat_template)
                formatted_observations.append(formatted_obs)
            
            # Check for observations that exceed max model length
            exceeds_lengths = []
            prompt_ids_list = []
            for formatted_obs in formatted_observations:
                ids = self.tokenizer(formatted_obs, add_special_tokens=False).input_ids
                exceeds_lengths.append(len(ids) >= max_model_len)
                prompt_ids_list.append(ids)
            
            # This will be filled by the rollout worker's generation logic
            # For now, we create placeholder actions and extras
            actions = []
            extras = []
            
            for i in range(self.env.num_envs):
                if exceeds_lengths[i]:
                    actions.append(INVALID_ACTION)
                    extras.append({"generation_failed": True})
                else:
                    # This would normally come from model generation
                    # Placeholder - will be replaced in actual rollout
                    actions.append("")
                    extras.append({
                        "formatted_observation": formatted_observations[i],
                        "prompt_ids": prompt_ids_list[i],
                        "response": "",
                        "response_ids": [],
                        "response_is_truncated": False,
                        "action_is_formatted": False,
                        "generation_failed": True,  # Will be updated by rollout worker
                    })
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated | truncated
            
            # Process transitions
            for i in range(self.env.num_envs):
                if extras[i]["generation_failed"]:
                    num_generation_failed += 1
                    if keep_generation_failed and episodes[i]:
                        episodes[i][-1].reward += reward[i]
                        episodes[i][-1].done = True
                        finished_episodes.append(episodes[i].copy())
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0) if done[i]
                            else info[i].get("tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0) if done[i]
                            else info[i].get("tool_success_counter", 0)
                        )
                    episodes[i].clear()
                    if not done[i]:
                        next_obs[i] = self.env.envs[i].reset()[0]
                else:
                    transition = GEMTransition(
                        obs=obs[i],
                        action=actions[i],
                        reward=reward[i],
                        done=done[i],
                        prompt=extras[i]["formatted_observation"],
                        prompt_ids=extras[i]["prompt_ids"],
                        response=extras[i]["response"],
                        response_ids=extras[i]["response_ids"],
                        response_is_truncated=extras[i]["response_is_truncated"],
                        action_is_formatted=extras[i]["action_is_formatted"],
                    )
                    episodes[i].append(transition)
                    
                    if done[i]:
                        finished_episodes.append(episodes[i].copy())
                        finished_episodes_tool_uses.append(
                            info[i].get("prev_ep_tool_use_counter", 0)
                        )
                        finished_episodes_tool_success.append(
                            info[i].get("prev_ep_tool_success_counter", 0)
                        )
                        episodes[i].clear()
            
            obs = next_obs
            if len([t for ep in finished_episodes for t in ep]) >= min_steps:
                break
        
        collection_info = {
            "num_generation_failed": num_generation_failed,
            "prop_generation_failed": (
                num_generation_failed / len(finished_episodes) if keep_generation_failed
                else num_generation_failed / (len(finished_episodes) + num_generation_failed)
            ) if (len(finished_episodes) + num_generation_failed) > 0 else 0,
            "num_tool_uses": sum(finished_episodes_tool_uses) / len(finished_episodes_tool_uses) if finished_episodes_tool_uses else 0,
            "num_tool_success": sum(finished_episodes_tool_success) / len(finished_episodes_tool_success) if finished_episodes_tool_success else 0,
        }
        
        return finished_episodes, collection_info
    
    def prepare_trajectories(self, episode: Sequence[GEMTransition], gamma: float) -> List[dict]:
        """
        Prepare language trajectories from episode transitions.
        
        Computes returns and prepares data for RL training.
        """
        trajectory_data = []
        rewards = [t.reward for t in episode]
        
        # Compute returns (discounted cumulative rewards)
        returns = [0.0] * len(rewards)
        cur = 0.0
        for i in reversed(range(len(rewards))):
            cur = rewards[i] + gamma * cur
            returns[i] = cur
        
        # Create trajectory data
        for i, step_data in enumerate(episode):
            trajectory_data.append({
                "prompt": step_data.prompt,
                "prompt_ids": step_data.prompt_ids,
                "response": step_data.response,
                "response_ids": step_data.response_ids,
                "reward": returns[i],  # Use discounted return as reward
                "info": {
                    "action_is_formatted": step_data.action_is_formatted,
                    "step_reward": rewards[i],
                    "discount_factor": gamma,
                    "discounted_step_return": returns[i],
                    "response_is_truncated": step_data.response_is_truncated,
                },
            })
        
        return trajectory_data