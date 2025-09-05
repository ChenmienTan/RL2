"""
GEM Rollout Worker for RL2

This module provides a specialized rollout worker for GEM environment integration.
It extends the base Rollout class to handle GEM environment interaction with
proper support for vectorized environments and parallel async generation.

This module is self-contained and includes all GEM environment functionality.
"""

import asyncio
import re
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tqdm.asyncio import tqdm

from RL2.workers.rollout import Rollout
from RL2.utils.comm import gather_and_concat_list
from RL2.utils.logging import time_logger, gather_and_log
from RL2.datasets import get_tensor_dict, pack_tensor_dicts

import gem
from gem.utils.parsing import extract_last_boxed_answer
from gem.wrappers.wrapper_factory import get_wrapper_fns


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
    llm_logps: list
    
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
        self.config = config
        self.tokenizer = tokenizer
        self.gem_config = config.gem_env
        
        # Get environment wrappers
        wrappers = get_wrapper_fns(self.gem_config.wrappers, tokenizer=tokenizer)
        
        # Instantiate vectorized environment
        # Note: gem.make_vec expects a single env_id and num_envs parameter, not a list of env_ids
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

class GEMRollout(Rollout):
    """
    Specialized rollout worker for GEM environment integration.
    
    Handles vectorized GEM environments with proper parallel async generation
    for all environment observations simultaneously.
    """
    
    def __init__(self, config):
        """Initialize GEM rollout worker."""
        super().__init__(config)
        
        # Initialize GEM environment manager on the primary device
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.gem_env_manager = GEMEnvironmentManager(config, self.tokenizer)
            self.num_envs = self.gem_env_manager.env.num_envs
    

    def prepare_environment(self):
        """Override to skip base class environment loading for GEM environments."""
        # GEM environments are handled by GEMEnvironmentManager
        # We don't need to load a Python module from env_path
        pass
    
    def tokenize_messages(self, messages, rm=False):
        prev_text, states, actions, action_mask = "", [], [], []
        for turn in range(len(messages)):
            is_this_turn_assistant = messages[turn]["role"] == "assistant"
            is_next_turn_assistant = turn + 1 < len(messages) and messages[turn + 1]["role"] == "assistant"

            text = self.tokenizer.apply_chat_template(
                messages[:turn + 1],
                add_generation_prompt=is_next_turn_assistant,
                tokenize=False
            )
            assert text[:len(prev_text)] == prev_text
            state = self.tokenizer.encode(
                text[len(prev_text):], add_special_tokens=False
            )
            states.extend(state)
            actions.extend(
                state
                if is_this_turn_assistant
                else len(state) * [0]
            )
            action_mask.extend(len(state) * [is_this_turn_assistant])
            prev_text = text

        return get_tensor_dict(
            states, actions, action_mask
        )

    async def generate_action_for_observation(
        self, 
        observation: str, 
        env_idx: int,
        train: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate action for a single observation from one environment.
        
        Args:
            observation: Raw observation from environment
            env_idx: Index of the environment this observation came from
            train: Whether this is training or evaluation
            
        Returns:
            Tuple of (action, extra_info)
        """
        # Format observation using templates
        formatted_obs = self.gem_env_manager.format_observation(
            observation,
            self.config.gem_env.get("prompt_template", "no"),
            self.config.gem_env.get("apply_chat_template", True)
        )
        
        # Tokenize to check length
        prompt_ids = self.tokenizer(formatted_obs, add_special_tokens=False).input_ids
        logps = len(prompt_ids) * [0]
        # Check if prompt exceeds max length
        max_model_len = self.config.gem_env.get("max_model_len", 12800)
        if len(prompt_ids) >= max_model_len:
            return INVALID_ACTION, {
                "formatted_observation": formatted_obs,
                "prompt_ids": prompt_ids,
                "response": "",
                "response_ids": [],
                "llm_logps": logps,
                "response_is_truncated": True,
                "action_is_formatted": False,
                "generation_failed": True,
                "env_idx": env_idx,
            }
        
        # Generate response using the language model
        try:
            response = await self.llm.async_generate(
                input_ids=prompt_ids,
                sampling_params=self.train_sampling_params if train else self.test_sampling_params,
                return_logprob=True
            )
            
            meta_info = response["meta_info"]
            logp, state, _ = map(list, zip(*meta_info["output_token_logprobs"]))

            response_ids = state
            content = response["text"]
            logps.extend(logp)

            response_is_truncated = meta_info["finish_reason"]["type"] == "length"
            
            # Extract structured action from response
            extracted_action = self.gem_env_manager.extract_action(
                content,
                self.config.gem_env.get("prompt_template", "no"),
                self.config.model_name
            )
            
            # Use raw response as action (environment handles extraction internally)
            executable_action = INVALID_ACTION if response_is_truncated else content
            
            return executable_action, {
                "formatted_observation": formatted_obs,
                "prompt_ids": prompt_ids,
                "response": content,
                "response_ids": response_ids,
                "llm_logps": logps,
                "response_is_truncated": response_is_truncated,
                "action_is_formatted": extracted_action != INVALID_ACTION,
                "generation_failed": False,
                "completion_tokens": meta_info["completion_tokens"],
                "finish_reason": meta_info["finish_reason"]["type"],
                "env_idx": env_idx,
            }
            
        except Exception:
            return INVALID_ACTION, {
                "formatted_observation": formatted_obs,
                "prompt_ids": prompt_ids,
                "response": "",
                "response_ids": [],
                "llm_logps": logps,
                "response_is_truncated": True,
                "action_is_formatted": False,
                "generation_failed": True,
                "env_idx": env_idx,
            }
    
    async def rollout(self, min_steps: int, train: bool) -> Tuple[List, dict]:
        """
        Collect episodes from GEM environments and return training-ready tensor dicts.
        
        This method handles vectorized environments by generating actions for ALL
        environment observations in parallel using async/await, and directly converts
        the collected data into training-ready format like rollout.py.
        
        Args:
            min_steps: Minimum number of transition steps to collect
            train: Whether this is training or evaluation
            
        Returns:
            Tuple of (tensor_dicts_list, collection_info)
            - tensor_dicts_list: List of lists of tensor dicts ready for training
            - collection_info: Dictionary containing collection statistics
        """
        # Reset all environments
        obs, _ = self.gem_env_manager.env.reset()
        episodes = [[] for _ in range(self.num_envs)]
        all_tensor_dicts = []  # Store lists of tensor dicts like rollout.py
        metrics = defaultdict(list)
        num_generation_failed = 0
        step_count = 0
        
        while True:
            step_count += 1
            
            # Generate actions for ALL observations in parallel
            # This is the key for efficiency with vectorized environments
            generation_tasks = []
            for env_idx, observation in enumerate(obs):
                task = self.generate_action_for_observation(observation, env_idx, train)
                generation_tasks.append(task)
                
            # Run all generation tasks in parallel and wait for ALL to complete
            # This ensures we utilize the full parallelism of async generation
            results = await tqdm.gather(
                *generation_tasks,
                desc=f"Step {step_count}: Generating actions for {self.num_envs} environments",
                leave=False,
                disable=(dist.get_rank() != 0)
            )
            
            # Extract actions and extras from results
            actions = []
            extras = []
            for action, extra in results:
                actions.append(action)
                extras.append(extra)
                
                # Collect generation metrics
                if not extra["generation_failed"]:
                    metrics["response_length"].append(extra.get("completion_tokens", 0))
                    metrics["length_clip_ratio"].append(
                        1 if extra.get("finish_reason") == "length" else 0
                    )
                else:
                    num_generation_failed += 1
                        
            next_obs, rewards, terminated, truncated, _ = self.gem_env_manager.env.step(actions)
            done = terminated | truncated
            
            # Process transitions for each environment
            episodes_completed_this_step = 0
            episodes_reset_this_step = 0
            
            for i in range(self.num_envs):
                if extras[i]["generation_failed"]:
                    # Handle generation failure
                    
                    if self.config.gem_env.get("keep_generation_failed", False) and episodes[i]:
                        # Add reward to last transition and mark episode as done
                        episodes[i][-1].reward += rewards[i]
                        episodes[i][-1].done = True
                        # Convert episode to tensor dicts and add to results
                        tensor_dicts = self._convert_episode_to_tensor_dicts(episodes[i])
                        if tensor_dicts:
                            all_tensor_dicts.append(tensor_dicts)
                        episodes_completed_this_step += 1
                    episodes[i].clear()
                    episodes_reset_this_step += 1
                    
                    # Reset this environment if not done
                    if not done[i]:
                        next_obs[i] = self.gem_env_manager.env.envs[i].reset()[0]
                else:
                    # Create transition for successful generation
                    transition = GEMTransition(
                        obs=obs[i],
                        action=actions[i],
                        reward=rewards[i],
                        done=done[i],
                        prompt=extras[i]["formatted_observation"],
                        prompt_ids=extras[i]["prompt_ids"],
                        response=extras[i]["response"],
                        response_ids=extras[i]["response_ids"],
                        llm_logps=extras[i]["llm_logps"],
                        response_is_truncated=extras[i]["response_is_truncated"],
                        action_is_formatted=extras[i]["action_is_formatted"],
                    )
                    episodes[i].append(transition)
                    
                    if done[i]:
                        # Episode finished - convert to tensor dicts
                        tensor_dicts = self._convert_episode_to_tensor_dicts(episodes[i])
                        if tensor_dicts:
                            all_tensor_dicts.append(tensor_dicts)
                        episodes_completed_this_step += 1
                        
                        metrics["episode_return"].append(sum(t.reward for t in episodes[i]))
                        metrics["episode_length"].append(len(episodes[i]))
                        metrics["episode_success"].append(episodes[i][-1].reward == 1.0)
                        
                        episodes[i].clear()
                        episodes_reset_this_step += 1
            
            # Update observations for next step
            obs = next_obs
            
            # Check if we've collected enough transitions
            total_transitions = sum(len(tensor_dicts) for tensor_dicts in all_tensor_dicts)
            if total_transitions >= min_steps:
                break
        
        # Compute collection statistics
        collection_info = {
            "num_generation_failed": num_generation_failed,
            "num_episodes": len(all_tensor_dicts),
            "mean_episode_return": sum(metrics["episode_return"]) / len(metrics["episode_return"]) if metrics["episode_return"] else 0,
            "mean_episode_length": sum(metrics["episode_length"]) / len(metrics["episode_length"]) if metrics["episode_length"] else 0,
            "mean_episode_success": sum(metrics["episode_success"]) / len(metrics["episode_success"]) if metrics["episode_success"] else 0,
            "mean_response_length": sum(metrics["response_length"]) / len(metrics["response_length"]) if metrics["response_length"] else 0,
            "length_clip_ratio": sum(metrics["length_clip_ratio"]) / len(metrics["length_clip_ratio"]) if metrics["length_clip_ratio"] else 0,
        }
        
        return all_tensor_dicts, collection_info
    
    def _convert_episode_to_tensor_dicts(self, episode: List) -> List[dict]:
        """
        Convert a single GEM episode to a list of tensor dicts ready for training.
        
        This method takes the logic from prepare_data_for_training and applies it
        to a single episode to directly create training-ready tensor dictionaries.
        
        Args:
            episode: List of GEMTransition objects for one episode
            
        Returns:
            List of tensor dictionaries ready for RL2 training
        """
        if not episode:
            return []
            
        tensor_dicts = []
        
        # Compute returns for the episode
        rewards = [t.reward for t in episode]
        gamma = self.config.gem_env.get("gamma", 1.0)
        
        returns = [0.0] * len(rewards)
        cur = 0.0
        for i in reversed(range(len(rewards))):
            cur = rewards[i] + gamma * cur
            returns[i] = cur
        
        # Process each transition in the episode
        for i, transition in enumerate(episode):
            # Skip if no response was generated
            if not transition.response_ids:
                continue
            
            states = transition.prompt_ids + transition.response_ids
            actions = len(transition.prompt_ids) * [0] + transition.response_ids
            action_mask = len(transition.prompt_ids) * [0] + len(transition.response_ids) * [1]
            ex = get_tensor_dict(states, actions, action_mask)
            
            # Add reward information
            # RL2 expects dense rewards - zeros for all tokens except the last
            num_tokens = ex["action_mask"].shape[0]
            dense_rewards = torch.zeros(num_tokens, dtype=torch.float32)
            
            # Find last action token (where action_mask is 1)
            last_action_idx = -1
            for j in reversed(range(num_tokens)):
                if ex["action_mask"][j] == 1:
                    last_action_idx = j
                    break
                    
            if last_action_idx >= 0:
                # Assign the discounted return to the last action token
                dense_rewards[last_action_idx] = returns[i]
            
            # Add reward and EOS mask
            ex["rewards"] = dense_rewards
            
            # EOS mask marks the end of sequences
            eos_mask = torch.zeros(num_tokens, dtype=torch.long)
            if last_action_idx >= 0:
                eos_mask[last_action_idx] = 1
            ex["eos_mask"] = eos_mask
            ex["llm_logps"] = torch.FloatTensor(transition.llm_logps[1:])
            
            assert ex["llm_logps"].shape[0] == ex["action_mask"].shape[0]
            tensor_dicts.append(ex)
        
        return tensor_dicts
    
    @time_logger("gem_rollout")
    def __call__(self, data_list, train: bool, step: int):
        """
        Main rollout function for GEM environments.
        
        Ignores input data_list and collects experiences directly from
        GEM environments using parallel async generation.
        
        Args:
            data_list: Ignored for GEM environments
            train: Whether this is training or evaluation
            step: Current training step
            
        Returns:
            List of training data or None for evaluation
        """
        # The data is distributed across ranks
        if self.device_mesh["tp"].get_local_rank() == 0:
            # For GEM environments, we collect from environments instead of using data_list
            
            # Determine how many transitions to collect
            rollout_batch_size = self.config.gem_env.get("rollout_batch_size", 128)
            
            
            # Run async episode collection - now returns tensor dicts directly
            loop = asyncio.get_event_loop()
            all_tensor_dicts, collection_info = loop.run_until_complete(
                self.rollout(rollout_batch_size, train)
            )
            
            # Release memory after training generation
            if train:
                self.llm.release_memory_occupation()
            
        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:
            # Prepare metrics for logging
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": [v] for k, v in collection_info.items()}
            gather_and_log(metrics, self.device_mesh["dp"], step)
            
            # For evaluation, just log metrics and return
            if not train:
                return None
            
            # Data is already prepared for training by collect_gem_episodes_async
            # Just gather across distributed processes
            all_tensor_dicts = gather_and_concat_list(all_tensor_dicts, self.device_mesh["dp"])
            
            if dist.get_rank() == 0:
                # Flatten the list of tensor dict lists to a single list
                tensor_dicts = sum(all_tensor_dicts, [])
                tensor_dict = pack_tensor_dicts(tensor_dicts)
                seqs = torch.LongTensor([
                    len(tensor_dicts) for tensor_dicts in all_tensor_dicts
                ])
                cu_seqs = torch.cumsum(
                    torch.cat((torch.LongTensor([0]), seqs)), dim=0
                )
                return tensor_dict, cu_seqs
        
        return None, None