"""
GEM Rollout Worker for RL2

This module provides a specialized rollout worker for GEM environment integration.
It extends the base Rollout class to handle GEM environment interaction with
proper support for vectorized environments and parallel async generation.
"""

import asyncio
import json
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from copy import deepcopy

import torch
import torch.distributed as dist
from tqdm.asyncio import tqdm

from RL2.workers.rollout import Rollout
from RL2.datasets import tokenize_messages
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.logging import time_logger, gather_and_log
from envs.gem_env import GEMEnvironmentManager, INVALID_ACTION, GEMTransition


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
            logging.info(f"Initialized GEM environment with {self.num_envs} parallel environments")
    
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
        
        # Check if prompt exceeds max length
        max_model_len = self.config.gem_env.get("max_model_len", 12800)
        if len(prompt_ids) >= max_model_len:
            logging.warning(f"Environment {env_idx}: Prompt exceeds max length ({len(prompt_ids)} >= {max_model_len})")
            return INVALID_ACTION, {
                "formatted_observation": formatted_obs,
                "prompt_ids": prompt_ids,
                "response": "",
                "response_ids": [],
                "response_is_truncated": True,
                "action_is_formatted": False,
                "generation_failed": True,
                "env_idx": env_idx,
            }
        
        # Generate response using the language model
        try:
            response = await self.llm.async_generate(
                formatted_obs,
                sampling_params=self.train_sampling_params if train else self.test_sampling_params
            )
            
            meta_info = response["meta_info"]
            
            # Truncate to actual completion tokens if needed
            content = self.tokenizer.decode(
                self.tokenizer.encode(
                    response["text"], add_special_tokens=False
                )[:meta_info["completion_tokens"]]
            )
            
            response_ids = self.tokenizer.encode(content, add_special_tokens=False)
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
                "response_is_truncated": response_is_truncated,
                "action_is_formatted": extracted_action != INVALID_ACTION,
                "generation_failed": False,
                "completion_tokens": meta_info["completion_tokens"],
                "finish_reason": meta_info["finish_reason"]["type"],
                "env_idx": env_idx,
            }
            
        except Exception as e:
            logging.error(f"Environment {env_idx}: Generation failed with error: {e}")
            return INVALID_ACTION, {
                "formatted_observation": formatted_obs,
                "prompt_ids": prompt_ids,
                "response": "",
                "response_ids": [],
                "response_is_truncated": True,
                "action_is_formatted": False,
                "generation_failed": True,
                "env_idx": env_idx,
            }
    
    async def collect_gem_episodes_async(self, min_steps: int, train: bool) -> Tuple[List, dict]:
        """
        Collect episodes from GEM environments with proper parallel async generation.
        
        This method handles vectorized environments by generating actions for ALL
        environment observations in parallel using async/await.
        
        Args:
            min_steps: Minimum number of transition steps to collect
            train: Whether this is training or evaluation
            
        Returns:
            Tuple of (finished_episodes, collection_info)
        """
        # Reset all environments
        obs, _ = self.gem_env_manager.env.reset()
        episodes = [[] for _ in range(self.num_envs)]
        finished_episodes = []
        metrics = defaultdict(list)
        num_generation_failed = 0
        
        logging.info(f"Starting episode collection from {self.num_envs} environments")
        
        while True:
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
                desc=f"Generating actions for {self.num_envs} environments",
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
            
            # Step ALL environments with their respective actions
            next_obs, rewards, terminated, truncated, info = self.gem_env_manager.env.step(actions)
            done = terminated | truncated
            
            # Process transitions for each environment
            for i in range(self.num_envs):
                if extras[i]["generation_failed"]:
                    # Handle generation failure
                    if self.config.gem_env.get("keep_generation_failed", False) and episodes[i]:
                        # Add reward to last transition and mark episode as done
                        episodes[i][-1].reward += rewards[i]
                        episodes[i][-1].done = True
                        finished_episodes.append(deepcopy(episodes[i]))
                    episodes[i].clear()
                    
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
                        response_is_truncated=extras[i]["response_is_truncated"],
                        action_is_formatted=extras[i]["action_is_formatted"],
                    )
                    episodes[i].append(transition)
                    
                    if done[i]:
                        # Episode finished - collect it
                        finished_episodes.append(deepcopy(episodes[i]))
                        
                        # Collect episode metrics
                        episode_return = sum(t.reward for t in episodes[i])
                        episode_length = len(episodes[i])
                        episode_success = episodes[i][-1].reward == 1.0  # Assuming success reward is 1
                        
                        metrics["episode_return"].append(episode_return)
                        metrics["episode_length"].append(episode_length)
                        metrics["episode_success"].append(episode_success)
                        
                        episodes[i].clear()
            
            # Update observations for next step
            obs = next_obs
            
            # Check if we've collected enough transitions
            total_transitions = sum(len(ep) for ep in finished_episodes)
            if total_transitions >= min_steps:
                logging.info(f"Collected {len(finished_episodes)} episodes with {total_transitions} transitions")
                break
        
        # Compute collection statistics
        collection_info = {
            "num_generation_failed": num_generation_failed,
            "num_episodes": len(finished_episodes),
            "mean_episode_return": sum(metrics["episode_return"]) / len(metrics["episode_return"]) if metrics["episode_return"] else 0,
            "mean_episode_length": sum(metrics["episode_length"]) / len(metrics["episode_length"]) if metrics["episode_length"] else 0,
            "mean_episode_success": sum(metrics["episode_success"]) / len(metrics["episode_success"]) if metrics["episode_success"] else 0,
            "mean_response_length": sum(metrics["response_length"]) / len(metrics["response_length"]) if metrics["response_length"] else 0,
            "length_clip_ratio": sum(metrics["length_clip_ratio"]) / len(metrics["length_clip_ratio"]) if metrics["length_clip_ratio"] else 0,
        }
        
        return finished_episodes, collection_info
    
    def prepare_data_for_training(self, finished_episodes: List) -> List[dict]:
        """
        Convert GEM episodes to RL2 training format.
        
        This carefully prepares data to match RL2's expected format:
        - states: input token IDs (all but last token)
        - actions: output token IDs (all but first token) 
        - action_mask: mask for assistant responses
        - position_ids: position indices
        - rewards: dense rewards (zeros except last token)
        - eos_mask: end of sequence mask
        - advantages: computed later by PPO trainer
        
        Args:
            finished_episodes: List of completed episodes
            
        Returns:
            List of data dictionaries for RL2 training
        """
        data_list = []
        
        # Process each episode
        for episode in finished_episodes:
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
                
                # Build messages for tokenization
                messages = [
                    {"role": "user", "content": transition.prompt},
                    {"role": "assistant", "content": transition.response}
                ]
                
                # Tokenize using RL2's tokenization function
                # This creates states, actions, action_mask, and position_ids
                ex = tokenize_messages(
                    self.tokenizer,
                    messages,
                    self.config.gem_env.get("apply_chat_template", True)
                )
                
                # Validate that we have the expected fields
                assert "states" in ex, "Missing states field from tokenization"
                assert "actions" in ex, "Missing actions field from tokenization"
                assert "action_mask" in ex, "Missing action_mask field from tokenization"
                assert "position_ids" in ex, "Missing position_ids field from tokenization"
                
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
                
                # Store additional info for debugging
                ex["gem_info"] = {
                    "step_reward": rewards[i],
                    "discounted_return": returns[i],
                    "action_is_formatted": transition.action_is_formatted,
                    "response_is_truncated": transition.response_is_truncated,
                }
                
                data_list.append(ex)
        
        # Subsample if we have too many trajectories
        rollout_batch_size = self.config.gem_env.get("rollout_batch_size", len(data_list))
        if len(data_list) > rollout_batch_size:
            import random
            data_list = random.sample(data_list, rollout_batch_size)
            logging.info(f"Subsampled {rollout_batch_size} from {len(data_list)} trajectories")
        
        # Validate data format
        if data_list:
            example = data_list[0]
            logging.debug(f"Data validation - Example fields: {example.keys()}")
            logging.debug(f"States shape: {example['states'].shape}")
            logging.debug(f"Actions shape: {example['actions'].shape}")
            logging.debug(f"Action mask shape: {example['action_mask'].shape}")
            logging.debug(f"Rewards shape: {example['rewards'].shape}")
            logging.debug(f"EOS mask shape: {example['eos_mask'].shape}")
        
        return data_list
    
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
            logging.info(f"Starting GEM rollout at step {step} (train={train})")
            
            # Determine how many transitions to collect
            rollout_batch_size = self.config.gem_env.get("rollout_batch_size", 128)
            
            # Run async episode collection
            loop = asyncio.get_event_loop()
            finished_episodes, collection_info = loop.run_until_complete(
                self.collect_gem_episodes_async(rollout_batch_size, train)
            )
            
            # Release memory after training generation
            if train:
                self.llm.release_memory_occupation()
            
            # Log collection metrics
            logging.info(f"Collection complete: {collection_info['num_episodes']} episodes")
            logging.info(f"Mean return: {collection_info['mean_episode_return']:.3f}")
            logging.info(f"Mean success: {collection_info['mean_episode_success']:.3f}")
            logging.info(f"Generation failures: {collection_info['num_generation_failed']}")
            
            # Prepare metrics for logging
            suffix = "train" if train else "test"
            metrics = {f"{k}/{suffix}": [v] for k, v in collection_info.items()}
            gather_and_log(metrics, self.device_mesh["dp"], step)
            
            # For evaluation, just log metrics and return
            if not train:
                return None
            
            # Convert episodes to training data
            data_list = self.prepare_data_for_training(finished_episodes)
            
            # Gather data across distributed processes
            data_list = gather_and_concat_list(data_list, self.device_mesh["dp"])
            
            # Log example data
            if dist.get_rank() == 0 and len(data_list) > 0:
                logging.info(f"Prepared {len(data_list)} training examples")
                example = data_list[0]
                logging.info(f"Example - States shape: {example['states'].shape}")
                logging.info(f"Example - Reward sum: {example['rewards'].sum().item():.3f}")
            
            return data_list if dist.get_rank() == 0 else None
            
        # Wait for primary rank
        dist.barrier()
        return None