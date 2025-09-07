from abc import ABC, abstractmethod
import importlib
import math
import asyncio
import torch
import torch.distributed as dist
from tqdm.asyncio import tqdm
from typing import List, Dict, Any, Tuple, Optional

from RL2.workers import Rollout
from RL2.utils.logging import time_logger
from RL2.datasets import get_tensor_dict, pack_tensor_dicts
from RL2.utils.comm import gather_and_concat_list
from RL2.utils.logging import gather_and_log


class AgentBase(ABC):
    @abstractmethod
    def __init__(self, config, **kwargs):
        self.config = config

    @abstractmethod
    async def reset(self, **kwargs) -> Dict[str, Any]:
        return {"obs": None}

    @abstractmethod
    async def step(self, action: str, train: bool = True, **kwargs) -> Dict[str, Any]:
        pass

    def format_observation(self, observation: str) -> str:
        return observation

class AgentRollout(Rollout):
    def __init__(self, config):
        super().__init__(config)
        self.agent_instances = []

    def _load_agent_class(self):
        agent_class_path = self.config.get('agent_class', None)
        assert agent_class_path is not None, "agent_class_path is not set"
        module_path, class_name = agent_class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def prepare_environment(self):
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.agent_class = self._load_agent_class()
            max_parallel_agents = self.config.get('max_parallel_agents', 4)
            loop = asyncio.get_event_loop()
            self.agent_instances = loop.run_until_complete(asyncio.gather(*[
                asyncio.to_thread(self.agent_class, config=self.config, tokenizer=self.tokenizer, agent_id=i)
                for i in range(max_parallel_agents)
            ]))

    async def rollout(self, agent_idx: int, min_trajectories: int, train: bool) -> Tuple[List[List[Dict]], List[Dict]]:
        agent = self.agent_instances[agent_idx]
        all_tensor_dicts = []
        all_metrics = []
        
        while len(all_tensor_dicts) < min_trajectories:
            current_obs = await agent.reset()
            trajectories = []
            
            while True:
                formatted_obs = agent.format_observation(current_obs)
                if self.config.get("apply_chat_template", True):
                    formatted_obs = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": formatted_obs}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                
                state = self.tokenizer(formatted_obs, add_special_tokens=False).input_ids
                sampling_params = self.config.get("train_sampling_params", {}) if train else self.config.get("test_sampling_params", {})
                response = await self.llm.async_generate(input_ids=state, sampling_params=sampling_params, return_logprob=True)
                
                action_text = response["text"]
                meta_info = response["meta_info"]
                logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
                
                step_result = await agent.step(action_text, train)
                done = step_result["done"]
                reward = step_result["reward"]

                trajectories.append({
                    "states": state + action,
                    "actions": len(state) * [0] + action,
                    "action_masks": len(state) * [0] + len(action) * [1],
                    "logps": [0.0] * len(state) + logp,
                    "rewards": (len(action) - 1) * [0] + [reward],
                    "response_length": meta_info["response_length"],
                    "finish_reason": meta_info["finish_reason"]["type"],
                })
                
                current_obs = step_result["obs"]
                
                if done:
                    break
            
            if trajectories:
                gamma = self.config.get("gamma", 1.0)
                cur = 0.0
                for i in reversed(range(len(trajectories))):
                    cur = trajectories[i]["rewards"][-1] + gamma * cur
                    trajectories[i]["rewards"][-1] = cur
                
                tensor_dicts = [self.get_tensor_dict(trajectory) for trajectory in trajectories]
                
                metrics = {
                    "episode_return": sum(t["rewards"][-1] for t in trajectories),
                    "episode_length": len(trajectories),
                    "response_length": sum(t["response_length"] for t in trajectories),
                    "length_clip_ratio": sum(t["finish_reason"] == "length" for t in trajectories),
                    "episode_success": 1.0 if sum(t["rewards"][-1] for t in trajectories) > 0 else 0.0,
                }
                
                all_tensor_dicts.extend(tensor_dicts)
                all_metrics.extend(metrics * len(tensor_dicts))
        
        if len(all_tensor_dicts) > min_trajectories:
            all_tensor_dicts = all_tensor_dicts[:min_trajectories]
            all_metrics = all_metrics[:min_trajectories]
        
        return all_tensor_dicts, all_metrics

    @time_logger("agent_rollout")
    def __call__(self, data_list, train: bool, step: int):
        if self.device_mesh["tp"].get_local_rank() == 0:
            rollout_batch_size = self.config.get('rollout_batch_size', 32)
            num_agents = len(self.agent_instances)
            min_trajectories = max(1, math.ceil(rollout_batch_size / num_agents))
            
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(i, min_trajectories, train) for i in range(num_agents)),
                    desc="Agent rollout",
                    position=1,
                    leave=False,
                    disable=(dist.get_rank() != 0)
                )
            )
            
            all_tensor_dicts = []
            all_metrics = []
            for tensor_dict_lists, metrics_list in results:
                all_tensor_dicts.extend(tensor_dict_lists)
                all_metrics.extend(metrics_list)
            
            if train:
                self.llm.release_memory_occupation()
        
        dist.barrier()
        
        if self.device_mesh["tp"].get_local_rank() == 0:
            suffix = "train" if train else "test"
            if all_metrics:
                logged_metrics = {}
                for k in all_metrics[0].keys():
                    logged_metrics[f"{k}/{suffix}"] = [m[k] for m in all_metrics]
                logged_metrics[f"num_episodes/{suffix}"] = [len(all_metrics)]
                gather_and_log(logged_metrics, self.device_mesh["dp"], step)
            
            if not train:
                return None
            
            all_tensor_dicts = gather_and_concat_list(all_tensor_dicts, self.device_mesh["dp"])
            
            if dist.get_rank() == 0:
                if not all_tensor_dicts:
                    return None, None
                
                tensor_dicts = []
                for tensor_dict_list in all_tensor_dicts:
                    tensor_dicts.extend(tensor_dict_list)
                
                if not tensor_dicts:
                    return None, None
                    
                tensor_dict = pack_tensor_dicts(tensor_dicts)
                seqs = torch.LongTensor([
                    len(tensor_dict_list) for tensor_dict_list in all_tensor_dicts
                ])
                cu_seqs = torch.cumsum(
                    torch.cat((torch.LongTensor([0]), seqs)), dim=0
                )
                
                return tensor_dict, cu_seqs
        
        return None, None