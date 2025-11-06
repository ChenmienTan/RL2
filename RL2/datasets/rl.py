from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from omegaconf import DictConfig
import asyncio
from copy import deepcopy
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from RL2.datasets import get_tensor_dict
from RL2.datasets.base import BaseDataset


class Experience:

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        state_text: Optional[str],
        extra_info: Dict[str, Any]
    ):

        self.config = config
        self.tokenizer = tokenizer
        self._initialize(state_text, extra_info)

        self.previous_action_text = ""
        self.previous_response_length = 0

        self.initial_state_text = state_text
        self.initial_extra_info = deepcopy(extra_info)

    def _initialize(self, state_text, extra_info):
        
        self.state_text = state_text
        self.extra_info = extra_info

        self.turn = 0
        if state_text is not None:
            self.state_dict = self._initialize_state_dict(state_text)
        self.state_dicts: List[Dict[str, List[Union[int, float]]]] = []
        self.rewards, self.scores = [], []
        self.metrics = defaultdict(list)

        self.done = False

    def _initialize_state_dict(
        self, state_text: str
    ) -> Dict[str, List[Union[int, float]]]:
        
        state = self.tokenizer.encode(state_text, add_special_tokens=False)
        return {
            "states": state,
            "actions": len(state) * [0],
            "action_mask": len(state) * [0],
            "logps": len(state) * [0.0],
            "rewards": len(state) * [0.0]
        }

    def _add_llm_response(self, payload: Dict[str, Any]) -> bool:

        # `previous_action_text` is not empty if aborted before
        self.action_text = self.previous_action_text + payload["text"]
        self.turn += 1
        # COMMENT: token-in-token-out
        meta_info = payload["meta_info"]
        if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"][0]) == 3:
            logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
            self.state_dict["states"].extend(action)
            self.state_dict["actions"].extend(action)
            self.state_dict["action_mask"].extend(len(action) * [1])
            self.state_dict["logps"].extend(logp)

        finish_reason = meta_info["finish_reason"]["type"]
        if finish_reason == "abort":
            if self.config.mask_offpolicy_data:
                # mask previous actions to guarantee fully onpolicy training
                length = len(self.state_dict["states"])
                self.state_dict["actions"] = length * [0]
                self.state_dict["action_mask"] = length * [0]
                self.state_dict["logps"] = length * [0.0]
                self.state_dicts = self.state_dicts[-1:]
            self.previous_action_text = self.action_text
            self.previous_response_length += meta_info["completion_tokens"]
            return True
        
        self.metrics["response_length"].append(
            self.previous_response_length + meta_info["completion_tokens"]
        )
        self.metrics["length_clip_ratio"].append(finish_reason == "length")

        # reset if not aborted
        self.previous_action_text = ""
        self.previous_response_length = 0
        return False

    def _add_env_response(self, payload: Dict[str, Any]) -> bool:
        
        self.extra_info = payload["extra_info"]
        self.state_dict["rewards"].extend(
            (
                len(self.state_dict["states"]) - len(self.state_dict["rewards"]) - 1
            ) * [0] + [payload["reward"]]
        )
        self.rewards.append(payload["reward"])
        self.scores.append(payload["score"])

        if self.turn == self.config.max_turns or payload["done"]:
            self.state_dicts.append(self.state_dict)
            self.metrics["n_turns"].append(self.turn)
            self.metrics["reward"].append(sum(self.rewards))
            self.metrics["scores"].append(sum(self.scores))
            return True
        if payload["next_state"].startswith(self.state_text + self.action_text):
            state_dict_delta = self._initialize_state_dict(
                payload["next_state"][len(self.state_text + self.action_text):]
            )
            for k, v in state_dict_delta.items():
                self.state_dict[k].extend(v)
        else:
            self.state_dicts.append(self.state_dict)
            self.state_dict = self._initialize_state_dict(payload["next_state"])
        self.state_text = payload["next_state"]
        return False

    async def make(
        self,
        async_generate_func: Callable,
        env_step_func: Callable,
        env_reset_func: Optional[Callable]
    ):
        
        if self.done:
            if not self.config.mask_offpolicy_data:
                return
            self._initialize(
                self.initial_state_text,
                self.initial_extra_info
            )

        if self.state_text is None:
            self.state_text, self.extra_info = await env_reset_func()
            self.state_dict = self._initialize_state_dict(self.state_text)

        while True:

            abort = self._add_llm_response(
                await async_generate_func(
                    self.state_dict["states"],
                    self.config.max_new_tokens - self.previous_response_length
                )
            )
            if abort:
                return
            self.done = self._add_env_response(
                await env_step_func(
                    self.state_text,
                    self.action_text,
                    self.extra_info
                )
            )
            if self.done:
                return

    def to_tensor_dicts_and_metrics(self) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, List[Union[float, int, bool]]]]:

        tensor_dicts = []
        for state_dict in self.state_dicts:
            tensor_dict = get_tensor_dict(
                state_dict["states"],
                state_dict["actions"],
                state_dict["action_mask"]
            )
            tensor_dict["llm_logps"] = torch.FloatTensor(
                state_dict["logps"][1:]
            )
            tensor_dict["rewards"] = torch.FloatTensor(
                state_dict["rewards"][1:]
            )
            tensor_dicts.append(tensor_dict)
        return tensor_dicts, self.metrics


class ExperienceGroup:

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        state_text: Optional[str],
        extra_info: Dict[str, Any]
    ):

        self.experiences = [
            Experience(
                config,
                tokenizer,
                state_text,
                deepcopy(extra_info)
            )
            for _ in range(config.responses_per_prompt)
        ]

    async def make(
        self,
        async_generate_func: Callable,
        env_step_func: Callable,
        env_reset_func: Optional[Callable]
    ) -> "ExperienceGroup":
        await asyncio.gather(*(
            experience.make(
                async_generate_func,
                env_step_func,
                env_reset_func
            )
            for experience in self.experiences
        ))
        return self
    
    def to_all_tensor_dicts_and_metrics(self) -> Tuple[List[List[Dict[str, torch.Tensor]]], Dict[str, List[Union[float, int, bool]]]]:
        
        all_tensor_dicts, metrics = [], defaultdict(list)
        for experience in self.experiences:
            tensor_dicts, metrics_delta = experience.to_tensor_dicts_and_metrics()
            all_tensor_dicts.append(tensor_dicts)
            for k, v in metrics_delta.items():
                metrics[k].extend(v)
        return all_tensor_dicts, metrics


class RLDataset(BaseDataset):

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Dict[str, Any]]]:

        ex = self.dataset[idx]

        if "prompt" in ex.keys():
            state_text = ex["prompt"]
        elif "messages" in ex.keys():
            state_text = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            state_text = None

        extra_info = ex.get("extra_info", {})
        return ExperienceGroup(
            self.config.experience,
            self.tokenizer,
            state_text,
            extra_info
        )