from typing import Dict, Any, List, Callable, Tuple
from omegaconf import DictConfig
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from copy import deepcopy
from collections import defaultdict
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from RL2.datasets import get_tensor_dict, BaseDataset


@dataclass
class Sample:

    class Status(Enum):

        RUNNING = "running"
        ABORTED = "aborted"
        DONE = "done"

    # for initialization
    sample: Dict[str, Any] = field(default_factory=dict)

    # for environment interaction
    state_text: str = ""
    action_text: str = ""

    # for training
    state_dict: Dict[str, List[int | float]] = field(default_factory=dict)
    state_dicts: List[Dict[str, List[int | float]]] = field(default_factory=list)

    # for logging
    turn: int = 0
    metrics: Dict[str, List[float | int | bool]] = field(default_factory=defaultdict(list))

    # for partial rollout
    status: Status = Status.RUNNING
    previous_action_text: str = ""
    previous_response_length: int = 0


class SampleGroup:

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        sample: Dict[str, Any]
    ):

        self.config = config
        self.tokenizer = tokenizer
        self.samples = [
            Sample(sample=deepcopy(sample))
            for _ in range(config.responses_per_prompt)
        ]

    async def generate(self, generate_fn: Callable) -> "SampleGroup":
        await asyncio.gather(*(
            generate_fn(self.config, self.tokenizer, sample)
            for sample in self.samples
        ))
        return self
    
    def to_all_tensor_dicts_and_metrics(self) -> Tuple[List[List[Dict[str, torch.Tensor]]], Dict[str, List[float | int | bool]]]:
        
        all_tensor_dicts, metrics = [], defaultdict(list)
        for sample in self.samples:
            tensor_dicts = []
            for state_dict in sample.state_dicts:
                tensor_dict = get_tensor_dict(
                    state_dict["states"],
                    state_dict["actions"],
                    state_dict["action_mask"],
                )
                tensor_dict["llm_logps"] = torch.FloatTensor(
                    tensor_dict["logps"][1:]
                )
                tensor_dict["rewards"] = torch.FloatTensor(
                    tensor_dict["rewards"][1:]
                )
                tensor_dicts.append(tensor_dict)
            all_tensor_dicts.append(tensor_dicts)
            for k, v in sample.metrics.items():
                metrics[k].extend(v)
        return all_tensor_dicts, metrics


class RLDataset(BaseDataset):

    def __getitem__(self, idx: int) -> SampleGroup:

        sample = self.dataset[idx]
        return SampleGroup(self.config, self.tokenizer, sample)


class StatefulCycleDataLoader(StatefulDataLoader):

    def __call__(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Fetch a variable number of data.
        """
        
        if not hasattr(self, "iterator"):
            self.iterator = iter(self)

        data_list = []
        for _ in range(batch_size):
            try:
                data = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self)
                data = next(self.iterator)
            data_list.append(data)
        return data_list