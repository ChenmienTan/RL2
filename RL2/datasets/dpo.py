from typing import Tuple, Dict
import torch
from RL2.datasets import RMDataset


class DPODataset(RMDataset):
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            chosen = self._tokenize_prompt_response(
                ex["prompt"], ex["chosen"]
            )
            rejected = self._tokenize_prompt_response(
                ex["prompt"], ex["rejected"]
            )
        else:
            chosen = self._tokenize_messages(ex["chosen"])
            rejected = self._tokenize_messages(ex["rejected"])
            assert len(chosen) == len(rejected) == 1
            chosen, rejected = chosen[0], rejected[0]
        return chosen, rejected