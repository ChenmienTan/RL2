from typing import Tuple, Dict
import torch
from RL2.datasets import RMDataset


class DPODataset(RMDataset):
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            chosen = self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"]
            )
            rejected = self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"]
            )
        else:
            chosen_messages = ex["messages"] + [
                {"role": "assistant", "content": ex["chosen"]}
            ]
            rejected_messages = ex["messages"] + [
                {"role": "assistant", "content": ex["rejected"]}
            ]
            chosen = self.tokenize_messages(chosen_messages)
            rejected = self.tokenize_messages(rejected_messages)
        return chosen, rejected