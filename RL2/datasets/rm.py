from typing import Tuple, Dict, List
import torch
from RL2.datasets import BaseDataset, pack_tensor_dicts


class RMDataset(BaseDataset):

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            chosen = self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"], rm=True
            )
            rejected = self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"], rm=True
            )
        else:
            chosen_messages = ex["messages"] + [
                {"role": "assistant", "content": ex["chosen"]}
            ]
            rejected_messages = ex["messages"] + [
                {"role": "assistant", "content": ex["rejected"]}
            ]
            chosen = self.tokenize_messages(
                chosen_messages, rm=True
            )
            rejected = self.tokenize_messages(
                rejected_messages, rm=True
            )
        return chosen, rejected
    
    def collate_fn(
        self, all_tensor_dicts: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        tensor_dicts: List[Dict[str, torch.Tensor]] = sum(
            [list(tds) for tds in all_tensor_dicts], []
        )
        return pack_tensor_dicts(tensor_dicts)