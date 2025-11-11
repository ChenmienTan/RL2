from typing import Tuple, Dict, List
import torch
from RL2.datasets import BaseDataset, pack_tensor_dicts


class RMDataset(BaseDataset):

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            chosen = self._tokenize_messages(
                ex[self.config.chosen_key], rm=True
            )
            rejected = self._tokenize_messages(
                ex[self.config.rejected_key], rm=True
            )
            assert len(chosen) == len(rejected) == 1
            chosen, rejected = chosen[0], rejected[0]
        else:
            chosen = self._tokenize_prompt_response(
                ex[self.config.prompt_key],
                ex[self.config.chosen_key],
                rm=True
            )
            rejected = self._tokenize_prompt_response(
                ex[self.config.prompt_key],
                ex[self.config.rejected_key], 
                rm=True
            )
        return chosen, rejected
    
    def collate_fn(
        self, all_tensor_dicts: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        tensor_dicts: List[Dict[str, torch.Tensor]] = [
            td for tds in all_tensor_dicts for td in tds
        ]
        return pack_tensor_dicts(tensor_dicts)