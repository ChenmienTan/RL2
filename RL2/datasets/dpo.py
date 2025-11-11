from typing import Tuple, Dict
import torch
from RL2.datasets import RMDataset


class DPODataset(RMDataset):
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor]]:

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            chosen = self._tokenize_messages(
                ex[self.config.chosen_key]
            )
            rejected = self._tokenize_messages(
                ex[self.config.rejected_key]
            )
            assert len(chosen) == len(rejected) == 1
            chosen, rejected = chosen[0], rejected[0]
        else:
            chosen = self._tokenize_prompt_response(
                ex[self.config.prompt_key],
                ex[self.config.chosen_key]
            )
            rejected = self._tokenize_prompt_response(
                ex[self.config.prompt_key],
                ex[self.config.rejected_key]
            )
        return chosen, rejected