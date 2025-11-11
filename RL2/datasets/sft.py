from typing import Tuple, Dict, List
import torch
from RL2.datasets import BaseDataset, pack_tensor_dicts


class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            tensor_dicts = self._tokenize_messages(
                ex[self.config.messages_key]
            )
        else:
            tensor_dicts = [
                self._tokenize_prompt_response(
                    ex[self.config.prompt_key],
                    ex[self.config.response_key]
                )
            ]
        return tensor_dicts
        
    def collate_fn(
        self, all_tensor_dicts: Tuple[List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        tensor_dicts = [td for tds in all_tensor_dicts for td in tds]
        return pack_tensor_dicts(tensor_dicts)