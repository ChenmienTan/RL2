from typing import Dict, Union, Any
from RL2.datasets.base import BaseDataset


class RLDataset(BaseDataset):

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Dict[str, Any]]]:

        ex = self.dataset[idx]
        data = {}

        if "prompt" in ex.keys():
            data["prompt"] = ex["prompt"]
        elif "messages" in ex.keys():
            data["prompt"] = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )

        data["extra_info"] = ex.get("extra_info", {})
        return data