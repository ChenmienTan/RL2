from typing import Tuple, Dict, List
import json
import uuid
from torch.utils.data import Dataset

class RLDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length,
        rollout_per_prompt
    ):

        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.trauncation_side = "left"
        self.rollout_per_prompt = rollout_per_prompt
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        message = ex["message"]
        answer = ex["answer"]

        prompt = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_id = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        return {
            "prompt": prompt,
            "answer": answer,
            "prompt_id": prompt_id
        }

    def collate_fn(self, data_tuple: Tuple[Dict]) -> List[Dict]:
        return sum([self.rollout_per_prompt * [ex] for ex in data_tuple], [])