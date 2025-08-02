import copy
from RL2.datasets.base import BaseDataset, load_dataset

class RLDataset(BaseDataset):
    
    def __init__(self, data_path, responses_per_prompt):

        self.dataset = load_dataset(data_path)
        self.responses_per_prompt = responses_per_prompt

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        answer = ex["answer"]

        return {
            "messages": messages,
            "answer": answer
        }

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]