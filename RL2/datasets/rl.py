import copy
from RL2.datasets.base import BaseDataset, load_dataset

class RLDataset(BaseDataset):
    
    def __init__(self, data_path, responses_per_prompt):
        # Handle None data_path for GEM environments
        if data_path:
            self.dataset = load_dataset(data_path)
        else:
            self.dataset = []  # Empty dataset for GEM environments
        self.responses_per_prompt = responses_per_prompt

    def __len__(self):
        # Return 1 for empty datasets to maintain training loop
        return len(self.dataset) if self.dataset else 1

    def __getitem__(self, idx):
        # Return empty data for GEM environments
        if not self.dataset:
            return {"messages": [], "answer": ""}
        
        ex = self.dataset[idx]
        messages = ex["messages"]
        answer = ex["answer"]

        return {
            "messages": messages,
            "answer": answer
        }

    def collate_fn(self, batch):
        # Handle empty batch for GEM environments
        if not batch or (len(batch) == 1 and not batch[0].get("messages")):
            return []
        
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]