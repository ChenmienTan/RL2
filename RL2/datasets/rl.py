import copy
from RL2.datasets.base import BaseDataset, load_dataset

class RLDataset(BaseDataset):
    
    def __init__(self, config, tokenizer):
        self.config = config
        # Handle None data_path for GEM environments
        if config.path:
            self.dataset = load_dataset(config.path)
        else:
            self.dataset = []  # Empty dataset for GEM environments
        self.tokenizer = tokenizer

    def __len__(self):
        # Return 1 for empty datasets to maintain training loop
        return len(self.dataset) if self.dataset else 1

    def __getitem__(self, idx):
        # Return empty data for GEM environments
        if not self.dataset:
            return {"messages": [], "answer": ""}
        
        ex = self.dataset[idx]
        
        if "prompt" in ex.keys():
            prompt = ex["prompt"]
        else:
            prompt = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
            
        answer = ex["answer"]

        return {
            "prompt": prompt,
            "answer": answer
        }

    def collate_fn(self, batch):
        # Handle empty batch for GEM environments
        if not batch or (len(batch) == 1 and not batch[0].get("messages")):
            return []
        
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]