from RL2.datasets import BaseDataset

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            return self.tokenize_messages(ex["messages"])
        else:
            return self.tokenize_prompt_response(
                ex["prompt"], ex["response"]
            )
    
    def collate_fn(self, batch):
        return list(batch)