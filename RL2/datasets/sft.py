from RL2.datasets import BaseDataset, tokenize_messages

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):

        messages = self.dataset[idx]["messages"]
        return tokenize_messages(
            self.tokenizer,
            messages,
            self.config.apply_chat_template,
            self.config.max_length
        )
    
    def collate_fn(self, batch):
        return list(batch)