from RL2.dataset import BaseDataset, tokenize_messages

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):
        # TODO (P0): perhaps do not apply chat template
        messages = self.dataset[idx]["messages"]
        ex = tokenize_messages(self.tokenizer, messages)
        return {k: v[:self.max_length] for k, v in ex.items()}
    
    def collate_fn(self, batch):
        return list(batch)