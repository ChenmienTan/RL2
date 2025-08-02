import torch
from RL2.datasets import BaseDataset


class RMDataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen = self.tokenize_messages_completion(messages, chosen)
        rejected = self.tokenize_messages_completion(messages, rejected)

        return chosen, rejected

    def tokenize_messages_completion(self, messages, completion):

        states = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}]
        )[:self.max_length]
        action_mask = (len(states) - 1) * [0] + [1]

        return {
            "states": torch.LongTensor(states),
            "action_mask": torch.LongTensor(action_mask),
            "position_ids": torch.arange(len(states))
        }
    
    def collate_fn(self, batch):
        return sum([list(ex) for ex in batch], [])