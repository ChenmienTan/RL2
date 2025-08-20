from RL2.datasets import BaseDataset, tokenize_messages


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
        return tokenize_messages(
            self.tokenizer,
            messages + [{"role": "assistant", "content": completion}],
            self.config.apply_chat_template,
            self.config.max_length,
            False
        )
    
    def collate_fn(self, batch):
        return sum([list(ex) for ex in batch], [])