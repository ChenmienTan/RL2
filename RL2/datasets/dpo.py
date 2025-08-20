from RL2.datasets import RMDataset, tokenize_messages


class DPODataset(RMDataset):
    
    def tokenize_messages_completion(self, messages, completion):
        return tokenize_messages(
            self.tokenizer,
            messages + [{"role": "assistant", "content": completion}],
            self.config.apply_chat_template,
            self.config.max_length
        )