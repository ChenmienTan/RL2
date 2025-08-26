from RL2.datasets import RMDataset


class DPODataset(RMDataset):
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            return self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["chosen"]}
                ]
            ), self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["rejected"]}
                ]
            )
        else:
            return self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"]
            ), self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"]
            )