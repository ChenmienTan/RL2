from RL2.datasets import BaseDataset


class RMDataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if self.config.apply_chat_template:
            return self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["chosen"]}
                ], rm=True
            ), self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["rejected"]}
                ], rm=True
            )
        else:
            return self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"], rm=True
            ), self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"], rm=True
            )
    
    def collate_fn(self, batch):
        return sum([list(ex) for ex in batch], [])