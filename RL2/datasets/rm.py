from RL2.datasets import BaseDataset, pack_tensor_dicts


class RMDataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            chosen = self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"], rm=True
            )
            rejected = self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"], rm=True
            )
        else:
            chosen = self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["chosen"]}
                ], rm=True
            )
            rejected = self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["rejected"]}
                ], rm=True
            )
        return chosen, rejected
    
    def collate_fn(self, batch):
        return pack_tensor_dicts(
            sum([list(b) for b in batch], [])
        )