from RL2.datasets import RMDataset


class DPODataset(RMDataset):
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            return self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"]
            ), self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"]
            )
        else:
            return self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["chosen"]}
                ]
            ), self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["rejected"]}
                ]
            )