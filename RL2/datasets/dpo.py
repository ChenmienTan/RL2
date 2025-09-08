from RL2.datasets import RMDataset


class DPODataset(RMDataset):
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            chosen = self.tokenize_prompt_response(
                ex["prompt"], ex["chosen"]
            )
            rejected = self.tokenize_prompt_response(
                ex["prompt"], ex["rejected"]
            )
        else:
            chosen = self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["chosen"]}
                ]
            )
            rejected = self.tokenize_messages(
                ex["messages"] + [
                    {"role": "assistant", "content": ex["rejected"]}
                ]
            )
        return chosen, rejected