from RL2.datasets.base import BaseDataset


class RLDataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        data = {}

        if "prompt" in ex.keys():
            data["prompt"] = ex["prompt"]
        elif "messages" in ex.keys():
            data["prompt"] = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )

        data["extra_info"] = ex.get("extra_info", {})
        return data
    
    def collate_fn(self, batch):
        return batch[0]