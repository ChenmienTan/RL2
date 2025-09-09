import copy
from RL2.datasets.base import BaseDataset


class RLDataset(BaseDataset):

    def __init__(self, config, tokenizer):
        if config.path:
            super().__init__(config, tokenizer)
        else:
            # When no data path provided, create dummy dataset for env-based training
            num_trajectories = config.get('prompts_per_rollout', 1)
            self.dataset = [
                {
                    'prompt': '', 
                    'extra_info': {'idx': i}
                } 
                for i in range(num_trajectories)
            ]
            self.tokenizer = tokenizer
            self.config = config

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        
        if "prompt" in ex.keys() and ex["prompt"]:
            prompt = ex["prompt"]
        elif "messages" in ex.keys():
            prompt = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = ""

        extra_info = ex.get("extra_info", {})
        if "idx" not in extra_info:
            extra_info["idx"] = idx

        return {
            "prompt": prompt,
            "extra_info": extra_info
        }

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]