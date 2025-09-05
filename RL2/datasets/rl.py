import copy
from RL2.datasets.base import BaseDataset


class RLDataset(BaseDataset):

    def __init__(self, config, tokenizer, is_custom_rollout=False):
        if not is_custom_rollout:
            super().__init__(config, tokenizer)
        else:
            self.dataset = [{'prompt': '', 'answer': ''}] * config.get('prompts_per_rollout', 1)
            self.tokenizer = tokenizer
            self.config = config

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        
        if "prompt" in ex.keys():
            prompt = ex["prompt"]
        else:
            prompt = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
            
        answer = ex["answer"]

        return {
            "prompt": prompt,
            "answer": answer
        }

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]