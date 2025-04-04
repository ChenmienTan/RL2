import json
from torch.utils.data import Dataset

class RLDataset(Dataset):

    def __init__(
        self,
        args,
        data_path,
        tokenizer,
        rollout_per_prompt
    ):

        self.args = args
        with open(data_path, "r") as f:
            self.dataset = json.load(f)
        self.tokenizer = tokenizer
        self.rollout_per_prompt = rollout_per_prompt
        self.tokenizer.trauncation_side = "left"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        conversation = [{"role": "user", "content": ex["prompt"]}]
        answer = ex["answer"]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        prompt_token_id = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            max_length=self.args.max_prompt_length,
            truncation=True
        )
        return {
            "prompts": prompt,
            "answers": answer,
            "prompt_token_ids": prompt_token_id
        }

    def collate_fn(self, batch):
        return {
            k: sum([self.rollout_per_prompt * [ex[k]] for ex in batch], [])
            for k in batch[0].keys()
        }