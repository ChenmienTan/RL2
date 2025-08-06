import os
import datasets
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

# TODO (P1): support concatnating multiple datasets
def load_dataset(data_path):

    if "@" in data_path:
        split, data_path = data_path.split("@")
    else:
        split = "train"
    
    ext = os.path.splitext(data_path)[-1].strip(".")
    if ext in ["json", "jsonl", "csv", "parquet", "arrow"]:
        if ext == "jsonl":
            ext = "json"
        return datasets.load_dataset(ext, data_files=data_path, split=split)
    else:
        return datasets.load_dataset(data_path, split=split)

def get_dataloader(dataset, batch_size):
    return StatefulDataLoader(
        dataset,
        batch_size,
        shuffle=True,
        drop_last=True if batch_size else False,
        collate_fn=dataset.collate_fn
    )

def tokenize_messages(
    tokenizer,
    messages,
    apply_chat_template=True
):

    states, actions, action_mask = [], [], []
    for idx, message in enumerate(messages):

        if message["role"] == "assistant":
            state = tokenizer.encode(
                message["content"], add_special_tokens=False
            )
            actions.extend(state)
            action_mask.extend(len(state) * [1])
        else:
            if apply_chat_template:
                next_states = tokenizer.apply_chat_template(
                    messages[:idx + 1],
                    add_generation_prompt=idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant"
                )
                assert next_states[:len(states)] == states, \
                    "Your tokenizer should be increasing, i.e., adding a new message should not change the tokenization of previous messages. For example, if you use Qwen3 in multi-turn cases, previous thinking may be eliminated. In this case, you may set `tokenizer_name=Chenmien/Qwen3-Increasing-Tokenizer`."
                state = next_states[len(states):]
            else:
                state = tokenizer.encode(
                    message["content"], add_special_tokens=False
                )
            actions.extend(len(state) * [0])
            action_mask.extend(len(state) * [0])

        states.extend(state)

    return {
        "states": torch.LongTensor(states[:-1]),
        "actions": torch.LongTensor(actions[1:]),
        "action_mask": torch.LongTensor(action_mask[1:]),
        "position_ids": torch.arange(len(states) - 1)
    }

class BaseDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length
    ):

        self.dataset = load_dataset(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)