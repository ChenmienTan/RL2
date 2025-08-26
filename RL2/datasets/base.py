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

def get_tensor_dict(
    states,
    actions,
    action_mask,
    max_length=None,
    rm=False
):

    if not rm:
        states = states[:-1]
        actions = actions[1:]
        action_mask = action_mask[1:]
    if max_length is not None:
        states = states[:max_length]
        actions = actions[:max_length]
        action_mask = action_mask[:max_length]

    tensor_dict = {
        "states": torch.LongTensor(states),
        "sos_mask": torch.LongTensor([1] + (len(states) - 1) * [0]),
        "position_ids": torch.arange(len(states))
    }
    if rm:
        tensor_dict["action_mask"] = torch.LongTensor(
            (len(states) - 1) * [0] + [1]
        )
    else:
        tensor_dict["actions"] = torch.LongTensor(actions)
        tensor_dict["action_mask"] = torch.LongTensor(action_mask)

    return tensor_dict

class BaseDataset(Dataset):
    
    def __init__(self, config, tokenizer):

        self.config = config
        self.dataset = load_dataset(config.path)
        self.tokenizer = tokenizer

    def tokenize_prompt_response(
        self, prompt, response, rm=False
    ):
        
        prompt = self.tokenizer.encode(
            prompt, add_special_tokens=False
        )
        response = self.tokenizer.encode(
            response + self.tokenizer.eos_token,
            add_special_tokens=False
        )
        
        states = prompt + response
        actions = len(states) * [0] + response
        action_mask = len(states) * [0] + len(response) * [1]
        
        return get_tensor_dict(
            states, actions, action_mask, self.config.max_length, rm
        )

    def tokenize_messages(self, messages, rm=False):

        prev_text, states, actions, action_mask = "", [], [], []
        for turn in range(len(messages)):
            is_this_turn_assistant = messages[turn]["role"] == "assistant"
            is_next_turn_assistant = turn + 1 < len(messages) and messages[turn + 1]["role"] == "assistant"

            text = self.tokenizer.apply_chat_template(
                messages[:turn + 1],
                add_generation_prompt=is_next_turn_assistant,
                tokenize=False
            )
            assert text[:len(prev_text)] == prev_text
            state = self.tokenizer.encode(
                text[len(prev_text):], add_special_tokens=False
            )
            states.extend(state)
            actions.extend(
                state
                if is_this_turn_assistant
                else len(state) * [0]
            )
            action_mask.extend(len(state) * [is_this_turn_assistant])
            prev_text = text

        return get_tensor_dict(
            states, actions, action_mask, self.config.max_length, rm
        )

    def __len__(self):
        return len(self.dataset)