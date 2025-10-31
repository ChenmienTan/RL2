import os
import datasets
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader

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
        "eos_mask": torch.LongTensor((len(states) - 1) * [0] + [1]),
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

def pack_tensor_dicts(tensor_dicts):
    return {
        k: pad_sequence(
            [tensor_dict[k] for tensor_dict in tensor_dicts], True
        )
        for k in tensor_dicts[0].keys()
    }


class BaseDataset(Dataset):
    
    def __init__(self, config, tokenizer):

        self.config = config
        self.tokenizer = tokenizer
        
        # TODO: support concatnating multiple datasets
        if not config.path:
            self.dataset = [{} for _ in range(42)] # for Gym like environments
        else:
            if "@" in config.path:
                split, path = config.path.split("@")
            else:
                split, path = "train", config.path
            ext = os.path.splitext(path)[-1].strip(".")
            if ext in ["json", "jsonl", "csv", "parquet", "arrow"]:
                if ext == "jsonl":
                    ext = "json"
                self.dataset = datasets.load_dataset(ext, data_files=path, split=split)
            else:
                self.dataset = datasets.load_dataset(path, split=split)

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
        actions = len(prompt) * [0] + response
        action_mask = len(prompt) * [0] + len(response) * [1]
        
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


class StatefulCycleDataLoader(StatefulDataLoader):

    def __call__(self, batch_size=None):
        
        if not hasattr(self, "iterator"):
            self.iterator = iter(self)

        data_list = []
        for _ in range(batch_size or len(self)):
            try:
                data = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self)
                data = next(self.iterator)
            data_list.append(data)
        return data_list


def get_dataloader(dataset, batch_size=None):

    kwargs = {
        "dataset": dataset,
        "shuffle": True,
        "drop_last": True
    }
    if batch_size is None:
        return StatefulCycleDataLoader(
            batch_size=1,
            **kwargs
        )
    else:
        return StatefulDataLoader(
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **kwargs
        )