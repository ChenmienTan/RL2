from typing import List, Optional, Dict, Sequence, Any, Tuple
from omegaconf import DictConfig
import os
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

def get_tensor_dict(
    states: List[int],
    actions: List[int],
    action_mask: List[int],
    max_length: Optional[int] = None,
    rm: bool = False
) -> Dict[str, torch.Tensor]:

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
        ) # rewards of non-terminal tokens are zeros
    else:
        tensor_dict["actions"] = torch.LongTensor(actions)
        tensor_dict["action_mask"] = torch.LongTensor(action_mask)

    return tensor_dict

def pack_tensor_dicts(
    tensor_dicts: Sequence[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    return {
        k: pad_sequence(
            [td[k] for td in tensor_dicts], True
        )
        for k in tensor_dicts[0].keys()
    }


class BaseDataset(Dataset):
    
    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        dataset: datasets.Dataset
    ):

        self.config = config
        self.tokenizer = tokenizer

    def _tokenize_prompt_response(
        self, prompt: str, response: str, rm: bool = False
    ) -> Dict[str, torch.Tensor]:
        
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

    def _tokenize_messages(
        self, messages: List[Dict[str, Any]], rm: bool = False
    ) -> List[Dict[str, torch.Tensor]]:

        prev_text, states, actions, action_mask = "", [], [], []
        tensor_dicts = []
        for turn in range(len(messages)):
            
            is_this_turn_assistant = messages[turn]["role"] == "assistant"
            is_next_turn_assistant = turn + 1 < len(messages) and messages[turn + 1]["role"] == "assistant"

            if not is_this_turn_assistant and not is_next_turn_assistant:
                continue

            text = self.tokenizer.apply_chat_template(
                messages[:turn + 1],
                add_generation_prompt=is_next_turn_assistant,
                tokenize=False
            )

            if text.startswith(prev_text):
        
                state = self.tokenizer.encode(
                    text[len(prev_text):], add_special_tokens=False
                )
                # This is NOT equivalent to 
                #     next_states = apply_chat_template(..., tokenize=True)
                #     state = next_states[len(states):]
                states.extend(state)
                actions.extend(
                    state if is_this_turn_assistant
                    else len(state) * [0]
                )
                action_mask.extend(
                    len(state) * [is_this_turn_assistant]
                )
            
            else:
                assert is_next_turn_assistant

                tensor_dicts.append(
                    get_tensor_dict(
                        states, actions, action_mask, self.config.max_length, rm
                    )
                )
                states = self.tokenizer.encode(
                    text, add_special_tokens=False
                )
                actions = len(states) * [0]
                action_mask = len(states) * [0]

            prev_text = text

        tensor_dicts.append(
            get_tensor_dict(
                states, actions, action_mask, self.config.max_length, rm
            )
        )

        return tensor_dicts

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(
    dataset_cls: BaseDataset,
    config: DictConfig,
    tokenizer: AutoTokenizer,
    batch_size: int = None
) -> Tuple[StatefulDataLoader, StatefulDataLoader]:

    def _load_dataset(path: str):

        # TODO: support concatnating multiple datasets
        if "@" in path:
            split, path = path.split("@")
        else:
            split, path = "train", path

        ext = os.path.splitext(path)[-1].strip(".")
        if ext in ["json", "jsonl", "csv", "parquet", "arrow"]:
            if ext == "jsonl":
                ext = "json"
            return datasets.load_dataset(ext, data_files=path, split=split)
        else:
            return datasets.load_dataset(path, split=split)

    def _get_dataloader(dataset: BaseDataset, batch_size: int):
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=dataset.collate_fn
        )

    train_dataset = _load_dataset(config.train.path)
    if config.test.path:
        test_dataset = _load_dataset(config.test.path)
    else:
        total_size = len(train_dataset)
        indices = np.arange(total_size)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_point = int(0.9 * total_size)
        train_indices, test_indices = indices[:split_point], indices[split_point:]
        test_dataset = train_dataset.select(test_indices)
        train_dataset = train_dataset.select(train_indices)

    train_dataset = dataset_cls(config.train, tokenizer, train_dataset)
    test_dataset = dataset_cls(config.test, tokenizer, test_dataset)

    train_dataloader = _get_dataloader(
        train_dataset, batch_size or config.train.batch_size
    )
    test_dataloader = _get_dataloader(
        test_dataset, batch_size or len(test_dataset)
    )
    return train_dataloader, test_dataloader