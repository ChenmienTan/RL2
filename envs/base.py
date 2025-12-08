from typing import Dict, List, Any
from transformers import AutoTokenizer
from RL2.datasets import Sample

def initialize_state_dict(
    tokenizer: AutoTokenizer,
    state_text: str
) -> Dict[str, List[int | float]]:
        
    state = tokenizer.encode(state_text, add_special_tokens=False)
    return {
        "states": state,
        "actions": len(state) * [0],
        "action_mask": len(state) * [0],
        "logps": len(state) * [0.0],
        "rewards": len(state) * [0.0]
    }

def add_llm_response(sample: Sample, response: Dict[str, Any]):

    # `previous_action_text` is non-empty if aborted before
    sample.action_text = sample.previous_action_text + response["text"]

    # encode(decode(tokens)) may not be identical to tokens. Therefore, 
    # token-in-token-out is necessary to guanartee that tokens fed into 
    # training and inference engines are identical
    # https://github.com/OpenRLHF/OpenRLHF/pull/1094
    # https://github.com/THUDM/slime/pull/117
    meta_info = response["meta_info"]
    if "output_token_logprobs" in meta_info and len(meta_info["output_token_logprobs"][0]) == 3: # TODO: is this condition correct?
        logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
        sample.state_dict["states"].extend(action)
        sample.state_dict["actions"].extend(action)
        sample.state_dict["action_mask"].extend(len(action) * [1])
        sample.state_dict["logps"].extend(logp)
        sample.state_dict["rewards"].extend(len(action) * [0.0])
        # actual rewards will be overwritten when scoring

    finish_reason = meta_info["finish_reason"]["type"]
    if finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
        sample.previous_action_text = sample.action_text
        sample.previous_response_length += meta_info["completion_tokens"]
        return
        
    sample.turn += 1
    sample.metrics["response_length"].append(
        sample.previous_response_length + meta_info["completion_tokens"]
    )
    sample.metrics["length_clip_ratio"].append(finish_reason == "length")

    # reset if not aborted
    sample.previous_action_text = ""
    sample.previous_response_length = 0

def add_env_response(
    tokenizer: AutoTokenizer,
    sample: Sample,
    response: Dict[str, Any]
):
    
    if response["done"]:

        sample.status = Sample.Status.DONE
        sample.state_dict["rewards"][-1] = response["reward"]
        sample.state_dicts.append(sample.state_dict)
        sample.metrics["turns"].append(sample.turn)
        sample.metrics["rewards"].append(response["reward"])
        return

    if response["next_state"].startswith(sample.state_text + sample.action_text):
        state_dict_delta = initialize_state_dict(
            tokenizer,
            response["next_state"][len(sample.state_text + sample.action_text):]
        )
        for k, v in state_dict_delta.items():
            sample.state_dict[k].extend(v)
    else:
        sample.state_dicts.append(sample.state_dict)
        sample.state_dict = initialize_state_dict(
            tokenizer, response["next_state"]
        )
    sample.state_text = response["next_state"]