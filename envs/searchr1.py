from typing import List, Dict, Any
from omegaconf import OmegaConf, DictConfig
import re
import string
import aiohttp
from transformers import AutoTokenizer
from RL2.datasets import Sample
from RL2.utils.communication import async_request
from .base import (
    initialize_state_dict,
    add_llm_response,
    add_env_response
)


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


async def env_step(
    state: str,
    action: str,
    answer: str | List[str]
) -> Dict[str, Any]:

    match = re.search(
        r"<(search|answer)>(.*?)</\1>", action, re.DOTALL
    )
    env_response = {"next_state": None, "done": False, "reward": 0.0}
    if match is None:
        env_response["next_state"] = state + action + "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
    elif match.group(1) == "search":
        query = match.group(2).strip()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:10000/search",
                json={"query": query}
            ) as response:
                try:
                    passage = (await response.json())["passage"].strip()
                    env_response["next_state"] = state + action + f"\n\n<information>{passage}</information>\n\n"
                except:
                    env_response["next_state"] = state + action + "\nThe query exceeded the maximum length allowed. Let me try again.\n"
    else:
        env_response["done"] = True
        pred = normalize_answer(match.group(2).strip())

        if isinstance(answer, str):
            answer = [answer]
        answer = [normalize_answer(a) for a in answer]

        reward = float(pred in answer)
        env_response["reward"] = reward

    return env_response


async def generate(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    sample: Sample
):
    sampling_params = OmegaConf.to_container(config.sampling_params)

    match sample.status:

        case Sample.Status.RUNNING:

            sample.state_text = tokenizer.apply_chat_template(
                sample.sample["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
            sample.state_dict = initialize_state_dict(
                tokenizer, sample.state_text
            )

        case Sample.Status.ABORTED:
            sample.status = Sample.Status.RUNNING

        case Sample.Status.DONE:
            return

    while True:
        
        response = await async_request(
            f"{config.router_url}/generate",
            json={
                "input_ids": sample.state_dict["states"],
                "sampling_params": {
                    **sampling_params,
                    "max_new_tokens": sampling_params["max_new_tokens"] - sample.previous_response_length
                },
                "return_logprob": True
            }
        )
        add_llm_response(sample, response)
        if sample.status == Sample.Status.ABORTED:
            return
        
        response = await env_step(
            sample.state_text,
            sample.action_text,
            sample.sample["answer"]
        )
        if sample.turn == config.max_turns:
            response["done"] = True
        add_env_response(tokenizer, sample, response)
        if sample.status == Sample.Status.DONE:
            return