from typing import List, Dict, Any
from omegaconf import OmegaConf, DictConfig
import os
import re
import json
import asyncio
from dotenv import load_dotenv
from transformers import AutoTokenizer
from RL2.datasets import (
    Sample,
    initialize_state_dict,
    add_llm_response
)
from RL2.utils.communication import async_request

load_dotenv("envs/.env")

SYSTEM_PROMPT = "You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."

SUMMARY_TEMPLATE = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
"""

# https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
SCORING_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "minItems": 1,
                        "description": "The list of search queries."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting webpage(s)."
                    }
                },
                "required": ["url", "goal"]
            }
        }
    }
]

TOOL_CALL_PARSER = "qwen25"

ROUTER_URL = None
MAX_TOKENS = None
SERVER_URL = None


async def _call_llm(prompt: str) -> str:

    response = await async_request(
        ROUTER_URL,
        "v1/chat/completions",
        json={
            "model": "",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )
    return response["choices"][0]["message"]["content"]


async def search(query: str | List[str]) -> str:

    if isinstance(query, list):
        results = await asyncio.gather(*[search(q) for q in query])
        return "\n=======\n".join(results)

    # TODO: condition on query language
    params = {
        "q": query,
        "api_key": os.environ["SERP_API_KEY"],
        "location": "United States",
        "hl": "en",
        "gl": "us",
    }
    result = await async_request(
        "https://serpapi.com",
        "search.json",
        "GET",
        params=params
    )
            
    if "organic_results" not in result:
        return f"No results found for '{query}'. Try with a more general query."
        
    results = []
    for idx, page in enumerate(result["organic_results"]):
        
        date = ""
        if "date" in page:
            date = f"\nDate published: {page['date']}"

        source = ""
        if "source" in page:
            source = f"\nSource: {page['source']}"

        snippet = ""
        if "snippet" in page:
            snippet = f"\n{page['snippet']}"

        results.append(f"{idx + 1}. [{page['title']}]({page['link']}){date}{source}\n{snippet}")

    return f"A Google search for '{query}' found {len(results)} results:\n\n## Web Results\n" + "\n\n".join(results)


async def visit(url: str | List[str], goal: str) -> str:

    if isinstance(url, list):
        results = await asyncio.gather(*[visit(u, goal) for u in url])
        return "\n=======\n".join(results)

    headers = {
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
        "X-Token-Budget": "256000"
        # TODO: truncate
    }
    webpage_content = await async_request(
        "https://r.jina.ai",
        url,
        "GET",
        headers=headers
    )

    response = await _call_llm(
        SUMMARY_TEMPLATE.format(
            webpage_content=webpage_content,
            goal=goal
        )
    )
    return response.split("</think>")[-1]


async def env_step(sample: Sample):

    env_response = {
        "next_state": None,
        "done": False,
        "reward": 0.0
    }

    response = await async_request(
        SERVER_URL,
        "parse_function_call",
        json={
            "text": sample.action_text,
            "tools": TOOLS,
            "tool_call_parser": TOOL_CALL_PARSER
        }
    )
    tool_calls: List[Dict[str, str]] = response["calls"]

    if len(tool_calls) == 0:

        response = await _call_llm(
            SCORING_TEMPLATE.format(
                question=sample.sample["prompt"],
                response=sample.action_text,
                correct_answer=sample.sample["answer"]
            )
        )
        match = re.search(r"correct: (yes|no)", response)

        env_response["done"] = True
        env_response["reward"] = float(
            match is not None and match.group(1) == "yes"
        )

        return env_response
    
    async def _call_tool(name: str, arguments: Dict[str, Any]) -> str:

        try:
            match name:
                case "search":
                    return await search(**arguments)
                case "visit":
                    return await visit(**arguments)
        except Exception as e:
            return str(e)

    tool_results: List[str] = await asyncio.gather(*[
        _call_tool(
            tool_call["name"],
            json.loads(tool_call["parameters"])
        )
        for tool_call in tool_calls
    ])

    next_state: str = "<|im_end|>\n<|im_start|>user\n".join(tool_results)
    next_state: str = sample.state_text + sample.action_text + "\n<|im_start|>user\n" + next_state + "<|im_end|>\n<|im_start|>assistant\n"
    env_response["next_state"] = next_state

    return env_response


def add_env_response(
    tokenizer: AutoTokenizer,
    sample: Sample,
    response: Dict[str, Any]
):
    def _process_completed_sample():

        sample.status = Sample.Status.DONE
        sample.metrics["turns"].append(sample.turn)
        sample.metrics["rewards"].append(
            sum([state_dict["rewards"][-1] for state_dict in sample.state_dicts])
        )

    sample.state_dict["rewards"][-1] = response["reward"]

    if response["done"]:

        sample.state_dicts.append(sample.state_dict)
        _process_completed_sample()
        return

    if response["next_state"].startswith(sample.state_text + sample.action_text):
        state_dict_delta = initialize_state_dict(
            tokenizer,
            response["next_state"][len(sample.state_text + sample.action_text):]
        )
        if len(sample.state_dict["states"]) + len(state_dict_delta["states"]) >= MAX_TOKENS - 1:
            sample.state_dicts.append(sample.state_dict)
            _process_completed_sample()
            return
        for k, v in state_dict_delta.items():
            sample.state_dict[k].extend(v)
    else:
        # If the previous state is not a prefix of the next state, the trajectory will 
        # contain multiple sequences
        sample.state_dicts.append(sample.state_dict)
        sample.state_dict = initialize_state_dict(
            tokenizer, response["next_state"]
        )
        if len(sample.state_dict["states"]) >= MAX_TOKENS - 1:
            _process_completed_sample()
            return
    sample.state_text = response["next_state"]


async def generate(
    config: DictConfig,
    tokenizer: AutoTokenizer,
    router_url: str,
    sample: Sample
):
    sampling_params = OmegaConf.to_container(config.sampling_params)

    # prepare global variables
    global ROUTER_URL, SERVER_URL, MAX_TOKENS
    if ROUTER_URL != router_url:
        ROUTER_URL = router_url
    if SERVER_URL is None:
        response = await async_request(
            ROUTER_URL, "list_workers", "GET"
        )
        SERVER_URL = response["urls"][0]
    if MAX_TOKENS is None:
        response = await async_request(
            SERVER_URL, "v1/models", "GET"
        )
        MAX_TOKENS = response["data"][0]["max_model_len"]

    match sample.status:

        case Sample.Status.RUNNING:

            sample.state_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": sample.sample["prompt"]}
                ],
                tools=TOOLS,
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

        # request exceeding max tokens allowed is illegal
        max_new_tokens = min(
            sampling_params["max_new_tokens"] - sample.previous_response_length,
            MAX_TOKENS - len(sample.state_dict["states"]) - 1
        )
        response = await async_request(
            ROUTER_URL,
            "generate",
            json={
                "input_ids": sample.state_dict["states"],
                "sampling_params": {
                    **sampling_params,
                    "max_new_tokens": max_new_tokens,
                    "no_stop_trim": True
                },
                "return_logprob": True
            }
        )
        add_llm_response(sample, response)
        if sample.status == Sample.Status.ABORTED:
            return
        
        response = await env_step(sample)
        add_env_response(tokenizer, sample, response)
        if sample.status == Sample.Status.DONE:
            return