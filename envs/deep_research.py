import json
import asyncio
from dotenv import load_dotenv
from envs.tools import search, visit
from RL2.utils.communication import async_request

load_dotenv("envs/tools/.env")
with open("envs/tools/descriptions.jsonl") as f:
    TOOLS = [json.loads(line) for line in f]
TOOL_PARSER = "qwen25"

async def step(state, action, extra_info):

    env_response = {
        "next_state": None,
        "reward": 0.0,
        "score": 0.0,
        "done": False,
        "extra_info": extra_info
    }
    
    response = await async_request(
        f"{extra_info['router_url']}/parse_function_call",
        payload={
            "text": action,
            "tools": TOOLS,
            "tool_parser": TOOL_PARSER
        }
    )

    if len(response["calls"]) == 0:
        # TODO: do evaluation
        env_response["done"] = True
        return env_response
    
    tool_results = []
    for call in response["calls"]:
        func_name = call["name"]
        arguments = json.loads(call["parameters"])

        if func_name == "search":
            result = await asyncio.to_thread(
                search,
                **arguments
            )
        elif func_name == "visit":
            result = await visit(
                summarizer_url=extra_info['router_url'],
                **arguments
            )
        else:
            NotImplementedError

        tool_results.append(result)
    tool_results = "<|im_end|>\n<|im_start|>user\n".join(tool_results)
    
    env_response["next_state"] = state + action + "\n<|im_start|>user\n" + tool_results + "<|im_end|>\n<|im_start|>assistant\n"
    return env_response