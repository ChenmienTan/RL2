from typing import Dict, Any
import re
from functools import partial
from RL2.datasets import Sample, base_generate

async def env_step(sample: Sample) -> Dict[str, Any]:

    env_response = {
        "next_state": None,
        "done": True,
        "reward": 0.0
    }

    match = re.search(r"<answer>(.*?)</answer>", sample.action_text)
    if match is None:
        return env_response
    equation = match.group(1).strip()
    env_response["reward"] = 0.1
    
    try:
        # maybe the number cannot be converted to integer
        numbers = [int(n) for n in re.findall(r"\d+", equation)]
        assert sorted(numbers) == sorted(sample.sample["numbers"])
    except:
        return env_response
        
    try:
        assert re.match(r"^[\d+\-*/().\s]+$", equation)
        # maybe the equation is illegal
        result = eval(equation, {"__builtins__": None}, {})
        assert abs(result - sample.sample["target"]) < 1e-5
        env_response["reward"] = 1.0
    except:
        pass
    return env_response

generate = partial(base_generate, env_step_fn=env_step)