from typing import Dict, Any
from omegaconf import DictConfig
from functools import partial
from math_verify import parse, verify
import logging
from RL2.datasets import Sample, base_generate

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

async def env_step(config: DictConfig, sample: Sample) -> Dict[str, Any]:

    reward = float(
        verify(
            parse(sample.sample["answer"]),
            parse(sample.action_text)
        )
    )
    return {
        "next_state": None,
        "done": True,
        "reward": reward
    }

generate = partial(base_generate, env_step_fn=env_step)