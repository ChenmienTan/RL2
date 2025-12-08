import logging
from math_verify import parse, verify
from functools import partial
from RL2.datasets import base_generate

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

async def env_step(state, action, answer):

    reward = float(
        verify(
            parse(answer),
            parse(action)
        )
    )
    return {
        "next_state": None,
        "done": True,
        "reward": reward
    }

generate = partial(base_generate, env_step_fn=env_step)