import logging
from math_verify import parse, verify

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

async def step(state, action, answer):

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