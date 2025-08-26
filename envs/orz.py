import logging
from math_verify import parse, verify

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

def reward_fn(prompt, answer):
    return verify(parse(answer), parse(prompt))