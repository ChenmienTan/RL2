from math_verify import parse, verify

def reward_fn(response, answer):
    return verify(parse(answer), parse(response))