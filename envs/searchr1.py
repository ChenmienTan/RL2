import re
import string
import aiohttp

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

async def step(state, action, extra_info):

    match = re.search(
        r"<(search|answer)>(.*?)</\1>", action, re.DOTALL
    )
    env_response = {
        "next_state": None,
        "reward": 0.0,
        "score": 0.0,
        "done": False,
        "extra_info": extra_info
    }
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
        pred = normalize_answer(match.group(2).strip())

        answer = extra_info["answer"]
        if isinstance(answer, str):
            answer = [answer]
        answer = [normalize_answer(a) for a in answer]

        reward = float(pred in answer)
        env_response["reward"] = reward
        env_response["score"] = reward
        env_response["done"] = True

    return env_response