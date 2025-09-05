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

async def step(state, action, answer):

    match = re.search(
        r"<(search|answer)>(.*?)</\1>", action, re.DOTALL
    )
    if match is None:
        next_state = state + action + "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        return next_state, False, False
    elif match.group(1) == "search":
        query = match.group(2).strip()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/search",
                json={"query": query}
            ) as response:
                passage = (await response.json())["passage"].strip()
        next_state = state + action + f"\n\n<information>{passage}</information>\n\n"
        return next_state, False, False
    else:
        preds = re.findall(
            r"<answer>(.*?)</answer>", action, re.DOTALL
        )
        pred = normalize_answer(preds[-1].strip())

        if isinstance(answer, str):
            answer = [answer]
        answer = [normalize_answer(a) for a in answer]
    
        reward = pred in answer
        return None, reward, True