import re
import string
import aiohttp

async def step(texts):

    match = re.search(
        r"<(search|answer)>(.*?)</\1>", texts[-1], re.DOTALL
    )
    if match is None:
        return "\nMy previous action is invalid. \
            If I want to search, I should put the query between <search> and </search>. \
            If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
    elif match.group(1) == "answer":
        return ""
    
    query = match.group(2)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/search",
            json={"query": query}
        ) as response:
            result = await response.json()
    return f"\n\n<information>{result.strip()}</information>\n\n"

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

def reward_fn(texts, answer):

    preds = re.findall(
        r"<answer>(.*?)</answer>", texts[-1], re.DOTALL
    )
    if len(preds) == 0:
        return False
    pred = normalize_answer(preds[-1].strip())

    if isinstance(answer, str):
        answer = [answer]
    answer = [normalize_answer(a) for a in answer]
  
    return pred in answer
