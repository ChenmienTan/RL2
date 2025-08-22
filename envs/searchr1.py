import re
import string
import requests

def interact(messages):

    match = re.search(
        r"<search>\s*(.*?)\s*</search>", messages[-1]["content"], re.DOTALL
    )
    if match is None:
        return []
    
    query = match.group(1)
    result = requests.post(
        "http://localhost:8000/search", json={
            "query": query
        }
    ).json()

    return [
        {"role": "tool", "content": result}
    ]

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

def reward_fn(messages, answer):

    preds = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", messages[-1]["content"], re.DOTALL
    )
    if len(preds) == 0:
        return False
    pred = normalize_answer(preds[-1])

    if isinstance(answer, str):
        answer = [answer]
    answer = [normalize_answer(a) for a in answer]
  
    return pred in answer
