import os
import json
import asyncio
from RL2.utils.communication import async_request

ERROR_TEMPLATE = """Visiting page {url} encountered error: {error}"""

SYSTEM_PROMPT = """You are a helpful assistant that follow the guidelines to extract goal-related information from webpage content.

<|The Start of Guidelines|>
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
Return your result strictly in the following JSON format
{
    "rational": "why the evidence is related to the goal",
    "evidence": "the most relevant information from the content",
    "summary": "a concise summary of the evidence"
}
<|The End of Guidelines|>"""

USER_PROMPT = """<|The Start of Webpage Content|>
{text}
<|The End of Webpage Content|>

<|The Start of Goal|>
{goal}
<|The End of Goal|>"""

SUMMARY_TEMPLATE = """The useful information in {url} for user goal {goal} as follows:

Evidence in page: 
{evidence}

Summary: 
{summary}

"""

async def read_page(url: str, timeout: int = 50) -> str:

    headers = {
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
        "X-Token-Budget": "256000"
        # avoid exceeding max tokens allowed
    }

    return await async_request(
        f"https://r.jina.ai/{url}",
        method="GET",
        headers=headers,
        timeout=timeout,
    )

async def summarize(url, summarizer_url, goal, text):

    # TODO: use `/generate` for training
    response = await async_request(
        f"{summarizer_url}/v1/chat/completions",
        payload={
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(text=text, goal=goal)
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        },
    )
    
    content = (
        response["choices"][0]["message"]["content"]
        .split("</think>")[-1]
        .strip()
        .removeprefix("```json")
        .removesuffix("```")
    )
    result = json.loads(content)
    return SUMMARY_TEMPLATE.format(
        url=url,
        goal=goal,
        evidence=result["evidence"],
        summary=result["summary"],
    )

async def visit(url, summarizer_url, goal):
    
    if isinstance(url, list):
        tasks = [visit(u, summarizer_url, goal) for u in url]
        results = await asyncio.gather(*tasks)
        return "\n=======\n".join(results)

    try:
        text = await read_page(url)
        return await summarize(url, summarizer_url, goal, text)
    except Exception as e:
        return ERROR_TEMPLATE.format(url=url, error=str(e))