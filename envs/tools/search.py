import os
import concurrent.futures
from serpapi import GoogleSearch

def search(query: str) -> str:
    
    if isinstance(query, list):
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(search, q) for q in query]
            responses = [future.result() for future in futures]
        return "\n=======\n".join(responses)

    # TODO: maybe conditioned on query language
    params = {
        "q": query,
        "location": "United States",
        "hl": "en",
        "gl": "us",
        "api_key": os.environ["SERP_API_KEY"]
    }
    result = GoogleSearch(params).get_dict() # TODO: maybe error
    if "organic_results" not in result:
        return f"No results found for '{query}'. Try with a more general query."
        
    results = []
    for idx, page in enumerate(result["organic_results"]):
        
        date = ""
        if "date" in page:
            date = f"\nDate published: {page['date']}"

        source = ""
        if "source" in page:
            source = f"\nSource: {page['source']}"

        snippet = ""
        if "snippet" in page:
            snippet = f"\n{page['snippet']}"

        results.append(f"{idx + 1}. [{page['title']}]({page['link']}){date}{source}\n{snippet}")

    return f"A Google search for '{query}' found {len(results)} results:\n\n## Web Results\n" + "\n\n".join(results)