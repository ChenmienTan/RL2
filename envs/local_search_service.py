import argparse
import json
import time
import requests
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, Request

def main(args):

    start = time.time()
    index = faiss.read_index(args.index_path)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    index = faiss.index_cpu_to_all_gpus(index, co=co)
    print(f"It takes {round(time.time() - start, 1)} seconds to load the index.")

    start = time.time()
    with open(args.corpus_path) as f:
        corpus = [json.loads(line) for line in f]
    print(f"It takes {round(time.time() - start, 1)} seconds to load the corpus.")

    app = FastAPI()

    @app.get("/health")
    async def check_health():
        return {"status": "ok"}

    @app.post("/search")
    async def local_search(request: Request):

        query = (await request.json())["query"]
        response = requests.post(
            f"http://{args.host}:{args.port}/v1/embeddings", json={
                "model": args.model_name,
                "input": query
            }
        ).json()
        embed = np.array(
            [response["data"][0]["embedding"]], dtype=np.float32
        )
        _, indices = index.search(embed, k=args.top_k)
        passages = []
        for local_idx, global_idx in enumerate(indices[0]):
            content = corpus[global_idx]["contents"].split("\n")
            title, text = content[0], "\n".join(content[1:])
            passages.append(f"Doc {local_idx + 1}(Title: {title}) {text}")
        return "\n".join(passages)

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--top_k", type=int)
    args = parser.parse_args()

    main(args)