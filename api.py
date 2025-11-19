from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import sys
import argparse
import uvicorn

from searcher import TenantSearch
from config import DATA_DIR
from pyngrok import ngrok

app = FastAPI(title="Semantic Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tenant_cache = {}


class SearchRequest(BaseModel):
    tenant_id: str
    dataset: str
    query: str
    k: Optional[int] = 5


def get_or_load_tenant(tenant_id: str, dataset: str):
    cache_key = f"{tenant_id}_{dataset}"

    if cache_key in tenant_cache:
        return tenant_cache[cache_key]

    tenant_path = os.path.join(DATA_DIR, tenant_id)
    if not os.path.exists(tenant_path):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    dataset_file = os.path.join(tenant_path, f"{dataset}.csv")
    if not os.path.exists(dataset_file):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found in tenant '{tenant_id}'")

    searcher = TenantSearch(tenant_id, dataset_file=dataset_file)

    if not searcher.load_data(use_cache=True):
        raise HTTPException(status_code=500, detail="Failed to load tenant data")

    searcher.create_embeddings()

    tenant_cache[cache_key] = searcher
    return searcher


@app.post("/search", response_model=List[Dict[str, Any]])
def search(request: SearchRequest):
    try:
        searcher = get_or_load_tenant(request.tenant_id, request.dataset)

        results, search_time = searcher.search(
            request.query,
            top_k=request.k,
            min_score=0.0
        )

        formatted_results = []
        for rank, result in enumerate(results, 1):
            doc = result['document']

            result_dict = {
                'rank': rank,
                'score': round(result['score'], 4)
            }

            for col in doc.index:
                value = doc[col]
                if pd.notna(value):
                    if isinstance(value, (int, float, bool)):
                        result_dict[col] = value
                    else:
                        result_dict[col] = str(value)

            formatted_results.append(result_dict)

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description='Semantic Search API')
    parser.add_argument('--ngrok', action='store_true', help='Enable ngrok')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    args = parser.parse_args()

    if args.ngrok:
        if ngrok is None:
            print("Error: Install pyngrok")
            sys.exit(1)

        try:
            token = os.getenv("NGROK_AUTHTOKEN")
            if token:
                ngrok.set_auth_token(token)

            public_url = ngrok.connect(args.port)

            print("Semantic Search API")
            print("-" * 50)
            print(f"Public: {public_url}")
            print(f"Local:  http://localhost:{args.port}")
            print(f"Docs:   {public_url}/docs")
            print("-" * 50)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Semantic Search API")
        print("-" * 50)
        print(f"Server: http://localhost:{args.port}")
        print(f"Docs:   http://localhost:{args.port}/docs")
        print("-" * 50)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
