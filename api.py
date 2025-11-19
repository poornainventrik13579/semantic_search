from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import sys
import argparse
import uvicorn
import logging
import json
import shutil

from searcher import TenantSearch
from config import DATA_DIR, CACHE_DIR
from cache import EmbeddingCache
# from pyngrok import ngrok

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


class DeleteRequest(BaseModel):
    tenant_id: str
    dataset: str


@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")
    if os.path.exists(DATA_DIR):
        logger.info(f"Contents: {os.listdir(DATA_DIR)}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")


def get_or_load_tenant(tenant_id: str, dataset: str):
    cache_key = f"{tenant_id}_{dataset}"
    logger.info(f"Loading tenant: {tenant_id}, dataset: {dataset}")

    if cache_key in tenant_cache:
        logger.info(f"Using cached tenant: {cache_key}")
        return tenant_cache[cache_key]

    tenant_path = os.path.join(DATA_DIR, tenant_id)
    logger.info(f"Tenant path: {tenant_path}")
    
    if not os.path.exists(tenant_path):
        logger.error(f"Tenant path not found: {tenant_path}")
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    dataset_file = os.path.join(tenant_path, f"{dataset}.csv")
    logger.info(f"Dataset file: {dataset_file}")
    
    if not os.path.exists(dataset_file):
        logger.error(f"Dataset file not found: {dataset_file}")
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found in tenant '{tenant_id}'")

    try:
        logger.info("Creating TenantSearch instance...")
        searcher = TenantSearch(tenant_id, dataset_file=dataset_file)

        logger.info("Loading data...")
        if not searcher.load_data(use_cache=True):
            logger.error("Failed to load tenant data")
            raise HTTPException(status_code=500, detail="Failed to load tenant data")

        logger.info("Creating embeddings...")
        searcher.create_embeddings()

        tenant_cache[cache_key] = searcher
        logger.info(f"Successfully loaded and cached tenant: {cache_key}")
        return searcher
    
    except Exception as e:
        logger.error(f"Error loading tenant: {str(e)}", exc_info=True)
        raise


@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "API is running", "status": "ok"}


@app.post("/search", response_model=List[Dict[str, Any]])
def search(request: SearchRequest):
    logger.info(f"Search request: tenant={request.tenant_id}, dataset={request.dataset}, query={request.query}, k={request.k}")
    
    try:
        searcher = get_or_load_tenant(request.tenant_id, request.dataset)

        logger.info("Performing search...")
        results, search_time = searcher.search(
            request.query,
            top_k=request.k,
            min_score=0.0
        )
        logger.info(f"Search completed in {search_time:.4f}s, found {len(results)} results")

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

        logger.info(f"Returning {len(formatted_results)} formatted results")
        return formatted_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    dataset: str = Form(...),
    schema: str = Form(...)
):
    logger.info(f"Upload request: tenant={tenant_id}, dataset={dataset}")

    try:
        schema_dict = json.loads(schema)

        tenant_path = os.path.join(DATA_DIR, tenant_id)
        os.makedirs(tenant_path, exist_ok=True)

        csv_path = os.path.join(tenant_path, f"{dataset}.csv")
        with open(csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        schema_path = os.path.join(tenant_path, f"{dataset}.json")
        with open(schema_path, "w") as f:
            json.dump(schema_dict, f, indent=2)

        logger.info(f"Dataset uploaded: {csv_path}")
        return {
            "message": "Dataset uploaded successfully",
            "tenant_id": tenant_id,
            "dataset": dataset,
            "csv_path": csv_path,
            "schema_path": schema_path
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid schema JSON")
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/dataset")
def delete_dataset(request: DeleteRequest):
    logger.info(f"Delete request: tenant={request.tenant_id}, dataset={request.dataset}")

    try:
        tenant_path = os.path.join(DATA_DIR, request.tenant_id)

        if not os.path.exists(tenant_path):
            raise HTTPException(status_code=404, detail=f"Tenant '{request.tenant_id}' not found")

        csv_path = os.path.join(tenant_path, f"{request.dataset}.csv")
        schema_path = os.path.join(tenant_path, f"{request.dataset}.json")

        deleted_files = []

        if os.path.exists(csv_path):
            os.remove(csv_path)
            deleted_files.append(csv_path)

        if os.path.exists(schema_path):
            os.remove(schema_path)
            deleted_files.append(schema_path)

        cache = EmbeddingCache(CACHE_DIR)
        cache_key = f"{request.tenant_id}_{request.dataset}"
        if cache.exists(cache_key):
            cache.clear(cache_key)
            logger.info(f"Cache cleared for: {cache_key}")

        if cache_key in tenant_cache:
            del tenant_cache[cache_key]
            logger.info(f"Removed from memory cache: {cache_key}")

        if not deleted_files:
            raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset}' not found")

        logger.info(f"Deleted files: {deleted_files}")
        return {
            "message": "Dataset deleted successfully",
            "tenant_id": request.tenant_id,
            "dataset": request.dataset,
            "deleted_files": deleted_files
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description='Semantic Search API')
    # parser.add_argument('--ngrok', action='store_true', help='Enable ngrok')
    parser.add_argument('--port', type=int, default=8000, help='Port')
    args = parser.parse_args()
    
    port = int(os.environ.get("PORT", args.port))
    logger.info(f"Starting server on port {port}")

    # if args.ngrok:
    #     if ngrok is None:
    #         logger.error("pyngrok not installed")
    #         print("Error: Install pyngrok")
    #         sys.exit(1)

    #     try:
    #         token = os.getenv("NGROK_AUTHTOKEN")
    #         if token:
    #             ngrok.set_auth_token(token)

    #         public_url = ngrok.connect(port)

    #         print("Semantic Search API")
    #         print("-" * 50)
    #         print(f"Public: {public_url}")
    #         print(f"Local:  http://localhost:{port}")
    #         print(f"Docs:   {public_url}/docs")
    #         print("-" * 50)

    #     except Exception as e:
    #         logger.error(f"Ngrok error: {e}", exc_info=True)
    #         print(f"Error: {e}")
    #         sys.exit(1)
    # else:
    print("Semantic Search API")
    print("-" * 50)
    print(f"Server: http://localhost:{port}")
    print(f"Docs:   http://localhost:{port}/docs")
    print("-" * 50)

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()