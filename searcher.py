import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

try:
    from .config import *
    from .cache import EmbeddingCache
except ImportError:
    from config import *
    from cache import EmbeddingCache


class TenantSearch:
    def __init__(self, tenant_name, enable_cache=ENABLE_CACHE, dataset_file=None):
        self.tenant_name = tenant_name
        self.tenant_folder = os.path.join(DATA_DIR, tenant_name)
        self.dataset_file = dataset_file
        self.cache_key = f"{tenant_name}_{os.path.splitext(os.path.basename(dataset_file))[0]}" if dataset_file else tenant_name

        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = None
        self.documents = None
        self.schema = None
        self.enable_cache = enable_cache

        if enable_cache:
            self.cache = EmbeddingCache(CACHE_DIR)

    def _load_schema(self):
        schema_file = self.dataset_file.replace('.csv', '.json')

        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        with open(schema_file, 'r') as f:
            self.schema = json.load(f)

        return self.schema

    def _get_embedding_columns(self):
        return [col for col, include in self.schema.items() if include]

    def _combine_text_columns(self, columns):
        return self.documents[columns].fillna('').astype(str).agg(' '.join, axis=1).tolist()

    def _csv_modified_time(self):
        return os.path.getmtime(self.dataset_file)

    def _is_cache_stale(self):
        if not self.cache.exists(self.cache_key):
            return True

        cache_file = os.path.join(self.cache.cache_dir, f"{self.cache_key}_embeddings.pkl")

        if not os.path.exists(cache_file):
            return True

        cache_time = os.path.getmtime(cache_file)
        csv_time = self._csv_modified_time()

        return csv_time > cache_time

    def load_data(self, use_cache=True):
        self._load_schema()

        if self.enable_cache and use_cache and not self._is_cache_stale():
            self.embeddings, self.documents, _ = self.cache.load(self.cache_key)
            if self.embeddings is not None:
                return True

        if not os.path.exists(self.dataset_file):
            print(f"Error: CSV file not found: {self.dataset_file}")
            return False

        self.documents = pd.read_csv(self.dataset_file)

        embedding_columns = self._get_embedding_columns()
        if not embedding_columns:
            print("Error: No columns marked for embedding")
            return False

        return True

    def create_embeddings(self):
        if self.embeddings is not None:
            return

        embedding_columns = self._get_embedding_columns()
        texts = self._combine_text_columns(embedding_columns)

        self.embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if self.enable_cache:
            self.cache.save(self.cache_key, self.embeddings, self.documents)

    def search(self, query, top_k=DEFAULT_TOP_K, min_score=DEFAULT_THRESHOLD):
        query_vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_vector)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append({
                    'index': int(idx),
                    'score': score,
                    'document': self.documents.iloc[idx]
                })

        return results, 0
