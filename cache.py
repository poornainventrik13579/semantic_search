import os
import pandas as pd
import time
import pickle
from pathlib import Path
import threading


class EmbeddingCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_paths(self, tenant_name):
        tenant_dir = self.cache_dir / tenant_name
        tenant_dir.mkdir(exist_ok=True)

        return {
            'embeddings': tenant_dir / 'embeddings.pkl',
            'documents': tenant_dir / 'documents.parquet'
        }

    def save(self, tenant_name, embeddings, documents):
        paths = self.get_paths(tenant_name)

        start = time.time()

        with open(paths['embeddings'], 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

        documents.to_parquet(paths['documents'], index=False)

        save_time = time.time() - start
        return save_time

    def load(self, tenant_name):
        paths = self.get_paths(tenant_name)

        if not paths['embeddings'].exists() or not paths['documents'].exists():
            return None, None, 0.0

        start = time.time()

        with open(paths['embeddings'], 'rb') as f:
            embeddings = pickle.load(f)

        documents = pd.read_parquet(paths['documents'])

        load_time = time.time() - start
        return embeddings, documents, load_time

    def exists(self, tenant_name):
        paths = self.get_paths(tenant_name)
        return paths['embeddings'].exists() and paths['documents'].exists()

    def clear(self, tenant_name):
        paths = self.get_paths(tenant_name)
        if paths['embeddings'].exists():
            paths['embeddings'].unlink()
        if paths['documents'].exists():
            paths['documents'].unlink()


# Periodic snapshot functionality (not currently used)
# class SnapshotManager:
#     def __init__(self, interval=300):
#         self.interval = interval
#         self.save_callback = None
#         self.timer = None
#
#     def register_callback(self, callback):
#         self.save_callback = callback
#
#     def start_periodic_save(self):
#         self._schedule_next_save()
#
#     def _schedule_next_save(self):
#         self.timer = threading.Timer(self.interval, self._periodic_save)
#         self.timer.daemon = True
#         self.timer.start()
#
#     def _periodic_save(self):
#         if self.save_callback:
#             threading.Thread(target=self.save_callback, daemon=True).start()
#         self._schedule_next_save()
#
#     def stop(self):
#         if self.timer:
#             self.timer.cancel()
