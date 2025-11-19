import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.0

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
CACHE_DIR = os.path.join(_BASE_DIR, "cache")

ENABLE_CACHE = True
# SNAPSHOT_INTERVAL = 300
