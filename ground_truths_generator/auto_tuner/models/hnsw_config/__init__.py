from .hnsw_config import HnswConfig, save_results_to_csv
from .hnsw_config_hnswlib import HnswConfigHnswlib
from .hnsw_config_faiss import HnswConfigFaiss
from .hnsw_config_weaviate import HnswConfigWeaviate
from .hnsw_config_milvus import HnswConfigMilvus

hnsw_config_mapping = {
    "faiss": HnswConfigFaiss,
    "hnswlib": HnswConfigHnswlib,
    "weaviate": HnswConfigWeaviate,
    "milvus": HnswConfigMilvus,
}
