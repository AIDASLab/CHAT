from .hnsw_config import HnswConfig, save_results_to_csv
from .hnsw_config_hnswlib import HnswConfigHnswlib
from .hnsw_config_faiss import HnswConfigFaiss

def _load_milvus():
    from .hnsw_config_milvus import HnswConfigMilvus
    return HnswConfigMilvus


def _load_weaviate():
    from .hnsw_config_weaviate import HnswConfigWeaviate
    return HnswConfigWeaviate


class HnswConfigMapping(dict):
    """Map implementation names to config classes, loading optional backends lazily."""

    _optional_loaders = {
        "milvus": _load_milvus,
        "weaviate": _load_weaviate,
    }

    def __getitem__(self, impl):
        if impl in self._optional_loaders and impl not in self:
            self[impl] = self._optional_loaders[impl]()
        return super().__getitem__(impl)

    def __contains__(self, impl):
        return super().__contains__(impl) or impl in self._optional_loaders

    def keys(self):
        return set(super().keys()) | set(self._optional_loaders.keys())


hnsw_config_mapping = HnswConfigMapping({
    "faiss": HnswConfigFaiss,
    "hnswlib": HnswConfigHnswlib,
})
