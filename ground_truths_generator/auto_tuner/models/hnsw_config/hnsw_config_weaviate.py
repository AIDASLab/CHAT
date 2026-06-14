import os
import time
from multiprocessing.pool import ThreadPool
import uuid

from auto_tuner.models.hnsw_config import HnswConfig
from auto_tuner.models.hnsw_result import HnswResult
from auto_tuner.constants import MAX_THREADS, ROOT_DIR
from auto_tuner.utils import log_execution

import weaviate
from weaviate.embedded import EmbeddedOptions

ENV_VARS = {
    "GOMEMLIMIT": str(100 * 1024)+"MiB",
    "GOMAXPROCS": str(MAX_THREADS),
}

class HnswConfigWeaviate(HnswConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = "weaviate"
        ####
        self.class_name = "Vector"
        self.__weaviate_path = ROOT_DIR + "/.weaviate"
        self._index = None
        self.metric = {
            "cosine": "cosine",
            "l2": "l2-squared",
        }[self.dataset.metric]
  
    def _clean_up(self):
        if self._index:
            self._index.schema.delete_all()
            del self._index
        # DELETE THE PERSISTENCE DATA
        if os.path.exists(self.__weaviate_path):
            import shutil
            shutil.rmtree(self.__weaviate_path)
    
    def _connect_weaviate(self):
        if self._index:
            return
        if not os.path.exists(self.__weaviate_path):
            os.makedirs(self.__weaviate_path)
        self._index = weaviate.Client(
            embedded_options=EmbeddedOptions(
                version="1.19.0-beta.1",
                persistence_data_path=self.__weaviate_path,
                additional_env_vars={
                    "GOMEMLIMIT": ENV_VARS["GOMEMLIMIT"],
                    "GOMAXPROCS": ENV_VARS["GOMAXPROCS"],
                    "LOG_LEVEL": "error",
                }
            )
        )
        self._index.schema.delete_all()
    
    @log_execution()
    def _build(self):
        self._connect_weaviate()
        _train_data = self.dataset.get_database()
        self._index.schema.create(
            {
                "classes": [
                    {
                        "class": self.class_name,
                        "properties": [
                            {
                                "name": "i",
                                "dataType": ["int"],
                            }
                        ],
                        "vectorIndexConfig": {
                            "distance": self.metric,
                            "maxConnections": self.M,
                            "efConstruction": self.efC,
                        },
                    }
                ]
            }
        )
        print("\nSTART OF BUILD ...\n")
        _start = time.time()
        with self._index.batch as batch:
            batch.batch_size = 10000 
            for i, x in enumerate(_train_data):
                batch.add_data_object(
                    data_object={
                        "i": i,
                    },
                    class_name=self.class_name,
                    uuid=uuid.UUID(int=i),
                    vector=x
                )
        self.build_time = time.time() - _start
        print("\nEND OF BUILD !!!\n")
        return self.build_time
    
    def get_batch_results(self, x)-> tuple[int, list]:
        i, v = x
        return i, (self._index.query.get(self.class_name, None).with_additional("id").with_near_vector(
            {
                "vector": v
            }
        ).do())
    
    
    @log_execution()
    def _evaluate(self, efS):
        _test_time = time.time()
        _test_data = self.dataset.get_queries()
        schema = self._index.schema.get(self.class_name)
        schema["vectorIndexConfig"]["ef"] = efS
        self._index.schema.update_config(self.class_name, schema)
        recall_qps = []
        search_times = []
        for _ in range(self.iters):
            _search_time = time.time()
            with ThreadPool(processes=MAX_THREADS) as pool:
                _results = pool.map(lambda x : self.get_batch_results(x), enumerate(_test_data))
            _search_time = (time.time() - _search_time)
            _qps = len(_test_data) / _search_time
            
            id_labels = list(map(
                lambda ret: 
                    (ret[0], 
                        list(map(lambda l : uuid.UUID(l["_additional"]["id"]).int, 
                        ret[1]["data"]["Get"][self.class_name]))), 
                _results
            ))
            gt = self.dataset.get_groundtruth()
            correct = 0
            for i, labels in id_labels:
                for label in labels:
                    for correct_label in gt[i]:
                        if label == correct_label:
                            correct += 1
                            break
            _recall = float(correct) / (self.dataset.k * len(_test_data))
            recall_qps.append((_recall, _qps))
            search_times.append(_search_time)
        test_time = time.time() - _test_time
        hnsw_result = HnswResult(self.M, self.efC, efS, self.build_time, self.index_size, test_time, search_times, recall_qps)
        self.results[efS] = hnsw_result
        print(f"efS: {efS}, recall: {hnsw_result.recall}, qps: {hnsw_result.qps}")
        return hnsw_result.recall, hnsw_result.qps
            