from ctypes import util
from operator import index
import os
from re import M
import time
from unittest import runner
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility, Collection
from sympy import limit
from auto_tuner.models.hnsw_config import HnswConfig
from auto_tuner.models.hnsw_result import HnswResult
from auto_tuner.utils import log_execution
from auto_tuner.constants import ROOT_DIR

class HnswConfigMilvus(HnswConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = "milvus"
        ####
        self.collection_name = "Vector"
        self.metric = {
            "cosine": "COSINE",
            "l2": "L2",
        }[self.dataset.metric]
        ####
        self.collection = None
    
    def _connect_milvus(self):
        # START MILVUS
        try:
            runner_filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "scripts", 
                "standalone_embed.sh"
            )
            os.system(f"bash {runner_filename} stop")
            os.system(f"bash {runner_filename} start")
        except Exception as e:
            print(f"Error starting Milvus: {e} !!!!")
        # CONNECT TO MILVUS
        self.__connects = connections
        for _ in range(10):
            try:
                self.__connects.connect(
                    alias="default",
                    host="localhost",
                    port="19530",
                )
                break
            except Exception as e:
                print(f"Error connecting to Milvus: {e}")
                time.sleep(1)
        self.__client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
    
    def _disconnect_milvus(self):
        runner_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts",
            "standalone_embed.sh"
        )
        os.system(f"bash {runner_filename} stop") 
     
    def step_1_create_collection(self):
        field_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
        )
        field_vector = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self.dataset.dim,
        )
        schema = CollectionSchema(
            fields=[field_id, field_vector],
            description="Vector collection",
        )
        self.collection = Collection(
            self.collection_name,
            schema=schema,
            consistence_level="Strong",
        )
   
    def step_2_insert(self, X):
        batch_size = 10000
        for i in range(0, len(X), batch_size):
            batch_range = range(i, min(i + batch_size, len(X)))
            batch_data = X[batch_range]
            entities = [
                [i for i in batch_range],
                batch_data.tolist(),
            ]
            self.collection.insert(entities)
        self.collection.flush()
    
    def step_3_create_index(self):
        index_params = {
            "index_type": "HNSW",
            "params": {
                "M": self.M,
                "efConstruction": self.efC,
            },
            "metric_type": self.metric,
        }
        self.collection.create_index(
            field_name="vector",
            index_params=index_params,
            index_name="hnsw_index",
        )
        utility.wait_for_index_building_complete(
            collection_name=self.collection_name,
            index_name="hnsw_index",
        )
        index = self.collection.index(index_name="hnsw_index")
        index_progess = utility.index_building_progress(
            collection_name=self.collection_name,
            index_name="hnsw_index",
        )
        print(f"Index progress: {index_progess}")
    
    def step_4_load_collection(self):
        self.collection.load()
    
    @log_execution() 
    def _build(self):
        self._connect_milvus()
        _start_time = time.time()
        self.step_1_create_collection()
        self.step_2_insert(self.dataset.get_database())
        self.step_3_create_index()
        self.step_4_load_collection()    
        self.build_time = time.time() - _start_time
        return self.build_time
    
    @log_execution()
    def _evaluate(self, efS):
        _test_data = self.dataset.get_queries()
        _test_time = time.time()
        recall_qps = []
        search_times = []
        for _ in range(self.iters):
            _search_time = time.time()
            results = self.collection.search(
                data = _test_data.tolist(),
                anns_field = "vector", 
                param = {
                    "metric_type": self.metric,
                    "params": {"ef": efS}
                },
                limit = self.dataset.k,
                output_fields=["id"],
            )
            _search_time = time.time() - _search_time
            _qps = len(_test_data) / _search_time
            labels = [[r.entity.get("id") for r in result] for result in results]
            gt = self.dataset.get_groundtruth()
            correct = 0
            for i in range(len(_test_data)):
                for label in labels[i]:
                    for correct_label in gt[i]:
                        if label == correct_label:
                            correct += 1
                            break
            _recall = float(correct) / (self.dataset.k * len(_test_data))
            recall_qps.append((_recall, _qps))
            search_times.append(_search_time)
        test_time = time.time() - _test_time
        hnsw_result = HnswResult(self.M, self.efC, efS, self.build_time, test_time, search_times, recall_qps)
        self.results[efS] = hnsw_result
        print(f"efS: {efS}, recall: {hnsw_result.recall}, qps: {hnsw_result.qps}")
        self.collection.release()
            
            
        test_time = time.time() - _test_time
        return 0, 0
        
    def _clean_up(self):
        if self.collection:
            self.collection.release()
            utility.drop_collection(self.collection_name)
            self._disconnect_milvus()