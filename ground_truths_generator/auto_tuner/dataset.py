import os
import numpy as np
import pickle as pkl
import h5py
# import devhnswlib as hnswlib
import hnswlib
import faiss

from auto_tuner.constants import DATA_DIR


class Dataset:
    
    def __init__(self, k=10, recompute=True, impl=None):
        if recompute and impl is None:
            raise ValueError("Recompute is set to True, please provide the implementation")
        self.k = k
        self.recompute=recompute
        self.impl = impl
        ##
        self.name = "Dataset"
        self.metric = "l2"  # or "cosine" or "ip"
        self.dim = -1
        self.nq = -1        # no. queries
        self.nb = -1        # no. base vectors
        self._recomputed_groundtruth = None
    
   
    def get_database(self):
        xb = np.array(self.data['train'])
        return xb
    
   
    def database_iterator(self, batch_size=100):
        xb = self.get_database()
        for i in range(0, len(xb), batch_size):
            yield xb[i:i + batch_size]
    
   
    def get_queries(self):
        xq = np.array(self.data['test'])
        return xq
    
    
    def __recompute_groundtruth_faiss(self):
        # Validate if the groundtruth is already computed
        if self._recomputed_groundtruth is not None:
            return self._recomputed_groundtruth
        # Validate if the groundtruth is already saved
        filename = f"{DATA_DIR}/{self.__class__.__name__}_{self.impl}_gt_{self.k}.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self._recomputed_groundtruth = pkl.load(f)
                return self._recomputed_groundtruth
        print("Recomputing groundtruth ...")
        xb = self.get_database()    # N x D
        xq = self.get_queries()     # Q x D
        if self.metric == "cosine" or self.metric == "ip":
            bf_index = faiss.IndexFlatIP(self.dim)
        if self.metric == "l2":
            bf_index = faiss.IndexFlatL2(self.dim)
        if xb.dtype != np.float32:
            xb = xb.astype(np.float32)
        if xq.dtype != np.float32:
            xq = xq.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(xb)
            faiss.normalize_L2(xq)
        bf_index.add(xb)
        _distances, _labels = bf_index.search(xq, self.k)
        labels = []
        for i in range(len(_distances)):
            r = []
            for l, d in zip(_labels[i], _distances[i]):
                r.append(-99 if l == -1 else l)
            labels.append(r)
        self._recomputed_groundtruth = labels
        with open(filename, "wb") as f:
            pkl.dump(self._recomputed_groundtruth, f)
        return self._recomputed_groundtruth
   
    
    def __recompute_groundtruth_hnswlib(self):
        # Validate if the groundtruth is already computed
        if self._recomputed_groundtruth is not None:
            return self._recomputed_groundtruth
        # Validate if the groundtruth is already saved
        filename = f"{DATA_DIR}/{self.__class__.__name__}_{self.impl}_gt_{self.k}.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self._recomputed_groundtruth = pkl.load(f)
                return self._recomputed_groundtruth
        # Recompute the groundtruth
        print("Recomputing groundtruth ...")
        xb = self.get_database()
        xq = self.get_queries()
        # hnswlib is automatically normalizing the vectors
        bf_index = hnswlib.BFIndex(space=self.metric, dim=self.dim)
        bf_index.init_index(max_elements=len(xb))
        bf_index.add_items(xb)
        labels, _distances = bf_index.knn_query(xq, self.k)
        ##
        self._recomputed_groundtruth = labels
        with open(filename, "wb") as f:
            pkl.dump(self._recomputed_groundtruth, f)
        return self._recomputed_groundtruth
   
    def _recompute_groundtruth(self):
        if self.impl == "faiss":
            return self.__recompute_groundtruth_faiss()
        if self.impl == "hnswlib":
            return self.__recompute_groundtruth_hnswlib()
        raise ValueError("Invalid implementation")
    
    
    def get_groundtruth(self):
        if self.recompute:
            return self._recompute_groundtruth()
        gt = np.array(self.data['neighbors'])
        if self.k is not None:
            assert self.k <= 100
            gt = gt[:, :self.k]
        return gt
   
class DatasetNytimes(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/nytimes-256-angular.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "nytimes-256-angular"
        self.metric = "cosine"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None
    
    
class DatasetGlove(Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/glove-100-angular.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "glove-100-angular"
        self.metric = "cosine"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None


class DatasetGloveIP(Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/glove-100-angular.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "glove-100-angular"
        self.metric = "ip"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None


class DatasetSift(Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/sift-128-euclidean.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "sift-128-euclidean"
        self.metric = "l2"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None


class DatasetYoutube(Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/youtube-1024-angular.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "youtube-1024-angular"
        self.metric = "cosine"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None

class DatasetMSMarco(Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/msmarco-384-angular.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "msmarco-384-angular"
        self.metric = "cosine"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None


class DatasetDBpediaEntity(Dataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loc = f"{DATA_DIR}/dbpediaentity-768-angular.hdf5"
        with h5py.File(loc, 'r') as f:
            self.data = {
                "train": np.array(f['train']),
                "test": np.array(f['test']),
                "neighbors": np.array(f['neighbors'])
            }
        self.name = "dbpediaentity-768-angular"
        self.metric = "cosine"
        self.dim = self.data['train'].shape[1]
        self.nb = self.data['train'].shape[0]
        self.nq = self.data['test'].shape[0]
        self._recomputed_groundtruth = None    

class DatasetMSMarco2(Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recompute = True
        loc = f"{DATA_DIR}/multi-qa-MiniLM-L6-cos-v1_all.npy"
        with open(loc, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        self.metric = "cosine"
        self.dim = data.shape[1]
        self.nb = 1_000_000
        self.nq = 10_000
        np.random.seed(42)
        indices = np.random.choice(data.shape[0], self.nb + self.nq, replace=False)
        data = data[indices]
        self.data = {
            "train": data[:self.nb],
            "test": data[self.nb:]
        }
    
    def save_as_hdf5(self):
        loc = f"{DATA_DIR}/msmarco-{self.dim}-angular.hdf5"
        with h5py.File(loc, 'w') as f:
            f.create_dataset("train", data=self.data["train"])
            f.create_dataset("test", data=self.data["test"])
            f.create_dataset("neighbors", data=self.get_groundtruth())  

class DatasetDbpediaEntity2(Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recompute = True
        loc = f"{DATA_DIR}/multi-qa-mpnet-base-cos-v1_all.npy"
        data = np.load(loc, allow_pickle=True)
        self.metric = "cosine"
        self.dim = data.shape[1]
        self.nb = 1_000_000
        self.nq = 10_000
        np.random.seed(42)
        indices = np.random.choice(data.shape[0], self.nb + self.nq, replace=False)
        data = data[indices]
        self.data = {
            "train": data[:self.nb],
            "test": data[self.nb:]
        }
    
    def save_as_hdf5(self):
        loc = f"{DATA_DIR}/fiqa-{self.dim}-angular.hdf5"
        with h5py.File(loc, 'w') as f:
            f.create_dataset("train", data=self.data["train"])
            f.create_dataset("test", data=self.data["test"])
            f.create_dataset("neighbors", data=self.get_groundtruth())


## UTILITY ##
class DatasetMapping:
    def __init__(self):
        self.dataset_mapping = {
            "nytimes-256-angular": DatasetNytimes,
            "glove-100-angular": DatasetGlove,
            "sift-128-euclidean": DatasetSift,
            "youtube-1024-angular": DatasetYoutube,
            "msmarco-384-angular": DatasetMSMarco,
            "dbpediaentity-768-angular": DatasetDBpediaEntity,
        }
        self._cache = {}  # (dataset_name, impl, recompute, k) -> dataset_instance

    def __getitem__(self, dataset_name):
        dataset_cls = self.dataset_mapping.get(dataset_name)
        if dataset_cls is None:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        def get_instance(impl="hnswlib", recompute=True, k=10):
            key = (dataset_name, impl, recompute, k)
            if key in self._cache:
                return self._cache[key]
            instance = dataset_cls(impl=impl, recompute=recompute, k=k)
            self._cache[key] = instance
            return instance
        return get_instance

dataset_mapping = DatasetMapping()