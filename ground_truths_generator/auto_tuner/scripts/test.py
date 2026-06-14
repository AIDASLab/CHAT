from auto_tuner.scripts import summary_from_csv, run_hnsw_config
from auto_tuner.scripts import dataset_nytimes_hnswlib, dataset_nytimes_faiss
from auto_tuner.scripts import dataset_glove_hnswlib, dataset_glove_faiss
from auto_tuner.scripts import dataset_sift_hnswlib, dataset_sift_faiss
from auto_tuner.scripts import dataset_youtube_hnswlib, dataset_youtube_faiss
from auto_tuner.scripts import dataset_msmarco_hnswlib, dataset_msmarco_faiss
from auto_tuner.scripts import dataset_dbpediaentity_hnswlib, dataset_dbpediaentity_faiss
from auto_tuner.scripts import dataset_og_nytimes, dataset_og_glove, dataset_og_sift, dataset_og_youtube
from auto_tuner.dataset import DatasetNytimes, DatasetGloveIP, DatasetSift


def test_nytimes_groundtruth():
    faiss_gd = dataset_nytimes_faiss.get_groundtruth()
    hswslib_gd = dataset_nytimes_hnswlib.get_groundtruth()
    gd = dataset_og_nytimes.get_groundtruth()
    
    # faiss 
    total, inter = 0, 0
    for i in range(len(faiss_gd)):
        s1 = set(faiss_gd[i])
        s2 = set(gd[i])
        len_intersection = len(s1.intersection(s2))
        len_union = len(s1.union(s2))
        total += len_union
        inter += len_intersection
    print(f"Total: {total}, Intersection: {inter}, Ratio: {inter/total}")
    
    total, inter = 0, 0
    for i in range(len(hswslib_gd)):
        s1 = set(hswslib_gd[i])
        s2 = set(gd[i])
        len_intersection = len(s1.intersection(s2))
        len_union = len(s1.union(s2))
        total += len_union
        inter += len_intersection
    print(f"Total: {total}, Intersection: {inter}, Ratio: {inter/total}")
    
    total, inter = 0, 0
    for i in range(len(hswslib_gd)):
        s1 = set(hswslib_gd[i])
        s2 = set(faiss_gd[i])
        len_intersection = len(s1.intersection(s2))
        len_union = len(s1.union(s2))
        total += len_union
        inter += len_intersection
    print(f"Total: {total}, Intersection: {inter}, Ratio: {inter/total}")
"""
Total: 101382, Intersection: 98537, Ratio: 0.9719378193367659
Total: 117652, Intersection: 82348, Ratio: 0.6999286029986741
""" 
"""
Total: 101382, Intersection: 98537, Ratio: 0.9719378193367659
Total: 101468, Intersection: 98532, Ratio: 0.9710647691883155
"""
"""
Total: 101382, Intersection: 98537, Ratio: 0.9719378193367659
Total: 101468, Intersection: 98532, Ratio: 0.9710647691883155
Total: 101254, Intersection: 98665, Ratio: 0.974430639777194
"""


def test_nytimes_groundtruth2():
    total, inter = 0, 0
    for i in range(dataset_og_nytimes.nq):
        s1 = set(dataset_og_nytimes.get_queries()[i])
        s2 = set(dataset_nytimes_hnswlib.get_queries()[i])
        len_intersection = len(s1.intersection(s2))
        len_union = len(s1.union(s2))
        total += len_union
        inter += len_intersection
    print(f"Total: {total}, Intersection: {inter}, Ratio: {inter/total}")
"""
Total: 2557700, Intersection: 2557700, Ratio: 1.0 # faiss
Total: 2557700, Intersection: 2557700, Ratio: 1.0 # hnswlib
"""

def test_nytimes_hnswlib():
    run_hnsw_config(dataset_nytimes_hnswlib, "hnswlib", M=[16], efC=[64], prefix="test_nytimes_hnswlib")
    
def test_nytimes_faiss():
    run_hnsw_config(DatasetNytimes(recompute=True, impl="faiss"), "faiss", M=[16], efC=[64], prefix="test_nytimes_faiss")

def test_glove_hnswlib():
    run_hnsw_config(dataset_glove_hnswlib, "hnswlib", M=[32], efC=[128], prefix="test_glove_hnswlib")

def test_glove_faiss():
    run_hnsw_config(dataset_glove_faiss, "faiss", M=[16], efC=[16], prefix="test_glove_faiss")
    
def test_sift_hnswlib():
    run_hnsw_config(dataset_sift_hnswlib, "hnswlib", M=[16], efC=[64], prefix="test_sift_hnswlib")
    
def test_sift_faiss():
    run_hnsw_config(dataset_sift_faiss, "faiss", M=[16], efC=[64], prefix="test_sift_faiss")
    
def test_youtube_hnswlib():
    run_hnsw_config(dataset_youtube_hnswlib, "hnswlib", M=[16], efC=[64], prefix="test_youtube_hnswlib")
    
def test_youtube_faiss():
    run_hnsw_config(dataset_youtube_faiss, "faiss", M=[16], efC=[64], prefix="test_youtube_faiss")
    
def test_glove_ip_faiss():
    run_hnsw_config(DatasetGloveIP(recompute=True, impl="faiss"), "faiss", M=[64], efC=[96, 126, 128, 130], prefix="test_glove_ip_faiss")

def main_glove_faiss_recompute():
    run_hnsw_config(dataset_glove_faiss, "faiss", M=[64], efC=list(range(62, 142+1, 4)), prefix="main_glove_faiss_recompute") 

def main_glove_hnswlib_recompute():
    run_hnsw_config(dataset_glove_hnswlib, "hnswlib", M=[64], efC=list(range(62, 142+1, 4)), prefix="main_glove_hnswlib_recompute")
    
def main_sift_hnswlib_recompute():
    run_hnsw_config(dataset_sift_hnswlib, "hnswlib", M=[56, 64], efC=[64, 96, 128, 160], prefix="main_sift_hnswlib_recompute") 
    
def main_sift_faiss_recompute():
    run_hnsw_config(DatasetSift(recompute=False, impl="faiss"), "faiss", M=[64], efC=[64, 80, 96, 112, 128, 144], prefix="main_sift_faiss_recompute")

def test_msmarco_hnswlib():
    _M = [8, 12, 16]
    _efC = [16, 20, 24, 28, 32]
    run_hnsw_config(dataset_msmarco_hnswlib, "hnswlib", M=_M, efC=_efC, prefix="test_msmarco_hnswlib")

def test_msmarco_faiss():
    _M = [8, 12, 16]
    _efC = [16, 20, 24, 28, 32]
    run_hnsw_config(dataset_msmarco_faiss, "faiss", M=_M, efC=_efC, prefix="test_msmarco_faiss")

def test_dbpediaEntity_hnswlib():
    from auto_tuner.constants import DEFAULT_M, DEFAULT_EFC
    _M = DEFAULT_M[::4]
    _efC = DEFAULT_EFC[::4]
    run_hnsw_config(dataset_dbpediaentity_hnswlib, "hnswlib", M=_M, efC=_efC, prefix="test_dbpediaentity_hnswlib")
    
def test_sift_weaviate():
    _M = [8] 
    _efC = [8]
    dataset = DatasetSift(recompute=False, impl="weaviate")
    run_hnsw_config(dataset, "weaviate", M=_M, efC=_efC, prefix="test_sift_weaviate")
    
def test_sift_milvus():
    _M = [8] 
    _efC = [8]
    dataset = DatasetSift(recompute=False, impl="milvus")
    run_hnsw_config(dataset, "milvus", M=_M, efC=_efC, prefix="test_sift_milvus")