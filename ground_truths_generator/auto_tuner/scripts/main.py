from auto_tuner.scripts import summary_from_csv, run_hnsw_config

from auto_tuner.scripts import dataset_nytimes_hnswlib, dataset_nytimes_faiss
from auto_tuner.scripts import dataset_glove_hnswlib, dataset_glove_faiss
from auto_tuner.scripts import dataset_sift_hnswlib, dataset_sift_faiss
from auto_tuner.scripts import dataset_youtube_hnswlib, dataset_youtube_faiss
from auto_tuner.scripts import dataset_msmarco_hnswlib, dataset_msmarco_faiss
from auto_tuner.scripts import dataset_dbpediaentity_hnswlib, dataset_dbpediaentity_faiss

from auto_tuner.constants import DEFAULT_M as M
from auto_tuner.constants import DEFAULT_EFC as efC
from auto_tuner.constants import DEFAULT_PARAMS as PARAMS

#### START OF MAIN FUNCTIONS ####
## nytimes
def main_nytimes_hnswlib():
    run_hnsw_config(dataset_nytimes_hnswlib, "hnswlib", M=M, efC=efC, params=PARAMS, prefix="main_nytimes_hnswlib")

def main_nytimes_faiss():
    run_hnsw_config(dataset_nytimes_faiss, "faiss", M=M, efC=efC, params=PARAMS, prefix="main_nytimes_faiss")

## glove    
def main_glove_hnswlib():
    run_hnsw_config(dataset_glove_hnswlib, "hnswlib", M=M, efC=efC, params=PARAMS, prefix="main_glove_hnswlib")
    
def main_glove_faiss():
    run_hnsw_config(dataset_glove_faiss, "faiss", M=M, efC=efC, params=PARAMS, prefix="main_glove_faiss")

## sift
def main_sift_hnswlib():
    run_hnsw_config(dataset_sift_hnswlib, "hnswlib", M=M, efC=efC, params=PARAMS, prefix="main_sift_hnswlib")

def main_sift_faiss():
    run_hnsw_config(dataset_sift_faiss, "faiss", M=M, efC=efC, params=PARAMS, prefix="main_sift_faiss")

## youtube
def main_youtube_hnswlib():
    run_hnsw_config(dataset_youtube_hnswlib, "hnswlib", M=M, efC=efC, params=PARAMS, prefix="main_youtube_hnswlib")
    
def main_youtube_faiss():
    run_hnsw_config(dataset_youtube_faiss, "faiss", M=M, efC=efC, params=PARAMS, prefix="main_youtube_faiss")

## msmarco 
def main_msmarco_hnswlib():
    run_hnsw_config(dataset_msmarco_hnswlib, "hnswlib", M=M, efC=efC, params=PARAMS, prefix="main_msmarco_hnswlib")

def main_msmarco_faiss():
    run_hnsw_config(dataset_msmarco_faiss, "faiss", M=M, efC=efC, params=PARAMS, prefix="main_msmarco_faiss")

## dbpediaentity
def main_dbpediaentity_hnswlib():
    run_hnsw_config(dataset_dbpediaentity_hnswlib, "hnswlib", M=M, efC=efC, params=PARAMS, prefix="main_dbpediaentity_hnswlib")
    
def main_dbpediaentity_faiss():
    run_hnsw_config(dataset_dbpediaentity_faiss, "faiss", M=M, efC=efC, params=PARAMS, prefix="main_dbpediaentity_faiss")
