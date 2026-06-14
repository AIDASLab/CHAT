import os
import psutil
import sys
from auto_tuner.scripts import summary_from_dir, summary_from_csv
from auto_tuner.scripts.main import \
    main_nytimes_hnswlib, main_nytimes_faiss,\
    main_glove_hnswlib, main_glove_faiss,\
    main_sift_hnswlib, main_sift_faiss,\
    main_youtube_hnswlib, main_youtube_faiss, \
    main_msmarco_hnswlib, main_msmarco_faiss,\
    main_dbpediaentity_hnswlib, main_dbpediaentity_faiss
from auto_tuner.constants import MAX_THREADS
p = psutil.Process(os.getpid())
p.cpu_affinity(list(range(MAX_THREADS)))

####
def func17():
    from auto_tuner.scripts.test import test_msmarco_hnswlib, test_msmarco_faiss
    test_msmarco_hnswlib()
    test_msmarco_faiss()

def func16():
    from auto_tuner.scripts import summary_from_csv
    summary_from_csv("test_msmarco_hnswlib_RQ_msmarco-384-angular_0408_115258", "./results/results-0408")

def func15():
    from auto_tuner.scripts.test import test_sift_milvus
    test_sift_milvus()

def func14():
    from auto_tuner.scripts.test import test_sift_weaviate
    test_sift_weaviate()

def func13():
    main_msmarco_faiss()

def func12():
    from auto_tuner.models.hnsw_config import HnswConfigWeaviate
    from auto_tuner.dataset import DatasetNytimes
    config = HnswConfigWeaviate(DatasetNytimes(impl="hnswlib"), 16, 128, batch=True)
    config._build()

def func11():
    main_dbpediaentity_hnswlib()

def func10():
    from auto_tuner.scripts.test import test_dbpediaEntity_hnswlib
    # test_dbpediaEntity_hnswlib()
    summary_from_dir("./results/gd")

def func9():
    from auto_tuner.scripts.test import test_msmarco_hnswlib
    test_msmarco_hnswlib()

def func8():
    from auto_tuner.scripts.main import main_sift_faiss_recompute
    main_sift_faiss_recompute()

def func7():
    from auto_tuner.scripts.test import test_glove_faiss
    test_glove_faiss()

def func6():
    from auto_tuner.scripts.test import test_glove_ip_faiss
    test_glove_ip_faiss()

def func5():
    summary_from_csv("main_sift_faiss_RQ_sift-128-euclidean_0316_162525.csv", "./results/results-0316")

def func4():
    from auto_tuner.scripts.main import main_sift_hnswlib_recompute
    main_sift_hnswlib_recompute()

def func3():
    from auto_tuner.scripts.main import main_glove_faiss_recompute, main_glove_hnswlib_recompute
    main_glove_faiss_recompute()
    # main_glove_hnswlib_recompute()

def func2():
    dir = "./results/results-0316"
    summary_from_dir(dir) 
 
def func1():
    opt = sys.argv[1]
    if opt == "1":
        main_nytimes_hnswlib()
    elif opt == "2":
        main_nytimes_faiss()
    elif opt == "3":
        main_glove_hnswlib()
    elif opt == "4":
        main_glove_faiss()
    elif opt == "5":
        main_sift_hnswlib()
    elif opt == "6":
        main_sift_faiss()
    elif opt == "7":
        main_youtube_hnswlib()
    elif opt == "8":
        main_youtube_faiss()
    else:
        print("Invalid option")

if __name__ == "__main__":
    func15()