import sys
import os
import csv
import time
import yaml 
from auto_tuner.constants import DEFAULT_EFC, DEFAULT_M, DEFAULT_EFS

LOG_FILENAME = "hp_check.log"
os.remove(LOG_FILENAME) if os.path.exists(LOG_FILENAME) else None

def merge_files_and_save(
    root_dir: str,
    filename_contains: str,
    header: str,
    exclude_line_contains: str,
    output_dir: str,
    output_file: str,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = f"{output_dir}/{output_file}"
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(header.split(","))
        for dirpath, _, filenames in os.walk(root_dir):
            if "__" in dirpath:
                continue
            for fname in filenames:
                if fname.endswith('.csv') and filename_contains in fname:
                    print(f"Processing file: {fname}")
                    file_path = os.path.join(dirpath, fname)
                    with open(file_path, 'r', encoding='utf-8') as in_file:
                        print(f"{output_file} => {file_path}")
                        csv_reader = csv.reader(in_file)
                        for row in csv_reader:
                            if exclude_line_contains in row[0]:
                                continue
                            if len(row) == 15:
                                csv_writer.writerow(row[:6+1] + row[12:13] + row[7:12])
                            elif len(row) == 14:
                                csv_writer.writerow(row[:6+1] + [0.0] +row[7:12])
                            elif len(row) == 13:
                                csv_writer.writerow(row[:])
                            else:
                                raise ValueError(f"[{file_path} : Row length is not 15: {row}")

def check_ground_truth_hp(filename: str):
    hp_check = dict()
    inner_hp_check = dict()
    target = "_".join(filename.split("/")[-1].split("_")[1:3])
    hp_check[target] = inner_hp_check
    inner_hp_check["Recompute"] = 0
    print(f"filename: {filename}")
    for m in DEFAULT_M:
        inner_hp_check[m] = {}
        for efc in DEFAULT_EFC:
            inner_hp_check[m][efc] = 2  # 기본값 2
            inner_hp_check["Recompute"] += 1

    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        _header = next(csv_reader)
        for row in csv_reader:
            m = int(row[4])
            efc = int(row[5])
            if m not in inner_hp_check:
                inner_hp_check[m] = {}
            if efc not in inner_hp_check[m]:
                inner_hp_check[m][efc] = 2
            inner_hp_check[m][efc] = 1      # Seen
            if 0 + 1e-6 < int(row[6]):
                inner_hp_check[m][efc] = 0  # 0 < IndexSize <=> Perfect
                inner_hp_check["Recompute"] -= 1

    with open(LOG_FILENAME, "a", encoding="utf-8") as log_file:
        def log_print(content=""):
            print(content)
            print(content, file=log_file)
            num_tab = content.count("\t")
            if 1 < num_tab:
                print("----" * (num_tab + 1), file=log_file)
            else:
                print("", file=log_file)

        log_print(f"\nTarget: {target}")
        log_print(f"* 0 : Perfect / 1 : IndexSize unknown / 2 : Not computed")
        efc_list = DEFAULT_EFC
        header_row = "M\\C\t" + "\t".join(str(efc) for efc in efc_list)
        log_print(header_row)

        for m in sorted(k for k in inner_hp_check.keys() if isinstance(k, int)):
            row = [str(m)]
            for efc in efc_list:
                val = inner_hp_check[m].get(efc, 0)
                row.append(str(val))
            log_print("\t".join(row))
        log_print(f"The number of Recomputing: {inner_hp_check['Recompute']}")



TARGET_FILES = [
    # nytimes-256-angular
    ("nytimes_hnswlib_RQ_nytimes-256-angular", "main_nytimes_hnswlib_RQ_nytimes-256-angular_0000_000000.csv"),
    ("nytimes_faiss_RQ", "main_nytimes_faiss_RQ_nytimes-256-angular_0000_000000.csv"),
    # glove-100-angular
    ("glove_hnswlib_RQ_glove-100-angular", "main_glove_hnswlib_RQ_glove-100-angular_0000_000000.csv"),
    ("glove_faiss_RQ", "main_glove_faiss_RQ_glove-100-angular_0000_000000.csv"),
    # sift-128-euclidean
    ("sift_hnswlib_RQ_sift-128-euclidean", "main_sift_hnswlib_RQ_sift-128-euclidean_0000_000000.csv"),
    ("sift_faiss_RQ", "main_sift_faiss_RQ_sift-128-euclidean_0000_000000.csv"),
    # youtube-1024-angular
    ("youtube_hnswlib_RQ", "main_youtube_hnswlib_RQ_youtube-1024-angular_0000_000000.csv"),
    ("youtube_faiss_RQ", "main_youtube_faiss_RQ_youtube-1024-angular_0000_000000.csv"),
    # msmarco-384-angular
    ("msmarco_hnswlib_RQ", "main_msmarco_hnswlib_RQ_msmarco-384-angular_0000_000000.csv"),
    ("msmarco_faiss_RQ", "main_msmarco_faiss_RQ_msmarco-384-angular_0000_000000.csv"),
    # dbpediaentity-768-angular
    ("dbpediaentity_hnswlib_RQ", "main_dbpediaentity_hnswlib_RQ_dbpediaentity-768-angular_0000_000000.csv"),
    ("dbpediaentity_faiss_RQ", "main_dbpediaentity_faiss_RQ_dbpediaentity-768-angular_0000_000000.csv"),
]

ROOT_DIR = "./results"
EXCLUDE_LINE_PATTERN = "Implementation"
EXCLUDE_FILE_PATTERN = "hnswlib" #! <-- Exclude file pattern
HEADER="Implementation,Dataset,Recomputed,Batch,M,efC,efS,IndexSize,BuildTime,TestIime,SearchTimes,Recall,QPS"
OUTPUT_DIR = "./gd-faiss-2"   #! <-- Output directory

for target_file in TARGET_FILES:
    if EXCLUDE_FILE_PATTERN in target_file[1]:
        continue
    output_dir = OUTPUT_DIR
    if len(sys.argv) == 1:
        _suffix = time.strftime("_%m%d%H%M")
        output_dir = output_dir + _suffix
    filename_contains, output_file = target_file
    merge_files_and_save(
        ROOT_DIR,
        filename_contains,
        HEADER,
        EXCLUDE_LINE_PATTERN,
        output_dir,
        output_file,
    )
    check_ground_truth_hp(f"{output_dir}/{output_file}")