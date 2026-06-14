"""Command line entrypoint for generating ground-truth HNSW CSV files.

Typical flow:
  1. Put the raw benchmark datasets under DATA_DIR, or export DATA_DIR.
  2. Run this file to generate backend-specific RQ/SCORE CSV files.
  3. Copy or symlink the final RQ CSVs into ../data/ground_truths/{impl}/{dataset}.csv.
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

DATASET_NAMES = (
    "dbpediaentity-768-angular",
    "deep1M-256-angular",
    "glove-100-angular",
    "msmarco-384-angular",
    "nytimes-256-angular",
    "sift-128-euclidean",
    "youtube-1024-angular",
)
IMPL_NAMES = ("faiss", "hnswlib", "milvus", "weaviate")
GROUND_TRUTH_IMPLS = {"hnswlib", "faiss"}


def _parse_int_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_param_pairs(value):
    """Parse CLI values like '16:64,32:128' into [(16, 64), (32, 128)]."""
    if not value:
        return []
    pairs = []
    for item in value.split(","):
        left, right = item.split(":", maxsplit=1)
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs


def _set_cpu_affinity():
    """Keep experiment threads on the same CPU set used by the original runs."""
    try:
        import psutil
        from auto_tuner.constants import MAX_THREADS

        process = psutil.Process(os.getpid())
        process.cpu_affinity(list(range(MAX_THREADS)))
    except Exception as exc:
        print(f"Warning: failed to set CPU affinity: {exc}")


def _default_prefix(dataset, impl):
    short_dataset = dataset.split("-")[0].replace("_", "")
    return f"main_{short_dataset}_{impl}"


def run_command(args):
    from auto_tuner.constants import DEFAULT_EFC, DEFAULT_M, DEFAULT_PARAMS
    from auto_tuner.dataset import dataset_mapping
    from auto_tuner.scripts import run_hnsw_config

    recompute = args.recompute_groundtruth
    if recompute is None:
        recompute = args.impl in GROUND_TRUTH_IMPLS
    if recompute and args.impl not in GROUND_TRUTH_IMPLS:
        raise SystemExit("--recompute-groundtruth is supported only for hnswlib/faiss")

    _set_cpu_affinity()
    dataset = dataset_mapping[args.dataset](
        impl=args.impl,
        recompute=recompute,
        k=args.k,
    )

    params = _parse_param_pairs(args.params)
    M = _parse_int_list(args.M) or DEFAULT_M
    efC = _parse_int_list(args.efC) or DEFAULT_EFC
    prefix = args.prefix or _default_prefix(args.dataset, args.impl)

    run_hnsw_config(
        dataset=dataset,
        impl=args.impl,
        M=M,
        efC=efC,
        params=params or DEFAULT_PARAMS,
        prefix=prefix,
        warmup=not args.no_warmup,
    )


def summary_csv_command(args):
    from auto_tuner.scripts import summary_from_csv

    summary_from_csv(args.filename, args.dir)


def summary_dir_command(args):
    from auto_tuner.scripts import summary_from_dir

    summary_from_dir(args.dir, include=args.include)


def list_command(_args):
    print("Datasets:")
    for dataset in DATASET_NAMES:
        print(f"  - {dataset}")
    print("\nImplementations:")
    for impl in IMPL_NAMES:
        print(f"  - {impl}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate and summarize HNSW ground-truth CSV files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run HNSW experiments")
    run_parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    run_parser.add_argument("--impl", required=True, choices=IMPL_NAMES)
    run_parser.add_argument("--M", default="", help="comma-separated M values; default uses DEFAULT_M")
    run_parser.add_argument("--efC", default="", help="comma-separated efConstruction values; default uses DEFAULT_EFC")
    run_parser.add_argument("--params", default="", help="comma-separated M:efC pairs; overrides --M/--efC grid")
    run_parser.add_argument("--prefix", default="", help="output filename prefix")
    run_parser.add_argument("--k", type=int, default=10, help="top-k ground-truth neighbors")
    run_parser.add_argument("--no-warmup", action="store_true", help="skip warmup build/search")
    run_parser.add_argument(
        "--recompute-groundtruth",
        dest="recompute_groundtruth",
        action="store_true",
        default=None,
        help="recompute exact neighbors with the selected backend",
    )
    run_parser.add_argument(
        "--use-file-groundtruth",
        dest="recompute_groundtruth",
        action="store_false",
        help="use neighbors embedded in the dataset file",
    )
    run_parser.set_defaults(func=run_command)

    csv_parser = subparsers.add_parser("summary-csv", help="summarize one RQ CSV")
    csv_parser.add_argument("filename")
    csv_parser.add_argument("dir")
    csv_parser.set_defaults(func=summary_csv_command)

    dir_parser = subparsers.add_parser("summary-dir", help="summarize matching RQ CSVs in a directory")
    dir_parser.add_argument("dir")
    dir_parser.add_argument("--include", nargs="*", default=[], help="filename patterns that must be present")
    dir_parser.set_defaults(func=summary_dir_command)

    list_parser = subparsers.add_parser("list", help="list supported datasets and implementations")
    list_parser.set_defaults(func=list_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
