# Ground Truths Generator

This directory generates the CSV files consumed by the main CHAT benchmark.

The expected output contract is:

```text
../data/ground_truths/{impl}/{dataset}.csv
```

Use the CLI entrypoint:

```bash
python main.py list
python main.py run --dataset nytimes-256-angular --impl hnswlib
python main.py run --dataset sift-128-euclidean --impl faiss --M 16,32 --efC 64,128
python main.py summary-dir ./results/results-0614 --include hnswlib nytimes
```

For `hnswlib` and `faiss`, the CLI recomputes exact neighbors by default. For
server-backed implementations such as Milvus, it uses the neighbors embedded in
the dataset file unless `--recompute-groundtruth` is explicitly provided.
