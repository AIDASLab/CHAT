# Benchmark Scripts

Run these scripts from the repository root after generating ground-truth CSVs.

1. Edit experiment lists in `scripts/run_main.py` or `scripts/run_milvus.py`.
2. Run the selected benchmark script.
3. Run `python results/postprocess.py` to aggregate saved results and plots.

Ground-truth CSVs must exist at `data/ground_truths/{impl}/{dataset}.csv`.
