# CHAT: Constraint-Aware HNSW Hyperparameter Tuning

Official artifact for **"Exploiting Structural Properties for Efficient
Constraint-Aware HNSW Hyperparameter Tuning"** (SIGMOD 2027).

> **TL;DR**: HNSW tuning is not an arbitrary three-dimensional black-box
> search. CHAT exploits HNSW-specific structure: `efSearch` induces a
> monotone feasibility boundary, `M` and `efConstruction` exhibit dominant
> unimodal trends under constraints, and build-time/index-size costs can be
> filtered before expensive index construction.

[Paper draft](contexts/_Revision___SIGMOD_2027__Exploiting_Structural_Properties_for_Efficient_Constraint_Aware_HNSW_Hyperparameter_Tuning.pdf)
| Venue: SIGMOD 2027
| Main entrypoints: `ground_truths_generator/main.py`, `scripts/run_main.py`

This repository contains the code needed to generate backend-specific HNSW
performance surfaces, run CHAT and competing tuners under the same constraints,
and aggregate the resulting tuning traces.

## Key Ideas

- **Boundary search over `efSearch`**: for a fixed HNSW graph, larger
  `efSearch` improves recall but increases query cost. CHAT searches the
  boundary value: the smallest feasible `efSearch` under a recall constraint,
  or the largest feasible `efSearch` under a QPS constraint.
- **Structure-aware search over `M` and `efConstruction`**: instead of sampling
  blindly, CHAT localizes promising construction settings using dominant
  unimodal trends observed in HNSW's constrained objective surface.
- **Resource-feasibility filtering**: build-time and index-size surrogates
  prune resource-infeasible `(M, efConstruction)` candidates before full index
  construction, while final selections still come from measured configurations.
- **API-level black-box design**: CHAT uses only standard build/query APIs and
  validation measurements such as recall, QPS, build time, and index size. It
  does not inspect adjacency lists, layers, or internal graph statistics.

## What This Artifact Supports

| Paper Component | Artifact Path |
| --- | --- |
| Ground-truth HNSW performance surface generation | `ground_truths_generator/` |
| Interpolated tuning-time simulator over generated CSVs | `data/ground_truths/` |
| CHAT implementation | `src/solutions/our_solution/` |
| Baselines: Oracle, Grid, Random, Optuna, NSGA-II, ECI, VDTuner | `src/solutions/` |
| Hnswlib/Faiss experiments | `scripts/run_main.py` |
| Milvus experiments | `scripts/run_milvus.py` |
| Aggregation and accumulated-performance plots | `results/postprocess.py`, `results/figures/` |

## Repository Structure

```text
.
|-- environment.yml                  # Main experiment environment
|-- data/
|   `-- ground_truths/               # Generated CSV loader and QPS target helper
|
|-- ground_truths_generator/         # Raw benchmark -> HNSW performance CSVs
|   |-- main.py                      # CLI: list, run, summary-csv, summary-dir
|   |-- main.sh                      # Thin wrapper around main.py
|   |-- environment.yml              # Generator/backend environment
|   `-- auto_tuner/                  # Backend adapters and CSV generation logic
|
|-- scripts/
|   |-- run_main.py                  # Paper matrix for hnswlib and Faiss
|   `-- run_milvus.py                # Paper matrix for Milvus
|
|-- src/
|   |-- constants.py                 # Search ranges, seed, tuning budget
|   |-- utils.py                     # Result saving, loading, plotting helpers
|   `-- solutions/                   # CHAT and baseline tuning algorithms
|
`-- results/
    |-- postprocess.py               # Aggregate saved traces
    `-- figures/                     # Figure builders and bundled fonts
```

## Experimental Matrix

The main paper experiments tune three HNSW parameters:

| Parameter | Range |
| --- | --- |
| `M` | `[4, 64]` |
| `efConstruction` / `efC` | `[8, 1024]` |
| `efSearch` / `efS` | `[10, 1024]` |

| Axis | Values |
| --- | --- |
| Backends | `hnswlib`, `faiss`, `milvus` |
| Paper datasets | `nytimes-256-angular`, `glove-100-angular`, `sift-128-euclidean`, `deep1M-256-angular`, `youtube-1024-angular` |
| Recall-constrained objective | maximize QPS subject to `Recall >= 0.95` |
| QPS-constrained objective | maximize Recall subject to `QPS >= q75` for the dataset/backend surface |
| Tuning budget | `4` hours per tuner/task |
| Seed and samples | `SEED = 42`, `sampling_count = 10` |

The generator also contains adapters for additional datasets used during
development, such as `msmarco-384-angular` and `dbpediaentity-768-angular`.
Use `bash ground_truths_generator/main.sh list` to see the exact local support
matrix.

## Ground-Truth CSV Policy

Generated ground-truth CSV files are intentionally **not** committed to this
repository. They are large, backend-specific, and in part machine-dependent.
The artifact expects reviewers and users to regenerate them with
`ground_truths_generator/`.

After generation, the main tuner consumes one CSV per backend and dataset at:

```text
data/ground_truths/{impl}/{dataset}.csv
```

For example:

```text
data/ground_truths/hnswlib/nytimes-256-angular.csv
data/ground_truths/faiss/glove-100-angular.csv
data/ground_truths/milvus/youtube-1024-angular.csv
```

The consumed CSV should be the generator's `RQ` output, whose rows contain
`M`, `efC`, `efS`, index size, build time, test/search time, recall samples, and
QPS samples. `data/ground_truths/ground_truth.py` loads this surface and
interpolates recall/QPS/resource values during tuning.

## Requirements

We recommend using two Conda environments: one for the main tuning experiments,
and one for the ground-truth generator. They can share the same machine, but
the split keeps backend dependencies easier to debug.

```bash
# Main CHAT experiments
conda env create -f environment.yml
conda activate tuning_env
```

```bash
# Ground-truth generation
conda env create -f ground_truths_generator/environment.yml
conda activate conda_env_2
```

Raw vector workloads should be available as HDF5 files under
`ground_truths_generator/data/`, or under a custom directory exported through
`DATA_DIR`:

```bash
export DATA_DIR=/path/to/hdf5/workloads
```

The default generator output directory is `ground_truths_generator/results/`.
Set `RESULTS_DIR` if you want generated CSVs on a larger disk:

```bash
export RESULTS_DIR=/path/to/generated/results
```

## Quick Start

The following commands run a compact sanity path for one backend/dataset pair.
Use this before launching the full paper matrix.

```bash
# 1. Confirm supported datasets and implementations.
bash ground_truths_generator/main.sh list

# 2. Generate a small hnswlib surface for nytimes.
#    Omit --M and --efC for the full paper-size surface.
bash ground_truths_generator/main.sh run \
  --dataset nytimes-256-angular \
  --impl hnswlib \
  --M 4,8,16,32,64 \
  --efC 16,64,128,256,512,1024

# 3. Install the generated RQ CSV at the path consumed by CHAT.
mkdir -p data/ground_truths/hnswlib
cp <generated_RQ_csv> data/ground_truths/hnswlib/nytimes-256-angular.csv

# 4. Run CHAT on the generated surface under Recall >= 0.95.
python -c "from src.solutions.our_solution.run import run; run(impl='hnswlib', dataset='nytimes-256-angular', recall_min=0.95, sampling_count=10)"
```

For the full surface, the generated file name has the form:

```text
ground_truths_generator/results/results-MMDD/main_<dataset>_<impl>_RQ_<dataset>_<timestamp>.csv
```

Copy or symlink that file into `data/ground_truths/{impl}/{dataset}.csv`.

## Reproducing the Main Experiments

1. Generate ground-truth CSVs for each backend/dataset pair.

```bash
# Hnswlib and Faiss paper surfaces
for impl in hnswlib faiss; do
  for dataset in nytimes-256-angular glove-100-angular sift-128-euclidean deep1M-256-angular youtube-1024-angular; do
    bash ground_truths_generator/main.sh run --dataset "$dataset" --impl "$impl"
  done
done
```

For Milvus, start the Milvus service required by your local setup, then generate
the corresponding `milvus` CSVs:

```bash
bash ground_truths_generator/main.sh run --dataset nytimes-256-angular --impl milvus --use-file-groundtruth
```

Repeat for the remaining paper datasets, then install the final `RQ` CSVs under
`data/ground_truths/milvus/`.

2. Run the hnswlib/Faiss experiment matrix.

```bash
python scripts/run_main.py
```

3. Run the Milvus experiment matrix.

```bash
python scripts/run_milvus.py
```

4. Aggregate traces and produce comparison plots.

```bash
python results/postprocess.py
```

## Outputs

Experiment traces and plots are written under `results/`:

| Output | Path Pattern |
| --- | --- |
| Raw tuning traces | `results/result/{seed}/{sampling_count}/{solution}/...csv` |
| Single-solution time plots | `results/timestamp/{seed}/{sampling_count}/{solution}/...png` |
| Accumulated best-so-far plots | `results/accumulated_timestamp/{seed}/{sampling_count}/...png` |
| Selected hyperparameters | `results/optimal_hyperparameters/{seed}/{sampling_count}/...csv` |

Each raw tuning trace stores:

```text
M, efC, efS, T_record, recall, qps, total_time, build_time, index_size
```

`brute_force` is the exhaustive-search Oracle used for normalization and
comparison. `our_solution` is CHAT.

## Customizing Runs

- Edit the benchmark matrix in `scripts/run_main.py` or `scripts/run_milvus.py`
  to run fewer datasets, fewer baselines, or a single constraint mode.
- Edit `src/constants.py` to change global ranges, seed, tolerance, or tuning
  budget.
- Use `data/ground_truths/get_qps_dataset.py` to inspect quantile-based QPS
  targets. By default, the paper scripts use `q75`.
- Use `sampling_count=1`, `3`, `5`, or `10` to study robustness to repeated
  measurement samples.

## Troubleshooting

| Symptom | Likely Cause and Fix |
| --- | --- |
| `Ground-truth CSV not found` | Generate the `RQ` CSV and place it at `data/ground_truths/{impl}/{dataset}.csv`. |
| `main.sh list` works, but `run` fails opening HDF5 | Set `DATA_DIR` or place raw HDF5 workloads under `ground_truths_generator/data/`. |
| QPS-constrained runs fail before starting | The q75 target is computed from the generated CSV, so the CSV must exist first. |
| Milvus run fails to connect | Start the local Milvus service and verify the `pymilvus` connection settings used by the backend adapter. |
| Full generation is very slow | Start with restricted `--M` and `--efC` ranges, then remove them for the full paper surface. |
| Optional backend imports fail | Use the generator Conda environment, or install only the backend dependencies needed for the selected `--impl`. |

## Citation

The final BibTeX entry should be updated after the SIGMOD 2027 proceedings
metadata is finalized.

```bibtex
@inproceedings{chat2027hnsw,
  title     = {Exploiting Structural Properties for Efficient Constraint-Aware
               HNSW Hyperparameter Tuning},
  booktitle = {Proceedings of the 2027 International Conference on Management
               of Data (SIGMOD)},
  year      = {2027}
}
```

## License

Please add the intended artifact license before public release.
