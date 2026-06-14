# How To Run Benchs
## 1. main experiment
1. Set `TUNING_BUDGET` and `SEED` at `src.constant`
2. Manipulate configs of `run_main.py` as you want
3. run `run_main.py`
4. run `n_seed_to_0.sh` with arg `{SEED}` to move created result into base result directory
5. In order to postprocess, run `python3 results/postprocess.py` after manipulating its configs
6. Done. Check the results