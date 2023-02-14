## Get started

1. Extract mnist.zip to <repo_root>/data folder.
2. Setup an environment with pytorch + ipex/cuda.
3. Open `run_perf_eval.sh`, leave only relevant devices in `devices` list and run the script.
4. At the end of each log file, the resulting performance numbers are provided.
Example:
```
Final avg batch time: 0.080862466460352
Epoch time: 75.99071860313416
Top-1 accuracy: 0.5615
Val time: 1.9733185768127441
```