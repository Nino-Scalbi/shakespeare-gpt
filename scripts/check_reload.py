"""Reload the first, middle and last snapshot of a run and check that they reproduce the logged eval losses.

Run from the repo root: python -m scripts.check_reload runs/<run_id>
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.dataset import prepare_data
from src.model import load_snapshot
from src.utils import evaluate

TOL = 1e-4  # also covers runs trained on GPU and checked on CPU

run_dir = Path(sys.argv[1])
config = yaml.safe_load(open(run_dir / "config.yaml"))
metrics = pd.read_parquet(run_dir / "metrics.parquet").set_index("step")
block_size = config["model"]["block_size"]
batch_size = config["model"]["batch_size"]

# Same data and eval windows as train.py
train_data, val_data, train_starts, val_starts, _, _ = prepare_data(config)
splits = {
    "val_loss": (val_data, val_starts),
    "train_eval_loss": (train_data, train_starts),
}

# First, middle and last saved snapshot
steps = sorted(int(p.stem.split("_")[-1]) for p in (run_dir / "weights").glob("step_*.pt"))
steps = sorted({steps[0], steps[len(steps) // 2], steps[-1]})

failed = False
for step in steps:
    model = load_snapshot(run_dir, step, "cpu")
    for column, (data, starts) in splits.items():
        loss = evaluate(model, data, starts, block_size, batch_size, "cpu")
        logged = metrics.loc[step, column]
        diff = abs(loss - logged)
        ok = diff <= TOL
        failed |= not ok
        print(f"step {step:6d}  {column:15s}  {loss:.6f}  logged {logged:.6f}  diff {diff:.1e}  {'OK' if ok else 'FAIL'}")

sys.exit(1 if failed else 0)
