import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import (get_batch,
                         get_windows
                         )


def get_save_steps(max_steps, n_steps):
    """Geometrically spaced snapshot steps: step 0 plus exactly n_steps distinct steps in [1, max_steps].
    The last step is always max_steps.
    """
    if n_steps > max_steps:
        raise ValueError(f"n_steps ({n_steps}) cannot exceed max_steps ({max_steps})")
    save_steps = {0}
    prev = 0
    for value in np.geomspace(1, max_steps, num=n_steps):
        step = max(int(round(value)), prev + 1)
        save_steps.add(step)
        prev = step
    return save_steps


def get_paths(cwd, run_id):
    cwd = Path(cwd)
    run_dir = cwd / "runs" / run_id
    paths = {
        "dataset": cwd / "dataset",
        "run": run_dir,
        "weights": run_dir / "weights",
        "metrics": run_dir / "metrics.parquet"
    }
    return paths


def _git(*args):
    """Output of a git command, or None if git is unavailable (e.g. no .git on the server)."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run_info(run_dir, config, seed, device, data_path):
    """Write the resolved config.yaml and a small metadata.json into run_dir."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    metadata = {
        "seed": seed,
        "git_commit": commit,
        "git_dirty": None if status is None else status != "",
        "torch_version": torch.__version__,
        "device": device,
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MetricsLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Load previous metrics if resuming training
        if self.path.exists():
            self.metrics = pd.read_parquet(self.path).to_dict("records")
        else:
            self.metrics = []

    def log(self, **metrics):
        self.metrics.append(metrics)

    def save(self):
        tmp_path = self.path.with_suffix(".tmp")

        df = pd.DataFrame(self.metrics)
        df = df.drop_duplicates(subset="step", keep="last")
        df.to_parquet(tmp_path, index=False)

        os.replace(tmp_path, self.path)


# Fixed seed for the evaluation batches. It is independent of the run seed so that
# different runs are evaluated on the same data for reproducibility.
EVAL_SEED = 1234


@torch.no_grad()
def estimate_loss(model, data, config, device, eval_iters):
    out = {}
    was_training = model.training
    model.eval()   # Turn the model into evaluation mode
    # New generator such that each evaluation sees the same batches
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split, data=data, config=config, generator=eval_gen)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out


@torch.no_grad()
def evaluate(model, data, starts, block_size, batch_size, device):
    """Token-weighted mean loss of the model on fixed windows of one split.

    Args:
        model: The GPT model. It is put in eval mode and restored afterwards.
        data: 1-D tensor of token ids for ONE split (train or val).
        starts: 1-D tensor of window start indices into data, e.g. from make_eval_starts.
        block_size: Number of tokens per window.
        batch_size: Number of windows per forward pass. Does not change the result.
        device: Device to run the model on.

    Returns:
        Mean cross-entropy per predicted token, as a float.
    """
    was_training = model.training
    model.eval()   # Turn the model into evaluation mode

    total_loss = 0.0
    total_tokens = 0
    for batch_starts in starts.split(batch_size):
        X, Y = get_windows(data, batch_starts, block_size)
        X, Y = X.to(device), Y.to(device)
        _, batch_loss = model(X, Y)
        # batch_loss is a mean over the batch's tokens: weight it by the token count
        total_loss += batch_loss.item() * Y.numel()
        total_tokens += Y.numel()

    model.train(was_training)
    return total_loss / total_tokens