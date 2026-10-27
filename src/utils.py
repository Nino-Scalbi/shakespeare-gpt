import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.dataset import get_batch

def get_save_steps(max_steps, n_steps):
    save_steps = set(
        np.geomspace(1, max_steps, num=n_steps).astype(int)
    )
    save_steps.update({0, max_steps})
    return save_steps



def get_paths(cwd, run_id):
    paths = {
        "dataset": cwd / "dataset",
        "run": cwd / "run_id",
        "weights": cwd / "run_id" / "weights"
    }
    return paths


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


@torch.no_grad()
def estimate_loss(model, data, config, device, eval_iters):
    out = {}
    was_training = model.training
    model.eval()   # Turn the model into evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split, data=data, config=config)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(was_training)
    return out