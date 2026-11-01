# Shakespeare-GPT

A small character-level GPT trained on the Shakespeare text in `data/input.txt`. The project uses this deliberately
small model to study the geometry of Transformer training: the loss landscape, connectivity between solutions,
parameter symmetries, and learned representations. Training therefore saves weight snapshots and metrics for later
offline analysis.

## Setup

```bash
conda activate s-gpt        # or any Python environment of your choice
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python -m src.train                                # default config, run id = timestamp
python -m src.train --run_id baseline              # named run
python -m src.train --config configs/test_config.yaml --run_id smoke   # quick smoke test
```

| Argument | Default | Description |
| --- | --- | --- |
| `--run_id` | `YYYYMMDD-HHMM` timestamp | Name of the run folder under `runs/`. The run fails if the folder already exists. |
| `--config` | `configs/default_config.yaml` | YAML config with `model` and `training` sections. |

The smoke-test config trains a tiny model for 20 steps and finishes in seconds on a CPU. Use it to check that
changes still produce weights and metrics. In VS Code, the "Train" and "Smoke Test" launch configurations in
`.vscode/launch.json` do the same.

At the end of training the script prints 500 generated characters.

## Configuration

`configs/default_config.yaml`:

- `model`: `batch_size`, `block_size` (context length), `n_embd`, `n_heads`, `dropout`, `n_layer`
- `training`: `max_iters`, `log_interval`, `save_interval`, `learning_rate`, `eval_iters`, `train_ratio`,
  `n_save_steps`

`n_save_steps` is the number of geometrically spaced steps at which weights are saved. Step 0 (the untrained model)
and the final step are always included. Rounding can merge the earliest steps, so the actual number of snapshots can
be smaller.

## Outputs

```
runs/<run_id>/
  metrics.parquet            # step, train_loss, val_loss, grad_norm, learning_rate
  weights/
    step_000000.pt           # {"step": int, "model": state_dict}
    step_000001.pt
    ...
```

Read the metrics with pandas:

```python
import pandas as pd
pd.read_parquet("runs/<run_id>/metrics.parquet")
```

`runs/` is git-ignored.

## Project layout

```
configs/    YAML configs (default and smoke test)
data/       input.txt (training text)
src/        dataset.py, model.py, train.py, utils.py
reports/    research scope and development plan
```

## Status

This is work in progress. Training and artifact saving work, but the evaluation is not yet deterministic, runs are not
seeded, and there is no resume support or run metadata. See `reports/shakespeare_gpt_development_plan.md` for the
planned work.
