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
python -m src.train --seed 0                                  # default config, run id = timestamp
python -m src.train --seed 0 --run_id baseline                # named run
python -m src.train --seed 0 --run_id smoke --config configs/test_config.yaml   # quick smoke test
```

| Argument | Default | Description |
| --- | --- | --- |
| `--seed` | required | Seeds `torch` / `numpy` / `random` and the training-batch generator. |
| `--run_id` | `YYYYMMDD-HHMM` timestamp | Name of the run folder under `runs/`. Free-form; the run fails if the folder already exists. |
| `--config` | `configs/default_config.yaml` | YAML config with `model` and `training` sections. |

The smoke-test config trains a tiny model for 20 steps and finishes in seconds on a CPU. Use it to check that
changes still produce weights and metrics. In VS Code, the "Train" and "Smoke Test" launch configurations in
`.vscode/launch.json` do the same.

At the end of training the script prints 500 generated characters.

## Configuration

`configs/default_config.yaml`:

- `model`: `batch_size`, `block_size` (context length), `n_embd`, `n_heads`, `dropout`, `n_layer`
- `training`: `max_iters`, `log_interval`, `save_interval`, `learning_rate`, `eval_max_windows`, `train_ratio`,
  `n_save_steps`

`n_save_steps` is the number of geometrically spaced snapshots after step 0. All are distinct and the last one is
`max_iters`, so a run has `n_save_steps + 1` weight files. `eval_max_windows` caps the number of fixed evaluation
windows (`null` = full pass over the validation split).

## Outputs

```
runs/<run_id>/
  config.yaml                # config actually used, including model.vocab_size
  metadata.json              # seed, git commit + dirty flag, torch version, device, SHA-256 of data/input.txt
  metrics.parquet            # step, batch_loss, grad_norm, train_eval_loss, val_loss, learning_rate
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

Work in progress. Training is seeded, evaluation is deterministic, and each run folder records its config and
metadata. Next: reloading snapshots and reproducing the logged losses. See
`reports/shakespeare_gpt_development_plan.md`.
