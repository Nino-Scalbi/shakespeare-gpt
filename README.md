# Shakespeare-GPT

A small character-level GPT trained on the works of Shakespeare. I use it to explore how Transformer training
behaves: the loss landscape, how different solutions are connected, parameter symmetries, and learned
representations. The model is kept deliberately small so that many runs can be trained and compared, and training
saves weights and metrics so that they can be analysed afterwards.

## Credits

The model and the core training loop come from Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course, specifically the lecture
[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
(code: [ng-video-lecture](https://github.com/karpathy/ng-video-lecture), see also
[nanoGPT](https://github.com/karpathy/nanoGPT)). The architecture follows the lecture: a decoder-only Transformer
with multi-head causal self-attention, pre-LayerNorm residual blocks and a ReLU feed-forward layer.

This repository adapts that code so that training runs can be analysed afterwards:

- **Configurable runs:** model and training settings come from YAML configs. Each run gets its own folder with the
  resolved config and metadata (seed, git commit, library versions, device, hash of the training data).
- **Reproducibility:** runs are seeded, training batches use a dedicated random generator, and evaluation uses
  fixed, non-overlapping windows, so the logged losses are deterministic.
- **Weight snapshots:** weights are saved at geometrically spaced steps (dense early in training, sparse later),
  starting from the untrained model at step 0.
- **Metrics logging:** batch loss, gradient norm, train and validation loss and learning rate are written to
  parquet.
- **Reload check:** a script reloads saved snapshots and checks that they reproduce the logged losses.

## Model

With the default config (`configs/default_config.yaml`) the model has 6 Transformer blocks, 6 attention heads, an
embedding size of 384 and a context length of 256 characters, about **10.8M parameters** in total. The vocabulary
consists of the 65 distinct characters in the text.

## Setup

Requires Python 3 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

**Train**

```bash
python -m src.train --seed 0                       # default config, run id = timestamp
python -m src.train --seed 0 --run_id baseline     # named run
```

| Argument | Default | Description |
| --- | --- | --- |
| `--seed` | required | Seeds `torch`, `numpy`, `random` and the training-batch generator. |
| `--run_id` | `YYYYMMDD-HHMM` timestamp | Name of the run folder under `runs/`. The run fails if the folder already exists. |
| `--config` | `configs/default_config.yaml` | YAML config with `model` and `training` sections. |

At the end of training the script prints 500 generated characters.

**Smoke test**

```bash
python -m src.train --seed 0 --run_id smoke --config configs/test_config.yaml
```

Trains a tiny model for 20 steps, which takes a few seconds on a CPU. Use it to check that the whole pipeline
(training, logging, snapshots) still works after a change.

**Check a run**

```bash
python -m scripts.check_reload runs/<run_id>
```

Reloads the first, middle and last snapshot of a run, recomputes the train and validation losses and compares them
with the logged values. Exits with a non-zero status if any of them differ.

## Configuration

- `model`: `batch_size`, `block_size` (context length), `n_embd`, `n_heads`, `dropout`, `n_layer`
- `training`: `max_iters`, `log_interval`, `save_interval`, `learning_rate`, `weight_decay`, `eval_max_windows`,
  `train_ratio`, `n_save_steps`

`n_save_steps` is the number of snapshots after step 0. They are all distinct and the last one is at `max_iters`, so
a run has `n_save_steps + 1` weight files. `eval_max_windows` caps the number of evaluation windows per split
(`null` evaluates the full split).

## Outputs

```
runs/<run_id>/
  config.yaml          # resolved config, including model.vocab_size (enough to rebuild the model)
  metadata.json        # seed, git commit and dirty flag, torch version, device, SHA-256 of the data
  metrics.parquet      # step, batch_loss, grad_norm, train_eval_loss, val_loss, learning_rate
  weights/
    step_000000.pt     # {"step": int, "model": state_dict}
    ...
```

In each metrics row, `batch_loss` and `grad_norm` are measured on the weights before that step's update, while
`train_eval_loss` and `val_loss` are measured on the weights after it, which are the ones saved as the snapshot.
At step 0, `batch_loss` and `grad_norm` are empty.

A snapshot can be reloaded with:

```python
from src.model import load_snapshot
model = load_snapshot("runs/<run_id>", step=0, device="cpu")
```

## Project layout

```
configs/    YAML configs (default and smoke test)
data/       input.txt (training text)
scripts/    check_reload.py
src/        dataset.py, model.py, train.py, utils.py
```

## Status

Work in progress. The training side is complete: runs are seeded and reproducible, and snapshots reproduce the
logged losses. Next, I want to use these runs to look at the loss landscape and the learned representations.
