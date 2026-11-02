import torch


DATA_PATH = "data/input.txt"


def load_text(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    """Character vocabulary of text. Returns (vocab_size, encode, decode)."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # string -> list of integers
    decode = lambda l: "".join(itos[i] for i in l)  # list of integers -> string
    return len(chars), encode, decode


# split data into train and validation sets
def split_data(data, train_ratio=0.9):
    """Split a 1-D tensor of token ids into (train, val) by ratio."""
    n = int(train_ratio*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def make_eval_starts(data, block_size, max_windows=None):
    """Start indices of non-overlapping eval windows of length block_size.

    Args:
        data: 1-D tensor of token ids for one split.
        block_size: Number of tokens per window (the model's context length).
        max_windows: Optional cap. If smaller than the number of windows, an
            evenly spaced subset is returned. None returns all windows.

    Returns:
        1-D tensor of window start indices.
    """
    starts = torch.arange(start=0, end=len(data) - block_size, step=block_size)
    if max_windows is None or max_windows >= len(starts):
        return starts
    starts_idx = torch.linspace(start=0, end=len(starts) - 1, steps=max_windows, dtype=torch.long)
    starts = starts[starts_idx]
    return starts


def get_windows(data, starts, block_size):
    """Return non-overlapping windows of length block_size starting at indices in starts.

    Args:
        data: 1-D tensor of token ids for one split.
        starts: 1-D tensor of window start indices.
        block_size: Number of tokens per window (the model's context length).

    Returns:
        tuple of tensors (x, y) where x is the input windows and y is the target windows.
        Both have shape (num_windows, block_size).
    """
    x = torch.stack([data[i:i+block_size] for i in starts])
    y = torch.stack([data[i+1:i+block_size+1] for i in starts])
    return x, y

# data loading
def get_batch(split, data, config, generator=None):
    train_data, val_data = data
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ), generator=generator)
    x, y = get_windows(data, ix, block_size)
    return x, y


# data preparation
def prepare_data(config):
    """Splits, fixed eval windows and vocab, built from the config.

    train.py and scripts/check_reload.py both use this, so they always see the same windows.
    The train eval subset has as many windows as the validation set, evenly spaced over the train split.
    """
    text = load_text(DATA_PATH)
    vocab_size, encode, decode = build_vocab(text)
    train_data, val_data = split_data(torch.tensor(encode(text), dtype=torch.long),
                                      train_ratio=config["training"]["train_ratio"])
    block_size = config["model"]["block_size"]
    val_starts = make_eval_starts(val_data, block_size, max_windows=config["training"]["eval_max_windows"])
    train_starts = make_eval_starts(train_data, block_size, max_windows=len(val_starts))
    return train_data, val_data, train_starts, val_starts, vocab_size, decode
