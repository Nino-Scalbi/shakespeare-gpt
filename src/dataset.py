import torch
batch_size = 32
block_size = 8 

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



#split data into train and validation sets
def split_data(data, train_ratio=0.9):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_ratio*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


# data loading
def get_batch(split, train_data, val_data, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
