import yaml
import torch
from src.model import GPT
from src.dataset import get_batch

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load config
with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

max_iters = config["training"]["max_iters"]
eval_interval = config["training"]["eval_interval"]
learning_rate = config["training"]["learning_rate"]
eval_iters = config["training"]["eval_iters"]



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()   # Turn the model into evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out




model = GPT(config['model'])
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


# Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of the data
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()