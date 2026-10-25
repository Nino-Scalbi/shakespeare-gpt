import torch
import torch.nn as nn
from torch.nn import functional as F
from dataset import vocab_size


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, config, head_size):
        super().__init__()
        n_embd = config["n_emnd"]
        block_size = config["block_size"]
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute the attention scores also called affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @  (B, head_size, T) = (B, T, T) the dot product between the keys and the queries
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # lower trinagular matric where evrything on the left of the diagonal is 0 and right -inf
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out



class MultiHeadAttention(nn.Module):
    """multiple parallel heads of self-attention which act independently"""

    def __init__(self, config, head_size):
        super().__init__()
        n_embd = config["n_emnd"]
        n_heads = config["n_heads"]
        self.heads = nn.ModuleList([Head(config=config, head_size=head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1) # Concatenate all the outputs of the different heads into a single tensor along the channel dimension
        out = self.proj(out) # project down to the residual pathway due to residual connection
        return out



class FeedForward(nn.Module):
    """A simple linear layer followed by a ReLu non-linearity"""

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_heads * n_embd), # The 4 * ... is due to the paper (Attention is all you need)
            nn.ReLU(),
            nn.Linear(n_heads * n_embd, n_embd), # Projection Layer down to the residual pathway
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    """Transformer Block: communication followed by computation"""

    def __init__(self, config):
        super().__init__()
        n_embd = config["n_emnd"]
        n_heads = config["n_heads"]
        head_size = n_embd // n_heads  # creating mutltiple parallel channels of self-attention instead of a single one of n_embd attention
        self.sa = MultiHeadAttention(config=config, head_size=head_size)
        self.ffwd = FeedForward(n_embd, n_heads=n_heads)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        n_embd = config["n_embd"]
        self._block_size = config["block_size"]
        # each token directly reads off the logits fro the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Embedding also the possible position (block_size)
        self.position_embedding_table = nn.Embedding(self._block_size, n_embd)
        # 4 Transformer Blocks
        self.blocks = nn.Sequential(
            Block(config),
            Block(config),
            Block(config),
            nn.LayerNorm(n_embd),

        )
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C = n_embed) -> B = Batches, T = Block Size/Timesteps, C = Channels (Embedding dimension)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T. C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Logits want B*T, C

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop ids to the last block_size tokens
            idx_cond = idx[:, -self._block_size:]
            # get the predicions
            logits, loss = self(idx_cond) # logits (B, T, C)
            # focus only on the last timestep
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx