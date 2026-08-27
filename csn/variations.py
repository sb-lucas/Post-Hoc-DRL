import torch
import torch.nn as nn

class OneLayer(nn.Module):
    """Single linear projection with residual connection, LayerNorm, GELU.
    Initialized near-identity so training starts close to the frozen embedding."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()

        # near-identity init: small weights, zero bias
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=1e-3)

    def forward(self, x):
        delta = self.act(self.proj(self.norm(x)))
        return x + delta

class MLP(nn.Module):
    """2-layer MLP adapter with residual connection, LayerNorm, dropout.
    Initialized near-identity so training starts close to the frozen embedding."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        hidden_dim = dim
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # zero-init last layer -> module starts as identity, adapter learns a delta
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        delta = self.net(self.norm(x))
        return x + delta