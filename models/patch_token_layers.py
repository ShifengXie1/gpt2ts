import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    def __init__(self, patch_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, patches):
        return self.net(patches)


class PatchProjector(nn.Module):
    def __init__(self, patch_dim, hidden_dim, gpt_dim):
        super().__init__()
        self.encoder = PatchEncoder(patch_dim, hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, gpt_dim),
            nn.LayerNorm(gpt_dim),
        )

    def forward(self, flat_patches):
        return self.projector(self.encoder(flat_patches))


class PatchBankDecoder(nn.Module):
    def __init__(self, gpt_dim, patch_dim, attn_dim, use_attention=True):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.query = nn.Linear(gpt_dim, attn_dim)
        self.key = nn.Linear(patch_dim, attn_dim)

    def forward(self, token_prob, hidden_state, token_patch_bank):
        # token_prob: [B,T,V], hidden_state: [B,T,D], token_patch_bank: [V,K,P]
        if not self.use_attention:
            token_base = token_patch_bank.mean(dim=1)
            return torch.matmul(token_prob, token_base)

        query = self.query(hidden_state)
        key = self.key(token_patch_bank)
        scores = torch.einsum("btd,vkd->btvk", query, key) / math.sqrt(max(query.shape[-1], 1))
        weights = torch.softmax(scores, dim=-1)
        token_base = torch.einsum("btvk,vkp->btvp", weights, token_patch_bank)
        return torch.einsum("btv,btvp->btp", token_prob, token_base)


class ResidualHead(nn.Module):
    def __init__(self, gpt_dim, patch_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gpt_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden_state):
        return self.net(hidden_state)


def l2_normalize(x):
    return F.normalize(x, dim=-1, eps=1e-12)
