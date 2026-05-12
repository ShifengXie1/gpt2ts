import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv1D(nn.Module):
    """LoRA adapter for GPT-2's Conv1D projections."""

    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.r = int(r)
        self.scaling = float(alpha) / max(self.r, 1)
        self.dropout = nn.Dropout(float(dropout))

        in_features = int(base_layer.weight.shape[0])
        out_features = int(base_layer.weight.shape[1])
        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = self.base_layer(x)
        if self.r <= 0:
            return base
        update = F.linear(self.dropout(x), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling
        return base + update
