from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyValueMemoryRetriever(nn.Module):
    def __init__(self, temperature=1.0, mode="straight_through", normalize=True):
        super().__init__()
        self.temperature = float(temperature)
        self.mode = str(mode)
        self.normalize = bool(normalize)
        if self.mode not in {"soft", "hard", "straight_through"}:
            raise ValueError(f"Invalid retrieval mode: {self.mode}")

    def forward(self, queries, keys, values):
        if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 4:
            raise ValueError("Expected queries [B,F,D], keys [B,N,D], and values [B,N,L,C].")
        if queries.shape[0] != keys.shape[0] or keys.shape[:2] != values.shape[:2]:
            raise ValueError("Query, key, and value batch/history dimensions must match.")

        score_queries = F.normalize(queries.float(), dim=-1) if self.normalize else queries.float()
        score_keys = F.normalize(keys.float(), dim=-1) if self.normalize else keys.float()
        temperature = max(self.temperature, 1e-6)
        scores = torch.matmul(score_queries, score_keys.transpose(-1, -2)) / temperature
        soft_weights = F.softmax(scores, dim=-1)
        indices = scores.argmax(dim=-1)

        if self.mode == "soft":
            weights = soft_weights
        else:
            hard_weights = F.one_hot(indices, num_classes=keys.shape[1]).to(dtype=soft_weights.dtype)
            if self.mode == "straight_through":
                weights = hard_weights - soft_weights.detach() + soft_weights
            else:
                weights = hard_weights

        batch, history_count, patch_len, channels = values.shape
        flat_values = values.reshape(batch, history_count, patch_len * channels)
        pred_flat = torch.bmm(weights.to(dtype=flat_values.dtype), flat_values)
        pred_patches = pred_flat.reshape(batch, queries.shape[1], patch_len, channels)
        aux = SimpleNamespace(indices=indices, weights=weights, scores=scores)
        return pred_patches, aux


class OverlapAddPatchDecoder(nn.Module):
    def __init__(self, patch_len, stride, pred_len):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.pred_len = int(pred_len)

    def forward(self, patches):
        batch, patch_count, _, channels = patches.shape
        output = patches.new_zeros(batch, self.pred_len, channels)
        counts = patches.new_zeros(batch, self.pred_len, channels)

        for patch_idx in range(patch_count):
            start = patch_idx * self.stride
            if start >= self.pred_len:
                break
            end = min(start + self.patch_len, self.pred_len)
            width = end - start
            output[:, start:end, :] += patches[:, patch_idx, :width, :]
            counts[:, start:end, :] += 1

        return output / counts.clamp_min(1.0)
