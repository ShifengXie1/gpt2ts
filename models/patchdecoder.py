import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryPatchDecoder(nn.Module):
    def __init__(self, patch_len, stride, pred_len, temperature=0.2, hard_lookup=False):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.pred_len = int(pred_len)
        self.temperature = max(float(temperature), 1e-6)
        self.hard_lookup = bool(hard_lookup)

    def forward(self, pred_embeds, history_embeds, history_patches):
        query = F.normalize(pred_embeds, dim=-1)
        key = F.normalize(history_embeds, dim=-1)
        logits = torch.matmul(query, key.transpose(1, 2)) / self.temperature

        if self.hard_lookup and not self.training:
            indices = logits.argmax(dim=-1)
            batch_ids = torch.arange(history_patches.shape[0], device=history_patches.device)[:, None]
            pred_patches = history_patches[batch_ids, indices]
        else:
            weights = F.softmax(logits, dim=-1)
            pred_patches = torch.einsum("bfh,bhpc->bfpc", weights, history_patches)

        return self._overlap_add(pred_patches)

    def _overlap_add(self, patches):
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
