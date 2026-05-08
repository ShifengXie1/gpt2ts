import torch.nn as nn


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
