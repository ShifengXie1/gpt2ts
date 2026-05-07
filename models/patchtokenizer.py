import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    def __init__(self, patch_len, stride, c_in, gpt_dim, dropout=0.05):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.c_in = int(c_in)
        patch_dim = self.patch_len * self.c_in
        self.encoder = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, gpt_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(gpt_dim, gpt_dim),
        )

    def patchify(self, x):
        if x.shape[1] < self.patch_len:
            pad_len = self.patch_len - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_len))
        else:
            remainder = (x.shape[1] - self.patch_len) % self.stride
            if remainder:
                x = F.pad(x, (0, 0, 0, self.stride - remainder))
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches

    def encode(self, patches):
        flat = patches.reshape(patches.shape[0], patches.shape[1], -1)
        return self.encoder(flat)
