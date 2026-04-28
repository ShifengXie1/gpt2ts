import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from transformers import AutoConfig, AutoModelForCausalLM

from .CasualTRM import CasualTRM
from .RevIN import RevIN


def _kmeans(x, k, iters=25, seed=2024):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    init_ids = torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]
    centers = x[init_ids].clone()

    for _ in range(max(1, int(iters))):
        distances = torch.cdist(x, centers)
        labels = distances.argmin(dim=1)
        new_centers = centers.clone()
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new_centers[idx] = x[mask].mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    distances = torch.cdist(x, centers)
    labels = distances.argmin(dim=1)
    nearest_token_ids = torch.empty(k, dtype=torch.long, device=x.device)
    for idx in range(k):
        cluster_dist = distances[:, idx]
        if (labels == idx).any():
            cluster_dist = cluster_dist.masked_fill(labels != idx, float("inf"))
        nearest_token_ids[idx] = cluster_dist.argmin()
    return centers, nearest_token_ids


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, channel_in, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 1 ** i
            in_channels = channel_in if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, chan_indep, channel_in, hidden_dim, block_num=3, kernel_size=3, dropout=0.2):
        super().__init__()
        self.chan_indep = chan_indep
        self.TCN = TemporalConvNet(channel_in, [hidden_dim] * block_num, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.chan_indep:
            x = x.reshape(-1, x.shape[-1]).unsqueeze(1)
        x = self.TCN(x)
        return x.permute(0, 2, 1)


class Decoder(nn.Module):
    def __init__(self, patch_len, enc_in, hidden_dim, n_heads=4, block_num=3, dropout=0.2):
        super().__init__()
        self.decoder = CasualTRM(
            dim=hidden_dim,
            d_ff=hidden_dim * 4,
            n_heads=n_heads,
            n_layers=block_num,
            dropout=dropout,
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_len * enc_in),
        )
        self.patch_len = patch_len
        self.enc_in = enc_in

    def forward(self, x):
        batch_size, token_count, _ = x.shape
        x, _ = self.decoder(x)
        x = self.linear(x)
        return x.view(batch_size, token_count * self.patch_len, self.enc_in)


class FixedGPT2Quantizer(nn.Module):
    def __init__(self, hidden_dim, gpt_dim, n_embed, configs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gpt_dim = gpt_dim
        self.n_embed = n_embed
        self.tau = max(float(getattr(configs, "cluster_tau", 0.2)), 1e-6)
        self.commitment_weight = float(getattr(configs, "commitment_weight", 0.25))
        self.entropy_penalty = float(getattr(configs, "entropy_penalty", 0.0))
        self.use_cosine = bool(getattr(configs, "cluster_cosine", True))

        self.ts_to_gpt = nn.Linear(hidden_dim, gpt_dim)
        self.gpt_to_ts = nn.Linear(gpt_dim, hidden_dim)
        centers, token_ids = self._init_centers(configs)
        self.register_buffer("gpt2_codebook", centers)
        self.register_buffer("gpt2_token_ids", token_ids)

    def _resolve_gpt2_path(self, configs):
        path = getattr(configs, "gpt_local_path", None)
        if path:
            return path
        path = getattr(configs, "local_model_path", None)
        if path:
            return path
        return getattr(configs, "gpt_model_name", "openai-community/gpt2")

    def _init_centers(self, configs):
        cluster_path = getattr(configs, "gpt2_cluster_path", None)
        if cluster_path and os.path.exists(cluster_path):
            payload = torch.load(cluster_path, map_location="cpu")
            return payload["centers"].float(), payload["token_ids"].long()

        if not bool(getattr(configs, "init_gpt2_codebook", True)):
            centers = torch.randn(self.n_embed, self.gpt_dim) * (self.gpt_dim ** -0.5)
            token_ids = torch.arange(self.n_embed, dtype=torch.long)
            return centers, token_ids

        gpt2_path = self._resolve_gpt2_path(configs)
        local_files_only = bool(getattr(configs, "gpt_local_files_only", True))
        gpt_config = AutoConfig.from_pretrained(gpt2_path, local_files_only=local_files_only)
        gpt = AutoModelForCausalLM.from_pretrained(
            gpt2_path,
            config=gpt_config,
            local_files_only=local_files_only,
        )
        embeddings = gpt.get_input_embeddings().weight.detach().float().cpu()
        if self.use_cosine:
            embeddings = F.normalize(embeddings, dim=-1)
        centers, token_ids = _kmeans(
            embeddings,
            self.n_embed,
            iters=getattr(configs, "gpt2_cluster_iters", 25),
            seed=getattr(configs, "seed", 2024),
        )
        if self.use_cosine:
            centers = F.normalize(centers, dim=-1)

        if cluster_path:
            os.makedirs(os.path.dirname(cluster_path) or ".", exist_ok=True)
            torch.save({"centers": centers, "token_ids": token_ids}, cluster_path)
        return centers, token_ids

    def forward(self, z_ts):
        z_gpt = self.ts_to_gpt(z_ts)
        codebook = self.gpt2_codebook.to(dtype=z_gpt.dtype, device=z_gpt.device)
        query = F.normalize(z_gpt, dim=-1) if self.use_cosine else z_gpt
        keys = F.normalize(codebook, dim=-1) if self.use_cosine else codebook
        distances = torch.cdist(query.reshape(-1, query.shape[-1]), keys)
        logits = -distances / self.tau
        probs = F.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1)
        hard = F.embedding(ids, codebook).view_as(z_gpt)
        soft = (probs @ codebook).view_as(z_gpt)
        quant_gpt = hard.detach() - soft.detach() + soft
        quant_ts = self.gpt_to_ts(quant_gpt)

        commit = F.mse_loss(z_gpt, quant_gpt.detach())
        usage = probs.mean(dim=0).clamp_min(1e-8)
        entropy = -(usage * torch.log(usage)).sum() / torch.log(
            torch.tensor(self.n_embed, dtype=usage.dtype, device=usage.device)
        )
        loss = self.commitment_weight * commit + self.entropy_penalty * (1.0 - entropy)
        return quant_ts, quant_gpt, loss, ids.view(z_ts.shape[0], z_ts.shape[1])

    def embed_ids_gpt(self, ids):
        codebook = self.gpt2_codebook.to(device=ids.device)
        return F.embedding(ids, codebook)


class VQVAE(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.wave_length
        self.revin = configs.revin
        self.chan_indep = configs.chan_indep

        hidden_dim = configs.d_model
        gpt_dim = int(getattr(configs, "gpt2_hidden_size", 768))
        n_embed = configs.n_embed
        enc_in = configs.enc_in if configs.chan_indep == 0 else 1

        self.enc = Encoder(self.chan_indep, enc_in, hidden_dim, configs.block_num)
        wave_patch = (self.patch_len, hidden_dim)
        self.quantize_input = nn.Conv2d(1, hidden_dim, kernel_size=wave_patch, stride=wave_patch)
        self.quantize = FixedGPT2Quantizer(hidden_dim, gpt_dim, n_embed, configs)
        self.dec = Decoder(self.patch_len, enc_in, hidden_dim)

        if self.revin:
            self.revin_layer = RevIN(
                enc_in,
                affine=configs.affine,
                subtract_last=configs.subtract_last,
            )

    def _encode_norm(self, x_norm):
        enc = self.enc(x_norm)
        z_ts = self.quantize_input(enc.unsqueeze(1)).squeeze(-1).transpose(1, 2)
        quant_ts, quant_gpt, loss, ids = self.quantize(z_ts)
        return quant_ts, quant_gpt, loss, ids

    def forward(self, x, y):
        if self.revin:
            x_norm = self.revin_layer(x, "norm")
            y_norm = self.revin_layer._normalize(y)
            seq = torch.cat([x_norm, y_norm], dim=1)
        else:
            seq = torch.cat([x, y], dim=1)

        n_var = seq.shape[-1]
        quant_ts, _, loss, ids = self._encode_norm(seq)
        dec = self.dec(quant_ts)
        if self.chan_indep:
            dec = dec.permute(0, 2, 1)
            dec = dec.reshape(-1, n_var, dec.shape[-1])
            dec = dec.permute(0, 2, 1)
        if self.revin:
            dec = self.revin_layer(dec, "denorm")
        return dec, loss, ids

    def get_name(self):
        return "gpt2_cluster_vq"

    @torch.no_grad()
    def encode_history_gpt(self, x):
        if self.revin:
            x = self.revin_layer(x, "norm")
        _, quant_gpt, _, ids = self._encode_norm(x)
        return quant_gpt, ids

    @torch.no_grad()
    def encode_future_target_gpt(self, x, y):
        if self.revin:
            x_norm = self.revin_layer(x, "norm")
            y_norm = self.revin_layer._normalize(y)
            seq = torch.cat([x_norm, y_norm], dim=1)
        else:
            seq = torch.cat([x, y], dim=1)
        _, quant_gpt, _, ids = self._encode_norm(seq)
        history_tokens = x.shape[1] // self.patch_len
        return quant_gpt[:, history_tokens:, :], ids[:, history_tokens:]

    def decode_future_gpt(self, x, history_gpt, future_gpt, pred_len):
        if self.revin:
            _ = self.revin_layer(x, "norm")
        full_gpt = torch.cat([history_gpt.detach(), future_gpt], dim=1)
        full_ts = self.quantize.gpt_to_ts(full_gpt)
        dec = self.dec(full_ts)
        if self.revin:
            dec = self.revin_layer(dec, "denorm")
        return dec[:, x.shape[1] : x.shape[1] + pred_len, :]

    def get_codebook_weight(self):
        return self.quantize.gpt2_codebook
