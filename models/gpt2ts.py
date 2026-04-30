import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def _get_bool(configs, name, default=False):
    return bool(getattr(configs, name, default))


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


class KMeansBridge(nn.Module):
    """Maps time-series patch embeddings into GPT vocabulary-cluster space."""

    def __init__(self, num_clusters=64, residual_scale=1.0, normalize=True, seed=0):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.residual_scale = float(residual_scale)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.register_buffer("ts_centers", torch.empty(0), persistent=True)
        self.register_buffer("vocab_centers", torch.empty(0), persistent=True)
        self.register_buffer("ts_to_vocab", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("cluster_fitted", torch.tensor(False), persistent=True)
        self.is_fitted = False

    @property
    def ready(self):
        return bool(
            self.cluster_fitted.item()
            and self.ts_centers.numel() > 0
            and self.vocab_centers.numel() > 0
            and self.ts_to_vocab.numel() > 0
        )

    def fit(self, ts_embeds, vocab_embeds, iters=8):
        device = ts_embeds.device
        k = min(self.num_clusters, ts_embeds.shape[0], vocab_embeds.shape[0])
        if k <= 0:
            raise ValueError("Cannot fit clusters from empty embeddings.")

        self.num_clusters = int(k)
        self.ts_centers = self._kmeans(ts_embeds.float(), k, int(iters), self.seed)
        self.vocab_centers = self._kmeans(vocab_embeds.float(), k, int(iters), self.seed + 13)

        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed + 29)
        self.ts_to_vocab = torch.randperm(k, generator=generator, device=device)
        self.cluster_fitted.fill_(True)
        self.is_fitted = True

    @torch.no_grad()
    def _kmeans(self, x, k, iters, seed):
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.detach()
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        if x.shape[0] >= k:
            indices = torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]
        else:
            indices = torch.randint(0, x.shape[0], (k,), generator=generator, device=x.device)
        centers = x[indices].clone()

        for _ in range(max(int(iters), 1)):
            distance_source = F.normalize(x, dim=-1) if self.normalize else x
            distance_centers = F.normalize(centers, dim=-1) if self.normalize else centers
            distances = torch.cdist(distance_source, distance_centers)
            assign = distances.argmin(dim=-1)

            new_centers = centers.clone()
            for cluster_id in range(k):
                mask = assign == cluster_id
                if mask.any():
                    new_centers[cluster_id] = x[mask].mean(dim=0)
            centers = new_centers
        return centers

    def _nearest_ts_center(self, embeds):
        source = F.normalize(embeds, dim=-1) if self.normalize else embeds
        centers = F.normalize(self.ts_centers, dim=-1) if self.normalize else self.ts_centers
        distances = torch.cdist(source.float(), centers.float())
        return distances.argmin(dim=-1), distances.min(dim=-1).values

    def _nearest_vocab_center(self, embeds):
        source = F.normalize(embeds, dim=-1) if self.normalize else embeds
        centers = F.normalize(self.vocab_centers, dim=-1) if self.normalize else self.vocab_centers
        distances = torch.cdist(source.float(), centers.float())
        return distances.argmin(dim=-1), distances.min(dim=-1).values

    def _inverse_mapping(self):
        inverse = torch.empty_like(self.ts_to_vocab)
        inverse[self.ts_to_vocab] = torch.arange(self.ts_to_vocab.numel(), device=self.ts_to_vocab.device)
        return inverse

    def map_ts_to_vocab_space(self, ts_embeds):
        if not self.ready:
            return ts_embeds
        center_ids, _ = self._nearest_ts_center(ts_embeds)
        ts_center = self.ts_centers[center_ids]
        vocab_center = self.vocab_centers[self.ts_to_vocab[center_ids]]
        residual = ts_embeds - ts_center.to(dtype=ts_embeds.dtype)
        direction = F.normalize(residual, dim=-1)
        distance = residual.norm(dim=-1, keepdim=True)
        mapped = vocab_center.to(dtype=ts_embeds.dtype) + direction * distance * self.residual_scale
        return mapped

    def map_vocab_to_ts_space(self, vocab_embeds):
        if not self.ready:
            return vocab_embeds
        vocab_center_ids, _ = self._nearest_vocab_center(vocab_embeds)
        inverse = self._inverse_mapping()
        ts_center_ids = inverse[vocab_center_ids]
        vocab_center = self.vocab_centers[vocab_center_ids]
        ts_center = self.ts_centers[ts_center_ids]
        residual = vocab_embeds - vocab_center.to(dtype=vocab_embeds.dtype)
        direction = F.normalize(residual, dim=-1)
        distance = residual.norm(dim=-1, keepdim=True)
        inverse_scale = 1.0 / max(self.residual_scale, 1e-6)
        return ts_center.to(dtype=vocab_embeds.dtype) + direction * distance * inverse_scale


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


class Model(nn.Module):
    """Patch-cluster GPT2 forecaster with LoRA-tuned attention."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.c_in = int(getattr(configs, "c_in", getattr(configs, "enc_in", 1)))
        self.c_out = int(getattr(configs, "c_out", self.c_in))
        self.patch_len = int(getattr(configs, "patch_len", getattr(configs, "patch_size", 16)))
        self.stride = int(getattr(configs, "stride", self.patch_len))

        self.gpt2_path = self._resolve_gpt2_path()
        self.local_files_only = _get_bool(configs, "gpt_local_files_only", True)
        gpt_config = AutoConfig.from_pretrained(self.gpt2_path, local_files_only=self.local_files_only)
        requested_layers = int(getattr(configs, "n_layers", 0))
        if requested_layers > 0:
            gpt_config.n_layer = min(requested_layers, int(gpt_config.n_layer))
            gpt_config.num_hidden_layers = gpt_config.n_layer
        gpt_config.output_hidden_states = True
        self.gpt_dim = int(gpt_config.hidden_size)

        self.gpt2 = self._load_gpt2(gpt_config)
        self._freeze_gpt2()
        self._inject_lora()

        self.patch_tokenizer = PatchTokenizer(
            patch_len=self.patch_len,
            stride=self.stride,
            c_in=self.c_in,
            gpt_dim=self.gpt_dim,
            dropout=getattr(configs, "embedding_dropout", getattr(configs, "dropout", 0.05)),
        )
        self.bridge = KMeansBridge(
            num_clusters=getattr(configs, "num_clusters", 64),
            residual_scale=getattr(configs, "cluster_residual_scale", 1.0),
            normalize=getattr(configs, "cluster_normalize", True),
            seed=getattr(configs, "cluster_seed", getattr(configs, "seed", 0)),
        )

        self.future_patch_count = self._patch_count_for_length(self.pred_len)
        self.future_queries = nn.Parameter(torch.empty(self.future_patch_count, self.gpt_dim))
        nn.init.normal_(self.future_queries, mean=0.0, std=self.gpt_dim ** -0.5)

        self.decoder = HistoryPatchDecoder(
            patch_len=self.patch_len,
            stride=self.stride,
            pred_len=self.pred_len,
            temperature=getattr(configs, "history_lookup_temperature", 0.2),
            hard_lookup=getattr(configs, "hard_patch_lookup", False),
        )
        self.output_dropout = nn.Dropout(float(getattr(configs, "dropout", 0.05)))

    def _patch_count_for_length(self, length):
        if length <= self.patch_len:
            return 1
        return math.ceil((length - self.patch_len) / self.stride) + 1

    def _resolve_gpt2_path(self):
        local_path = getattr(self.configs, "gpt_local_path", None)
        if local_path:
            return local_path
        if os.path.isdir("./gpt"):
            return "./gpt"
        return getattr(self.configs, "gpt_model_name", "openai-community/gpt2")

    def _load_gpt2(self, config):
        if _get_bool(self.configs, "use_pretrained_gpt2", True):
            return AutoModelForCausalLM.from_pretrained(
                self.gpt2_path,
                config=config,
                local_files_only=self.local_files_only,
            )
        return AutoModelForCausalLM.from_config(config)

    def _freeze_gpt2(self):
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def _inject_lora(self):
        r = int(getattr(self.configs, "lora_r", 8))
        if r <= 0:
            return
        alpha = float(getattr(self.configs, "lora_alpha", 16))
        dropout = float(getattr(self.configs, "lora_dropout", 0.05))
        targets = str(getattr(self.configs, "lora_target", "c_attn,c_proj")).split(",")
        targets = {target.strip() for target in targets if target.strip()}

        for block in self.gpt2.transformer.h:
            if "c_attn" in targets:
                block.attn.c_attn = LoRAConv1D(block.attn.c_attn, r=r, alpha=alpha, dropout=dropout)
            if "c_proj" in targets:
                block.attn.c_proj = LoRAConv1D(block.attn.c_proj, r=r, alpha=alpha, dropout=dropout)

    def _vocab_weight(self):
        return self.gpt2.get_input_embeddings().weight.detach()

    @torch.no_grad()
    def fit_token_clusters(self, data_loader, device=None):
        device = device or next(self.parameters()).device
        sample_limit = int(getattr(self.configs, "cluster_sample_size", 8192))
        vocab_limit = int(getattr(self.configs, "vocab_cluster_sample_size", 20000))
        kmeans_iters = int(getattr(self.configs, "kmeans_iters", 8))

        was_training = self.training
        self.eval()
        samples = []
        seen = 0
        for batch in data_loader:
            batch_x = batch[0].to(device=device, dtype=torch.float)
            patches = self.patch_tokenizer.patchify(batch_x)
            embeds = self.patch_tokenizer.encode(patches).reshape(-1, self.gpt_dim)
            samples.append(embeds.detach())
            seen += embeds.shape[0]
            if seen >= sample_limit:
                break

        if not samples:
            raise ValueError("No patches found for fitting time-series clusters.")

        ts_embeds = torch.cat(samples, dim=0)
        if ts_embeds.shape[0] > sample_limit:
            ts_embeds = ts_embeds[:sample_limit]

        vocab_embeds = self._vocab_weight().to(device=device, dtype=torch.float)
        if vocab_limit > 0 and vocab_embeds.shape[0] > vocab_limit:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(getattr(self.configs, "cluster_seed", getattr(self.configs, "seed", 0))) + 71)
            ids = torch.randperm(vocab_embeds.shape[0], generator=generator, device=device)[:vocab_limit]
            vocab_embeds = vocab_embeds[ids]

        self.bridge.fit(ts_embeds, vocab_embeds, iters=kmeans_iters)
        self.train(was_training)

    def _predicted_vocab_embeddings(self, logits):
        temperature = max(float(getattr(self.configs, "forecast_temperature", 1.0)), 1e-6)
        top_k = int(getattr(self.configs, "forecast_top_k", 64))
        vocab_weight = self._vocab_weight().to(device=logits.device, dtype=logits.dtype)

        if top_k > 0 and top_k < logits.shape[-1]:
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(values / temperature, dim=-1)
            selected = vocab_weight[indices]
            embeds = torch.einsum("bfk,bfkd->bfd", probs, selected)
            token_ids = indices[..., 0]
            return embeds, token_ids

        probs = F.softmax(logits / temperature, dim=-1)
        embeds = torch.matmul(probs, vocab_weight)
        token_ids = probs.argmax(dim=-1)
        return embeds, token_ids

    def forecast(self, batch_x):
        history_patches = self.patch_tokenizer.patchify(batch_x)
        history_ts_embeds = self.patch_tokenizer.encode(history_patches)
        history_llm_embeds = self.bridge.map_ts_to_vocab_space(history_ts_embeds)
        history_llm_embeds = self.output_dropout(history_llm_embeds)

        future_queries = self.future_queries.unsqueeze(0).expand(batch_x.shape[0], -1, -1)
        inputs_embeds = torch.cat([history_llm_embeds, future_queries], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=batch_x.device)

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        future_logits = outputs.logits[:, -self.future_patch_count :, :]
        pred_vocab_embeds, pred_token_ids = self._predicted_vocab_embeddings(future_logits)
        pred_ts_embeds = self.bridge.map_vocab_to_ts_space(pred_vocab_embeds)

        pred = self.decoder(pred_ts_embeds, history_ts_embeds, history_patches)
        pred = pred[:, : self.pred_len, : self.c_out]
        aux = SimpleNamespace(
            pred_token_ids=pred_token_ids,
            pred_vocab_embeds=pred_vocab_embeds,
            pred_ts_embeds=pred_ts_embeds,
            history_llm_embeds=history_llm_embeds,
            history_ts_embeds=history_ts_embeds,
            mapped_ts_embeds=pred_ts_embeds,
        )
        return pred, aux

    def forward(self, batch_x, batch_y=None):
        pred, aux = self.forecast(batch_x)
        loss = None
        if batch_y is not None:
            target = batch_y[:, -self.pred_len :, : self.c_out]
            loss = F.mse_loss(pred, target)
        return SimpleNamespace(pred=pred, loss=loss, aux=aux)
