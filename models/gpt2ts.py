import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


KMEANS_TOL = 1e-6
KMEANS_MAX_ITERS = 100


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


class PatchTokenDictionary(nn.Module):
    """Full-train-set patch/token conversion table."""

    def __init__(self, cluster_num, patch_len, c_in, normalize=True, seed=0, match_tol=1e-6):
        super().__init__()
        self.cluster_num = int(cluster_num)
        self.patch_len = int(patch_len)
        self.c_in = int(c_in)
        self.normalize = bool(normalize)
        self.seed = 0 if seed is None else int(seed)
        self.match_tol = float(match_tol)

        self.register_buffer("train_patches", torch.empty(0, self.patch_len, self.c_in), persistent=True)
        self.register_buffer("train_patch_token_ids", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("patch_centers", torch.empty(self.cluster_num, self.patch_len * self.c_in), persistent=True)
        self.register_buffer("patch_cluster_ids", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("patch_center_distances", torch.empty(0), persistent=True)
        self.register_buffer("vocab_centers", torch.empty(0, 0), persistent=True)
        self.register_buffer("vocab_token_cluster_ids", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("vocab_token_center_distances", torch.empty(0), persistent=True)
        self.register_buffer("vocab_token_ranks", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("vocab_cluster_sizes", torch.empty(self.cluster_num, dtype=torch.long), persistent=True)
        self.register_buffer("patch_to_vocab_cluster", torch.empty(self.cluster_num, dtype=torch.long), persistent=True)
        self.register_buffer("vocab_to_patch_cluster", torch.empty(self.cluster_num, dtype=torch.long), persistent=True)
        self.register_buffer("fitted", torch.tensor(False), persistent=True)
        self.last_exact_match_count = 0
        self.last_total_match_count = 0

    @property
    def ready(self):
        return bool(self.fitted.item() and self.train_patches.numel() > 0 and self.train_patch_token_ids.numel() > 0)

    def _resize_for_checkpoint(self, state_dict, prefix):
        centers_key = prefix + "patch_centers"
        if centers_key in state_dict:
            self.cluster_num = int(state_dict[centers_key].shape[0])
        for name in (
            "train_patches",
            "train_patch_token_ids",
            "patch_centers",
            "patch_cluster_ids",
            "patch_center_distances",
            "vocab_centers",
            "vocab_token_cluster_ids",
            "vocab_token_center_distances",
            "vocab_token_ranks",
            "vocab_cluster_sizes",
            "patch_to_vocab_cluster",
            "vocab_to_patch_cluster",
            "fitted",
        ):
            key = prefix + name
            if key in state_dict and getattr(self, name).shape != state_dict[key].shape:
                setattr(self, name, torch.empty_like(state_dict[key]))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self._resize_for_checkpoint(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _distance_source(self, x):
        return F.normalize(x, dim=-1) if self.normalize else x

    @torch.no_grad()
    def _kmeans(self, x, k, seed):
        x = x.detach().float().reshape(-1, x.shape[-1])
        if x.shape[0] == 0:
            raise ValueError("Cannot run k-means on an empty tensor.")

        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        if x.shape[0] >= k:
            indices = torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]
        else:
            indices = torch.randint(0, x.shape[0], (k,), generator=generator, device=x.device)
        centers = x[indices].clone()

        previous_assign = None
        for _ in range(KMEANS_MAX_ITERS):
            distances = torch.cdist(self._distance_source(x), self._distance_source(centers))
            assign = distances.argmin(dim=-1)

            if previous_assign is not None and torch.equal(assign, previous_assign):
                break

            new_centers = centers.clone()
            for cluster_id in range(k):
                mask = assign == cluster_id
                if mask.any():
                    new_centers[cluster_id] = x[mask].mean(dim=0)

            center_shift = (new_centers - centers).abs().max()
            centers = new_centers
            if center_shift.item() <= KMEANS_TOL:
                break
            previous_assign = assign
        return centers

    @torch.no_grad()
    def _assign_to_centers(self, x, centers):
        distances = torch.cdist(self._distance_source(x.float()), self._distance_source(centers.float()))
        cluster_ids = distances.argmin(dim=-1)
        center_distances = (x.float() - centers[cluster_ids].float()).norm(dim=-1)
        return cluster_ids, center_distances

    @torch.no_grad()
    def _patchify_series(self, series):
        series = torch.as_tensor(series, dtype=torch.float32, device=self.train_patches.device)
        if series.ndim == 2:
            series = series.unsqueeze(0)
        if series.ndim != 3:
            raise ValueError("Expected train series with shape [T,C] or [1,T,C].")
        if series.shape[-1] != self.c_in:
            raise ValueError(f"Expected {self.c_in} input channel(s), got {series.shape[-1]}.")

        remainder = series.shape[1] % self.patch_len
        if remainder:
            pad_len = self.patch_len - remainder
            series = F.pad(series, (0, 0, 0, pad_len))
        patches = series.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.reshape(-1, self.patch_len, self.c_in)

    @torch.no_grad()
    def _build_vocab_ranks(self, token_cluster_ids, token_distances):
        vocab_ranks = torch.zeros_like(token_cluster_ids)
        cluster_sizes = torch.zeros(self.cluster_num, dtype=torch.long, device=token_cluster_ids.device)
        for cluster_id in range(self.cluster_num):
            indices = torch.nonzero(token_cluster_ids == cluster_id, as_tuple=False).flatten()
            cluster_sizes[cluster_id] = indices.numel()
            if indices.numel() == 0:
                continue
            sorted_indices = indices[torch.argsort(token_distances[indices])]
            vocab_ranks[sorted_indices] = torch.arange(indices.numel(), device=token_cluster_ids.device)
        return vocab_ranks, cluster_sizes

    @torch.no_grad()
    def _map_train_patches_to_tokens(self):
        token_ids = torch.empty(self.train_patches.shape[0], dtype=torch.long, device=self.train_patches.device)
        non_empty_vocab_clusters = torch.nonzero(self.vocab_cluster_sizes > 0, as_tuple=False).flatten()
        for patch_cluster_id in range(self.cluster_num):
            patch_indices = torch.nonzero(self.patch_cluster_ids == patch_cluster_id, as_tuple=False).flatten()
            if patch_indices.numel() == 0:
                continue

            vocab_cluster_id = int(self.patch_to_vocab_cluster[patch_cluster_id].item())
            vocab_indices = torch.nonzero(self.vocab_token_cluster_ids == vocab_cluster_id, as_tuple=False).flatten()
            if vocab_indices.numel() == 0:
                if non_empty_vocab_clusters.numel() == 0:
                    raise RuntimeError("No GPT vocab cluster has assigned tokens.")
                center = self.vocab_centers[vocab_cluster_id].unsqueeze(0)
                candidate_centers = self.vocab_centers[non_empty_vocab_clusters]
                nearest_cluster_pos = torch.cdist(center.float(), candidate_centers.float()).argmin(dim=-1)
                nearest_cluster_id = non_empty_vocab_clusters[nearest_cluster_pos].item()
                vocab_indices = torch.nonzero(self.vocab_token_cluster_ids == nearest_cluster_id, as_tuple=False).flatten()

            sorted_patch_indices = patch_indices[torch.argsort(self.patch_center_distances[patch_indices])]
            sorted_vocab_indices = vocab_indices[torch.argsort(self.vocab_token_center_distances[vocab_indices])]

            if sorted_patch_indices.numel() == 1:
                mapped_positions = torch.full((1,), sorted_vocab_indices.numel() // 2, dtype=torch.long, device=token_ids.device)
            else:
                patch_positions = torch.arange(sorted_patch_indices.numel(), device=token_ids.device, dtype=torch.float32)
                mapped_positions = torch.round(
                    patch_positions * float(sorted_vocab_indices.numel() - 1) / float(sorted_patch_indices.numel() - 1)
                ).long()
            token_ids[sorted_patch_indices] = sorted_vocab_indices[mapped_positions]
        return token_ids

    @torch.no_grad()
    def fit(self, train_series, vocab_embeds):
        train_patches = self._patchify_series(train_series)
        flat_patches = train_patches.reshape(train_patches.shape[0], -1).float()
        vocab_embeds = vocab_embeds.detach().float().to(flat_patches.device)

        patch_centers = self._kmeans(flat_patches, self.cluster_num, self.seed + 11)
        patch_cluster_ids, patch_distances = self._assign_to_centers(flat_patches, patch_centers)

        vocab_centers = self._kmeans(vocab_embeds, self.cluster_num, self.seed + 17)
        vocab_cluster_ids, vocab_distances = self._assign_to_centers(vocab_embeds, vocab_centers)
        vocab_ranks, vocab_cluster_sizes = self._build_vocab_ranks(vocab_cluster_ids, vocab_distances)

        generator = torch.Generator(device=flat_patches.device)
        generator.manual_seed(self.seed + 23)
        patch_to_vocab = torch.randperm(self.cluster_num, generator=generator, device=flat_patches.device)
        vocab_to_patch = torch.empty_like(patch_to_vocab)
        vocab_to_patch[patch_to_vocab] = torch.arange(self.cluster_num, device=flat_patches.device)

        self.train_patches = train_patches.detach()
        self.patch_centers = patch_centers.detach()
        self.patch_cluster_ids = patch_cluster_ids.detach()
        self.patch_center_distances = patch_distances.detach()
        self.vocab_centers = vocab_centers.detach()
        self.vocab_token_cluster_ids = vocab_cluster_ids.detach()
        self.vocab_token_center_distances = vocab_distances.detach()
        self.vocab_token_ranks = vocab_ranks.detach()
        self.vocab_cluster_sizes = vocab_cluster_sizes.detach()
        self.patch_to_vocab_cluster = patch_to_vocab.detach()
        self.vocab_to_patch_cluster = vocab_to_patch.detach()
        self.train_patch_token_ids = self._map_train_patches_to_tokens().detach()
        self.fitted.fill_(True)

    @torch.no_grad()
    def patches_to_token_ids(self, patches):
        if not self.ready:
            raise RuntimeError("PatchTokenDictionary must be fitted before converting patches to tokens.")
        flat = patches.reshape(-1, self.patch_len * self.c_in).float()
        query_cluster_ids, _ = self._assign_to_centers(flat, self.patch_centers)
        selected_indices = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        selected_distances = torch.empty(flat.shape[0], dtype=flat.dtype, device=flat.device)
        non_empty_clusters = torch.nonzero(
            torch.bincount(self.patch_cluster_ids, minlength=self.cluster_num) > 0,
            as_tuple=False,
        ).flatten()
        if non_empty_clusters.numel() == 0:
            raise RuntimeError("No train patch cluster has assigned patches.")

        train_flat = self.train_patches.reshape(self.train_patches.shape[0], -1).float()
        chunk_size = max(int(getattr(self, "nearest_chunk_size", 1024)), 1)

        for cluster_id in torch.unique(query_cluster_ids):
            query_indices = torch.nonzero(query_cluster_ids == cluster_id, as_tuple=False).flatten()
            candidate_indices = torch.nonzero(self.patch_cluster_ids == cluster_id, as_tuple=False).flatten()

            if candidate_indices.numel() == 0:
                center = self.patch_centers[cluster_id].unsqueeze(0)
                candidate_centers = self.patch_centers[non_empty_clusters]
                fallback_pos = torch.cdist(
                    self._distance_source(center.float()),
                    self._distance_source(candidate_centers.float()),
                ).argmin(dim=-1)
                fallback_cluster_id = non_empty_clusters[fallback_pos].item()
                candidate_indices = torch.nonzero(
                    self.patch_cluster_ids == fallback_cluster_id,
                    as_tuple=False,
                ).flatten()

            candidate_flat = train_flat[candidate_indices]
            for start in range(0, query_indices.numel(), chunk_size):
                end = min(start + chunk_size, query_indices.numel())
                query_chunk_indices = query_indices[start:end]
                distances = torch.cdist(flat[query_chunk_indices], candidate_flat)
                local_indices = distances.argmin(dim=-1)
                selected_indices[query_chunk_indices] = candidate_indices[local_indices]
                selected_distances[query_chunk_indices] = distances.gather(1, local_indices.unsqueeze(-1)).squeeze(-1)

        nearest_indices = selected_indices
        nearest_distances = selected_distances
        self.last_exact_match_count = int((nearest_distances <= self.match_tol).sum().item())
        self.last_total_match_count = int(flat.shape[0])
        token_ids = self.train_patch_token_ids[nearest_indices]
        return token_ids.reshape(patches.shape[0], patches.shape[1])

    @torch.no_grad()
    def token_ids_to_patches(self, token_ids):
        if not self.ready:
            raise RuntimeError("PatchTokenDictionary must be fitted before converting tokens to patches.")

        flat_tokens = token_ids.reshape(-1).long().clamp(min=0, max=self.vocab_token_cluster_ids.shape[0] - 1)
        selected_patch_indices = torch.empty(flat_tokens.shape[0], dtype=torch.long, device=flat_tokens.device)

        for pos, token_id in enumerate(flat_tokens):
            vocab_cluster_id = int(self.vocab_token_cluster_ids[token_id].item())
            patch_cluster_id = int(self.vocab_to_patch_cluster[vocab_cluster_id].item())
            patch_indices = torch.nonzero(self.patch_cluster_ids == patch_cluster_id, as_tuple=False).flatten()
            if patch_indices.numel() == 0:
                patch_center = self.patch_centers[patch_cluster_id].unsqueeze(0)
                train_flat = self.train_patches.reshape(self.train_patches.shape[0], -1).float()
                selected_patch_indices[pos] = torch.cdist(patch_center.float(), train_flat).argmin(dim=-1).item()
                continue

            sorted_patch_indices = patch_indices[torch.argsort(self.patch_center_distances[patch_indices])]
            token_cluster_size = int(self.vocab_cluster_sizes[vocab_cluster_id].item())
            token_rank = int(self.vocab_token_ranks[token_id].item())
            if token_cluster_size <= 1 or sorted_patch_indices.numel() == 1:
                patch_pos = sorted_patch_indices.numel() // 2
            else:
                patch_pos = int(round(token_rank * float(sorted_patch_indices.numel() - 1) / float(token_cluster_size - 1)))
            selected_patch_indices[pos] = sorted_patch_indices[patch_pos]

        patches = self.train_patches[selected_patch_indices]
        return patches.reshape(*token_ids.shape, self.patch_len, self.c_in)

    @torch.no_grad()
    def save_npz(self, path):
        if not self.ready:
            raise RuntimeError("Cannot save an unfitted PatchTokenDictionary.")
        np.savez_compressed(
            path,
            train_patches=self.train_patches.detach().cpu().numpy(),
            train_patch_token_ids=self.train_patch_token_ids.detach().cpu().numpy(),
            patch_centers=self.patch_centers.detach().cpu().numpy(),
            patch_cluster_ids=self.patch_cluster_ids.detach().cpu().numpy(),
            patch_center_distances=self.patch_center_distances.detach().cpu().numpy(),
            vocab_centers=self.vocab_centers.detach().cpu().numpy(),
            vocab_token_cluster_ids=self.vocab_token_cluster_ids.detach().cpu().numpy(),
            vocab_token_center_distances=self.vocab_token_center_distances.detach().cpu().numpy(),
            vocab_token_ranks=self.vocab_token_ranks.detach().cpu().numpy(),
            vocab_cluster_sizes=self.vocab_cluster_sizes.detach().cpu().numpy(),
            patch_to_vocab_cluster=self.patch_to_vocab_cluster.detach().cpu().numpy(),
            vocab_to_patch_cluster=self.vocab_to_patch_cluster.detach().cpu().numpy(),
        )


class GPT2TS(nn.Module):
    """Patch-token GPT-2 forecaster using a full-train-set conversion table."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.c_in = int(configs.c_in)
        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)

        if getattr(configs, "features", "S") != "S" or self.c_in != 1:
            raise ValueError("The patch-token GPT flow currently supports only features='S' with c_in=1.")
        if self.stride != self.patch_len:
            raise ValueError("This patch-token GPT flow requires non-overlap patches, so stride must equal patch_len.")
        if self.seq_len % self.patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len for non-overlap tokenization.")
        if self.pred_len % self.patch_len != 0:
            raise ValueError("pred_len must be divisible by patch_len for direct patch concatenation.")

        self.history_patch_count = self.seq_len // self.patch_len
        self.future_patch_count = self.pred_len // self.patch_len

        self.gpt2_path = getattr(configs, "gpt_local_path", "./gpt")
        self.local_files_only = getattr(configs, "gpt_local_files_only", True)
        gpt_config = AutoConfig.from_pretrained(self.gpt2_path, local_files_only=self.local_files_only)
        requested_layers = int(getattr(configs, "n_layers", 0))
        if requested_layers > 0:
            gpt_config.n_layer = min(requested_layers, int(gpt_config.n_layer))
            gpt_config.num_hidden_layers = gpt_config.n_layer
        self.gpt2 = self._load_gpt2(gpt_config)
        self._freeze_gpt2()
        self._inject_lora()

        cluster_num = int(getattr(configs, "cluster_num", 128))
        self.dictionary = PatchTokenDictionary(
            cluster_num=cluster_num,
            patch_len=self.patch_len,
            c_in=self.c_in,
            normalize=getattr(configs, "cluster_normalize", True),
            seed=getattr(configs, "cluster_seed", 0),
            match_tol=getattr(configs, "patch_match_tol", 1e-6),
        )

    def _load_gpt2(self, config):
        if getattr(self.configs, "use_pretrained_gpt2", False):
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
        r = int(getattr(self.configs, "lora_r", 0))
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
    def fit_patch_token_map(self, train_series):
        self.dictionary.fit(train_series, self._vocab_weight())

    @torch.no_grad()
    def save_patch_token_map(self, path):
        self.dictionary.save_npz(path)

    @torch.no_grad()
    def build_lm_training_tensors(self):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before building token LM training data.")

        tokens = self.dictionary.train_patch_token_ids.detach().long()
        window_size = self.history_patch_count + self.future_patch_count
        if tokens.numel() < window_size:
            raise ValueError(
                f"Training token sequence has {tokens.numel()} patches, but at least {window_size} are required."
            )

        max_positions = int(getattr(self.gpt2.config, "n_positions", window_size))
        if window_size - 1 > max_positions:
            raise ValueError(
                f"Token training window length {window_size - 1} exceeds GPT context length {max_positions}."
            )

        step = max(int(getattr(self.configs, "token_train_stride", 1)), 1)
        windows = tokens.unfold(dimension=0, size=window_size, step=step).contiguous()
        input_ids = windows[:, :-1].contiguous()
        labels = windows[:, 1:].contiguous()
        labels[:, : self.history_patch_count - 1] = -100
        return input_ids, labels

    def token_lm_loss(self, input_ids, labels):
        attention_mask = torch.ones_like(input_ids)
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def _patchify_batch(self, batch_x):
        if batch_x.shape[1] != self.seq_len:
            raise ValueError(f"Expected batch_x length {self.seq_len}, got {batch_x.shape[1]}.")
        patches = batch_x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches

    def _concat_patches(self, patches):
        return patches.reshape(patches.shape[0], patches.shape[1] * self.patch_len, self.c_in)[:, : self.pred_len, :]

    @torch.no_grad()
    def _generate_future_tokens(self, history_token_ids):
        max_positions = int(getattr(self.gpt2.config, "n_positions", history_token_ids.shape[1] + self.future_patch_count))
        max_history = max_positions - self.future_patch_count
        if max_history <= 0:
            raise ValueError("GPT context length is smaller than the requested future patch count.")
        if history_token_ids.shape[1] > max_history:
            history_token_ids = history_token_ids[:, -max_history:]

        eos_token_id = self.gpt2.config.eos_token_id
        pad_token_id = eos_token_id if eos_token_id is not None else 0
        attention_mask = torch.ones_like(history_token_ids)
        top_k = int(getattr(self.configs, "forecast_top_k", 1))
        temperature = max(float(getattr(self.configs, "forecast_temperature", 1.0)), 1e-6)
        do_sample = top_k > 1 or abs(temperature - 1.0) > 1e-6
        generate_kwargs = {}
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if top_k > 0:
                generate_kwargs["top_k"] = top_k
        generated = self.gpt2.generate(
            input_ids=history_token_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.future_patch_count,
            do_sample=do_sample,
            eos_token_id=None,
            pad_token_id=pad_token_id,
            **generate_kwargs,
        )
        return generated[:, -self.future_patch_count :]

    @torch.no_grad()
    def forecast(self, batch_x):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map with the full training set before forecasting.")

        history_patches = self._patchify_batch(batch_x)
        history_token_ids = self.dictionary.patches_to_token_ids(history_patches)
        future_token_ids = self._generate_future_tokens(history_token_ids)
        future_patches = self.dictionary.token_ids_to_patches(future_token_ids)
        pred = self._concat_patches(future_patches)
        aux = SimpleNamespace(history_token_ids=history_token_ids, future_token_ids=future_token_ids)
        return pred, aux

    def forward(self, batch_x, batch_y=None):
        pred, aux = self.forecast(batch_x)
        return SimpleNamespace(pred=pred, loss=None, aux=aux)
