import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from models.patchdecoder import OverlapAddPatchDecoder
from models.patchtokenizer import PatchTokenizer


KMEANS_TOL = 1e-6
KMEANS_MAX_ITERS = 100


class LoRAConv1D(nn.Module):
    """LoRA adapter for GPT-2's Conv1D projections."""

    # Initialize LoRA for Conv1D.
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

    # Add the LoRA update.
    def forward(self, x):
        base = self.base_layer(x)
        if self.r <= 0:
            return base
        update = F.linear(self.dropout(x), self.lora_A)
        update = F.linear(update, self.lora_B) * self.scaling
        return base + update


class KMeansBridge(nn.Module):
    """Maps time-series patch embeddings into GPT vocabulary-cluster space."""

    # Initialize the cluster bridge.
    def __init__(self, num_clusters=64, embed_dim=None, residual_scale=1.0, normalize=True, seed=0):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.embed_dim = None if embed_dim is None else int(embed_dim)
        self.residual_scale = float(residual_scale)
        self.normalize = bool(normalize)
        self.seed = 0 if seed is None else int(seed)
        if self.embed_dim is None or self.num_clusters <= 0:
            self.register_buffer("ts_centers", torch.empty(0), persistent=True)
            self.register_buffer("vocab_centers", torch.empty(0), persistent=True)
            self.register_buffer("ts_to_vocab", torch.empty(0, dtype=torch.long), persistent=True)
        else:
            self.register_buffer("ts_centers", torch.empty(self.num_clusters, self.embed_dim), persistent=True)
            self.register_buffer("vocab_centers", torch.empty(self.num_clusters, self.embed_dim), persistent=True)
            self.register_buffer("ts_to_vocab", torch.arange(self.num_clusters, dtype=torch.long), persistent=True)
        self.register_buffer("cluster_fitted", torch.tensor(False), persistent=True)
        self.register_buffer("vocab_token_center_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_token_center_distances", torch.empty(0), persistent=False)
        self.is_fitted = False

    def _resize_for_checkpoint(self, state_dict, prefix):
        ts_key = prefix + "ts_centers"
        vocab_key = prefix + "vocab_centers"
        map_key = prefix + "ts_to_vocab"

        if ts_key in state_dict and self.ts_centers.shape != state_dict[ts_key].shape:
            self.ts_centers = torch.empty_like(state_dict[ts_key])
        if vocab_key in state_dict and self.vocab_centers.shape != state_dict[vocab_key].shape:
            self.vocab_centers = torch.empty_like(state_dict[vocab_key])
        if map_key in state_dict and self.ts_to_vocab.shape != state_dict[map_key].shape:
            self.ts_to_vocab = torch.empty_like(state_dict[map_key])

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

    # Check whether both cluster sets are ready.
    @property
    def ready(self):
        return bool(
            self.cluster_fitted.item()
            and self.ts_centers.numel() > 0
            and self.vocab_centers.numel() > 0
            and self.ts_to_vocab.numel() > 0
        )

    @property
    def vocab_ready(self):
        return bool(self.vocab_centers.numel() > 0)

    # Fit time-series clusters and randomly map them one-to-one to vocab clusters.
    def fit_ts_to_vocab(self, ts_embeds):
        if not self.vocab_ready:
            raise RuntimeError("Cannot fit time-series clusters before vocab clusters are fitted.")
        if ts_embeds.ndim != 2:
            ts_embeds = ts_embeds.reshape(-1, ts_embeds.shape[-1])

        k = self.num_clusters
        if k <= 0:
            raise ValueError("Cannot fit time-series clusters from empty embeddings.")

        self.ts_centers = self._kmeans(ts_embeds.float(), k, self.seed)
        generator = torch.Generator(device=ts_embeds.device)
        generator.manual_seed(self.seed + 29)
        self.ts_to_vocab = torch.randperm(k, generator=generator, device=ts_embeds.device)
        self.cluster_fitted.fill_(True)
        self.is_fitted = True

    # Fit GPT vocab centers only.
    def fit_vocab(self, vocab_embeds):
        k = self.num_clusters
        if k <= 0:
            raise ValueError("Cannot fit vocab clusters from empty embeddings.")

        self.num_clusters = int(k)
        self.vocab_centers = self._kmeans(vocab_embeds.float(), k, self.seed + 13)
        self._assign_vocab_tokens(vocab_embeds.float())

    # Estimate centers with simple KMeans.
    @torch.no_grad()
    def _kmeans(self, x, k, seed):
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

        previous_assign = None
        for _ in range(KMEANS_MAX_ITERS):
            distance_source = F.normalize(x, dim=-1) if self.normalize else x
            distance_centers = F.normalize(centers, dim=-1) if self.normalize else centers
            distances = torch.cdist(distance_source, distance_centers)
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

    # Find nearest time-series centers.
    def _nearest_ts_center(self, embeds):
        source = F.normalize(embeds, dim=-1) if self.normalize else embeds
        centers = F.normalize(self.ts_centers, dim=-1) if self.normalize else self.ts_centers
        distances = torch.cdist(source.float(), centers.float())
        return distances.argmin(dim=-1), distances.min(dim=-1).values

    # Find nearest vocab centers.
    def _nearest_vocab_center(self, embeds):
        source = F.normalize(embeds, dim=-1) if self.normalize else embeds
        centers = F.normalize(self.vocab_centers, dim=-1) if self.normalize else self.vocab_centers
        distances = torch.cdist(source.float(), centers.float())
        return distances.argmin(dim=-1), distances.min(dim=-1).values

    # Assign each real GPT vocab embedding to its nearest vocab cluster center.
    @torch.no_grad()
    def _assign_vocab_tokens(self, vocab_embeds):
        center_ids, _ = self._nearest_vocab_center(vocab_embeds)
        centers = self.vocab_centers[center_ids].to(device=vocab_embeds.device, dtype=vocab_embeds.dtype)
        center_distances = (vocab_embeds - centers).norm(dim=-1)
        self.vocab_token_center_ids = center_ids.detach()
        self.vocab_token_center_distances = center_distances.detach()

    def _ensure_vocab_token_assignments(self, vocab_embeds):
        if (
            self.vocab_token_center_ids.shape[:1] != vocab_embeds.shape[:1]
            or self.vocab_token_center_ids.device != vocab_embeds.device
            or self.vocab_token_center_distances.device != vocab_embeds.device
        ):
            self._assign_vocab_tokens(vocab_embeds.float())

    def _match_by_center_distance(self, query_center_ids, target_distances, candidate_center_ids, candidate_distances):
        flat_center_ids = query_center_ids.reshape(-1)
        flat_target_distances = target_distances.detach().reshape(-1).float()
        candidate_center_ids = candidate_center_ids.to(device=flat_center_ids.device)
        candidate_distances = candidate_distances.detach().to(device=flat_center_ids.device).float()

        selected_ids = []
        selected_distances = []
        selected_errors = []
        used_fallback = []
        chunk_size = 256
        max_error = torch.finfo(flat_target_distances.dtype).max

        for start in range(0, flat_center_ids.numel(), chunk_size):
            end = min(start + chunk_size, flat_center_ids.numel())
            center_chunk = flat_center_ids[start:end]
            target_chunk = flat_target_distances[start:end]
            same_cluster = candidate_center_ids.unsqueeze(0) == center_chunk.unsqueeze(-1)
            has_cluster_candidate = same_cluster.any(dim=-1, keepdim=True)
            all_candidates = torch.ones_like(same_cluster, dtype=torch.bool)
            candidate_mask = torch.where(has_cluster_candidate, same_cluster, all_candidates)
            errors = (candidate_distances.unsqueeze(0) - target_chunk.unsqueeze(-1)).abs()
            masked_errors = errors.masked_fill(~candidate_mask, max_error)

            chunk_selected_ids = masked_errors.argmin(dim=-1)
            selected_ids.append(chunk_selected_ids)
            selected_distances.append(candidate_distances[chunk_selected_ids])
            selected_errors.append(masked_errors.gather(dim=1, index=chunk_selected_ids.unsqueeze(-1)).squeeze(-1))
            used_fallback.append(~has_cluster_candidate.squeeze(-1))

        output_shape = query_center_ids.shape
        return SimpleNamespace(
            selected_ids=torch.cat(selected_ids, dim=0).reshape(output_shape),
            selected_distances=torch.cat(selected_distances, dim=0).reshape(output_shape),
            selected_errors=torch.cat(selected_errors, dim=0).reshape(output_shape),
            used_fallback=torch.cat(used_fallback, dim=0).reshape(output_shape),
        )

    # Build vocab-to-time inverse map.
    def _inverse_mapping(self):
        inverse = torch.empty_like(self.ts_to_vocab)
        inverse[self.ts_to_vocab] = torch.arange(self.ts_to_vocab.numel(), device=self.ts_to_vocab.device)
        return inverse

    # Map time-series embeds to vocab space.
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

    # Map time-series embeds to real GPT vocab embeddings by distance matching.
    def map_ts_to_vocab_embedding(self, ts_embeds, vocab_embeds):
        if ts_embeds.ndim != 3 or vocab_embeds.ndim != 2:
            raise ValueError("Expected ts_embeds [B,N,D] and vocab_embeds [V,D].")
        if ts_embeds.shape[-1] != vocab_embeds.shape[-1]:
            raise ValueError("Time-series and vocab embeddings must share embedding dimension.")
        if not self.ready:
            source = F.normalize(ts_embeds.float(), dim=-1)
            vocab = F.normalize(vocab_embeds.float(), dim=-1)
            scores = torch.matmul(source.reshape(-1, source.shape[-1]), vocab.transpose(0, 1))
            selected_vocab_token_ids = scores.argmax(dim=-1).reshape(ts_embeds.shape[:-1])
            selected_vocab_embeds = vocab_embeds[selected_vocab_token_ids].to(device=ts_embeds.device, dtype=ts_embeds.dtype)
            unknown_ids = torch.full(ts_embeds.shape[:-1], -1, dtype=torch.long, device=ts_embeds.device)
            zero_distances = torch.zeros(ts_embeds.shape[:-1], device=ts_embeds.device, dtype=ts_embeds.dtype)
            aux = SimpleNamespace(
                ts_center_ids=unknown_ids,
                vocab_center_ids=unknown_ids,
                ts_center_distances=zero_distances,
                ts_residual_distances=zero_distances,
                target_vocab_distances=zero_distances,
                selected_vocab_token_ids=selected_vocab_token_ids,
                selected_vocab_center_ids=unknown_ids,
                selected_vocab_distances=zero_distances,
                selected_distance_errors=zero_distances,
                used_vocab_cluster_fallback=torch.ones(ts_embeds.shape[:-1], dtype=torch.bool, device=ts_embeds.device),
            )
            return selected_vocab_embeds, aux

        self._ensure_vocab_token_assignments(vocab_embeds)
        ts_center_ids, ts_center_distances = self._nearest_ts_center(ts_embeds)
        vocab_center_ids = self.ts_to_vocab[ts_center_ids]
        ts_center = self.ts_centers[ts_center_ids].to(device=ts_embeds.device, dtype=ts_embeds.dtype)

        ts_residual = ts_embeds - ts_center
        ts_residual_distances = ts_residual.norm(dim=-1)
        target_vocab_distances = ts_residual_distances * self.residual_scale
        matched = self._match_by_center_distance(
            vocab_center_ids,
            target_vocab_distances,
            self.vocab_token_center_ids,
            self.vocab_token_center_distances,
        )

        selected_vocab_embeds = vocab_embeds[matched.selected_ids].to(device=ts_embeds.device, dtype=ts_embeds.dtype)
        selected_vocab_center_ids = self.vocab_token_center_ids[matched.selected_ids].to(device=ts_embeds.device)
        aux = SimpleNamespace(
            ts_center_ids=ts_center_ids,
            vocab_center_ids=vocab_center_ids,
            ts_center_distances=ts_center_distances,
            ts_residual_distances=ts_residual_distances,
            target_vocab_distances=target_vocab_distances,
            selected_vocab_token_ids=matched.selected_ids,
            selected_vocab_center_ids=selected_vocab_center_ids,
            selected_vocab_distances=matched.selected_distances.to(device=ts_embeds.device),
            selected_distance_errors=matched.selected_errors.to(device=ts_embeds.device),
            used_vocab_cluster_fallback=matched.used_fallback.to(device=ts_embeds.device),
        )
        return selected_vocab_embeds, aux

    # Map vocab embeds to real history time-series embeddings by inverse distance matching.
    def map_vocab_to_history_ts_embedding(self, vocab_embeds, history_ts_embeds):
        if not self.ready:
            empty_ids = torch.empty(vocab_embeds.shape[:-1], dtype=torch.long, device=vocab_embeds.device)
            empty_distances = torch.empty(vocab_embeds.shape[:-1], device=vocab_embeds.device)
            aux = SimpleNamespace(
                vocab_center_ids=empty_ids,
                ts_center_ids=empty_ids,
                vocab_center_distances=empty_distances,
                vocab_residual_distances=empty_distances,
                target_ts_distances=empty_distances,
                selected_history_indices=empty_ids,
                selected_history_center_ids=empty_ids,
                selected_ts_distances=empty_distances,
                selected_distance_errors=empty_distances,
                used_cluster_fallback=torch.empty(vocab_embeds.shape[:-1], dtype=torch.bool, device=vocab_embeds.device),
            )
            return vocab_embeds, aux
        if vocab_embeds.ndim != 3 or history_ts_embeds.ndim != 3:
            raise ValueError("Expected vocab_embeds [B,F,D] and history_ts_embeds [B,N,D].")
        if vocab_embeds.shape[0] != history_ts_embeds.shape[0] or vocab_embeds.shape[-1] != history_ts_embeds.shape[-1]:
            raise ValueError("Vocab and history embeddings must share batch and embedding dimensions.")

        vocab_center_ids, vocab_center_distances = self._nearest_vocab_center(vocab_embeds)
        inverse = self._inverse_mapping()
        ts_center_ids = inverse[vocab_center_ids]
        vocab_center = self.vocab_centers[vocab_center_ids].to(device=vocab_embeds.device, dtype=vocab_embeds.dtype)
        ts_center = self.ts_centers[ts_center_ids].to(device=vocab_embeds.device, dtype=vocab_embeds.dtype)

        vocab_residual = vocab_embeds - vocab_center
        vocab_residual_distances = vocab_residual.norm(dim=-1)
        inverse_scale = 1.0 / max(self.residual_scale, 1e-6)
        target_ts_distances = vocab_residual_distances * inverse_scale

        detached_history_ts_embeds = history_ts_embeds.detach()
        detached_ts_center = ts_center.detach()
        history_center_ids, _ = self._nearest_ts_center(detached_history_ts_embeds)
        history_distances = (detached_history_ts_embeds.unsqueeze(1) - detached_ts_center.unsqueeze(2)).norm(dim=-1)
        distance_errors = (history_distances - target_ts_distances.detach().unsqueeze(-1)).abs()

        same_cluster = history_center_ids.unsqueeze(1) == ts_center_ids.unsqueeze(-1)
        has_cluster_candidate = same_cluster.any(dim=-1, keepdim=True)
        all_candidates = torch.ones_like(same_cluster, dtype=torch.bool)
        candidate_mask = torch.where(has_cluster_candidate, same_cluster, all_candidates)
        max_error = torch.finfo(distance_errors.dtype).max
        masked_errors = distance_errors.masked_fill(~candidate_mask, max_error)

        selected_history_indices = masked_errors.argmin(dim=-1)
        gather_index = selected_history_indices.unsqueeze(-1).expand(-1, -1, history_ts_embeds.shape[-1])
        selected_ts_embeds = history_ts_embeds.gather(dim=1, index=gather_index)
        selected_ts_distances = history_distances.gather(dim=2, index=selected_history_indices.unsqueeze(-1)).squeeze(-1)
        selected_distance_errors = masked_errors.gather(dim=2, index=selected_history_indices.unsqueeze(-1)).squeeze(-1)
        selected_history_center_ids = history_center_ids.gather(dim=1, index=selected_history_indices)

        aux = SimpleNamespace(
            vocab_center_ids=vocab_center_ids,
            ts_center_ids=ts_center_ids,
            vocab_center_distances=vocab_center_distances,
            vocab_residual_distances=vocab_residual_distances,
            target_ts_distances=target_ts_distances,
            selected_history_indices=selected_history_indices,
            selected_history_center_ids=selected_history_center_ids,
            selected_ts_distances=selected_ts_distances,
            selected_distance_errors=selected_distance_errors,
            used_cluster_fallback=~has_cluster_candidate.squeeze(-1),
        )
        return selected_ts_embeds, aux

    # Map vocab embeds to the paired time-series cluster centers only.
    def map_vocab_to_ts_center(self, vocab_embeds):
        if not self.ready:
            empty_ids = torch.empty(vocab_embeds.shape[:-1], dtype=torch.long, device=vocab_embeds.device)
            aux = SimpleNamespace(
                vocab_center_ids=empty_ids,
                ts_center_ids=empty_ids,
                vocab_center_distances=torch.empty(vocab_embeds.shape[:-1], device=vocab_embeds.device),
            )
            return vocab_embeds, aux

        vocab_center_ids, vocab_center_distances = self._nearest_vocab_center(vocab_embeds)
        inverse = self._inverse_mapping()
        ts_center_ids = inverse[vocab_center_ids]
        ts_centers = self.ts_centers[ts_center_ids].to(device=vocab_embeds.device, dtype=vocab_embeds.dtype)
        aux = SimpleNamespace(
            vocab_center_ids=vocab_center_ids,
            ts_center_ids=ts_center_ids,
            vocab_center_distances=vocab_center_distances,
        )
        return ts_centers, aux


class VocabPatchBridge(nn.Module):
    """Bridge patches through GPT vocab clusters without clustering time-series inputs."""

    def __init__(self, num_clusters, embed_dim, normalize=True, seed=0):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.embed_dim = int(embed_dim)
        self.normalize = bool(normalize)
        self.seed = 0 if seed is None else int(seed)
        self.register_buffer("vocab_centers", torch.empty(self.num_clusters, self.embed_dim), persistent=True)
        self.register_buffer("vocab_token_center_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_token_center_distances", torch.empty(0), persistent=False)
        self.register_buffer("cluster_fitted", torch.tensor(False), persistent=True)

    @property
    def ready(self):
        return bool(
            self.cluster_fitted.item()
            and self.vocab_centers.numel() > 0
            and self.vocab_token_center_ids.numel() > 0
        )

    def _resize_for_checkpoint(self, state_dict, prefix):
        vocab_key = prefix + "vocab_centers"
        if vocab_key in state_dict and self.vocab_centers.shape != state_dict[vocab_key].shape:
            self.vocab_centers = torch.empty_like(state_dict[vocab_key])
            self.num_clusters = int(self.vocab_centers.shape[0])
            self.embed_dim = int(self.vocab_centers.shape[1])

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

    def fit_vocab(self, vocab_embeds):
        self.vocab_centers = self._kmeans(vocab_embeds.float(), self.num_clusters, self.seed + 13)
        self._assign_vocab_tokens(vocab_embeds.float())
        self.cluster_fitted.fill_(True)

    @torch.no_grad()
    def _kmeans(self, x, k, seed):
        x = x.detach().reshape(-1, x.shape[-1])
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        if x.shape[0] >= k:
            indices = torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]
        else:
            indices = torch.randint(0, x.shape[0], (k,), generator=generator, device=x.device)
        centers = x[indices].clone()

        previous_assign = None
        for _ in range(KMEANS_MAX_ITERS):
            distance_source = F.normalize(x, dim=-1) if self.normalize else x
            distance_centers = F.normalize(centers, dim=-1) if self.normalize else centers
            distances = torch.cdist(distance_source, distance_centers)
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

    def _nearest_vocab_center(self, embeds):
        source = F.normalize(embeds, dim=-1) if self.normalize else embeds
        centers = F.normalize(self.vocab_centers, dim=-1) if self.normalize else self.vocab_centers
        distances = torch.cdist(source.float(), centers.float())
        return distances.argmin(dim=-1), distances.min(dim=-1).values

    @torch.no_grad()
    def _assign_vocab_tokens(self, vocab_embeds):
        center_ids, _ = self._nearest_vocab_center(vocab_embeds)
        centers = self.vocab_centers[center_ids].to(device=vocab_embeds.device, dtype=vocab_embeds.dtype)
        center_distances = (vocab_embeds - centers).norm(dim=-1)
        self.vocab_token_center_ids = center_ids.detach()
        self.vocab_token_center_distances = center_distances.detach()

    def _ensure_vocab_token_assignments(self, vocab_embeds):
        if (
            self.vocab_token_center_ids.shape[:1] != vocab_embeds.shape[:1]
            or self.vocab_token_center_ids.device != vocab_embeds.device
            or self.vocab_token_center_distances.device != vocab_embeds.device
        ):
            self._assign_vocab_tokens(vocab_embeds.float())

    def patch_cluster_ids(self, patch_count, device):
        if patch_count > self.num_clusters:
            raise ValueError(
                f"Patch count {patch_count} exceeds vocab cluster count {self.num_clusters}; "
                "increase the cluster count or reduce patch count."
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed + 29 + int(patch_count))
        return torch.randperm(self.num_clusters, generator=generator, device=device)[:patch_count]

    def _match_vocab_tokens(self, cluster_ids, target_distances, vocab_embeds):
        self._ensure_vocab_token_assignments(vocab_embeds)
        flat_cluster_ids = cluster_ids.reshape(-1)
        flat_target_distances = target_distances.detach().reshape(-1).float()
        token_center_ids = self.vocab_token_center_ids.to(device=flat_cluster_ids.device)
        token_distances = self.vocab_token_center_distances.detach().to(device=flat_cluster_ids.device).float()

        selected_ids = []
        selected_distances = []
        selected_errors = []
        used_fallback = []
        max_error = torch.finfo(flat_target_distances.dtype).max
        chunk_size = 256

        for start in range(0, flat_cluster_ids.numel(), chunk_size):
            end = min(start + chunk_size, flat_cluster_ids.numel())
            cluster_chunk = flat_cluster_ids[start:end]
            target_chunk = flat_target_distances[start:end]
            same_cluster = token_center_ids.unsqueeze(0) == cluster_chunk.unsqueeze(-1)
            has_cluster_candidate = same_cluster.any(dim=-1, keepdim=True)
            all_candidates = torch.ones_like(same_cluster, dtype=torch.bool)
            candidate_mask = torch.where(has_cluster_candidate, same_cluster, all_candidates)
            errors = (token_distances.unsqueeze(0) - target_chunk.unsqueeze(-1)).abs()
            masked_errors = errors.masked_fill(~candidate_mask, max_error)
            chunk_ids = masked_errors.argmin(dim=-1)

            selected_ids.append(chunk_ids)
            selected_distances.append(token_distances[chunk_ids])
            selected_errors.append(masked_errors.gather(1, chunk_ids.unsqueeze(-1)).squeeze(-1))
            used_fallback.append(~has_cluster_candidate.squeeze(-1))

        output_shape = cluster_ids.shape
        return SimpleNamespace(
            token_ids=torch.cat(selected_ids, dim=0).reshape(output_shape),
            token_distances=torch.cat(selected_distances, dim=0).reshape(output_shape),
            distance_errors=torch.cat(selected_errors, dim=0).reshape(output_shape),
            used_fallback=torch.cat(used_fallback, dim=0).reshape(output_shape),
        )

    def z_to_vocab_embeddings(self, z_embeds, vocab_embeds):
        if not self.ready:
            raise RuntimeError("VocabPatchBridge must be fitted before mapping patches.")
        if z_embeds.ndim != 3 or vocab_embeds.ndim != 2:
            raise ValueError("Expected z_embeds [B,N,D] and vocab_embeds [V,D].")
        if z_embeds.shape[-1] != vocab_embeds.shape[-1]:
            raise ValueError("Patch and vocab embeddings must share embedding dimension.")

        batch, patch_count, _ = z_embeds.shape
        cluster_ids = self.patch_cluster_ids(patch_count, z_embeds.device)
        cluster_ids = cluster_ids.unsqueeze(0).expand(batch, -1)
        centers = self.vocab_centers[cluster_ids].to(device=z_embeds.device, dtype=z_embeds.dtype)
        target_distances = (z_embeds - centers).norm(dim=-1)
        matched = self._match_vocab_tokens(cluster_ids, target_distances, vocab_embeds)
        selected_vocab_embeds = vocab_embeds[matched.token_ids].to(device=z_embeds.device, dtype=z_embeds.dtype)

        aux = SimpleNamespace(
            cluster_ids=cluster_ids,
            target_distances=target_distances,
            selected_vocab_token_ids=matched.token_ids,
            selected_vocab_distances=matched.token_distances.to(device=z_embeds.device),
            selected_distance_errors=matched.distance_errors.to(device=z_embeds.device),
            used_vocab_cluster_fallback=matched.used_fallback.to(device=z_embeds.device),
        )
        return selected_vocab_embeds, aux

    def vocab_to_history_z(self, vocab_embeds, history_z_embeds):
        if not self.ready:
            raise RuntimeError("VocabPatchBridge must be fitted before inverse mapping.")
        if vocab_embeds.ndim != 3 or history_z_embeds.ndim != 3:
            raise ValueError("Expected vocab_embeds [B,F,D] and history_z_embeds [B,N,D].")
        if vocab_embeds.shape[0] != history_z_embeds.shape[0] or vocab_embeds.shape[-1] != history_z_embeds.shape[-1]:
            raise ValueError("Vocab and history-Z embeddings must share batch and embedding dimensions.")

        batch, history_count, _ = history_z_embeds.shape
        history_cluster_ids = self.patch_cluster_ids(history_count, vocab_embeds.device)
        cluster_to_patch = torch.full((self.num_clusters,), -1, dtype=torch.long, device=vocab_embeds.device)
        cluster_to_patch[history_cluster_ids] = torch.arange(history_count, device=vocab_embeds.device)

        pred_cluster_ids, pred_center_distances = self._nearest_vocab_center(vocab_embeds)
        selected_patch_indices = cluster_to_patch[pred_cluster_ids]
        used_unassigned_fallback = selected_patch_indices < 0
        if used_unassigned_fallback.any():
            assigned_centers = self.vocab_centers[history_cluster_ids].to(device=vocab_embeds.device, dtype=vocab_embeds.dtype)
            source = F.normalize(vocab_embeds, dim=-1) if self.normalize else vocab_embeds
            centers = F.normalize(assigned_centers, dim=-1) if self.normalize else assigned_centers
            assigned_distances = torch.cdist(source.float(), centers.float())
            fallback_indices = assigned_distances.argmin(dim=-1)
            selected_patch_indices = torch.where(used_unassigned_fallback, fallback_indices, selected_patch_indices)

        gather_index = selected_patch_indices.unsqueeze(-1).expand(-1, -1, history_z_embeds.shape[-1])
        selected_z = history_z_embeds.gather(dim=1, index=gather_index)
        aux = SimpleNamespace(
            vocab_center_ids=pred_cluster_ids,
            vocab_center_distances=pred_center_distances,
            selected_history_indices=selected_patch_indices,
            selected_history_cluster_ids=history_cluster_ids[selected_patch_indices],
            used_unassigned_fallback=used_unassigned_fallback,
        )
        return selected_z, aux


class GPT2TS(nn.Module):
    """Patch-cluster GPT2 forecaster with LoRA-tuned attention."""

    # Initialize GPT2TS modules.
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_in = configs.c_in
        self.c_out = configs.c_out
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.history_patch_count = self._patch_count(self.seq_len)

        # gpt
        self.gpt2_path = getattr(configs, "gpt_local_path", "./gpt") 
        self.local_files_only = getattr(configs, "gpt_local_files_only", True) 
        gpt_config = AutoConfig.from_pretrained(self.gpt2_path, local_files_only=self.local_files_only)
        requested_layers = configs.n_layers
        if requested_layers > 0:
            gpt_config.n_layer = min(requested_layers, int(gpt_config.n_layer))
            gpt_config.num_hidden_layers = gpt_config.n_layer
        gpt_config.output_hidden_states = True
        self.gpt_dim = int(gpt_config.hidden_size)
        self.gpt2 = self._load_gpt2(gpt_config)
        self._freeze_gpt2()
        self._inject_lora()
        self.bridge_weight = nn.Parameter(torch.eye(self.gpt_dim))
        self.bridge_bias = nn.Parameter(torch.zeros(self.gpt_dim))

        # tokenizer
        self.patch_tokenizer = PatchTokenizer(
            patch_len=self.patch_len,
            stride=self.stride,
            c_in=self.c_in,
            gpt_dim=self.gpt_dim,
            dropout=getattr(configs, "embedding_dropout", 0.05),
        )
        
        # vocab-cluster bridge
        cluster_seed = getattr(configs, "cluster_seed", 0)
        self.bridge = VocabPatchBridge(
            num_clusters=self.history_patch_count,
            embed_dim=self.gpt_dim,
            normalize=getattr(configs, "cluster_normalize", True),
            seed=cluster_seed,
        )
        self._fit_vocab_clusters()

        if self.pred_len <= self.patch_len:
            self.future_patch_count = 1
        else:
            self.future_patch_count = math.ceil((self.pred_len - self.patch_len) / self.stride) + 1
        self.future_query = nn.Parameter(torch.empty(1, self.gpt_dim))
        nn.init.normal_(self.future_query, mean=0.0, std=self.gpt_dim ** -0.5)

        # decoder
        self.decoder = OverlapAddPatchDecoder(
            patch_len=self.patch_len,
            stride=self.stride,
            pred_len=self.pred_len,
        )
        self.output_dropout = nn.Dropout(float(getattr(configs, "dropout", 0.05)))

    def _patch_count(self, length):
        if length <= self.patch_len:
            return 1
        return math.ceil((length - self.patch_len) / self.stride) + 1

    def _x_to_z(self, x_embeds):
        return torch.matmul(x_embeds, self.bridge_weight.to(device=x_embeds.device, dtype=x_embeds.dtype)) + self.bridge_bias.to(
            device=x_embeds.device,
            dtype=x_embeds.dtype,
        )

    def _z_to_x(self, z_embeds):
        weight = self.bridge_weight.float()
        inverse = torch.linalg.pinv(weight).to(device=z_embeds.device, dtype=z_embeds.dtype)
        bias = self.bridge_bias.to(device=z_embeds.device, dtype=z_embeds.dtype)
        return torch.matmul(z_embeds - bias, inverse)

    # Load pretrained or fresh GPT-2.
    def _load_gpt2(self, config):  
        if getattr(self.configs, "use_pretrained_gpt2", False):
            return AutoModelForCausalLM.from_pretrained(
                self.gpt2_path,
                config=config,
                local_files_only=self.local_files_only,
            )
        return AutoModelForCausalLM.from_config(config)

    # Freeze base GPT-2 params.
    def _freeze_gpt2(self):
        for param in self.gpt2.parameters():
            param.requires_grad = False

    # Inject LoRA into attention projections.
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

    # Get GPT input embeddings.
    def _vocab_weight(self):
        return self.gpt2.get_input_embeddings().weight.detach()

    # Fit vocab clusters at init.
    @torch.no_grad()
    def _fit_vocab_clusters(self):
        if self.bridge.num_clusters <= 0:
            return

        vocab_embeds = self._vocab_weight().to(dtype=torch.float)
        self.bridge.fit_vocab(vocab_embeds)

    @torch.no_grad()
    def _fit_history_clusters(self, history_ts_embeds):
        return

    # Convert logits to predicted vocab embeddings.
    def _predicted_vocab_embeddings(self, logits):
        temperature = max(float(getattr(self.configs, "forecast_temperature", 1.0)), 1e-6)
        top_k = int(getattr(self.configs, "forecast_top_k", 64))
        vocab_weight = self._vocab_weight().to(device=logits.device, dtype=logits.dtype)

        if top_k == 1:
            token_ids = logits.argmax(dim=-1)
            embeds = vocab_weight[token_ids]
            return embeds, token_ids

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

    def _soft_vocab_embeddings(self, logits):
        temperature = max(float(getattr(self.configs, "aux_temperature", 1.0)), 1e-6)
        top_k = int(getattr(self.configs, "aux_embed_top_k", 64))
        vocab_weight = self._vocab_weight().to(device=logits.device, dtype=logits.dtype)

        if top_k > 0 and top_k < logits.shape[-1]:
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            probs = F.softmax(values / temperature, dim=-1)
            selected = vocab_weight[indices]
            return torch.einsum("bfk,bfkd->bfd", probs, selected)

        probs = F.softmax(logits / temperature, dim=-1)
        return torch.matmul(probs, vocab_weight)

    @torch.no_grad()
    def _nearest_vocab_token_ids(self, vocab_embeds):
        vocab_weight = self._vocab_weight().to(device=vocab_embeds.device, dtype=torch.float)
        source = F.normalize(vocab_embeds.float(), dim=-1)
        vocab = F.normalize(vocab_weight, dim=-1)
        scores = torch.matmul(source.reshape(-1, source.shape[-1]), vocab.transpose(0, 1))
        return scores.argmax(dim=-1).reshape(vocab_embeds.shape[:-1])

    @torch.no_grad()
    def _target_future_embeddings(self, batch_y):
        target = batch_y[:, -self.pred_len :, :]
        if target.shape[-1] != self.c_in:
            return None, None, None

        target_patches = self.patch_tokenizer.patchify(target)
        target_patches = target_patches[:, : self.future_patch_count, :, :]
        target_ts_embeds = self.patch_tokenizer.encode(target_patches, apply_dropout=False)
        target_z_embeds = self._x_to_z(target_ts_embeds)
        vocab_weight = self._vocab_weight().to(device=target_ts_embeds.device, dtype=target_ts_embeds.dtype)
        target_vocab_embeds, target_vocab_mapping = self.bridge.z_to_vocab_embeddings(target_z_embeds, vocab_weight)
        return (
            target_ts_embeds.detach(),
            target_vocab_embeds.detach(),
            target_vocab_mapping.selected_vocab_token_ids.detach(),
        )

    def _auxiliary_loss(self, batch_y, aux):
        token_weight = float(getattr(self.configs, "aux_token_loss_weight", 0.0))
        embed_weight = float(getattr(self.configs, "aux_embed_loss_weight", 0.0))
        if token_weight <= 0.0 and embed_weight <= 0.0:
            return None, {}

        target_ts_embeds, target_vocab_embeds, target_token_ids = self._target_future_embeddings(batch_y)
        if target_ts_embeds is None:
            return None, {}

        total = aux.future_logits.new_zeros(())
        losses = {}

        if token_weight > 0.0:
            token_loss = F.cross_entropy(
                aux.future_logits.reshape(-1, aux.future_logits.shape[-1]).float(),
                target_token_ids.reshape(-1),
            )
            total = total + token_weight * token_loss.to(dtype=total.dtype)
            losses["aux_token_loss"] = token_loss.detach()
            losses["aux_target_token_ids"] = target_token_ids.detach()

        if embed_weight > 0.0:
            pred_vocab_embeds = self._soft_vocab_embeddings(aux.future_logits)
            pred = F.normalize(pred_vocab_embeds.float(), dim=-1)
            target = F.normalize(target_vocab_embeds.float(), dim=-1)
            embed_loss = F.mse_loss(pred, target)
            total = total + embed_weight * embed_loss.to(dtype=total.dtype)
            losses["aux_embed_loss"] = embed_loss.detach()

        losses["aux_total_loss"] = total.detach()
        losses["target_ts_embeds"] = target_ts_embeds
        losses["target_vocab_embeds"] = target_vocab_embeds
        return total, losses

    # Forecast future values.
    def forecast(self, batch_x):
        history_patches = self.patch_tokenizer.patchify(batch_x)
        history_ts_embeds = self.patch_tokenizer.encode(history_patches)
        history_z_embeds = self._x_to_z(history_ts_embeds)
        vocab_weight = self._vocab_weight().to(device=batch_x.device, dtype=history_ts_embeds.dtype)
        history_llm_embeds, history_vocab_mapping = self.bridge.z_to_vocab_embeddings(
            history_z_embeds,
            vocab_weight,
        )
        history_llm_embeds = self.output_dropout(history_llm_embeds)

        future_queries = self.future_query.unsqueeze(0).expand(batch_x.shape[0], self.future_patch_count, -1)
        inputs_embeds = torch.cat([history_llm_embeds, future_queries], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=batch_x.device)

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        future_logits = outputs.logits[:, -self.future_patch_count :, :]
        pred_vocab_embeds, pred_token_ids = self._predicted_vocab_embeddings(future_logits)
        pred_z_embeds, cluster_mapping = self.bridge.vocab_to_history_z(
            pred_vocab_embeds,
            history_z_embeds,
        )
        pred_x_queries = self._z_to_x(pred_z_embeds)
        x_distances = torch.cdist(pred_x_queries.float(), history_ts_embeds.float())
        selected_patch_indices = x_distances.argmin(dim=-1)
        gather_index = selected_patch_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            history_patches.shape[-2],
            history_patches.shape[-1],
        )
        pred_patches = history_patches.gather(dim=1, index=gather_index)
        pred_ts_embeds = history_ts_embeds.gather(
            dim=1,
            index=selected_patch_indices.unsqueeze(-1).expand(-1, -1, history_ts_embeds.shape[-1]),
        )

        pred = self.decoder(pred_patches)
        pred = pred[:, : self.pred_len, :]
        aux = SimpleNamespace(
            future_logits=future_logits,
            pred_token_ids=pred_token_ids,
            pred_vocab_embeds=pred_vocab_embeds,
            pred_z_embeds=pred_z_embeds,
            pred_x_queries=pred_x_queries,
            pred_ts_embeds=pred_ts_embeds,
            pred_patches=pred_patches,
            pred_vocab_center_ids=cluster_mapping.vocab_center_ids,
            pred_vocab_center_distances=cluster_mapping.vocab_center_distances,
            pred_selected_z_history_indices=cluster_mapping.selected_history_indices,
            pred_selected_z_history_cluster_ids=cluster_mapping.selected_history_cluster_ids,
            pred_used_unassigned_cluster_fallback=cluster_mapping.used_unassigned_fallback,
            retrieval_indices=selected_patch_indices,
            retrieval_weights=F.one_hot(selected_patch_indices, num_classes=history_patches.shape[1]).to(dtype=history_patches.dtype),
            retrieval_scores=-x_distances,
            history_patches=history_patches,
            history_llm_embeds=history_llm_embeds,
            history_ts_embeds=history_ts_embeds,
            history_z_embeds=history_z_embeds,
            history_vocab_center_ids=history_vocab_mapping.cluster_ids,
            history_selected_vocab_token_ids=history_vocab_mapping.selected_vocab_token_ids,
            history_target_vocab_distances=history_vocab_mapping.target_vocab_distances,
            history_selected_vocab_distances=history_vocab_mapping.selected_vocab_distances,
            history_selected_vocab_distance_errors=history_vocab_mapping.selected_distance_errors,
            history_used_vocab_cluster_fallback=history_vocab_mapping.used_vocab_cluster_fallback,
            mapped_ts_embeds=pred_ts_embeds,
        )
        return pred, aux

    # Return prediction, optional loss, and aux data.
    def forward(self, batch_x, batch_y=None):
        pred, aux = self.forecast(batch_x)
        loss = None
        if batch_y is not None:
            loss, aux_losses = self._auxiliary_loss(batch_y, aux)
            for name, value in aux_losses.items():
                setattr(aux, name, value)
        return SimpleNamespace(pred=pred, loss=loss, aux=aux)
