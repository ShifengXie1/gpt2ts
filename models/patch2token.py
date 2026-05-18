import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_EPS = 1e-12


class PatchTokenDictionary(nn.Module):
    """Historical patch motif <-> existing GPT token conversion table."""

    def __init__(
        self,
        cluster_num,
        patch_len,
        c_in,
        stride=None,
        normalize=True,
        seed=0,
        match_tol=0.001,
        kmeans_iters=30,
    ):
        super().__init__()
        self.cluster_num = max(int(cluster_num), 1)
        self.patch_len = int(patch_len)
        self.stride = int(stride) if stride is not None else int(patch_len)
        self.c_in = int(c_in)
        self.normalize = bool(normalize)
        self.seed = 0 if seed is None else int(seed)
        self.match_tol = float(match_tol)
        self.kmeans_iters = max(int(kmeans_iters), 1)
        self.nearest_chunk_size = 1024
        self.last_exact_match_count = 0
        self.last_total_match_count = 0
        self.last_dropped_points = 0

        self.register_buffer("train_patches", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("train_patch_features", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("train_patch_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("train_patch_motif_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("train_patch_match_scores", torch.empty(0), persistent=False)

        self.register_buffer("patch_centers", torch.empty(0, self.patch_len * self.c_in), persistent=False)
        self.register_buffer("patch_cluster_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("patch_cluster_sizes", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("motif_patches", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("motif_features", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("motif_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("token_to_motif_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("valid_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("fitted", torch.tensor(False), persistent=False)

    @property
    def ready(self):
        return bool(
            self.fitted.item()
            and self.train_patches.numel() > 0
            and self.train_patch_token_ids.numel() > 0
            and self.motif_patches.numel() > 0
            and self.motif_token_ids.numel() > 0
            and self.valid_token_ids.numel() > 0
        )

    def _resize_for_checkpoint(self, state_dict, prefix):
        dynamic_names = (
            "train_patches",
            "train_patch_features",
            "train_patch_token_ids",
            "train_patch_motif_ids",
            "train_patch_match_scores",
            "patch_centers",
            "patch_cluster_ids",
            "patch_cluster_sizes",
            "motif_patches",
            "motif_features",
            "motif_token_ids",
            "token_to_motif_ids",
            "valid_token_ids",
            "fitted",
        )
        for name in dynamic_names:
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

    def _patch_features(self, patches):
        patches = patches.float()
        if not self.normalize:
            return patches
        mean = patches.mean(dim=1, keepdim=True)
        std = patches.std(dim=1, keepdim=True, unbiased=False).clamp_min(FEATURE_EPS)
        return (patches - mean) / std

    def _pad_patch_features(self, flat_patches, target_dim):
        if flat_patches.shape[-1] > target_dim:
            raise ValueError(
                "Cannot zero-pad patch features because patch dimension "
                f"{flat_patches.shape[-1]} is larger than token embedding dimension {target_dim}."
            )
        if flat_patches.shape[-1] == target_dim:
            return flat_patches
        return F.pad(flat_patches, (0, target_dim - flat_patches.shape[-1]))

    def _nearest_center_ids(self, flat, centers):
        ids = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        distances_out = torch.empty(flat.shape[0], dtype=flat.dtype, device=flat.device)
        chunk_size = max(int(self.nearest_chunk_size), 1)
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            distances = torch.cdist(flat[start:end], centers)
            best_distances, best_ids = distances.min(dim=-1)
            ids[start:end] = best_ids
            distances_out[start:end] = best_distances
        return ids, distances_out

    def _kmeans(self, flat_patches):
        num_patches = flat_patches.shape[0]
        cluster_num = min(self.cluster_num, num_patches)
        if cluster_num <= 0:
            raise ValueError("No patches are available for motif clustering.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        initial_indices = torch.randperm(num_patches, generator=generator)[:cluster_num].to(flat_patches.device)
        centers = flat_patches[initial_indices].clone()

        cluster_ids = None
        for _ in range(self.kmeans_iters):
            cluster_ids, _ = self._nearest_center_ids(flat_patches, centers)
            counts = torch.bincount(cluster_ids, minlength=cluster_num).to(flat_patches.dtype)
            new_centers = torch.zeros_like(centers)
            new_centers.index_add_(0, cluster_ids, flat_patches)

            non_empty = counts > 0
            new_centers[non_empty] = new_centers[non_empty] / counts[non_empty].unsqueeze(-1)
            if bool((~non_empty).any().item()):
                replacement_indices = torch.randperm(num_patches, generator=generator)[: int((~non_empty).sum().item())]
                replacement_indices = replacement_indices.to(flat_patches.device)
                new_centers[~non_empty] = flat_patches[replacement_indices]

            if torch.allclose(new_centers, centers, atol=1e-6, rtol=1e-6):
                centers = new_centers
                break
            centers = new_centers

        cluster_ids, distances = self._nearest_center_ids(flat_patches, centers)
        return cluster_ids, centers, distances

    def _select_medoids(self, train_patches, train_patch_features, flat_features, cluster_ids, centers):
        motif_patches = []
        motif_features = []
        cluster_sizes = []

        for motif_id in range(centers.shape[0]):
            member_indices = torch.nonzero(cluster_ids == motif_id, as_tuple=False).flatten()
            if member_indices.numel() == 0:
                distances = torch.cdist(centers[motif_id : motif_id + 1], flat_features).squeeze(0)
                medoid_index = int(distances.argmin().item())
                cluster_sizes.append(0)
            else:
                member_flat = flat_features[member_indices]
                distances = torch.cdist(centers[motif_id : motif_id + 1], member_flat).squeeze(0)
                medoid_index = int(member_indices[distances.argmin()].item())
                cluster_sizes.append(int(member_indices.numel()))

            motif_patches.append(train_patches[medoid_index])
            motif_features.append(train_patch_features[medoid_index])

        return (
            torch.stack(motif_patches, dim=0),
            torch.stack(motif_features, dim=0),
            torch.tensor(cluster_sizes, dtype=torch.long, device=train_patches.device),
        )

    def _assign_motif_tokens(self, motif_features, vocab_embeds, candidate_token_ids):
        vocab_embeds = vocab_embeds.detach().float()
        vocab_size = vocab_embeds.shape[0]
        if candidate_token_ids is None:
            candidate_token_ids = torch.arange(vocab_size, dtype=torch.long, device=vocab_embeds.device)
        else:
            candidate_token_ids = torch.as_tensor(candidate_token_ids, dtype=torch.long, device=vocab_embeds.device)
            candidate_token_ids = candidate_token_ids[(candidate_token_ids >= 0) & (candidate_token_ids < vocab_size)]
            candidate_token_ids = torch.unique(candidate_token_ids, sorted=True)

        if candidate_token_ids.numel() < motif_features.shape[0]:
            raise ValueError(
                f"Need at least {motif_features.shape[0]} candidate GPT tokens, "
                f"but only {candidate_token_ids.numel()} are available."
            )

        motif_flat = motif_features.reshape(motif_features.shape[0], -1).float()
        motif_flat = self._pad_patch_features(motif_flat, vocab_embeds.shape[-1])
        motif_flat = F.normalize(motif_flat, dim=-1, eps=FEATURE_EPS)
        candidate_embeds = F.normalize(vocab_embeds[candidate_token_ids], dim=-1, eps=FEATURE_EPS)
        scores = 1.0 - torch.matmul(motif_flat, candidate_embeds.transpose(0, 1))

        motif_token_ids = torch.empty(motif_features.shape[0], dtype=torch.long, device=motif_features.device)
        used = torch.zeros(candidate_token_ids.shape[0], dtype=torch.bool, device=motif_features.device)
        order = torch.argsort(self.patch_cluster_sizes, descending=True)

        for motif_id in order.tolist():
            motif_scores = scores[motif_id].clone()
            motif_scores[used] = torch.inf
            candidate_pos = int(motif_scores.argmin().item())
            motif_token_ids[motif_id] = candidate_token_ids[candidate_pos]
            used[candidate_pos] = True

        token_to_motif_ids = torch.full((vocab_size,), -1, dtype=torch.long, device=motif_features.device)
        token_to_motif_ids[motif_token_ids] = torch.arange(motif_features.shape[0], device=motif_features.device)
        return motif_token_ids, token_to_motif_ids

    @torch.no_grad()
    def _patchify_series(self, series):
        series = torch.as_tensor(series, dtype=torch.float32, device=self.train_patches.device)
        if series.ndim == 2:
            series = series.unsqueeze(0)
        if series.ndim != 3:
            raise ValueError("Expected train series with shape [T,C] or [1,T,C].")
        if series.shape[-1] != self.c_in:
            raise ValueError(f"Expected {self.c_in} input channel(s), got {series.shape[-1]}.")
        if series.shape[1] < self.patch_len:
            raise ValueError(
                f"Training series length {series.shape[1]} is shorter than patch_len {self.patch_len}."
            )

        usable_length = ((series.shape[1] - self.patch_len) // self.stride) * self.stride + self.patch_len
        self.last_dropped_points = int(series.shape[1] - usable_length)
        series = series[:, :usable_length, :]
        patches = series.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.reshape(-1, self.patch_len, self.c_in)

    @torch.no_grad()
    def fit(self, train_series, vocab_embeds, candidate_token_ids=None):
        train_patches = self._patchify_series(train_series)
        train_patch_features = self._patch_features(train_patches)
        flat_features = train_patch_features.reshape(train_patch_features.shape[0], -1).float()

        cluster_ids, centers, distances = self._kmeans(flat_features)
        motif_patches, motif_features, cluster_sizes = self._select_medoids(
            train_patches,
            train_patch_features,
            flat_features,
            cluster_ids,
            centers,
        )

        self.patch_cluster_sizes = cluster_sizes.detach()
        motif_token_ids, token_to_motif_ids = self._assign_motif_tokens(
            motif_features,
            vocab_embeds.detach().float().to(flat_features.device),
            candidate_token_ids,
        )

        train_token_ids = motif_token_ids[cluster_ids]

        self.train_patches = train_patches.detach()
        self.train_patch_features = train_patch_features.detach()
        self.train_patch_motif_ids = cluster_ids.detach()
        self.train_patch_token_ids = train_token_ids.detach()
        self.train_patch_match_scores = distances.detach()
        self.patch_centers = centers.detach()
        self.patch_cluster_ids = cluster_ids.detach()
        self.motif_patches = motif_patches.detach()
        self.motif_features = motif_features.detach()
        self.motif_token_ids = motif_token_ids.detach()
        self.token_to_motif_ids = token_to_motif_ids.detach()
        self.valid_token_ids = motif_token_ids.detach()
        self.cluster_num = int(motif_token_ids.shape[0])
        self.fitted.fill_(True)

    @torch.no_grad()
    def patches_to_token_ids(self, patches):
        if not self.ready:
            raise RuntimeError("PatchTokenDictionary must be fitted before converting patches to tokens.")
        patch_features = self._patch_features(patches.reshape(-1, self.patch_len, self.c_in))
        flat = patch_features.reshape(patch_features.shape[0], -1).float()
        motif_ids, nearest_distances = self._nearest_center_ids(flat, self.patch_centers.float())
        self.last_exact_match_count = int((nearest_distances <= self.match_tol).sum().item())
        self.last_total_match_count = int(flat.shape[0])
        token_ids = self.motif_token_ids[motif_ids]
        return token_ids.reshape(patches.shape[0], patches.shape[1])

    @torch.no_grad()
    def token_ids_to_patches(self, token_ids, vocab_embeds=None):
        if not self.ready:
            raise RuntimeError("PatchTokenDictionary must be fitted before converting tokens to patches.")

        device = self.motif_patches.device
        flat_tokens = token_ids.reshape(-1).long().to(device)
        motif_ids = torch.full_like(flat_tokens, -1)
        known = (flat_tokens >= 0) & (flat_tokens < self.token_to_motif_ids.shape[0])
        motif_ids[known] = self.token_to_motif_ids[flat_tokens[known]]

        missing = motif_ids < 0
        if bool(missing.any().item()):
            if vocab_embeds is None:
                raise RuntimeError(
                    "Generated token is not in the motif-token table; pass vocab_embeds to find "
                    "the nearest used token."
                )
            vocab_embeds = vocab_embeds.detach().float().to(device)
            used_token_ids = self.valid_token_ids.to(device)
            used_vocab_embeds = vocab_embeds[used_token_ids]
            missing_token_ids = flat_tokens[missing].clamp(min=0, max=vocab_embeds.shape[0] - 1)
            replacement = torch.empty_like(missing_token_ids)

            chunk_size = max(int(self.nearest_chunk_size), 1)
            for start in range(0, missing_token_ids.shape[0], chunk_size):
                end = min(start + chunk_size, missing_token_ids.shape[0])
                distances = torch.cdist(vocab_embeds[missing_token_ids[start:end]], used_vocab_embeds)
                replacement[start:end] = used_token_ids[distances.argmin(dim=-1)]

            motif_ids[missing] = self.token_to_motif_ids[replacement]

        if bool((motif_ids < 0).any().item()):
            raise RuntimeError("Some generated tokens could not be mapped back to motifs.")

        selected_patches = self.motif_patches[motif_ids]
        return selected_patches.reshape(*token_ids.shape, self.patch_len, self.c_in)

    @torch.no_grad()
    def save_npz(self, path):
        if not self.ready:
            raise RuntimeError("Cannot save an unfitted PatchTokenDictionary.")
        np.savez_compressed(
            path,
            train_patches=self.train_patches.detach().cpu().numpy(),
            train_patch_features=self.train_patch_features.detach().cpu().numpy(),
            train_patch_token_ids=self.train_patch_token_ids.detach().cpu().numpy(),
            train_patch_motif_ids=self.train_patch_motif_ids.detach().cpu().numpy(),
            train_patch_match_scores=self.train_patch_match_scores.detach().cpu().numpy(),
            patch_centers=self.patch_centers.detach().cpu().numpy(),
            patch_cluster_ids=self.patch_cluster_ids.detach().cpu().numpy(),
            patch_cluster_sizes=self.patch_cluster_sizes.detach().cpu().numpy(),
            motif_patches=self.motif_patches.detach().cpu().numpy(),
            motif_features=self.motif_features.detach().cpu().numpy(),
            motif_token_ids=self.motif_token_ids.detach().cpu().numpy(),
            token_to_motif_ids=self.token_to_motif_ids.detach().cpu().numpy(),
            valid_token_ids=self.valid_token_ids.detach().cpu().numpy(),
            map_version=np.array(4),
            cluster_num=np.array(self.cluster_num),
            patch_len=np.array(self.patch_len),
            stride=np.array(self.stride),
            c_in=np.array(self.c_in),
            normalize=np.array(self.normalize),
            seed=np.array(self.seed),
            match_tol=np.array(self.match_tol),
            kmeans_iters=np.array(self.kmeans_iters),
            dropped_points=np.array(self.last_dropped_points),
        )

    @torch.no_grad()
    def load_npz(self, path):
        data = np.load(path)
        device = self.train_patches.device
        map_version = int(data["map_version"]) if "map_version" in data else 1
        if map_version < 4:
            raise ValueError(
                "This patch-token map was saved with an old mapping format. "
                "Refit the patch-token map to use motif-to-token assignments."
            )

        self.cluster_num = int(data["cluster_num"])
        self.patch_len = int(data["patch_len"])
        self.stride = int(data["stride"]) if "stride" in data else self.patch_len
        self.c_in = int(data["c_in"])
        self.normalize = bool(data["normalize"])
        self.seed = int(data["seed"])
        self.match_tol = float(data["match_tol"])
        self.kmeans_iters = int(data["kmeans_iters"]) if "kmeans_iters" in data else self.kmeans_iters
        self.last_dropped_points = int(data["dropped_points"]) if "dropped_points" in data else 0

        self.train_patches = torch.as_tensor(data["train_patches"], dtype=torch.float32, device=device)
        self.train_patch_features = torch.as_tensor(data["train_patch_features"], dtype=torch.float32, device=device)
        self.train_patch_token_ids = torch.as_tensor(data["train_patch_token_ids"], dtype=torch.long, device=device)
        self.train_patch_motif_ids = torch.as_tensor(data["train_patch_motif_ids"], dtype=torch.long, device=device)
        self.train_patch_match_scores = torch.as_tensor(data["train_patch_match_scores"], dtype=torch.float32, device=device)
        self.patch_centers = torch.as_tensor(data["patch_centers"], dtype=torch.float32, device=device)
        self.patch_cluster_ids = torch.as_tensor(data["patch_cluster_ids"], dtype=torch.long, device=device)
        self.patch_cluster_sizes = torch.as_tensor(data["patch_cluster_sizes"], dtype=torch.long, device=device)
        self.motif_patches = torch.as_tensor(data["motif_patches"], dtype=torch.float32, device=device)
        self.motif_features = torch.as_tensor(data["motif_features"], dtype=torch.float32, device=device)
        self.motif_token_ids = torch.as_tensor(data["motif_token_ids"], dtype=torch.long, device=device)
        self.token_to_motif_ids = torch.as_tensor(data["token_to_motif_ids"], dtype=torch.long, device=device)
        self.valid_token_ids = torch.as_tensor(data["valid_token_ids"], dtype=torch.long, device=device)
        self.fitted.fill_(True)
