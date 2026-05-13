import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_EPS = 1e-12


class PatchTokenDictionary(nn.Module):
    """Full-train-set patch/token conversion table."""

    def __init__(
        self,
        cluster_num,
        patch_len,
        c_in,
        normalize=True,
        seed=0,
        match_tol=1e-6,
    ):
        super().__init__()
        self.cluster_num = 1
        self.patch_len = int(patch_len)
        self.c_in = int(c_in)
        self.normalize = bool(normalize)
        self.seed = 0 if seed is None else int(seed)
        self.match_tol = float(match_tol)
        self.match_alpha = 1.0
        self.match_beta = 1.0

        self.register_buffer("train_patches", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("train_patch_features", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("train_patch_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("patch_centers", torch.empty(self.cluster_num, self.patch_len * self.c_in), persistent=False)
        self.register_buffer("patch_cluster_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("patch_center_distances", torch.empty(0), persistent=False)
        self.register_buffer("patch_sort_ranks", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("patch_center_cosines", torch.empty(0), persistent=False)
        self.register_buffer("patch_center_distance_norms", torch.empty(0), persistent=False)
        self.register_buffer("vocab_centers", torch.empty(0, 0), persistent=False)
        self.register_buffer("vocab_token_cluster_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_token_center_distances", torch.empty(0), persistent=False)
        self.register_buffer("vocab_token_ranks", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_token_center_cosines", torch.empty(0), persistent=False)
        self.register_buffer("vocab_token_distance_norms", torch.empty(0), persistent=False)
        self.register_buffer("vocab_cluster_sizes", torch.empty(self.cluster_num, dtype=torch.long), persistent=False)
        self.register_buffer("patch_to_vocab_cluster", torch.empty(self.cluster_num, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_to_patch_cluster", torch.empty(self.cluster_num, dtype=torch.long), persistent=False)
        self.register_buffer("mean_patch_center_distance", torch.tensor(1.0), persistent=False)
        self.register_buffer("mean_vocab_center_distance", torch.tensor(1.0), persistent=False)
        self.register_buffer("valid_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("train_patch_match_scores", torch.empty(0), persistent=False)
        self.register_buffer("fitted", torch.tensor(False), persistent=False)
        self.last_exact_match_count = 0
        self.last_total_match_count = 0
        self.last_dropped_points = 0

    @property
    def ready(self):
        return bool(
            self.fitted.item()
            and self.train_patches.numel() > 0
            and self.train_patch_features.numel() > 0
            and self.train_patch_token_ids.numel() > 0
            and self.valid_token_ids.numel() > 0
        )

    def _resize_for_checkpoint(self, state_dict, prefix):
        centers_key = prefix + "patch_centers"
        if centers_key in state_dict:
            self.cluster_num = int(state_dict[centers_key].shape[0])
        for name in (
            "train_patches",
            "train_patch_features",
            "train_patch_token_ids",
            "patch_centers",
            "patch_cluster_ids",
            "patch_center_distances",
            "patch_sort_ranks",
            "patch_center_cosines",
            "patch_center_distance_norms",
            "vocab_centers",
            "vocab_token_cluster_ids",
            "vocab_token_center_distances",
            "vocab_token_ranks",
            "vocab_token_center_cosines",
            "vocab_token_distance_norms",
            "vocab_cluster_sizes",
            "patch_to_vocab_cluster",
            "vocab_to_patch_cluster",
            "mean_patch_center_distance",
            "mean_vocab_center_distance",
            "valid_token_ids",
            "train_patch_match_scores",
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

    def _patch_features(self, patches):
        # Data loaders already scale the series; do not standardize each patch again.
        return patches.float()

    def _pad_patch_features(self, flat_patches, target_dim):
        if flat_patches.shape[-1] > target_dim:
            raise ValueError(
                "Cannot zero-pad patch features because patch dimension "
                f"{flat_patches.shape[-1]} is larger than token embedding dimension {target_dim}."
            )
        if flat_patches.shape[-1] == target_dim:
            return flat_patches
        return F.pad(flat_patches, (0, target_dim - flat_patches.shape[-1]))

    def _center_stats(self, x, center):
        center = center.float()
        distances = (x.float() - center.unsqueeze(0)).norm(dim=-1)
        mean_distance = distances.mean().clamp_min(FEATURE_EPS)
        distance_norms = distances / mean_distance
        cosines = F.cosine_similarity(
            x.float(),
            center.unsqueeze(0).expand_as(x).float(),
            dim=-1,
            eps=FEATURE_EPS,
        )
        return cosines, distances, distance_norms, mean_distance

    @torch.no_grad()
    def _match_patch_stats_to_tokens(self, patch_cosines, patch_distance_norms, token_cosines, token_distance_norms):
        token_ids = torch.empty(patch_cosines.shape[0], dtype=torch.long, device=patch_cosines.device)
        match_scores = torch.empty(patch_cosines.shape[0], dtype=patch_cosines.dtype, device=patch_cosines.device)
        chunk_size = max(int(getattr(self, "token_match_chunk_size", 256)), 1)

        for start in range(0, patch_cosines.shape[0], chunk_size):
            end = min(start + chunk_size, patch_cosines.shape[0])
            score = (
                self.match_alpha * (token_cosines.unsqueeze(0) - patch_cosines[start:end].unsqueeze(1)).abs()
                + self.match_beta
                * (token_distance_norms.unsqueeze(0) - patch_distance_norms[start:end].unsqueeze(1)).abs()
            )
            best_scores, best_token_ids = score.min(dim=-1)
            token_ids[start:end] = best_token_ids
            match_scores[start:end] = best_scores

        return token_ids, match_scores

    @torch.no_grad()
    def _find_prior_matching_patch(self, flat_patches, patch_idx):
        if patch_idx <= 0:
            return None

        current = flat_patches[patch_idx : patch_idx + 1].float()
        previous = flat_patches[:patch_idx].float()
        chunk_size = max(int(getattr(self, "nearest_chunk_size", 1024)), 1)
        best_distance = None
        best_index = None

        for start in range(0, previous.shape[0], chunk_size):
            end = min(start + chunk_size, previous.shape[0])
            distances = torch.cdist(current, previous[start:end]).squeeze(0)
            distance, local_index = distances.min(dim=0)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = start + int(local_index.item())

        if best_distance is not None and float(best_distance.item()) <= self.match_tol:
            return best_index
        return None

    def _patch_token_score(self, patch_idx, token_id, patch_cosines, patch_distance_norms, token_cosines, token_distance_norms):
        return (
            self.match_alpha * (token_cosines[token_id] - patch_cosines[patch_idx]).abs()
            + self.match_beta * (token_distance_norms[token_id] - patch_distance_norms[patch_idx]).abs()
        )

    @torch.no_grad()
    def _assign_unique_patch_tokens(
        self,
        flat_patches,
        vocab_embeds,
        patch_cosines,
        patch_distance_norms,
        token_cosines,
        token_distance_norms,
    ):
        best_token_ids, _ = self._match_patch_stats_to_tokens(
            patch_cosines,
            patch_distance_norms,
            token_cosines,
            token_distance_norms,
        )
        train_token_ids = torch.empty(flat_patches.shape[0], dtype=torch.long, device=flat_patches.device)
        patch_match_scores = torch.empty(flat_patches.shape[0], dtype=patch_cosines.dtype, device=flat_patches.device)
        used_tokens = torch.zeros(vocab_embeds.shape[0], dtype=torch.bool, device=flat_patches.device)

        for patch_idx in range(flat_patches.shape[0]):
            matching_patch_idx = self._find_prior_matching_patch(flat_patches, patch_idx)
            if matching_patch_idx is not None:
                token_id = int(train_token_ids[matching_patch_idx].item())
            else:
                best_token_id = int(best_token_ids[patch_idx].item())
                if not bool(used_tokens[best_token_id].item()):
                    token_id = best_token_id
                else:
                    unused_token_ids = torch.nonzero(~used_tokens, as_tuple=False).flatten()
                    if unused_token_ids.numel() == 0:
                        raise RuntimeError(
                            "Cannot assign unique tokens because all GPT vocab tokens are already used."
                        )
                    distances = torch.cdist(
                        vocab_embeds[best_token_id].float().unsqueeze(0),
                        vocab_embeds[unused_token_ids].float(),
                    ).squeeze(0)
                    token_id = int(unused_token_ids[distances.argmin()].item())
                used_tokens[token_id] = True

            train_token_ids[patch_idx] = token_id
            patch_match_scores[patch_idx] = self._patch_token_score(
                patch_idx,
                token_id,
                patch_cosines,
                patch_distance_norms,
                token_cosines,
                token_distance_norms,
            )

        valid_token_ids = torch.nonzero(used_tokens, as_tuple=False).flatten()
        if valid_token_ids.numel() == 0:
            raise RuntimeError("No valid patch-token assignments were produced.")
        return train_token_ids, patch_match_scores, valid_token_ids

    def _distance_ranks(self, distances):
        ranks = torch.empty(distances.shape[0], dtype=torch.long, device=distances.device)
        if distances.numel() == 0:
            return ranks
        ranks[torch.argsort(distances)] = torch.arange(distances.numel(), device=distances.device)
        return ranks

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
        self.last_dropped_points = int(remainder)
        if remainder:
            usable_length = series.shape[1] - remainder
            if usable_length <= 0:
                raise ValueError(
                    f"Training series length {series.shape[1]} is shorter than patch_len {self.patch_len}; "
                    "cannot build any full patch with drop_last."
                )
            series = series[:, :usable_length, :]
        patches = series.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.reshape(-1, self.patch_len, self.c_in)

    @torch.no_grad()
    def fit(self, train_series, vocab_embeds):
        train_patches = self._patchify_series(train_series)
        train_patch_features = self._patch_features(train_patches)
        flat_patches = train_patch_features.reshape(train_patch_features.shape[0], -1).float()
        vocab_embeds = vocab_embeds.detach().float().to(flat_patches.device)

        padded_patches = self._pad_patch_features(flat_patches, vocab_embeds.shape[-1])
        patch_centers = padded_patches.mean(dim=0, keepdim=True)
        vocab_centers = vocab_embeds.mean(dim=0, keepdim=True)

        patch_cosines, patch_distances, patch_distance_norms, mean_patch_distance = self._center_stats(
            padded_patches,
            patch_centers[0],
        )
        vocab_cosines, vocab_distances, vocab_distance_norms, mean_vocab_distance = self._center_stats(
            vocab_embeds,
            vocab_centers[0],
        )
        train_token_ids, patch_match_scores, valid_token_ids = self._assign_unique_patch_tokens(
            flat_patches,
            vocab_embeds,
            patch_cosines,
            patch_distance_norms,
            vocab_cosines,
            vocab_distance_norms,
        )

        self.train_patches = train_patches.detach()
        self.train_patch_features = train_patch_features.detach()
        self.patch_centers = patch_centers.detach()
        self.patch_cluster_ids = torch.zeros(flat_patches.shape[0], dtype=torch.long, device=flat_patches.device)
        self.patch_center_distances = patch_distances.detach()
        self.patch_sort_ranks = self._distance_ranks(patch_distances).detach()
        self.patch_center_cosines = patch_cosines.detach()
        self.patch_center_distance_norms = patch_distance_norms.detach()
        self.vocab_centers = vocab_centers.detach()
        self.vocab_token_cluster_ids = torch.zeros(vocab_embeds.shape[0], dtype=torch.long, device=flat_patches.device)
        self.vocab_token_center_distances = vocab_distances.detach()
        self.vocab_token_ranks = self._distance_ranks(vocab_distances).detach()
        self.vocab_token_center_cosines = vocab_cosines.detach()
        self.vocab_token_distance_norms = vocab_distance_norms.detach()
        self.vocab_cluster_sizes = torch.tensor([vocab_embeds.shape[0]], dtype=torch.long, device=flat_patches.device)
        self.patch_to_vocab_cluster = torch.zeros(1, dtype=torch.long, device=flat_patches.device)
        self.vocab_to_patch_cluster = torch.zeros(1, dtype=torch.long, device=flat_patches.device)
        self.mean_patch_center_distance = mean_patch_distance.detach()
        self.mean_vocab_center_distance = mean_vocab_distance.detach()
        self.valid_token_ids = valid_token_ids.detach()
        self.train_patch_match_scores = patch_match_scores.detach()
        self.train_patch_token_ids = train_token_ids.detach()
        self.fitted.fill_(True)

    @torch.no_grad()
    def patches_to_token_ids(self, patches):
        if not self.ready:
            raise RuntimeError("PatchTokenDictionary must be fitted before converting patches to tokens.")
        patch_features = self._patch_features(patches.reshape(-1, self.patch_len, self.c_in))
        flat = patch_features.reshape(patch_features.shape[0], -1).float()
        selected_indices = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        selected_distances = torch.empty(flat.shape[0], dtype=flat.dtype, device=flat.device)

        train_flat = self.train_patch_features.reshape(self.train_patch_features.shape[0], -1).float()
        chunk_size = max(int(getattr(self, "nearest_chunk_size", 1024)), 1)

        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            distances = torch.cdist(flat[start:end], train_flat)
            nearest_indices = distances.argmin(dim=-1)
            selected_indices[start:end] = nearest_indices
            selected_distances[start:end] = distances.gather(1, nearest_indices.unsqueeze(-1)).squeeze(-1)

        nearest_indices = selected_indices
        nearest_distances = selected_distances
        self.last_exact_match_count = int((nearest_distances <= self.match_tol).sum().item())
        self.last_total_match_count = int(flat.shape[0])
        token_ids = self.train_patch_token_ids[nearest_indices]
        return token_ids.reshape(patches.shape[0], patches.shape[1])

    @torch.no_grad()
    def token_ids_to_patches(self, token_ids, vocab_embeds=None):
        if not self.ready:
            raise RuntimeError("PatchTokenDictionary must be fitted before converting tokens to patches.")

        device = self.train_patch_token_ids.device
        flat_tokens = token_ids.reshape(-1).long().to(device)
        train_token_ids = self.train_patch_token_ids.to(device)
        used_token_ids = self.valid_token_ids.to(device)
        selected_patches = torch.empty(
            flat_tokens.shape[0],
            self.patch_len,
            self.c_in,
            dtype=self.train_patches.dtype,
            device=device,
        )

        if vocab_embeds is not None:
            vocab_embeds = vocab_embeds.detach().float().to(device)
            used_vocab_embeds = vocab_embeds[used_token_ids]
        else:
            used_vocab_embeds = None

        for pos, token_id in enumerate(flat_tokens):
            token_id = int(token_id.item())
            patch_indices = torch.nonzero(train_token_ids == token_id, as_tuple=False).flatten()
            if patch_indices.numel() == 0:
                if used_vocab_embeds is None:
                    raise RuntimeError(
                        "Generated token is not in the patch-token table; pass vocab_embeds to find "
                        "the nearest used token."
                    )
                token_id = max(0, min(token_id, vocab_embeds.shape[0] - 1))
                distances = torch.cdist(vocab_embeds[token_id].unsqueeze(0), used_vocab_embeds).squeeze(0)
                nearest_used_token = used_token_ids[distances.argmin()]
                patch_indices = torch.nonzero(train_token_ids == nearest_used_token, as_tuple=False).flatten()
                if patch_indices.numel() == 0:
                    raise RuntimeError("Nearest used token has no patch entry. Refit the patch-token map.")

            selected_patches[pos] = self.train_patches[patch_indices].mean(dim=0)

        return selected_patches.reshape(*token_ids.shape, self.patch_len, self.c_in)

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
            patch_sort_ranks=self.patch_sort_ranks.detach().cpu().numpy(),
            patch_center_cosines=self.patch_center_cosines.detach().cpu().numpy(),
            patch_center_distance_norms=self.patch_center_distance_norms.detach().cpu().numpy(),
            vocab_centers=self.vocab_centers.detach().cpu().numpy(),
            vocab_token_cluster_ids=self.vocab_token_cluster_ids.detach().cpu().numpy(),
            vocab_token_center_distances=self.vocab_token_center_distances.detach().cpu().numpy(),
            vocab_token_ranks=self.vocab_token_ranks.detach().cpu().numpy(),
            vocab_token_center_cosines=self.vocab_token_center_cosines.detach().cpu().numpy(),
            vocab_token_distance_norms=self.vocab_token_distance_norms.detach().cpu().numpy(),
            vocab_cluster_sizes=self.vocab_cluster_sizes.detach().cpu().numpy(),
            patch_to_vocab_cluster=self.patch_to_vocab_cluster.detach().cpu().numpy(),
            vocab_to_patch_cluster=self.vocab_to_patch_cluster.detach().cpu().numpy(),
            mean_patch_center_distance=self.mean_patch_center_distance.detach().cpu().numpy(),
            mean_vocab_center_distance=self.mean_vocab_center_distance.detach().cpu().numpy(),
            valid_token_ids=self.valid_token_ids.detach().cpu().numpy(),
            train_patch_match_scores=self.train_patch_match_scores.detach().cpu().numpy(),
            map_version=np.array(3),
            cluster_num=np.array(self.cluster_num),
            patch_len=np.array(self.patch_len),
            c_in=np.array(self.c_in),
            normalize=np.array(self.normalize),
            seed=np.array(self.seed),
            match_tol=np.array(self.match_tol),
            dropped_points=np.array(self.last_dropped_points),
        )

    @torch.no_grad()
    def load_npz(self, path):
        data = np.load(path)
        device = self.train_patches.device
        map_version = int(data["map_version"]) if "map_version" in data else 1
        if map_version < 3:
            raise ValueError(
                "This patch-token map was saved with an old mapping format. "
                "Refit the patch-token map to use unique patch-token assignments."
            )
        self.cluster_num = int(data["cluster_num"]) if "cluster_num" in data else int(data["patch_centers"].shape[0])
        self.patch_len = int(data["patch_len"]) if "patch_len" in data else self.patch_len
        self.c_in = int(data["c_in"]) if "c_in" in data else self.c_in
        self.normalize = bool(data["normalize"]) if "normalize" in data else self.normalize
        if "patch_level_normalize" in data and bool(data["patch_level_normalize"]):
            raise ValueError(
                "This patch-token map was saved with patch-level normalization. "
                "Refit the patch-token map because patch-level normalization has been removed."
            )
        self.seed = int(data["seed"]) if "seed" in data else self.seed
        self.match_tol = float(data["match_tol"]) if "match_tol" in data else self.match_tol
        self.last_dropped_points = int(data["dropped_points"]) if "dropped_points" in data else 0

        self.train_patches = torch.as_tensor(data["train_patches"], dtype=torch.float32, device=device)
        self.train_patch_features = self._patch_features(self.train_patches).detach()
        self.train_patch_token_ids = torch.as_tensor(data["train_patch_token_ids"], dtype=torch.long, device=device)
        self.patch_centers = torch.as_tensor(data["patch_centers"], dtype=torch.float32, device=device)
        self.patch_cluster_ids = torch.as_tensor(data["patch_cluster_ids"], dtype=torch.long, device=device)
        self.patch_center_distances = torch.as_tensor(data["patch_center_distances"], dtype=torch.float32, device=device)
        self.patch_sort_ranks = torch.as_tensor(data["patch_sort_ranks"], dtype=torch.long, device=device)
        self.patch_center_cosines = torch.as_tensor(data["patch_center_cosines"], dtype=torch.float32, device=device)
        self.patch_center_distance_norms = torch.as_tensor(
            data["patch_center_distance_norms"],
            dtype=torch.float32,
            device=device,
        )
        self.vocab_centers = torch.as_tensor(data["vocab_centers"], dtype=torch.float32, device=device)
        self.vocab_token_cluster_ids = torch.as_tensor(data["vocab_token_cluster_ids"], dtype=torch.long, device=device)
        self.vocab_token_center_distances = torch.as_tensor(
            data["vocab_token_center_distances"],
            dtype=torch.float32,
            device=device,
        )
        self.vocab_token_ranks = torch.as_tensor(data["vocab_token_ranks"], dtype=torch.long, device=device)
        self.vocab_token_center_cosines = torch.as_tensor(
            data["vocab_token_center_cosines"],
            dtype=torch.float32,
            device=device,
        )
        self.vocab_token_distance_norms = torch.as_tensor(
            data["vocab_token_distance_norms"],
            dtype=torch.float32,
            device=device,
        )
        self.vocab_cluster_sizes = torch.as_tensor(data["vocab_cluster_sizes"], dtype=torch.long, device=device)
        self.patch_to_vocab_cluster = torch.as_tensor(data["patch_to_vocab_cluster"], dtype=torch.long, device=device)
        self.vocab_to_patch_cluster = torch.as_tensor(data["vocab_to_patch_cluster"], dtype=torch.long, device=device)
        self.mean_patch_center_distance = torch.as_tensor(
            data["mean_patch_center_distance"],
            dtype=torch.float32,
            device=device,
        )
        self.mean_vocab_center_distance = torch.as_tensor(
            data["mean_vocab_center_distance"],
            dtype=torch.float32,
            device=device,
        )
        self.valid_token_ids = torch.as_tensor(data["valid_token_ids"], dtype=torch.long, device=device)
        self.train_patch_match_scores = torch.as_tensor(data["train_patch_match_scores"], dtype=torch.float32, device=device)
        self.fitted.fill_(True)
