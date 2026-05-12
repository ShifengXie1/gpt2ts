import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KMEANS_TOL = 1e-6
KMEANS_MAX_ITERS = 100


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
        self.cluster_num = int(cluster_num)
        self.patch_len = int(patch_len)
        self.c_in = int(c_in)
        self.normalize = bool(normalize)
        self.seed = 0 if seed is None else int(seed)
        self.match_tol = float(match_tol)

        self.register_buffer("train_patches", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("train_patch_features", torch.empty(0, self.patch_len, self.c_in), persistent=False)
        self.register_buffer("train_patch_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("patch_centers", torch.empty(self.cluster_num, self.patch_len * self.c_in), persistent=False)
        self.register_buffer("patch_cluster_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("patch_center_distances", torch.empty(0), persistent=False)
        self.register_buffer("vocab_centers", torch.empty(0, 0), persistent=False)
        self.register_buffer("vocab_token_cluster_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_token_center_distances", torch.empty(0), persistent=False)
        self.register_buffer("vocab_token_ranks", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_cluster_sizes", torch.empty(self.cluster_num, dtype=torch.long), persistent=False)
        self.register_buffer("patch_to_vocab_cluster", torch.empty(self.cluster_num, dtype=torch.long), persistent=False)
        self.register_buffer("vocab_to_patch_cluster", torch.empty(self.cluster_num, dtype=torch.long), persistent=False)
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

    def _patch_features(self, patches):
        # Data loaders already scale the series; do not standardize each patch again.
        return patches.float()

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

            if sorted_vocab_indices.numel() < sorted_patch_indices.numel():
                raise ValueError(
                    "Cannot build a one-to-one patch-token map because the mapped GPT vocab cluster "
                    f"has {sorted_vocab_indices.numel()} token(s), but the patch cluster has "
                    f"{sorted_patch_indices.numel()} patch(es)."
                )
            token_ids[sorted_patch_indices] = sorted_vocab_indices[: sorted_patch_indices.numel()]
        return token_ids

    @torch.no_grad()
    def fit(self, train_series, vocab_embeds):
        train_patches = self._patchify_series(train_series)
        train_patch_features = self._patch_features(train_patches)
        flat_patches = train_patch_features.reshape(train_patch_features.shape[0], -1).float()
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
        self.train_patch_features = train_patch_features.detach()
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
                train_flat = self.train_patch_features.reshape(self.train_patch_features.shape[0], -1).float()
                selected_patch_indices[pos] = torch.cdist(patch_center.float(), train_flat).argmin(dim=-1).item()
                continue

            sorted_patch_indices = patch_indices[torch.argsort(self.patch_center_distances[patch_indices])]
            token_rank = int(self.vocab_token_ranks[token_id].item())
            patch_pos = min(token_rank, sorted_patch_indices.numel() - 1)
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
        self.vocab_centers = torch.as_tensor(data["vocab_centers"], dtype=torch.float32, device=device)
        self.vocab_token_cluster_ids = torch.as_tensor(data["vocab_token_cluster_ids"], dtype=torch.long, device=device)
        self.vocab_token_center_distances = torch.as_tensor(
            data["vocab_token_center_distances"],
            dtype=torch.float32,
            device=device,
        )
        self.vocab_token_ranks = torch.as_tensor(data["vocab_token_ranks"], dtype=torch.long, device=device)
        self.vocab_cluster_sizes = torch.as_tensor(data["vocab_cluster_sizes"], dtype=torch.long, device=device)
        self.patch_to_vocab_cluster = torch.as_tensor(data["patch_to_vocab_cluster"], dtype=torch.long, device=device)
        self.vocab_to_patch_cluster = torch.as_tensor(data["vocab_to_patch_cluster"], dtype=torch.long, device=device)
        self.fitted.fill_(True)
