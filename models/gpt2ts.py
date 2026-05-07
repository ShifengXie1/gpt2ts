import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from models.patchdecoder import KeyValueMemoryRetriever, OverlapAddPatchDecoder
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

    # Map vocab embeds back to time-series space.
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

        # tokenizer
        self.patch_tokenizer = PatchTokenizer(
            patch_len=self.patch_len,
            stride=self.stride,
            c_in=self.c_in,
            gpt_dim=self.gpt_dim,
            dropout=getattr(configs, "embedding_dropout", 0.05),
        )
        
        # kmeans bridge
        cluster_seed = getattr(configs, "cluster_seed", 0)
        self.bridge = KMeansBridge(
            num_clusters=getattr(configs, "num_clusters", 64),
            embed_dim=self.gpt_dim,
            residual_scale=getattr(configs, "cluster_residual_scale", 1.0),
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
        self.memory_retriever = KeyValueMemoryRetriever(
            temperature=getattr(configs, "retrieval_temperature", 1.0),
            mode=getattr(configs, "retrieval_mode", "straight_through"),
            normalize=getattr(configs, "retrieval_normalize", True),
        )
        self.decoder = OverlapAddPatchDecoder(
            patch_len=self.patch_len,
            stride=self.stride,
            pred_len=self.pred_len,
        )
        self.output_dropout = nn.Dropout(float(getattr(configs, "dropout", 0.05)))

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
        num_clusters = int(getattr(self.configs, "num_clusters", 64))
        if num_clusters <= 0:
            return

        vocab_embeds = self._vocab_weight().to(dtype=torch.float)
        self.bridge.fit_vocab(vocab_embeds)

    @torch.no_grad()
    def _fit_history_clusters(self, history_ts_embeds):
        if int(getattr(self.configs, "num_clusters", 64)) <= 0:
            return
        self.bridge.fit_ts_to_vocab(history_ts_embeds.detach())

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

    # Forecast future values.
    def forecast(self, batch_x):
        history_patches = self.patch_tokenizer.patchify(batch_x)
        history_ts_embeds = self.patch_tokenizer.encode(history_patches)
        self._fit_history_clusters(history_ts_embeds)
        history_llm_embeds = self.bridge.map_ts_to_vocab_space(history_ts_embeds)
        history_llm_embeds = self.output_dropout(history_llm_embeds)

        future_queries = self.future_query.unsqueeze(0).expand(batch_x.shape[0], self.future_patch_count, -1)
        inputs_embeds = torch.cat([history_llm_embeds, future_queries], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=batch_x.device)

        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        future_logits = outputs.logits[:, -self.future_patch_count :, :]
        pred_vocab_embeds, pred_token_ids = self._predicted_vocab_embeddings(future_logits)
        pred_ts_embeds, cluster_mapping = self.bridge.map_vocab_to_ts_center(pred_vocab_embeds)

        pred_patches, retrieval = self.memory_retriever(pred_ts_embeds, history_ts_embeds, history_patches)
        pred = self.decoder(pred_patches)
        pred = pred[:, : self.pred_len, :]
        aux = SimpleNamespace(
            pred_token_ids=pred_token_ids,
            pred_vocab_embeds=pred_vocab_embeds,
            pred_ts_embeds=pred_ts_embeds,
            pred_patches=pred_patches,
            pred_vocab_center_ids=cluster_mapping.vocab_center_ids,
            pred_ts_center_ids=cluster_mapping.ts_center_ids,
            pred_vocab_center_distances=cluster_mapping.vocab_center_distances,
            retrieval_indices=retrieval.indices,
            retrieval_weights=retrieval.weights,
            retrieval_scores=retrieval.scores,
            history_patches=history_patches,
            history_llm_embeds=history_llm_embeds,
            history_ts_embeds=history_ts_embeds,
            mapped_ts_embeds=pred_ts_embeds,
        )
        return pred, aux

    # Return prediction, optional loss, and aux data.
    def forward(self, batch_x, batch_y=None):
        pred, aux = self.forecast(batch_x)
        loss = None
        if batch_y is not None:
            loss = F.mse_loss(pred, batch_y)
        return SimpleNamespace(pred=pred, loss=loss, aux=aux)
