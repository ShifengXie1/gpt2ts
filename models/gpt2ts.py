from types import SimpleNamespace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from models.lora import LoRAConv1D
from models.native_gpt_vocab import (
    NativeGPTVocabQuantizer,
    NativePatchDecoder,
    NativePatchEncoder,
    build_candidate_token_ids,
)
from models.patch2token import PatchTokenDictionary
from models.patch_token_layers import PatchBankDecoder, PatchProjector, ResidualHead, l2_normalize


class ValidTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, valid_token_ids):
        self.valid_token_ids = valid_token_ids.detach().reshape(-1).long()

    def __call__(self, input_ids, scores):
        valid_token_ids = self.valid_token_ids.to(scores.device)
        valid_token_ids = valid_token_ids[(valid_token_ids >= 0) & (valid_token_ids < scores.shape[-1])]
        if valid_token_ids.numel() == 0:
            raise RuntimeError("No valid token ids are available for constrained GPT generation.")

        masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked_scores[:, valid_token_ids] = scores[:, valid_token_ids]
        return masked_scores


class GPT2TS(nn.Module):
    """Patch-token GPT-2 forecaster using original GPT token ids."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.c_in = int(configs.c_in)
        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)
        self.patch_tokenizer = str(getattr(configs, "patch_tokenizer", "patch2token"))
        self.use_native_gpt_vocab = self.patch_tokenizer == "native_gpt_vocab"
        if self.patch_tokenizer not in {"patch2token", "native_gpt_vocab"}:
            raise ValueError("Unsupported patch_tokenizer '{}'. ".format(self.patch_tokenizer))

        if not self.use_native_gpt_vocab and (getattr(configs, "features", "S") != "S" or self.c_in != 1):
            raise ValueError("The patch-token GPT flow currently supports only features='S' with c_in=1.")
        if self.patch_len <= 0 or self.stride <= 0:
            raise ValueError("patch_len and stride must be positive.")
        if self.seq_len < self.patch_len:
            raise ValueError("seq_len must be at least patch_len.")
        if self.stride > self.patch_len:
            raise ValueError("stride must be <= patch_len for patch-token forecasting.")
        if self.patch_len % self.stride != 0:
            raise ValueError("patch_len must be divisible by stride for aligned patch-token forecasting.")
        if self.seq_len % self.stride != 0:
            raise ValueError("seq_len must be divisible by stride for aligned patch-token forecasting.")

        self.history_patch_count = self._history_patch_count(self.seq_len)
        self.future_patch_count = self._future_patch_count(self.pred_len)
        self.boundary_patch_count = max(self.seq_len // self.stride - self.history_patch_count, 0)
        self.generated_patch_count = self.boundary_patch_count + self.future_patch_count
        self.patch_dim = self.patch_len * self.c_in

        self.lambda_ce = float(getattr(configs, "lambda_ce", 0.3))
        self.lambda_mse = float(getattr(configs, "lambda_mse", 1.0))
        self.lambda_align = float(getattr(configs, "lambda_align", 0.1))
        self.lambda_smooth = float(getattr(configs, "lambda_smooth", 0.05))
        self.mse_temperature = float(getattr(configs, "mse_temperature", 1.0))
        self.align_temperature = float(getattr(configs, "align_temperature", 1.0))
        self.residual_scale = float(getattr(configs, "residual_scale", 0.1))
        self.use_trainable_patch_projector = bool(getattr(configs, "use_trainable_patch_projector", True))
        self.use_patch_bank_attention = bool(getattr(configs, "use_patch_bank_attention", True))

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

        self.original_vocab_size = int(self.gpt2.get_input_embeddings().weight.shape[0])
        self.gpt_dim = int(self.gpt2.get_input_embeddings().weight.shape[-1])
        self.tokenizer = self._load_tokenizer()
        if self.use_native_gpt_vocab:
            self._init_native_gpt_vocab()
        else:
            self._init_patch2token_legacy()

    def _init_patch2token_legacy(self):
        patch_encoder_dim = int(getattr(self.configs, "patch_encoder_dim", 256))
        patch_bank_attn_dim = int(getattr(self.configs, "patch_bank_attn_dim", 128))
        self.patch_projector = PatchProjector(self.patch_dim, patch_encoder_dim, self.gpt_dim)
        self.patch_bank_decoder = PatchBankDecoder(
            self.gpt_dim,
            self.patch_dim,
            patch_bank_attn_dim,
            use_attention=self.use_patch_bank_attention,
        )
        self.residual_head = ResidualHead(self.gpt_dim, self.patch_dim, patch_encoder_dim)
        if not self.use_trainable_patch_projector:
            for param in self.patch_projector.parameters():
                param.requires_grad = False

        cluster_num = int(getattr(self.configs, "cluster_num", 1))
        self.dictionary = PatchTokenDictionary(
            cluster_num=cluster_num,
            patch_len=self.patch_len,
            c_in=self.c_in,
            stride=self.stride,
            normalize=getattr(self.configs, "cluster_normalize", False),
            seed=getattr(self.configs, "cluster_seed", 0),
            kmeans_iters=getattr(self.configs, "kmeans_iters", 30),
            patch_bank_topk=getattr(self.configs, "patch_bank_topk", 8),
            assignment_method=getattr(self.configs, "assignment_method", "hungarian"),
        )

    def _init_native_gpt_vocab(self):
        if int(getattr(self.configs, "native_token_k", 4)) <= 0:
            raise ValueError("--native_token_k must be positive.")

        self.native_token_k = int(getattr(self.configs, "native_token_k", 4))
        self.use_inputs_embeds_for_training = bool(getattr(self.configs, "use_inputs_embeds_for_training", True))
        self.recon_loss_weight = float(getattr(self.configs, "recon_loss_weight", 1.0))
        self.token_ce_loss_weight = float(getattr(self.configs, "token_ce_loss_weight", 1.0))
        self.forecast_loss_weight = float(getattr(self.configs, "forecast_loss_weight", 1.0))
        self.commitment_loss_weight = float(getattr(self.configs, "commitment_loss_weight", 0.25))
        self.usage_loss_weight = float(getattr(self.configs, "usage_loss_weight", 0.01))
        self.train_stage = str(getattr(self.configs, "train_stage", "joint_train"))
        self.freeze_gpt_codebook = bool(getattr(self.configs, "freeze_gpt_codebook", True))

        allowed_token_ids = build_candidate_token_ids(
            self.tokenizer,
            self.original_vocab_size,
            mode=getattr(self.configs, "candidate_token_mode", "all"),
            candidate_token_file=getattr(self.configs, "candidate_token_file", None),
            device=self.gpt2.get_input_embeddings().weight.device,
        )
        self.native_patch_encoder = NativePatchEncoder(
            self.patch_len,
            self.c_in,
            int(getattr(self.configs, "patch_encoder_hidden_dim", 512)),
            self.native_token_k,
            self.gpt_dim,
        )
        self.native_quantizer = NativeGPTVocabQuantizer(
            allowed_token_ids,
            self.original_vocab_size,
            tau=float(getattr(self.configs, "vq_tau", 1.0)),
            tau_min=float(getattr(self.configs, "vq_tau_min", 0.05)),
            use_straight_through=bool(getattr(self.configs, "use_straight_through", True)),
            freeze_codebook=self.freeze_gpt_codebook,
        )
        self.native_patch_decoder = NativePatchDecoder(
            self.patch_len,
            self.c_in,
            int(getattr(self.configs, "patch_decoder_hidden_dim", 512)),
            self.native_token_k,
            self.gpt_dim,
        )
        self._configure_native_train_stage()
        print(
            "Native GPT vocab tokenizer: candidate_tokens={} | K={} | original_vocab_size={}".format(
                int(allowed_token_ids.numel()),
                self.native_token_k,
                self.original_vocab_size,
            )
        )

    def _load_gpt2(self, config):
        if getattr(self.configs, "use_pretrained_gpt2", False):
            return AutoModelForCausalLM.from_pretrained(
                self.gpt2_path,
                config=config,
                local_files_only=self.local_files_only,
            )
        return AutoModelForCausalLM.from_config(config)

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.gpt2_path, local_files_only=self.local_files_only)
        except Exception:
            return None

    def _freeze_gpt2(self):
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def _set_module_trainable(self, module, trainable):
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    def _freeze_gpt_codebook_params(self):
        embedding = self.gpt2.get_input_embeddings()
        if embedding is not None:
            for param in embedding.parameters():
                param.requires_grad = False
        output_embedding = self.gpt2.get_output_embeddings()
        if output_embedding is not None:
            for param in output_embedding.parameters():
                param.requires_grad = False

    def _configure_native_train_stage(self):
        if not self.use_native_gpt_vocab:
            return
        if self.train_stage not in {"tokenizer_pretrain", "gpt_train", "joint_train"}:
            raise ValueError("Unsupported train_stage '{}'.".format(self.train_stage))

        train_tokenizer = self.train_stage in {"tokenizer_pretrain", "joint_train"}
        train_gpt = self.train_stage in {"gpt_train", "joint_train"}
        self._set_module_trainable(self.native_patch_encoder, train_tokenizer)
        self._set_module_trainable(self.native_patch_decoder, train_tokenizer)

        if not train_gpt:
            self._freeze_gpt2()
        if self.freeze_gpt_codebook:
            self._freeze_gpt_codebook_params()

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

    def _candidate_token_ids(self):
        vocab_weight = self._vocab_weight()
        vocab_size = vocab_weight.shape[0]
        special_ids = {
            token_id
            for token_id in [
                getattr(self.gpt2.config, "bos_token_id", None),
                getattr(self.gpt2.config, "eos_token_id", None),
                getattr(self.gpt2.config, "pad_token_id", None),
                getattr(self.gpt2.config, "unk_token_id", None),
            ]
            if token_id is not None
        }
        if self.tokenizer is not None:
            special_ids.update(int(token_id) for token_id in getattr(self.tokenizer, "all_special_ids", []))

        candidate_limit = int(getattr(self.configs, "candidate_token_num", 4096) or 0)
        if candidate_limit <= 0:
            candidate_limit = int(getattr(self.configs, "candidate_token_count", 0) or vocab_size)

        selected = []
        for token_id in range(vocab_size):
            if token_id in special_ids:
                continue
            if self.tokenizer is not None:
                text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                if text.strip() == "":
                    continue
                if not any(ch.isprintable() for ch in text):
                    continue
            selected.append(token_id)
            if len(selected) >= candidate_limit:
                break

        if not selected:
            selected = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
        return torch.tensor(selected, dtype=torch.long, device=vocab_weight.device)

    def _patch_features(self, patches):
        flat_shape = patches.shape[:-2]
        features = self.dictionary._patch_features(patches.reshape(-1, self.patch_len, self.c_in))
        return features.reshape(*flat_shape, self.patch_len, self.c_in)

    def _project_patch_features(self, patch_features):
        original_shape = patch_features.shape[:-2]
        flat = patch_features.reshape(-1, self.patch_dim).float()
        projected = self.patch_projector(flat)
        projected = l2_normalize(projected)
        return projected.reshape(*original_shape, self.gpt_dim)

    def _project_motif_features_for_assignment(self, motif_features):
        return self._project_patch_features(motif_features)

    def _native_vocab_weight(self, device=None, dtype=None, detach=None):
        weight = self.gpt2.get_input_embeddings().weight
        should_detach = self.freeze_gpt_codebook if detach is None else bool(detach)
        if should_detach:
            weight = weight.detach()
        if device is not None or dtype is not None:
            weight = weight.to(device=device if device is not None else weight.device, dtype=dtype if dtype is not None else weight.dtype)
        return weight

    def _native_check_token_ids(self, token_ids):
        if token_ids.numel() == 0:
            raise RuntimeError("Native GPT vocab tokenizer produced no token ids.")
        assert int(token_ids.min().item()) >= 0
        assert int(token_ids.max().item()) < self.original_vocab_size

    def _native_pad_to_patch_coverage(self, series, patch_count):
        required_len = (int(patch_count) - 1) * self.stride + self.patch_len
        if series.shape[1] >= required_len:
            return series[:, :required_len, :]
        pad_len = required_len - series.shape[1]
        pad_value = series[:, -1:, :].expand(series.shape[0], pad_len, series.shape[2])
        return torch.cat([series, pad_value], dim=1)

    def _native_patchify_future(self, target):
        target = target[:, -self.pred_len :, :]
        target = self._native_pad_to_patch_coverage(target, self.future_patch_count)
        patches = self._patchify_any(target)
        patches = patches[:, : self.future_patch_count].contiguous()
        if patches.shape[1] != self.future_patch_count:
            raise RuntimeError(
                "Expected {} future patches, got {}.".format(self.future_patch_count, patches.shape[1])
            )
        return patches

    def _native_quantize_patches(self, patches):
        patch_repr = self.native_patch_encoder(patches)
        quantized = self.native_quantizer(
            patch_repr,
            self._native_vocab_weight(device=patch_repr.device, dtype=patch_repr.dtype),
        )
        batch_size, patch_count = patches.shape[:2]
        assert quantized.token_ids.shape == (batch_size, patch_count, self.native_token_k)
        assert quantized.token_embeds.shape == (batch_size, patch_count, self.native_token_k, self.gpt_dim)
        self._native_check_token_ids(quantized.token_ids)
        quantized.patch_repr = patch_repr
        return quantized

    def _native_flatten_token_ids(self, token_ids):
        flat = token_ids.reshape(token_ids.shape[0], token_ids.shape[1] * token_ids.shape[2]).long()
        self._native_check_token_ids(flat)
        return flat

    def _native_flatten_token_embeds(self, token_embeds):
        return token_embeds.reshape(token_embeds.shape[0], token_embeds.shape[1] * token_embeds.shape[2], token_embeds.shape[-1])

    def _native_assert_context_length(self, token_count):
        max_positions = int(getattr(self.gpt2.config, "n_positions", token_count))
        if int(token_count) > max_positions:
            raise ValueError("Native token sequence length {} exceeds GPT context length {}.".format(token_count, max_positions))

    def _native_ce_from_logits(self, logits, target_token_ids):
        if logits.shape[:2] != target_token_ids.shape:
            raise RuntimeError(
                "Native CE shape mismatch: logits {} target {}.".format(
                    tuple(logits.shape),
                    tuple(target_token_ids.shape),
                )
            )
        self._native_check_token_ids(target_token_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_token_ids.reshape(-1),
        )

    def _native_gpt_next_token_logits(self, flat_token_ids, flat_token_embeds):
        if flat_token_ids.shape[1] < 2:
            raise RuntimeError("Need at least two native GPT tokens for next-token training.")

        self._native_assert_context_length(flat_token_ids.shape[1])
        gpt_inputs = flat_token_ids[:, :-1].contiguous()
        targets = flat_token_ids[:, 1:].contiguous()
        attention_mask = torch.ones_like(gpt_inputs)
        if self.use_inputs_embeds_for_training:
            input_embeds = flat_token_embeds[:, :-1, :].contiguous()
            outputs = self.gpt2(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            outputs = self.gpt2(
                input_ids=gpt_inputs,
                attention_mask=attention_mask,
                return_dict=True,
            )
        if outputs.logits.shape[-1] != self.original_vocab_size:
            raise RuntimeError("GPT logits vocab size changed; resizing token embeddings is not allowed.")
        return outputs.logits, targets

    def native_training_loss(self, batch_x, target, force_gpt=False):
        if not self.use_native_gpt_vocab:
            raise RuntimeError("native_training_loss is only available with --patch_tokenizer native_gpt_vocab.")

        history_patches = self._patchify_batch(batch_x)
        future_patches = self._native_patchify_future(target)
        all_patches = torch.cat([history_patches, future_patches], dim=1)
        quantized = self._native_quantize_patches(all_patches)

        history_token_embeds = quantized.token_embeds[:, : self.history_patch_count]
        recon_patches = self.native_patch_decoder(history_token_embeds)
        loss_recon = F.mse_loss(recon_patches, history_patches)

        zero = quantized.commitment_loss.new_zeros(())
        loss_ce = zero
        loss_forecast = zero
        pred_patches = None
        run_gpt = force_gpt or self.train_stage in {"gpt_train", "joint_train"}
        flat_token_ids = self._native_flatten_token_ids(quantized.token_ids)
        flat_token_embeds = self._native_flatten_token_embeds(quantized.token_embeds)

        if run_gpt:
            logits, targets = self._native_gpt_next_token_logits(flat_token_ids, flat_token_embeds)
            loss_ce = self._native_ce_from_logits(logits, targets)

            history_token_count = self.history_patch_count * self.native_token_k
            future_token_count = self.future_patch_count * self.native_token_k
            start = history_token_count - 1
            end = start + future_token_count
            if end <= logits.shape[1]:
                future_logits = logits[:, start:end, :]
                future_probs = torch.softmax(future_logits, dim=-1)
                codebook = self._native_vocab_weight(
                    device=future_probs.device,
                    dtype=future_probs.dtype,
                    detach=True,
                )
                pred_embeds = torch.matmul(future_probs, codebook)
                pred_embeds = pred_embeds.reshape(
                    batch_x.shape[0],
                    self.future_patch_count,
                    self.native_token_k,
                    self.gpt_dim,
                )
                pred_patches = self.native_patch_decoder(pred_embeds)
                loss_forecast = F.mse_loss(pred_patches, future_patches)

        if self.train_stage == "tokenizer_pretrain":
            total_loss = (
                self.recon_loss_weight * loss_recon
                + self.commitment_loss_weight * quantized.commitment_loss
                + self.usage_loss_weight * quantized.usage_loss
            )
        elif self.train_stage == "gpt_train":
            total_loss = self.token_ce_loss_weight * loss_ce
        elif self.train_stage == "joint_train":
            total_loss = (
                self.token_ce_loss_weight * loss_ce
                + self.recon_loss_weight * loss_recon
                + self.forecast_loss_weight * loss_forecast
                + self.commitment_loss_weight * quantized.commitment_loss
                + self.usage_loss_weight * quantized.usage_loss
            )
        else:
            raise ValueError("Unsupported train_stage '{}'.".format(self.train_stage))

        return SimpleNamespace(
            total_loss=total_loss,
            loss_ce=loss_ce,
            loss_recon=loss_recon,
            loss_forecast=loss_forecast,
            commitment_loss=quantized.commitment_loss,
            usage_loss=quantized.usage_loss,
            history_patches=history_patches,
            future_patches=future_patches,
            recon_patches=recon_patches,
            pred_patches=pred_patches,
            token_ids=quantized.token_ids,
        )

    @torch.no_grad()
    def native_debug_token_check(self, batch_x, max_items=12):
        history_patches = self._patchify_batch(batch_x)
        quantized = self._native_quantize_patches(history_patches)
        flat_ids = quantized.token_ids.reshape(-1)
        self._native_check_token_ids(flat_ids)
        sample_ids = flat_ids[: int(max_items)].detach().cpu().tolist()
        return {
            "min_token_id": int(flat_ids.min().detach().cpu().item()),
            "max_token_id": int(flat_ids.max().detach().cpu().item()),
            "original_vocab_size": int(self.original_vocab_size),
            "candidate_token_count": int(self.native_quantizer.allowed_token_ids.numel()),
            "sample_token_ids": sample_ids,
        }

    @torch.no_grad()
    def fit_patch_token_map(self, train_series):
        motif_projector = self._project_motif_features_for_assignment if self.use_trainable_patch_projector else None
        self.dictionary.fit(
            train_series,
            self._vocab_weight(),
            self._candidate_token_ids(),
            motif_projector=motif_projector,
        )
        valid_count = int(self.dictionary.valid_token_ids.numel())
        if valid_count < 2:
            raise ValueError(
                "Patch-token map has fewer than 2 valid GPT tokens. "
                "Increase --cluster_num; using --cluster_num 1 makes token loss exactly 0 "
                "and forces every forecast patch to be identical."
            )

    @torch.no_grad()
    def save_patch_token_map(self, path):
        self.dictionary.save_npz(path)

    @torch.no_grad()
    def load_patch_token_map(self, path):
        self.dictionary.load_npz(path)

    @torch.no_grad()
    def refresh_patch_token_assignment(self):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before refreshing patch-token assignments.")
        motif_projector = self._project_motif_features_for_assignment if self.use_trainable_patch_projector else None
        return self.dictionary.refresh_assignment(
            self._vocab_weight(),
            candidate_token_ids=self.dictionary.candidate_token_ids,
            motif_projector=motif_projector,
        )

    @torch.no_grad()
    def build_lm_training_tensors(self):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before building token LM training data.")

        tokens = self.dictionary.train_patch_token_ids.detach().long()
        patches = self.dictionary.train_patches.detach().float()
        candidate_indices = self.dictionary.train_patch_candidate_indices.detach().long()
        window_size = self.history_patch_count + self.generated_patch_count
        if tokens.numel() < window_size:
            raise ValueError(
                f"Training token sequence has {tokens.numel()} patches, but at least {window_size} are required."
            )

        max_positions = int(getattr(self.gpt2.config, "n_positions", window_size))
        if window_size > max_positions:
            raise ValueError(
                f"Token training window length {window_size} exceeds GPT context length {max_positions}."
            )

        step = max(int(getattr(self.configs, "token_train_stride", 1)), 1)
        window_indices = torch.arange(tokens.numel(), device=tokens.device).unfold(0, window_size, step).contiguous()
        input_ids = tokens[window_indices].contiguous()
        labels = input_ids.clone().contiguous()
        labels[:, : self.history_patch_count] = -100
        patch_windows = patches[window_indices].contiguous()
        candidate_windows = candidate_indices[window_indices].contiguous()
        return SimpleNamespace(
            input_ids=input_ids,
            labels=labels,
            future_patches=patch_windows[:, -self.future_patch_count :].contiguous(),
            align_patches=patch_windows,
            align_candidate_indices=candidate_windows,
        )

    def _valid_token_ids(self, device):
        valid_token_ids = self.dictionary.valid_token_ids.to(device=device).long()
        return valid_token_ids[(valid_token_ids >= 0) & (valid_token_ids < self._vocab_weight().shape[0])]

    def _token_ce_from_logits(self, logits, labels):
        valid_token_ids = self._valid_token_ids(logits.device)
        if valid_token_ids.numel() == 0:
            raise RuntimeError("No valid token ids are available for masked token LM training.")
        if valid_token_ids.numel() < 2:
            raise RuntimeError(
                "Only one valid patch token is available, so cross-entropy is always 0. "
                "Increase --cluster_num before training."
            )

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].to(device=logits.device, dtype=torch.long).contiguous()
        active_positions = shift_labels != -100
        if not bool(active_positions.any().item()):
            raise RuntimeError("No active labels are available for token LM training.")

        active_labels = shift_labels[active_positions]
        if bool(((active_labels < 0) | (active_labels >= logits.shape[-1])).any().item()):
            raise RuntimeError("Token LM labels contain ids outside the GPT vocabulary.")

        label_to_valid_index = torch.full(
            (logits.shape[-1],),
            -100,
            dtype=torch.long,
            device=logits.device,
        )
        label_to_valid_index[valid_token_ids] = torch.arange(valid_token_ids.numel(), device=logits.device)

        remapped_labels = torch.full_like(shift_labels, -100)
        remapped_labels[active_positions] = label_to_valid_index[active_labels]
        if bool((remapped_labels[active_positions] < 0).any().item()):
            raise RuntimeError("Token LM labels contain ids outside the valid patch-token set.")

        valid_logits = shift_logits.index_select(dim=-1, index=valid_token_ids).contiguous()
        return F.cross_entropy(
            valid_logits.view(-1, valid_logits.shape[-1]),
            remapped_labels.view(-1),
            ignore_index=-100,
        )

    def _candidate_scores(self, patches):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before scoring patches.")
        patch_features = self._patch_features(patches)
        projected = self._project_patch_features(patch_features)
        candidate_ids = self.dictionary.candidate_token_ids.to(projected.device).long()
        candidate_embeds = l2_normalize(self._vocab_weight().to(projected.device)[candidate_ids])
        return torch.matmul(projected, candidate_embeds.transpose(0, 1))

    def _hard_patch_token_ids(self, patches):
        if self.use_trainable_patch_projector:
            scores = self._candidate_scores(patches)
            valid_candidate_indices = self.dictionary.valid_candidate_indices.to(scores.device).long()
            valid_scores = scores.index_select(dim=-1, index=valid_candidate_indices)
            motif_ids = valid_scores.argmax(dim=-1)
            valid_token_ids = self.dictionary.valid_token_ids.to(scores.device).long()
            return valid_token_ids[motif_ids]
        return self.dictionary.patches_to_token_ids(patches)

    def _future_prediction_positions(self, input_length, device):
        label_positions = torch.arange(input_length - self.future_patch_count, input_length, device=device)
        return label_positions - 1

    def _valid_patch_bank_flat(self, device):
        return self.dictionary.patch_bank.to(device=device).reshape(
            self.dictionary.patch_bank.shape[0],
            self.dictionary.patch_bank.shape[1],
            self.patch_dim,
        )

    def _decode_patches_from_prob(self, token_prob, hidden_state):
        token_patch_bank = self._valid_patch_bank_flat(token_prob.device)
        base_patch = self.patch_bank_decoder(token_prob, hidden_state, token_patch_bank)
        residual_patch = self.residual_head(hidden_state) * self.residual_scale
        pred_patch = base_patch + residual_patch
        return pred_patch.reshape(*pred_patch.shape[:-1], self.patch_len, self.c_in)

    def _one_hot_valid_token_prob(self, token_ids):
        motif_ids = self.dictionary.token_ids_to_motif_ids(token_ids).to(token_ids.device)
        return F.one_hot(motif_ids, num_classes=int(self.dictionary.valid_token_ids.numel())).float()

    def _sequence_from_patches(self, patches, history_anchor=None):
        return self._concat_patches(patches, history_anchor=history_anchor)

    def token_lm_loss(self, input_ids, labels):
        attention_mask = torch.ones_like(input_ids)
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return self._token_ce_from_logits(outputs.logits, labels)

    def joint_training_loss(self, input_ids, labels, future_patches, align_patches, align_candidate_indices):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before training the token LM.")

        attention_mask = torch.ones_like(input_ids)
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        loss_ce = self._token_ce_from_logits(logits, labels)
        future_positions = self._future_prediction_positions(input_ids.shape[1], input_ids.device)
        future_logits = logits.index_select(dim=1, index=future_positions)
        future_hidden = hidden.index_select(dim=1, index=future_positions)
        valid_token_ids = self._valid_token_ids(logits.device)
        valid_logits = future_logits.index_select(dim=-1, index=valid_token_ids)
        token_prob = torch.softmax(valid_logits / max(self.mse_temperature, 1e-6), dim=-1)
        pred_patches = self._decode_patches_from_prob(token_prob, future_hidden)

        future_patches = future_patches.to(device=pred_patches.device, dtype=pred_patches.dtype)
        patch_mse = F.mse_loss(pred_patches, future_patches)
        pred_series = self._sequence_from_patches(pred_patches)
        true_series = self._sequence_from_patches(future_patches)
        sequence_mse = F.mse_loss(pred_series, true_series)
        loss_mse = patch_mse + sequence_mse

        if self.lambda_smooth > 0:
            pred_diff = pred_patches[:, :, 1:, :] - pred_patches[:, :, :-1, :]
            true_diff = future_patches[:, :, 1:, :] - future_patches[:, :, :-1, :]
            loss_smooth = F.mse_loss(pred_diff, true_diff)
        else:
            loss_smooth = torch.zeros((), device=input_ids.device, dtype=logits.dtype)

        if self.use_trainable_patch_projector and self.lambda_align > 0:
            align_scores = self._candidate_scores(align_patches.to(input_ids.device))
            align_scores = align_scores / max(self.align_temperature, 1e-6)
            align_targets = align_candidate_indices.to(device=input_ids.device, dtype=torch.long)
            loss_align = F.cross_entropy(
                align_scores.reshape(-1, align_scores.shape[-1]),
                align_targets.reshape(-1),
            )
        else:
            loss_align = torch.zeros((), device=input_ids.device, dtype=logits.dtype)

        total_loss = (
            self.lambda_ce * loss_ce
            + self.lambda_mse * loss_mse
            + self.lambda_align * loss_align
            + self.lambda_smooth * loss_smooth
        )
        return SimpleNamespace(
            total_loss=total_loss,
            loss_ce=loss_ce,
            loss_mse=loss_mse,
            patch_mse=patch_mse,
            sequence_mse=sequence_mse,
            loss_align=loss_align,
            loss_smooth=loss_smooth,
        )

    def _patchify_batch(self, batch_x):
        if batch_x.shape[1] != self.seq_len:
            raise ValueError(f"Expected batch_x length {self.seq_len}, got {batch_x.shape[1]}.")
        patches = batch_x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        if patches.shape[1] != self.history_patch_count:
            raise RuntimeError(
                f"Expected {self.history_patch_count} history patches, got {patches.shape[1]}."
            )
        return patches

    def _patchify_any(self, series):
        patches = series.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches.permute(0, 1, 3, 2).contiguous()

    def _concat_patches(self, patches, history_anchor=None):
        if self.stride == self.patch_len:
            return patches.reshape(patches.shape[0], patches.shape[1] * self.patch_len, self.c_in)[:, : self.pred_len, :]

        total_len = (patches.shape[1] - 1) * self.stride + self.patch_len
        out = patches.new_zeros(patches.shape[0], total_len, self.c_in)
        weight = patches.new_zeros(total_len)
        window = torch.hann_window(self.patch_len, periodic=False, device=patches.device, dtype=patches.dtype)
        window = window.clamp_min(1e-3)

        for patch_idx in range(patches.shape[1]):
            start = patch_idx * self.stride
            patch = patches[:, patch_idx].clone()

            if patch_idx == 0 and history_anchor is not None:
                patch = patch - patch[:, :1, :] + history_anchor
            elif start > 0:
                overlap_len = min(self.patch_len - self.stride, self.patch_len)
                if overlap_len > 0:
                    existing_weight = weight[start:start + overlap_len].clamp_min(1e-6).view(1, -1, 1)
                    existing = out[:, start:start + overlap_len, :] / existing_weight
                    offset = (existing - patch[:, :overlap_len, :]).mean(dim=1, keepdim=True)
                    patch = patch + offset

            out[:, start:start + self.patch_len, :] += patch * window.view(1, -1, 1)
            weight[start:start + self.patch_len] += window

        out = out / weight.clamp_min(1e-6).view(1, -1, 1)
        return out[:, : self.pred_len, :]

    def _history_patch_count(self, length):
        return (int(length) - self.patch_len) // self.stride + 1

    def _future_patch_count(self, length):
        length = int(length)
        if length <= self.patch_len:
            return 1
        return math.ceil((length - self.patch_len) / self.stride) + 1

    @torch.no_grad()
    def _generate_native_future_tokens(self, context_ids):
        future_token_count = self.future_patch_count * self.native_token_k
        max_positions = int(getattr(self.gpt2.config, "n_positions", context_ids.shape[1] + future_token_count))
        max_history = max_positions - future_token_count
        max_history = (max_history // self.native_token_k) * self.native_token_k
        if max_history <= 0:
            raise ValueError("GPT context length is smaller than the requested native future token count.")
        if context_ids.shape[1] > max_history:
            context_ids = context_ids[:, -max_history:]

        self._native_check_token_ids(context_ids)
        attention_mask = torch.ones_like(context_ids)
        logits_processor = LogitsProcessorList([
            ValidTokenLogitsProcessor(self.native_quantizer.allowed_token_ids)
        ])
        generated = self.gpt2.generate(
            input_ids=context_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            min_new_tokens=future_token_count,
            max_new_tokens=future_token_count,
            do_sample=False,
            pad_token_id=getattr(self.gpt2.config, "pad_token_id", None) or getattr(self.gpt2.config, "eos_token_id", 0),
            eos_token_id=getattr(self.gpt2.config, "eos_token_id", None),
        )
        future_ids = generated[:, context_ids.shape[1] :]
        if future_ids.shape[1] != future_token_count:
            raise RuntimeError(
                "Expected {} generated native tokens, got {}.".format(future_token_count, future_ids.shape[1])
            )
        self._native_check_token_ids(future_ids)
        return SimpleNamespace(
            used_context_ids=context_ids,
            full_token_ids=generated,
            future_token_ids=future_ids.reshape(context_ids.shape[0], self.future_patch_count, self.native_token_k),
        )

    @torch.no_grad()
    def native_forecast(self, batch_x):
        history_patches = self._patchify_batch(batch_x)
        quantized = self._native_quantize_patches(history_patches)
        context_ids = self._native_flatten_token_ids(quantized.token_ids)
        generated = self._generate_native_future_tokens(context_ids)
        future_ids = generated.future_token_ids
        codebook = self._native_vocab_weight(device=batch_x.device, dtype=batch_x.dtype, detach=True)
        future_embeds = codebook.index_select(dim=0, index=future_ids.reshape(-1))
        future_embeds = future_embeds.reshape(
            batch_x.shape[0],
            self.future_patch_count,
            self.native_token_k,
            self.gpt_dim,
        )
        future_patches = self.native_patch_decoder(future_embeds)
        pred = self._concat_patches(future_patches, history_anchor=batch_x[:, -1:, :])
        pred = pred[:, : self.pred_len, :]
        aux = SimpleNamespace(
            history_token_ids=quantized.token_ids,
            future_token_ids=future_ids,
            future_patches=future_patches,
            allowed_token_ids=self.native_quantizer.allowed_token_ids,
        )
        return pred, aux

    @torch.no_grad()
    def _generate_future_tokens(self, history_token_ids):
        max_positions = int(getattr(self.gpt2.config, "n_positions", history_token_ids.shape[1] + self.generated_patch_count))
        max_history = max_positions - self.generated_patch_count
        if max_history <= 0:
            raise ValueError("GPT context length is smaller than the requested future patch count.")
        if history_token_ids.shape[1] > max_history:
            history_token_ids = history_token_ids[:, -max_history:]

        attention_mask = torch.ones_like(history_token_ids)
        logits_processor = LogitsProcessorList([
            ValidTokenLogitsProcessor(self.dictionary.valid_token_ids)
        ])
        generated = self.gpt2.generate(
            input_ids=history_token_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            min_new_tokens=self.generated_patch_count,
            max_new_tokens=self.generated_patch_count,
            do_sample=False,
            pad_token_id=getattr(self.gpt2.config, "pad_token_id", None) or getattr(self.gpt2.config, "eos_token_id", 0),
            eos_token_id=getattr(self.gpt2.config, "eos_token_id", None),
        )
        return SimpleNamespace(
            used_history_token_ids=history_token_ids,
            full_token_ids=generated,
            future_token_ids=generated[:, -self.future_patch_count :],
        )

    @torch.no_grad()
    def forecast(self, batch_x):
        if self.use_native_gpt_vocab:
            return self.native_forecast(batch_x)
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map with the full training set before forecasting.")

        history_patches = self._patchify_batch(batch_x)
        history_token_ids = self._hard_patch_token_ids(history_patches)
        generated = self._generate_future_tokens(history_token_ids)

        outputs = self.gpt2(
            input_ids=generated.full_token_ids,
            attention_mask=torch.ones_like(generated.full_token_ids),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        future_positions = self._future_prediction_positions(generated.full_token_ids.shape[1], generated.full_token_ids.device)
        future_hidden = hidden.index_select(dim=1, index=future_positions)
        token_prob = self._one_hot_valid_token_prob(generated.future_token_ids).to(future_hidden.device)
        future_patches = self._decode_patches_from_prob(token_prob, future_hidden)
        pred = self._concat_patches(future_patches, history_anchor=batch_x[:, -1:, :])
        aux = SimpleNamespace(
            history_token_ids=history_token_ids,
            future_token_ids=generated.future_token_ids,
            future_patches=future_patches,
        )
        return pred, aux

    @torch.no_grad()
    def eval_token_ce(self, batch_x, target):
        if self.use_native_gpt_vocab:
            losses = self.native_training_loss(batch_x, target, force_gpt=True)
            return float(losses.loss_ce.detach().cpu().item())
        if not self.dictionary.ready:
            return float("nan")
        target = target[:, -self.pred_len :, :]
        full_series = torch.cat([batch_x, target], dim=1)
        patches = self._patchify_any(full_series)
        expected_count = self.history_patch_count + self.generated_patch_count
        if patches.shape[1] != expected_count:
            return float("nan")
        input_ids = self.dictionary.patches_to_token_ids(patches)
        labels = input_ids.clone()
        labels[:, : self.history_patch_count] = -100
        outputs = self.gpt2(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        return float(self._token_ce_from_logits(outputs.logits, labels).detach().cpu().item())

    def debug_token_map(self, max_items=8):
        if self.use_native_gpt_vocab:
            allowed_token_ids = self.native_quantizer.allowed_token_ids.detach().cpu()
            lines = [
                f"native_candidate_token_ids: {int(allowed_token_ids.numel())}",
                f"native_token_k: {int(self.native_token_k)}",
                f"original_vocab_size: {int(self.original_vocab_size)}",
                f"train_stage: {self.train_stage}",
            ]
            for token_id in allowed_token_ids[: int(max_items)].tolist():
                text = ""
                if self.tokenizer is not None:
                    text = self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
                lines.append(f"candidate token {int(token_id)} -> {text!r}")
            return lines
        if not self.dictionary.ready:
            return ["Patch-token map is not fitted."]
        lines = [
            f"candidate_token_ids: {int(self.dictionary.candidate_token_ids.numel())}",
            f"valid_token_ids: {int(self.dictionary.valid_token_ids.numel())}",
            f"patch_bank: {tuple(self.dictionary.patch_bank.shape)}",
            f"assignment_method: {self.dictionary.last_assignment_method}",
        ]
        limit = min(int(max_items), int(self.dictionary.valid_token_ids.numel()))
        for motif_id in range(limit):
            token_id = int(self.dictionary.valid_token_ids[motif_id].detach().cpu().item())
            text = ""
            if self.tokenizer is not None:
                text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            lines.append(f"motif {motif_id} -> token {token_id} -> {text!r}")
        return lines

    def forward(self, batch_x):
        pred, aux = self.forecast(batch_x)
        return SimpleNamespace(pred=pred, aux=aux)
