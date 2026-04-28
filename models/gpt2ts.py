import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from Tokenizer.models.GPT2ClusterVQ import VQVAE as GPT2ClusterTokenizer


class TimeSeriesVocabMapper(nn.Module):
    def __init__(self, mode="soft_vocab", tau=0.2, top_k=0, normalize=True):
        super().__init__()
        self.mode = str(mode)
        self.tau = max(float(tau), 1e-6)
        self.top_k = max(0, int(top_k))
        self.normalize = bool(normalize)

    def forward(self, ts_embeds, vocab_weight):
        queries = ts_embeds
        keys = vocab_weight
        if self.normalize:
            queries = F.normalize(queries, dim=-1)
            keys = F.normalize(keys, dim=-1)

        logits = queries @ keys.transpose(0, 1)
        if self.top_k > 0 and self.top_k < logits.shape[-1]:
            top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            logits = masked_logits.scatter(-1, top_indices, top_values)

        probs = F.softmax(logits / self.tau, dim=-1)
        soft_embeds = probs @ vocab_weight

        if self.mode in {"st_vocab", "hard_vocab"}:
            hard_ids = probs.argmax(dim=-1)
            hard_embeds = F.embedding(hard_ids, vocab_weight)
            mapped = hard_embeds.detach() - soft_embeds.detach() + soft_embeds
        elif self.mode == "none":
            mapped = ts_embeds
        else:
            mapped = soft_embeds

        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        return mapped, entropy


class FrozenTokenizerAdapter(nn.Module):
    """Wraps the trained TokenCast tokenizer while keeping its parameters frozen."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.cfg = self._build_tokenizer_config(configs)
        self.vq_model = getattr(configs, "vq_model", "GPT2ClusterVQ")
        if self.vq_model != "GPT2ClusterVQ":
            raise ValueError(f"Unsupported tokenizer model: {self.vq_model}. Use `GPT2ClusterVQ`.")
        self.model = GPT2ClusterTokenizer(self.cfg)
        self._load_checkpoint(configs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = int(self.cfg.d_model)
        self.patch_len = int(self.cfg.wave_length)

    def _build_tokenizer_config(self, configs):
        return SimpleNamespace(
            seq_len=int(configs.seq_len),
            pred_len=int(configs.pred_len),
            d_model=int(getattr(configs, "tokenizer_d_model", getattr(configs, "d_model", 64))),
            n_embed=int(getattr(configs, "n_embed", getattr(configs, "num_ts_clusters", 256))),
            block_num=int(getattr(configs, "tokenizer_block_num", getattr(configs, "block_num", 2))),
            wave_length=int(getattr(configs, "wave_length", getattr(configs, "patch_len", 8))),
            revin=int(getattr(configs, "revin", 1)),
            affine=int(getattr(configs, "affine", 0)),
            subtract_last=int(getattr(configs, "subtract_last", 0)),
            chan_indep=int(getattr(configs, "chan_indep", 0)),
            enc_in=int(getattr(configs, "enc_in", getattr(configs, "c_in", 1))),
            entropy_penalty=float(getattr(configs, "entropy_penalty", 0.0)),
            entropy_temp=float(getattr(configs, "entropy_temp", 0.5)),
            gpt_local_path=getattr(configs, "gpt_local_path", None),
            gpt_model_name=getattr(configs, "gpt_model_name", "openai-community/gpt2"),
            gpt_local_files_only=getattr(configs, "gpt_local_files_only", True),
            gpt2_hidden_size=int(getattr(configs, "gpt2_hidden_size", 768)),
            init_gpt2_codebook=False,
            cluster_tau=float(getattr(configs, "cluster_tau", 0.2)),
            cluster_cosine=bool(getattr(configs, "cluster_cosine", True)),
            commitment_weight=float(getattr(configs, "commitment_weight", 0.25)),
        )

    def _resolve_checkpoint(self, configs):
        path = getattr(configs, "tokenizer_path", None) or getattr(configs, "vqvae_model_path", None)
        if not path:
            return None
        if os.path.isdir(path):
            return os.path.join(path, "model.pkl")
        return path

    def _load_checkpoint(self, configs):
        checkpoint = self._resolve_checkpoint(configs)
        if not checkpoint:
            raise ValueError("Stage-2 GPT2TS requires `--tokenizer_path` or `--vqvae_model_path`.")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded frozen tokenizer from {checkpoint}")

    def _normalize_history(self, x):
        if self.model.revin:
            return self.model.revin_layer(x, "norm")
        return x

    def _normalize_future_with_history_stats(self, y):
        if self.model.revin:
            return self.model.revin_layer._normalize(y)
        return y

    def _encode_normalized(self, x_norm):
        enc = self.model.enc(x_norm)
        quant_input = self.model.quantize_input(enc.unsqueeze(1)).squeeze(-1).transpose(1, 2)
        quant, _, ids = self.model.quantize(quant_input)
        return quant.detach(), ids.detach()

    @torch.no_grad()
    def encode_history(self, x):
        if hasattr(self.model, "encode_history_gpt"):
            return self.model.encode_history_gpt(x)
        x_norm = self._normalize_history(x)
        return self._encode_normalized(x_norm)

    @torch.no_grad()
    def encode_future_target(self, x, y):
        if hasattr(self.model, "encode_future_target_gpt"):
            return self.model.encode_future_target_gpt(x, y)
        x_norm = self._normalize_history(x)
        y_norm = self._normalize_future_with_history_stats(y)
        full_norm = torch.cat([x_norm, y_norm], dim=1)
        full_quant, full_ids = self._encode_normalized(full_norm)
        history_tokens = x.shape[1] // self.patch_len
        return full_quant[:, history_tokens:, :], full_ids[:, history_tokens:]

    def decode_future(self, x, history_quant, future_quant, pred_len):
        if hasattr(self.model, "decode_future_gpt"):
            return self.model.decode_future_gpt(x, history_quant, future_quant, pred_len)
        # Reset RevIN statistics from the current history before denormalizing decoder output.
        _ = self._normalize_history(x)
        full_quant = torch.cat([history_quant.detach(), future_quant], dim=1)
        decoded = self.model.dec(full_quant)
        if self.model.revin:
            decoded = self.model.revin_layer(decoded, "denorm")
        return decoded[:, x.shape[1] : x.shape[1] + pred_len, :]

    def codebook(self):
        if hasattr(self.model, "get_codebook_weight"):
            return self.model.get_codebook_weight()
        return None


class Model(nn.Module):
    """Stage-2 TokenCast-style forecaster: frozen tokenizer + fully frozen GPT-2."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.c_out = int(getattr(configs, "c_out", getattr(configs, "enc_in", getattr(configs, "c_in", 1))))

        self.tokenizer_adapter = FrozenTokenizerAdapter(configs)
        self.tokenizer_dim = self.tokenizer_adapter.hidden_dim
        self.patch_len = self.tokenizer_adapter.patch_len

        self.gpt2_path = self._resolve_gpt2_path()
        self.local_files_only = bool(getattr(configs, "gpt_local_files_only", True))
        gpt2_config = AutoConfig.from_pretrained(
            self.gpt2_path,
            local_files_only=self.local_files_only,
        )
        requested_layers = int(getattr(configs, "n_layers", 0))
        if requested_layers > 0:
            gpt2_config.n_layer = min(requested_layers, gpt2_config.n_layer)
            gpt2_config.num_hidden_layers = gpt2_config.n_layer
        gpt2_config.output_hidden_states = True
        self.gpt_dim = int(gpt2_config.hidden_size)

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.gpt2_path,
            local_files_only=self.local_files_only,
        )
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self.gpt2 = self._load_gpt2(gpt2_config)
        self._freeze_gpt2_completely()

        tokenizer_codebook = self.tokenizer_adapter.codebook()
        self.uses_gpt2_cluster_codebook = tokenizer_codebook is not None
        if self.uses_gpt2_cluster_codebook:
            self.register_buffer("gpt2_cluster_codebook", tokenizer_codebook.detach().float())
            self.ts_to_gpt = nn.Identity()
            self.hidden_to_code = nn.Sequential(nn.LayerNorm(self.gpt_dim), nn.Linear(self.gpt_dim, self.gpt_dim))
        else:
            self.ts_to_gpt = nn.Sequential(
                nn.LayerNorm(self.tokenizer_dim),
                nn.Linear(self.tokenizer_dim, self.gpt_dim),
            )
            self.hidden_to_code = nn.Sequential(nn.LayerNorm(self.gpt_dim), nn.Linear(self.gpt_dim, self.tokenizer_dim))
        self.mapper = TimeSeriesVocabMapper(
            mode=getattr(configs, "ts_embedding_mode", "soft_vocab"),
            tau=getattr(configs, "ts_mapping_tau", getattr(configs, "cluster_tau", 0.2)),
            top_k=getattr(configs, "ts_mapping_top_k", 0),
            normalize=getattr(configs, "ts_mapping_normalize", True),
        )

        future_token_count = math.ceil(self.pred_len / self.patch_len)
        self.future_queries = nn.Parameter(torch.empty(future_token_count, self.gpt_dim))
        nn.init.normal_(self.future_queries, mean=0.0, std=self.gpt_dim ** -0.5)

    def _resolve_gpt2_path(self):
        path = getattr(self.configs, "gpt_local_path", None)
        if path:
            return path
        path = getattr(self.configs, "local_model_path", None)
        if path:
            return path
        return getattr(self.configs, "gpt_model_name", "openai-community/gpt2")

    def _load_gpt2(self, config):
        if bool(getattr(self.configs, "use_pretrained_gpt2", True)):
            return AutoModelForCausalLM.from_pretrained(
                self.gpt2_path,
                config=config,
                local_files_only=self.local_files_only,
            )
        return AutoModelForCausalLM.from_config(config)

    def _freeze_gpt2_completely(self):
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.gpt2.eval()

    def _codebook_weight(self):
        if self.uses_gpt2_cluster_codebook:
            return self.gpt2_cluster_codebook.to(device=next(self.parameters()).device)
        return self.gpt2.get_input_embeddings().weight.detach()

    def _map_to_codebook(self, projected):
        return self.mapper(projected, self._codebook_weight())

    def forecast(self, batch_x):
        history_quant, history_ids = self.tokenizer_adapter.encode_history(batch_x)
        history_projected = self.ts_to_gpt(history_quant)
        history_embeds, history_entropy = self._map_to_codebook(history_projected)

        future_queries = self.future_queries.unsqueeze(0).expand(batch_x.shape[0], -1, -1)
        future_embeds, future_entropy = self._map_to_codebook(future_queries)
        inputs_embeds = torch.cat([history_embeds, future_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pred_hidden = outputs.hidden_states[-1][:, -self.future_queries.shape[0] :, :]
        future_quant_raw = self.hidden_to_code(pred_hidden)
        future_quant, output_entropy = self._map_to_codebook(future_quant_raw)
        pred = self.tokenizer_adapter.decode_future(
            batch_x,
            history_quant,
            future_quant,
            self.pred_len,
        )
        return pred[:, :, : self.c_out], {
            "history_ids": history_ids,
            "history_quant": history_quant,
            "future_quant": future_quant,
            "mapping_entropy": history_entropy + future_entropy + output_entropy,
        }

    def forward(self, batch_x, batch_y=None):
        pred, aux = self.forecast(batch_x)
        loss = None
        if batch_y is not None:
            target = batch_y[:, -self.pred_len :, : self.c_out]
            loss = F.mse_loss(pred, target)

            lambda_latent = float(getattr(self.configs, "lambda_tokenizer_latent", 0.1))
            if lambda_latent > 0:
                target_future_quant, _ = self.tokenizer_adapter.encode_future_target(batch_x, target)
                token_count = min(aux["future_quant"].shape[1], target_future_quant.shape[1])
                loss = loss + lambda_latent * F.mse_loss(
                    aux["future_quant"][:, :token_count, :],
                    target_future_quant[:, :token_count, :],
                )

            lambda_diff = float(getattr(self.configs, "lambda_diff", 0.0))
            if lambda_diff > 0 and self.pred_len > 1:
                loss = loss + lambda_diff * F.mse_loss(
                    pred[:, 1:, :] - pred[:, :-1, :],
                    target[:, 1:, :] - target[:, :-1, :],
                )

            lambda_entropy = float(getattr(self.configs, "lambda_ts_mapping_entropy", 0.0))
            if lambda_entropy != 0.0:
                loss = loss + lambda_entropy * aux["mapping_entropy"]

        return SimpleNamespace(pred=pred, loss=loss, aux=aux)
