from types import SimpleNamespace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

from models.lora import LoRAConv1D
from models.patch2token import PatchTokenDictionary


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

        cluster_num = int(getattr(configs, "cluster_num", 1))
        self.dictionary = PatchTokenDictionary(
            cluster_num=cluster_num,
            patch_len=self.patch_len,
            c_in=self.c_in,
            stride=self.stride,
            normalize=getattr(configs, "cluster_normalize", False),
            seed=getattr(configs, "cluster_seed", 0),
            match_tol=getattr(configs, "patch_match_tol", 1e-6),
            kmeans_iters=getattr(configs, "kmeans_iters", 30),
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

    def _candidate_token_ids(self):
        vocab_size = self._vocab_weight().shape[0]
        candidate_token_ids = torch.arange(vocab_size, dtype=torch.long, device=self._vocab_weight().device)

        special_ids = [
            getattr(self.gpt2.config, "bos_token_id", None),
            getattr(self.gpt2.config, "eos_token_id", None),
            getattr(self.gpt2.config, "pad_token_id", None),
        ]
        keep = torch.ones(vocab_size, dtype=torch.bool, device=candidate_token_ids.device)
        for token_id in special_ids:
            if token_id is not None and 0 <= int(token_id) < vocab_size:
                keep[int(token_id)] = False
        candidate_token_ids = candidate_token_ids[keep]

        candidate_count = int(getattr(self.configs, "candidate_token_count", 0) or 0)
        if candidate_count > 0:
            candidate_token_ids = candidate_token_ids[:candidate_count]
        return candidate_token_ids

    @torch.no_grad()
    def fit_patch_token_map(self, train_series):
        self.dictionary.fit(train_series, self._vocab_weight(), self._candidate_token_ids())

    @torch.no_grad()
    def print_patch_token_distribution(self):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before printing patch-token distribution.")

        token_ids = self.dictionary.train_patch_token_ids.detach().cpu()
        unique_ids, counts = torch.unique(token_ids, return_counts=True)
        order = torch.argsort(counts, descending=True)
        token_counts = [
            (int(unique_ids[idx].item()), int(counts[idx].item()))
            for idx in order
        ]

        print(
            "\tPatch-token distribution | total patches: {0} | unique tokens: {1}".format(
                int(token_ids.numel()),
                len(token_counts),
            )
        )
        print("\tPatch-token ids/counts:", token_counts)

    @torch.no_grad()
    def save_patch_token_map(self, path):
        self.dictionary.save_npz(path)

    @torch.no_grad()
    def load_patch_token_map(self, path):
        self.dictionary.load_npz(path)

    @torch.no_grad()
    def build_lm_training_tensors(self):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before building token LM training data.")

        tokens = self.dictionary.train_patch_token_ids.detach().long()
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
        windows = tokens.unfold(dimension=0, size=window_size, step=step).contiguous()
        input_ids = windows.contiguous()
        labels = windows.clone().contiguous()
        labels[:, : self.history_patch_count] = -100
        return input_ids, labels

    def token_lm_loss(self, input_ids, labels):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map before training the token LM.")

        attention_mask = torch.ones_like(input_ids)
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        valid_token_ids = self.dictionary.valid_token_ids.to(device=logits.device).long()
        valid_token_ids = valid_token_ids[(valid_token_ids >= 0) & (valid_token_ids < logits.shape[-1])]
        if valid_token_ids.numel() == 0:
            raise RuntimeError("No valid token ids are available for masked token LM training.")

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

    def _concat_patches(self, patches):
        if self.stride == self.patch_len:
            return patches.reshape(patches.shape[0], patches.shape[1] * self.patch_len, self.c_in)[:, : self.pred_len, :]

        total_len = (patches.shape[1] - 1) * self.stride + self.patch_len
        out = patches.new_zeros(patches.shape[0], total_len, self.c_in)
        weight = patches.new_zeros(total_len)
        window = torch.hann_window(self.patch_len, periodic=False, device=patches.device, dtype=patches.dtype)
        window = window.clamp_min(1e-3)

        for patch_idx in range(patches.shape[1]):
            start = patch_idx * self.stride
            out[:, start:start + self.patch_len, :] += patches[:, patch_idx] * window.view(1, -1, 1)
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
        return generated[:, -self.future_patch_count :]

    @torch.no_grad()
    def forecast(self, batch_x):
        if not self.dictionary.ready:
            raise RuntimeError("Call fit_patch_token_map with the full training set before forecasting.")

        history_patches = self._patchify_batch(batch_x)
        history_token_ids = self.dictionary.patches_to_token_ids(history_patches)
        future_token_ids = self._generate_future_tokens(history_token_ids)
        future_patches = self.dictionary.token_ids_to_patches(future_token_ids, self._vocab_weight())
        pred = self._concat_patches(future_patches)
        aux = SimpleNamespace(history_token_ids=history_token_ids, future_token_ids=future_token_ids)
        return pred, aux

    def forward(self, batch_x, batch_y=None):
        pred, aux = self.forecast(batch_x)
        return SimpleNamespace(pred=pred, loss=None, aux=aux)
