import json
import unicodedata
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def _decode_token(tokenizer, token_id):
    if tokenizer is None:
        return ""
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _special_token_ids(tokenizer):
    if tokenizer is None:
        return set()
    return {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}


def _has_control_character(text):
    for char in text:
        if char in ("\t", "\n", "\r"):
            return True
        if unicodedata.category(char).startswith("C"):
            return True
    return False


def _looks_like_common_text(text):
    if text == "" or text.strip() == "":
        return False
    if "\ufffd" in text:
        return False
    if _has_control_character(text):
        return False
    printable_count = sum(1 for char in text if char.isprintable())
    return printable_count == len(text)


def _read_candidate_file(path):
    if path is None or str(path).lower() in ("", "none", "null"):
        raise ValueError("--candidate_token_file must be set when candidate_token_mode='file'.")

    path = str(path)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            for key in ("allowed_token_ids", "candidate_token_ids", "token_ids", "ids"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("Candidate token json must be a list or contain a token-id list field.")
        return [int(item) for item in payload]

    token_ids = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token_ids.append(int(line.split()[0]))
    return token_ids


def build_candidate_token_ids(tokenizer, original_vocab_size, mode="all", candidate_token_file=None, device=None):
    """Build real GPT token ids for the native GPT vocab quantizer."""

    original_vocab_size = int(original_vocab_size)
    mode = str(mode or "all").lower()
    special_ids = _special_token_ids(tokenizer)

    if mode == "all":
        selected = list(range(original_vocab_size))
    elif mode == "numeric":
        selected = []
        for token_id in range(original_vocab_size):
            text = _decode_token(tokenizer, token_id)
            if any(char.isdigit() for char in text):
                selected.append(token_id)
    elif mode == "common_text":
        selected = []
        for token_id in range(original_vocab_size):
            if token_id in special_ids:
                continue
            text = _decode_token(tokenizer, token_id)
            if _looks_like_common_text(text):
                selected.append(token_id)
    elif mode == "file":
        selected = _read_candidate_file(candidate_token_file)
    else:
        raise ValueError(
            "Unsupported candidate_token_mode '{}'. Use all, numeric, common_text, or file.".format(mode)
        )

    allowed_token_ids = torch.as_tensor(selected, dtype=torch.long, device=device)
    allowed_token_ids = torch.unique(allowed_token_ids, sorted=True)
    if allowed_token_ids.numel() == 0:
        raise ValueError("No candidate GPT token ids were selected.")

    assert int(allowed_token_ids.min().item()) >= 0
    assert int(allowed_token_ids.max().item()) < original_vocab_size
    return allowed_token_ids


class NativePatchEncoder(nn.Module):
    def __init__(self, patch_len, c_in, hidden_dim, native_token_k, d_model):
        super().__init__()
        self.patch_len = int(patch_len)
        self.c_in = int(c_in)
        self.native_token_k = int(native_token_k)
        self.d_model = int(d_model)
        self.patch_dim = self.patch_len * self.c_in
        self.net = nn.Sequential(
            nn.Linear(self.patch_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.native_token_k * self.d_model),
        )

    def forward(self, patches):
        if patches.ndim != 4:
            raise ValueError("Expected patches with shape [B,N,patch_len,C].")
        batch_size, patch_count, patch_len, channel_count = patches.shape
        if patch_len != self.patch_len or channel_count != self.c_in:
            raise ValueError(
                "Expected patch shape (*,*,{},{}), got {}.".format(
                    self.patch_len,
                    self.c_in,
                    tuple(patches.shape),
                )
            )

        flat = patches.reshape(batch_size, patch_count, self.patch_dim).float()
        out = self.net(flat)
        out = out.reshape(batch_size, patch_count, self.native_token_k, self.d_model)
        assert out.shape == (batch_size, patch_count, self.native_token_k, self.d_model)
        return out


class NativeGPTVocabQuantizer(nn.Module):
    def __init__(
        self,
        allowed_token_ids,
        original_vocab_size,
        tau=1.0,
        tau_min=0.05,
        use_straight_through=True,
        freeze_codebook=True,
    ):
        super().__init__()
        self.original_vocab_size = int(original_vocab_size)
        self.tau = float(tau)
        self.tau_min = float(tau_min)
        self.use_straight_through = bool(use_straight_through)
        self.freeze_codebook = bool(freeze_codebook)
        allowed_token_ids = torch.as_tensor(allowed_token_ids, dtype=torch.long)
        assert int(allowed_token_ids.min().item()) >= 0
        assert int(allowed_token_ids.max().item()) < self.original_vocab_size
        self.register_buffer("allowed_token_ids", allowed_token_ids, persistent=True)

    def forward(self, patch_repr, gpt_embedding, tau=None):
        if patch_repr.ndim != 4:
            raise ValueError("Expected patch_repr with shape [B,N,K,d_model].")
        if gpt_embedding.ndim != 2:
            raise ValueError("Expected GPT embedding matrix with shape [V,d_model].")

        batch_size, patch_count, native_token_k, d_model = patch_repr.shape
        original_vocab_size = int(gpt_embedding.shape[0])
        if original_vocab_size != self.original_vocab_size:
            raise RuntimeError(
                "GPT vocab size changed from {} to {}. Native GPT vocab tokenizer forbids resizing.".format(
                    self.original_vocab_size,
                    original_vocab_size,
                )
            )
        if int(gpt_embedding.shape[-1]) != d_model:
            raise RuntimeError("patch_repr d_model does not match GPT embedding dimension.")

        allowed_token_ids = self.allowed_token_ids.to(device=patch_repr.device, dtype=torch.long)
        assert int(allowed_token_ids.min().item()) >= 0
        assert int(allowed_token_ids.max().item()) < original_vocab_size

        codebook = gpt_embedding.detach() if self.freeze_codebook else gpt_embedding
        candidate_embeds = codebook.index_select(dim=0, index=allowed_token_ids).to(dtype=patch_repr.dtype)

        z_norm = F.normalize(patch_repr, dim=-1, eps=1e-12)
        e_norm = F.normalize(candidate_embeds, dim=-1, eps=1e-12)
        logits = torch.matmul(z_norm, e_norm.transpose(0, 1))

        temperature = max(float(self.tau if tau is None else tau), self.tau_min)
        probs = torch.softmax(logits / temperature, dim=-1)
        soft_embed = torch.matmul(probs, candidate_embeds)

        local_ids = torch.argmax(logits, dim=-1)
        token_ids = allowed_token_ids[local_ids]
        hard_embed = codebook.index_select(dim=0, index=token_ids.reshape(-1))
        hard_embed = hard_embed.reshape(batch_size, patch_count, native_token_k, d_model).to(dtype=patch_repr.dtype)

        if self.use_straight_through:
            token_embeds = soft_embed + (hard_embed - soft_embed).detach()
        else:
            token_embeds = soft_embed

        commitment_loss = F.mse_loss(patch_repr.float(), hard_embed.detach().float())
        avg_probs = probs.float().mean(dim=(0, 1, 2))
        usage_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        assert token_ids.shape == (batch_size, patch_count, native_token_k)
        assert int(token_ids.min().item()) >= 0
        assert int(token_ids.max().item()) < original_vocab_size
        assert token_embeds.shape == (batch_size, patch_count, native_token_k, d_model)

        return SimpleNamespace(
            token_ids=token_ids,
            token_embeds=token_embeds,
            logits=logits,
            probs=probs,
            commitment_loss=commitment_loss,
            usage_loss=usage_loss,
        )


class NativePatchDecoder(nn.Module):
    def __init__(self, patch_len, c_in, hidden_dim, native_token_k, d_model):
        super().__init__()
        self.patch_len = int(patch_len)
        self.c_in = int(c_in)
        self.native_token_k = int(native_token_k)
        self.d_model = int(d_model)
        self.patch_dim = self.patch_len * self.c_in
        self.net = nn.Sequential(
            nn.Linear(self.native_token_k * self.d_model, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.patch_dim),
        )

    def forward(self, token_embeds):
        if token_embeds.ndim != 4:
            raise ValueError("Expected token_embeds with shape [B,N,K,d_model].")
        batch_size, patch_count, native_token_k, d_model = token_embeds.shape
        if native_token_k != self.native_token_k or d_model != self.d_model:
            raise ValueError(
                "Expected token embed shape (*,*,{},{}), got {}.".format(
                    self.native_token_k,
                    self.d_model,
                    tuple(token_embeds.shape),
                )
            )

        flat = token_embeds.reshape(batch_size, patch_count, native_token_k * d_model)
        out = self.net(flat)
        recon = out.reshape(batch_size, patch_count, self.patch_len, self.c_in)
        assert recon.shape == (batch_size, patch_count, self.patch_len, self.c_in)
        return recon
