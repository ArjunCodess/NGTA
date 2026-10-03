from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class ModelOutput:
    logits: torch.Tensor
    attention: torch.Tensor
    token_scores: torch.Tensor
    cls_logit: torch.Tensor


@dataclass
class MCPredictionSummary:
    labels: np.ndarray
    probabilities_mean: np.ndarray
    probabilities_var: np.ndarray
    attention_mean: np.ndarray
    attention_var: np.ndarray
    token_score_mean: np.ndarray
    cls_logit_mean: np.ndarray
    probability_passes: np.ndarray
    attention_passes: np.ndarray
    token_score_passes: np.ndarray
    cls_logit_passes: np.ndarray
    rng_states: list[tuple[torch.Tensor, list[torch.Tensor]]]


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, key_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = None
        if key_bias is not None:
            batch, length, _ = src.shape
            attention_mask = key_bias[:, None, :].expand(batch, length, length)
            attention_mask = attention_mask.repeat_interleave(self.self_attn.num_heads, dim=0)
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            need_weights=True,
            average_attn_weights=False,
            attn_mask=attention_mask,
        )
        src = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_output))
        return src, attn_weights


class TabularTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [AttentionEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.cls_head = nn.Linear(d_model, 1)
        self.token_scorer = nn.Linear(d_model, 1)

    def _embed_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"Expected 2D feature tensor, got shape {tuple(inputs.shape)}")
        return (inputs.unsqueeze(-1) * self.feature_weight.unsqueeze(0)) + self.feature_bias.unsqueeze(0)

    def forward(self, inputs: torch.Tensor, feature_confidence: torch.Tensor | None = None,
                gamma: float = 2.0) -> ModelOutput:
        feature_tokens = self._embed_features(inputs)
        batch_size = feature_tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls_token, feature_tokens], dim=1)
        hidden = hidden + self.position_embedding[:, : hidden.shape[1], :]
        hidden = self.input_dropout(hidden)

        key_bias = None
        if feature_confidence is not None:
            if feature_confidence.shape != inputs.shape or not torch.isfinite(feature_confidence).all():
                raise ValueError("Encoder confidence must match finite input feature dimensions")
            if not np.isfinite(gamma) or gamma < 0:
                raise ValueError("Gamma must be finite and nonnegative")
            feature_bias = gamma * torch.log(feature_confidence.clamp(min=1e-12, max=1.0))
            key_bias = torch.cat([torch.zeros_like(feature_bias[:, :1]), feature_bias], dim=1)

        last_attention = None
        for layer in self.layers:
            hidden, last_attention = layer(hidden, key_bias)

        if last_attention is None:
            raise RuntimeError("Encoder stack did not produce attention weights.")

        attention = last_attention.mean(dim=1)[:, 0, 1:]
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        feature_hidden = hidden[:, 1:, :]
        cls_hidden = hidden[:, 0, :]
        token_scores = self.token_scorer(feature_hidden).squeeze(-1)
        cls_logit = self.cls_head(cls_hidden).squeeze(-1)
        logits = cls_logit + (attention * token_scores).sum(dim=1)
        return ModelOutput(
            logits=logits,
            attention=attention,
            token_scores=token_scores,
            cls_logit=cls_logit,
        )

    def predict_with_mc_dropout(
        self,
        loader: DataLoader,
        device: torch.device,
        mc_samples: int,
        feature_confidence: np.ndarray | None = None,
        gamma: float = 2.0,
        replay_rng: list[tuple[torch.Tensor, list[torch.Tensor]]] | None = None,
    ) -> MCPredictionSummary:
        if mc_samples < 2:
            raise ValueError("MC uncertainty requires at least two passes")
        from torch.utils.data import SequentialSampler
        if not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("MC inference requires a sequential loader to align cases across passes")
        if feature_confidence is not None and np.asarray(feature_confidence).shape != (len(loader.dataset), self.input_dim):
            raise ValueError("Encoder confidence must align with all inference cases")
        if replay_rng is not None and len(replay_rng) != mc_samples:
            raise ValueError("Replay RNG states must match the number of passes")
        was_training = self.training
        labels = None
        probability_passes = []
        attention_passes = []
        token_score_passes = []
        cls_logit_passes = []
        rng_states = []
        initial_cpu = torch.get_rng_state()
        initial_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

        try:
            with torch.no_grad():
                for pass_index in range(mc_samples):
                    if replay_rng is not None:
                        cpu_state, cuda_states = replay_rng[pass_index]
                        torch.set_rng_state(cpu_state)
                        if cuda_states:
                            torch.cuda.set_rng_state_all(cuda_states)
                    rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
                    self.train()
                    pass_probabilities = []
                    pass_attention = []
                    pass_token_scores = []
                    pass_cls_logits = []
                    pass_labels = []

                    case_offset = 0
                    for features, target in loader:
                        features = features.to(device)
                        confidence = None if feature_confidence is None else torch.as_tensor(
                            feature_confidence[case_offset:case_offset+len(features)], dtype=features.dtype, device=device)
                        output = self(features, confidence, gamma)
                        case_offset += len(features)
                        pass_probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
                        pass_attention.append(output.attention.cpu().numpy())
                        pass_token_scores.append(output.token_scores.cpu().numpy())
                        pass_cls_logits.append(output.cls_logit.cpu().numpy())
                        pass_labels.append(target.cpu().numpy())

                    probability_passes.append(np.concatenate(pass_probabilities, axis=0))
                    attention_passes.append(np.concatenate(pass_attention, axis=0))
                    token_score_passes.append(np.concatenate(pass_token_scores, axis=0))
                    cls_logit_passes.append(np.concatenate(pass_cls_logits, axis=0))
                    if labels is None:
                        labels = np.concatenate(pass_labels, axis=0)

        finally:
            self.train(was_training)
            if replay_rng is not None:
                torch.set_rng_state(initial_cpu)
                if initial_cuda:
                    torch.cuda.set_rng_state_all(initial_cuda)

        probabilities = np.stack(probability_passes, axis=0)
        attentions = np.stack(attention_passes, axis=0)
        token_scores = np.stack(token_score_passes, axis=0)
        cls_logits = np.stack(cls_logit_passes, axis=0)

        return MCPredictionSummary(
            labels=np.asarray(labels),
            probabilities_mean=probabilities.mean(axis=0),
            probabilities_var=probabilities.var(axis=0),
            attention_mean=attentions.mean(axis=0),
            attention_var=attentions.var(axis=0),
            token_score_mean=token_scores.mean(axis=0),
            cls_logit_mean=cls_logits.mean(axis=0),
            probability_passes=probabilities,
            attention_passes=attentions,
            token_score_passes=token_scores,
            cls_logit_passes=cls_logits,
            rng_states=rng_states,
        )

    def predict_proba(self, features: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
        """Deterministic probabilities. Used for ensemble members and frozen shift scoring."""
        was_training = self.training
        self.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
                outputs.append(torch.sigmoid(self(batch).logits).cpu().numpy())
        if not was_training:
            self.eval()
        else:
            self.train()
        if not outputs:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(outputs, axis=0)
