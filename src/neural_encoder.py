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

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            need_weights=True,
            average_attn_weights=False,
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

    def forward(self, inputs: torch.Tensor) -> ModelOutput:
        feature_tokens = self._embed_features(inputs)
        batch_size = feature_tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls_token, feature_tokens], dim=1)
        hidden = hidden + self.position_embedding[:, : hidden.shape[1], :]
        hidden = self.input_dropout(hidden)

        last_attention = None
        for layer in self.layers:
            hidden, last_attention = layer(hidden)

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
    ) -> MCPredictionSummary:
        was_training = self.training
        labels = None
        probability_passes = []
        attention_passes = []
        token_score_passes = []
        cls_logit_passes = []

        with torch.no_grad():
            for _ in range(mc_samples):
                self.train()
                pass_probabilities = []
                pass_attention = []
                pass_token_scores = []
                pass_cls_logits = []
                pass_labels = []

                for features, target in loader:
                    features = features.to(device)
                    output = self(features)
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

        if not was_training:
            self.eval()

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
        )
