import torch
import torch.nn as nn

from .config import DiffusionLMConfig


class DiffusionTransformerLM(nn.Module):
    """A compact bidirectional Transformer for masked diffusion language modeling."""

    def __init__(self, config: DiffusionLMConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.time_emb = nn.Embedding(config.diffusion_steps + 1, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(config.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        """Return token logits for each position.

        Args:
            input_ids: Tensor of token ids with shape ``[batch, seq_len]``.
            timesteps: Integer diffusion step tensor with shape ``[batch]``.
            attention_mask: Boolean tensor where True means a valid token.
        """
        _, length = input_ids.shape
        if length > self.config.seq_len:
            raise ValueError(f"Sequence length {length} > config.seq_len {self.config.seq_len}")

        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = x + self.time_emb(timesteps).unsqueeze(1)
        x = self.drop(x)

        src_key_padding_mask = None if attention_mask is None else ~attention_mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        return self.lm_head(x)
