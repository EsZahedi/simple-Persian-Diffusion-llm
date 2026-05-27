from dataclasses import dataclass


@dataclass
class RunConfig:
    """Training/runtime configuration for a named experiment profile."""

    train_examples: int
    val_examples: int
    tokenizer_train_examples: int
    seq_len: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    diffusion_steps: int
    train_steps: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    warmup_steps: int
    dropout: float = 0.1
    dataset_name: str = "taesiri/TinyStories-Farsi"
    text_column: str = "Persian"


@dataclass
class DiffusionLMConfig:
    """Model architecture configuration."""

    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    diffusion_steps: int
    dropout: float = 0.1


def get_run_config(run_mode: str = "quick") -> RunConfig:
    """Return a reproducible configuration profile.

    Args:
        run_mode: Either ``quick`` for end-to-end validation or ``budget_100``
            for a heavier training run.
    """
    if run_mode == "quick":
        return RunConfig(
            train_examples=120_000,
            val_examples=2_000,
            tokenizer_train_examples=100_000,
            seq_len=256,
            vocab_size=26_000,
            d_model=512,
            n_layers=10,
            n_heads=8,
            d_ff=4 * 512,
            diffusion_steps=85,
            train_steps=10_000,
            batch_size=64,
            grad_accum=2,
            lr=4e-4,
            weight_decay=0.1,
            warmup_steps=200,
        )

    if run_mode == "budget_100":
        return RunConfig(
            train_examples=100_000,
            val_examples=10_000,
            tokenizer_train_examples=150_000,
            seq_len=256,
            vocab_size=26_000,
            d_model=512,
            n_layers=10,
            n_heads=8,
            d_ff=4 * 512,
            diffusion_steps=128,
            train_steps=60_000,
            batch_size=48,
            grad_accum=2,
            lr=2e-4,
            weight_decay=0.1,
            warmup_steps=1000,
        )

    raise ValueError("run_mode must be 'quick' or 'budget_100'")
