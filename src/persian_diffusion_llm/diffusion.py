import torch
import torch.nn.functional as F


def mask_ratio_schedule(t, total_steps: int):
    """Linear schedule: later diffusion steps mask more tokens."""
    return t.float() / float(total_steps)


@torch.no_grad()
def corrupt_with_mask(input_ids, attention_mask, timesteps, mask_token_id: int, total_steps: int, protected_token_ids=None):
    """Replace a scheduled fraction of tokens with [MASK]."""
    if protected_token_ids is None:
        protected_token_ids = []

    batch_size, seq_len = input_ids.shape
    ratio = mask_ratio_schedule(timesteps, total_steps).unsqueeze(1)

    can_mask = attention_mask.clone()
    for token_id in protected_token_ids:
        can_mask &= input_ids != token_id

    rand = torch.rand((batch_size, seq_len), device=input_ids.device)
    mask_positions = (rand < ratio) & can_mask

    noisy_ids = input_ids.clone()
    noisy_ids[mask_positions] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]
    return noisy_ids, labels, mask_positions


def diffusion_loss(model, batch, total_steps: int, mask_token_id: int, protected_token_ids=None):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size = input_ids.size(0)
    timesteps = torch.randint(1, total_steps + 1, (batch_size,), device=input_ids.device)

    noisy_ids, labels, _ = corrupt_with_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        timesteps=timesteps,
        mask_token_id=mask_token_id,
        total_steps=total_steps,
        protected_token_ids=protected_token_ids,
    )

    logits = model(noisy_ids, timesteps, attention_mask)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
