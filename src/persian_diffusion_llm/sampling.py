import math

import torch
import torch.nn.functional as F


def chat_prompt(user_msg: str, system_msg: str | None = None) -> str:
    parts = []
    if system_msg:
        parts.append(f"<|system|>\n{system_msg}\n")
    parts.append(f"<|user|>\n{user_msg}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


@torch.no_grad()
def diffusion_generate(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    diffusion_steps: int | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    record_steps: bool = True,
):
    """Generate text by iterative denoising/unmasking."""
    model.eval()
    device = next(model.parameters()).device
    diffusion_steps = diffusion_steps or model.config.diffusion_steps

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    prompt_len = prompt_ids.size(1)
    length = min(model.config.seq_len, prompt_len + max_new_tokens)
    gen_len = max(0, length - prompt_len)

    x = torch.full((1, length), tokenizer.mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids[:, :prompt_len]

    fixed = torch.zeros((1, length), dtype=torch.bool, device=device)
    fixed[:, :prompt_len] = True
    attention_mask = torch.ones((1, length), dtype=torch.bool, device=device)
    frames = []

    def sample_from_logits(logits):
        if temperature != 1.0:
            logits = logits / temperature
        if top_k and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            logits = filtered
        probs = F.softmax(logits, dim=-1)
        flat = probs.reshape(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1).reshape(1, length)
        sampled_prob = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, sampled_prob

    for step in range(diffusion_steps, 0, -1):
        timestep = torch.tensor([step], device=device, dtype=torch.long)
        logits = model(x, timesteps=timestep, attention_mask=attention_mask)
        sampled, confidence = sample_from_logits(logits)

        update_pos = ~fixed
        x[update_pos] = sampled[update_pos]

        next_ratio = float(step - 1) / float(diffusion_steps)
        target_masks = int(math.ceil(gen_len * next_ratio))

        generated_positions = torch.arange(length, device=device) >= prompt_len
        candidates = generated_positions & (~fixed[0])
        candidate_idx = torch.where(candidates)[0]

        if target_masks > 0 and candidate_idx.numel() > 0:
            candidate_conf = confidence[0, candidate_idx]
            k = min(target_masks, candidate_idx.numel())
            _, low_idx = torch.topk(candidate_conf, k=k, largest=False)
            remask_positions = candidate_idx[low_idx]
            x[0, remask_positions] = tokenizer.mask_token_id

        if record_steps:
            decoded = tokenizer.decode(x[0].tolist()).replace("[MASK]", "■")
            frames.append((step, decoded))

    final = tokenizer.decode(x[0].tolist())
    model.train()
    return final, frames
