import argparse
import json
import os

import torch

from .config import DiffusionLMConfig
from .model import DiffusionTransformerLM
from .sampling import chat_prompt, diffusion_generate
from .tokenizer import load_fast_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Persian text with a trained diffusion LM checkpoint.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/final")
    parser.add_argument("--prompt", default="روزی روزگاری")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.checkpoint_dir, "config.json"), "r", encoding="utf-8") as f:
        config = DiffusionLMConfig(**json.load(f))

    tokenizer = load_fast_tokenizer(os.path.join(args.checkpoint_dir, "tokenizer"))
    model = DiffusionTransformerLM(config).to(device)
    state_dict = torch.load(os.path.join(args.checkpoint_dir, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)

    final_text, _ = diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_text=chat_prompt(args.prompt),
        max_new_tokens=args.max_new_tokens,
        diffusion_steps=config.diffusion_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        record_steps=False,
    )
    print(final_text)


if __name__ == "__main__":
    main()
