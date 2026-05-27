import argparse
import json
import os

import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from .config import DiffusionLMConfig, get_run_config
from .data import load_farsi_tinystories, make_dataloader
from .diffusion import diffusion_loss
from .model import DiffusionTransformerLM
from .tokenizer import load_fast_tokenizer, train_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Persian diffusion language model from scratch.")
    parser.add_argument("--run-mode", default="quick", choices=["quick", "budget_100"])
    parser.add_argument("--output-dir", default="checkpoints/final")
    parser.add_argument("--tokenizer-dir", default="tokenizer_from_scratch")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_config = get_run_config(args.run_mode)
    train_ds, val_ds = load_farsi_tinystories(run_config.dataset_name, run_config.train_examples, run_config.val_examples)

    tokenizer_file = train_tokenizer(
        train_ds,
        output_dir=args.tokenizer_dir,
        vocab_size=run_config.vocab_size,
        n_examples=run_config.tokenizer_train_examples,
        text_column=run_config.text_column,
    )
    tokenizer = load_fast_tokenizer(tokenizer_file)

    model_config = DiffusionLMConfig(
        vocab_size=len(tokenizer),
        seq_len=run_config.seq_len,
        d_model=run_config.d_model,
        n_layers=run_config.n_layers,
        n_heads=run_config.n_heads,
        d_ff=run_config.d_ff,
        dropout=run_config.dropout,
        diffusion_steps=run_config.diffusion_steps,
    )
    model = DiffusionTransformerLM(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_loader = make_dataloader(train_ds, tokenizer, run_config.seq_len, run_config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, tokenizer, run_config.seq_len, run_config.batch_size, shuffle=False)

    mixed_precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=run_config.warmup_steps,
        num_training_steps=run_config.train_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    protected_token_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]

    def eval_loss(n_batches: int = 20):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                loss = diffusion_loss(model, batch, run_config.diffusion_steps, tokenizer.mask_token_id, protected_token_ids)
                losses.append(accelerator.gather(loss.detach().float().reshape(1)).cpu())
        model.train()
        return float("nan") if not losses else torch.cat(losses).mean().item()

    model.train()
    val_history = []
    train_iter = iter(train_loader)
    running = []
    pbar = tqdm(range(run_config.train_steps), disable=not accelerator.is_main_process)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = diffusion_loss(model, batch, run_config.diffusion_steps, tokenizer.mask_token_id, protected_token_ids)
        loss = loss / run_config.grad_accum
        accelerator.backward(loss)

        if (step + 1) % run_config.grad_accum == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running.append(loss.item())
        if (step + 1) % 50 == 0 and accelerator.is_main_process:
            pbar.set_description(f"loss={np.mean(running[-50:]):.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if (step + 1) % 5000 == 0 and accelerator.is_main_process:
            val_loss = eval_loss(n_batches=20)
            print(f"Step {step + 1} | val_loss ~ {val_loss:.4f}")
            val_history.append({"step": step + 1, "val_loss": round(val_loss, 4)})

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(args.output_dir, "model.pt"))
        with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(model_config.__dict__, f, indent=2)
        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        with open(os.path.join(args.output_dir, "val_loss_history.json"), "w", encoding="utf-8") as f:
            json.dump(val_history, f, indent=2)
        print("Saved final checkpoint to:", args.output_dir)


if __name__ == "__main__":
    main()
