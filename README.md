# Simple Persian Diffusion LLM

A clean, resume-ready PyTorch project that trains a **Persian masked diffusion language model** from scratch on the `TinyStories-Farsi` dataset. The project includes tokenizer training, data streaming, model training, diffusion-style generation, and optional GIF visualization of the progressive unmasking process.

## Highlights

- Byte-level BPE tokenizer trained from scratch for Persian text
- Compact Transformer encoder language model with timestep embeddings
- Masked diffusion objective: predict original tokens at corrupted positions
- End-to-end training with Hugging Face `datasets`, `tokenizers`, `transformers`, and `accelerate`
- CLI scripts for training, inference, and GIF generation
- Professional Python package layout suitable for GitHub and resume presentation

## Repository Structure

```text
persian-diffusion-llm/
├── src/persian_diffusion_llm/
│   ├── config.py          # Experiment and model configs
│   ├── data.py            # Dataset loading and token block streaming
│   ├── diffusion.py       # Mask corruption schedule and loss
│   ├── model.py           # Diffusion Transformer LM
│   ├── tokenizer.py       # Byte-level BPE tokenizer training/loading
│   ├── sampling.py        # Iterative diffusion generation
│   ├── visualize.py       # Persian RTL GIF rendering helpers
│   ├── train.py           # Training CLI
│   ├── generate.py        # Text generation CLI
│   └── make_gif.py        # GIF generation CLI
├── scripts/               # Thin script wrappers
├── checkpoints/           # Local checkpoints, ignored by git
├── assets/                # Generated visualizations, ignored by git
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
git clone https://github.com/your-username/persian-diffusion-llm.git
cd persian-diffusion-llm
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

Or install from requirements only:

```bash
pip install -r requirements.txt
```

## Train

Quick end-to-end run:

```bash
python scripts/train.py --run-mode quick --output-dir checkpoints/
```

Heavier run:

```bash
python scripts/train.py --run-mode budget_100 --output-dir checkpoints/
```

The training command saves:

```text
checkpoints/model.pt
checkpoints/config.json
checkpoints/tokenizer/
checkpoints/val_loss_history.json
```

## Generate Text

```bash
python scripts/generate.py \
  --checkpoint-dir checkpoints/final \
  --prompt "روزی روزگاری" \
  --max-new-tokens 128 \
  --top-k 50
```

## Create an Inference GIF

```bash
python scripts/make_gif.py \
  --checkpoint-dir checkpoints/final \
  --prompt "روزی روزگاری" \
  --output assets/inference.gif
```

For Linux systems without Persian-compatible fonts, install fonts first:

```bash
sudo apt-get update
sudo apt-get install -y fonts-dejavu-core fonts-freefont-ttf
```

## Model Overview

1. Sample a diffusion timestep `t`.
2. Replace a timestep-dependent ratio of non-special tokens with `[MASK]`.
3. Predict the original token IDs only at masked positions.
4. During generation, begin with masked assistant tokens and progressively fill high-confidence tokens while re-masking lower-confidence ones.

## Resume Description

**Persian Diffusion LLM** — Built a from-scratch Persian text generation system using PyTorch, a custom byte-level BPE tokenizer, Transformer encoder architecture, and masked diffusion training objective. Implemented scalable data loading, mixed-precision training with Accelerate, iterative denoising inference, and GIF visualization for model behavior.

