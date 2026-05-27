import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


def load_farsi_tinystories(dataset_name: str, train_examples: int, val_examples: int):
    train_ds = load_dataset(dataset_name, split=f"train[:{train_examples}]")
    val_ds = load_dataset(dataset_name, split=f"validation[:{val_examples}]")
    return train_ds, val_ds


def format_as_chat(story_text: str) -> str:
    story_text = story_text.strip()
    return f"<|user|>یه داستان کوتاه بنویس.\n<|assistant|>\n{story_text}\n<|end|>\n"


class TokenBlockDataset(IterableDataset):
    """Stream tokenized Persian stories into fixed-length token blocks."""

    def __init__(self, hf_ds, tokenizer, seq_len: int, text_column: str = "Persian", shuffle: bool = False, seed: int = 0):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(len(self.hf_ds)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        buffer = []
        for idx in indices:
            text = format_as_chat(self.hf_ds[idx][self.text_column])
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len:
                block = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                yield torch.tensor(block, dtype=torch.long)


def collate_blocks(batch, pad_id: int):
    input_ids = torch.stack(batch, dim=0)
    attention_mask = input_ids != pad_id
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_dataloader(dataset, tokenizer, seq_len: int, batch_size: int, shuffle: bool, seed: int = 42, text_column: str = "Persian"):
    blocks = TokenBlockDataset(dataset, tokenizer, seq_len, text_column=text_column, shuffle=shuffle, seed=seed)
    return DataLoader(blocks, batch_size=batch_size, collate_fn=lambda batch: collate_blocks(batch, tokenizer.pad_token_id))
