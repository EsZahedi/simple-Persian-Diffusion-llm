import os
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[MASK]",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
]


def iter_tokenizer_texts(dataset, n_examples: int, text_column: str = "Persian") -> Iterable[str]:
    for i in range(min(n_examples, len(dataset))):
        story = dataset[i][text_column].strip()
        yield f"<|user|>Write a short story.\n<|assistant|>\n{story}\n<|end|>\n"


def train_tokenizer(dataset, output_dir: str, vocab_size: int, n_examples: int, text_column: str = "Persian") -> str:
    """Train a byte-level BPE tokenizer and return the tokenizer.json path."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_tokenizer_texts(dataset, n_examples, text_column), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]"))],
    )
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    return tokenizer_path


def load_fast_tokenizer(tokenizer_file_or_dir: str) -> PreTrainedTokenizerFast:
    """Load a Hugging Face fast tokenizer with all special tokens configured."""
    tokenizer_file = tokenizer_file_or_dir
    if os.path.isdir(tokenizer_file_or_dir):
        tokenizer_file = os.path.join(tokenizer_file_or_dir, "tokenizer.json")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]}
    )
    return tokenizer
