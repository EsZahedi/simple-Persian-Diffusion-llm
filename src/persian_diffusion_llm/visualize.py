import os
from typing import Iterable

import arabic_reshaper
import imageio.v2 as imageio
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont


def fix_persian(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def get_font(size: int = 24):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text_to_width(text: str, max_chars: int = 90) -> list[str]:
    out = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            out.append("")
            continue
        while len(paragraph) > max_chars:
            out.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        out.append(paragraph)
    return out


def make_chat_lines(user_msg: str, assistant_text: str) -> list[str]:
    header = "================== multi-turn chat mode ===================="
    sub = "<Starting a new chat. Type your message.>"
    if "<|assistant|>" in assistant_text:
        assistant_text = assistant_text.split("<|assistant|>", 1)[1]
    assistant_text = assistant_text.replace("<|end|>", "").strip()
    return [header, sub, f"> {user_msg}", ""] + wrap_text_to_width(assistant_text, max_chars=90)


def render_terminal_frame(lines: list[str], width: int = 1200, height: int = 700, font_size: int = 24, margin: int = 20):
    image = Image.new("RGB", (width, height), (10, 10, 10))
    draw = ImageDraw.Draw(image)
    font = get_font(font_size)
    y = margin
    for line in lines:
        draw.text((margin, y), fix_persian(line), font=font, fill=(230, 230, 230), direction="rtl", language="fa")
        y += font_size + 8
    return image


def save_inference_gif(frames: Iterable[tuple[int, str]], user_prompt: str, output_path: str, total_steps: int, duration: float = 0.08):
    gif_frames = []
    for step, decoded in frames:
        lines = make_chat_lines(user_prompt, decoded)
        lines.insert(2, f"(diffusion step {step:03d}/{total_steps:03d})")
        gif_frames.append(np.array(render_terminal_frame(lines)))
    imageio.mimsave(output_path, gif_frames, duration=duration)
    return output_path
