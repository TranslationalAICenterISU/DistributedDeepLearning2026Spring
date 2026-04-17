"""
Module 06 — Multi-Model LLM Inference with Ollama
---------------------------------------------------
Demonstrates three open-weight models, each highlighting a different
capability tier:

  Section 1 — gpt-oss:20b
    OpenAI's open-weight model, optimised for reasoning and agentic tasks.
    Demos: chain-of-thought reasoning, multi-step task planning.

  Section 2 — qwen3.6
    Qwen3.6, strong at agentic coding and preserving reasoning chains.
    Demos: code generation, streaming code explanation.

  Section 3 — gemma4  (multimodal)
    Google Gemma 4, supports image + text inputs.
    Demos: describe a Fashion MNIST sample, answer a specific visual question.

Prerequisites (handled by inference.sh):
  • ollama serve  is running in the background on localhost:11434
  • All three models have been pulled

Run manually:
  ollama serve &
  python inference.py --host http://localhost:11434
"""
import argparse
import base64
import io
import os
import subprocess
import sys
import time

import ollama

GPT_OSS = "gpt-oss:20b"
QWEN    = "qwen3.6"
GEMMA   = "gemma4"


# ── Utilities ─────────────────────────────────────────────────────

def separator(title=""):
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print(f"\n{'─'*width}")


def check_server(host: str) -> bool:
    try:
        ollama.Client(host=host).list()
        return True
    except Exception as e:
        print(f"[ERROR] Cannot reach Ollama server at {host}: {e}")
        return False


def nvidia_smi_vram() -> str:
    """Return a compact VRAM summary from nvidia-smi, or empty string if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        lines = []
        for row in out.splitlines():
            idx, name, used, total = [x.strip() for x in row.split(",")]
            lines.append(f"    GPU {idx} ({name}): {used} / {total} MiB used")
        return "\n".join(lines)
    except Exception:
        return ""


def check_gpu_placement(client: ollama.Client, model_name: str):
    """
    Query ollama /api/ps to confirm the model loaded onto VRAM.
    Prints placement status and current nvidia-smi VRAM usage.
    """
    separator("GPU placement check")
    try:
        ps = client.ps()
        loaded = {m.model: m for m in ps.models}
        # ollama ps uses short names; match on prefix
        match = next((m for k, m in loaded.items() if model_name.split(":")[0] in k), None)
        if match:
            size_gb  = (match.size      or 0) / 1e9
            vram_gb  = (match.size_vram or 0) / 1e9
            on_gpu   = vram_gb > 0.1
            location = "GPU" if on_gpu else "CPU (no VRAM allocated — check GPU availability)"
            print(f"  Model      : {match.model}")
            print(f"  Total size : {size_gb:.2f} GB")
            print(f"  VRAM used  : {vram_gb:.2f} GB  →  [{location}]")
        else:
            print(f"  {model_name} not found in ollama ps (may have already been unloaded)")
    except Exception as e:
        print(f"  ollama ps failed: {e}")

    vram = nvidia_smi_vram()
    if vram:
        print(f"\n  nvidia-smi snapshot:\n{vram}")


def timed_generate(client, model, prompt, stream=False, system=None):
    """Generate text, return (response_text, tokens, elapsed_s)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    if stream:
        text = ""
        for chunk in client.chat(model=model, messages=messages, stream=True):
            delta = chunk.message.content or ""
            print(delta, end="", flush=True)
            text += delta
            if chunk.done:
                break
        print()
        elapsed = time.time() - t0
        return text, None, elapsed
    else:
        resp = client.chat(model=model, messages=messages)
        elapsed = time.time() - t0
        return resp.message.content.strip(), getattr(resp, "eval_count", None), elapsed


# ── Section 1: gpt-oss:20b — Reasoning & Agentic Planning ─────────

def demo_gpt_oss(client):
    print("\n" + "=" * 64)
    print(f"  SECTION 1 — {GPT_OSS}")
    print("  Reasoning & Agentic Task Planning")
    print("=" * 64)

    # 1a. Chain-of-thought reasoning
    separator("1a. Chain-of-Thought Reasoning")
    prompt = (
        "A distributed training job uses 8 GPUs across 2 nodes. "
        "Each GPU processes a local batch of 32 samples. "
        "After each backward pass the gradients are all-reduced. "
        "If the all-reduce bandwidth between nodes is 25 GB/s and each "
        "gradient tensor is 400 MB, how long does the all-reduce take? "
        "Show your reasoning step by step."
    )
    print(f"  Prompt: {prompt}\n")
    text, tokens, elapsed = timed_generate(client, GPT_OSS, prompt)
    print(f"  {text}")
    print(f"\n  [{elapsed:.1f}s]")

    # 1b. Agentic task planning
    separator("1b. Agentic Task Decomposition")
    prompt = (
        "I want to fine-tune a 7B parameter language model on a custom dataset "
        "using 4 A100 GPUs. Break this down into concrete numbered steps, "
        "including data preparation, environment setup, training configuration, "
        "and evaluation. Be specific about tool choices."
    )
    print(f"  Prompt: {prompt}\n")
    text, tokens, elapsed = timed_generate(client, GPT_OSS, prompt)
    print(f"  {text}")
    print(f"\n  [{elapsed:.1f}s]")

    check_gpu_placement(client, GPT_OSS)


# ── Section 2: qwen3.6 — Coding & Thinking ────────────────────────

def demo_qwen(client):
    print("\n" + "=" * 64)
    print(f"  SECTION 2 — {QWEN}")
    print("  Agentic Coding & Reasoning Preservation")
    print("=" * 64)

    # 2a. Code generation
    separator("2a. Code Generation")
    prompt = (
        "Write a Python function `sync_gradients(model, world_size)` that "
        "manually all-reduces gradients across ranks using torch.distributed. "
        "Include type hints and a brief docstring. No extra explanation needed."
    )
    print(f"  Prompt: {prompt}\n")
    text, tokens, elapsed = timed_generate(client, QWEN, prompt)
    print(f"  {text}")
    print(f"\n  [{elapsed:.1f}s]")

    # 2b. Streaming code explanation
    separator("2b. Code Explanation (streaming)")
    code = """
def save_fsdp_checkpoint(model, save_path, rank):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state = model.state_dict()
    if rank == 0:
        torch.save(state, save_path)
"""
    prompt = (
        f"Explain what this PyTorch FSDP checkpoint function does and why "
        f"`state_dict_type` must be called outside `if rank == 0:`:\n{code}"
    )
    print(f"  Prompt: {prompt}\n  Response (streaming):\n")
    timed_generate(client, QWEN, prompt, stream=True)

    # 2c. Debugging with thinking
    separator("2c. Bug Identification")
    buggy_code = """
# Bug: only rank 0 saves, but state_dict gathering is collective
if rank == 0:
    if val_acc > best_acc:
        save_fsdp_checkpoint(model, ckpt_path, rank)
"""
    prompt = (
        f"Identify the bug in this FSDP training loop snippet and explain "
        f"how to fix it:\n{buggy_code}"
    )
    print(f"  Prompt: {prompt}\n")
    text, tokens, elapsed = timed_generate(client, QWEN, prompt)
    print(f"  {text}")
    print(f"\n  [{elapsed:.1f}s]")

    check_gpu_placement(client, QWEN)


# ── Section 3: gemma4 — Multimodal ────────────────────────────────

def load_fashion_mnist_image(data_dir: str):
    """
    Return (b64_string, label_name) for the first Fashion MNIST test sample.
    Falls back to a synthetic matplotlib image if the dataset is not present.
    """
    LABELS = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    # Try loading from torchvision dataset
    try:
        from torchvision import datasets
        from PIL import Image as PILImage

        ds = datasets.FashionMNIST(data_dir, train=False, download=False)
        img_tensor, label_idx = ds[0]
        # ds returns a PIL image when no transform is set
        if not hasattr(img_tensor, "save"):
            # convert tensor → PIL
            import torchvision.transforms.functional as TF
            img_pil = TF.to_pil_image(img_tensor)
        else:
            img_pil = img_tensor

        # Upscale for better visibility (28→224)
        img_pil = img_pil.resize((224, 224), PILImage.NEAREST)

        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), LABELS[label_idx]

    except Exception:
        pass

    # Fallback: synthetic matplotlib figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(3, 3))
        data = np.random.rand(28, 28)
        ax.imshow(data, cmap="gray")
        ax.set_title("Synthetic grayscale image")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode(), "synthetic"

    except Exception as e:
        return None, f"(image creation failed: {e})"


def demo_gemma(client, data_dir: str):
    print("\n" + "=" * 64)
    print(f"  SECTION 3 — {GEMMA}")
    print("  Multimodal: Vision + Language")
    print("=" * 64)

    img_b64, label = load_fashion_mnist_image(data_dir)

    if img_b64 is None:
        print(f"  [SKIP] Could not load image: {label}")
        return

    source = "Fashion MNIST test set" if label != "synthetic" else "synthetic fallback"
    print(f"\n  Image source : {source}")
    if label != "synthetic":
        print(f"  True label   : {label}  (hidden from model)")

    # 3a. Open-ended description
    separator("3a. Image Description")
    prompt = "Describe what you see in this image in two or three sentences."
    print(f"  Prompt: {prompt}\n")
    t0 = time.time()
    resp = client.chat(
        model=GEMMA,
        messages=[{
            "role":    "user",
            "content": prompt,
            "images":  [img_b64],
        }],
    )
    elapsed = time.time() - t0
    print(f"  {resp.message.content.strip()}")
    print(f"\n  [{elapsed:.1f}s]")

    # 3b. Specific visual question
    separator("3b. Visual Classification")
    prompt = (
        "This is a grayscale image from the Fashion MNIST dataset. "
        "Which clothing category does it belong to? Choose from: "
        "T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, "
        "Sneaker, Bag, Ankle boot. State only the category name and your confidence."
    )
    print(f"  Prompt: {prompt}\n")
    t0 = time.time()
    resp = client.chat(
        model=GEMMA,
        messages=[{
            "role":    "user",
            "content": prompt,
            "images":  [img_b64],
        }],
    )
    elapsed = time.time() - t0
    print(f"  Model answer : {resp.message.content.strip()}")
    if label != "synthetic":
        print(f"  True label   : {label}")
    print(f"\n  [{elapsed:.1f}s]")

    check_gpu_placement(client, GEMMA)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 06 — Multi-Model Ollama Inference")
    parser.add_argument("--host",     default="http://localhost:11434")
    parser.add_argument("--data-dir", default="/work/mech-ai/Aditya/data",
                        help="Path to the Fashion MNIST dataset root (for multimodal demo)")
    parser.add_argument("--skip",     default="",
                        help="Comma-separated sections to skip: gpt,qwen,gemma")
    args = parser.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    print("=" * 64)
    print("  Module 06 — Multi-Model LLM Inference with Ollama")
    print("=" * 64)
    print(f"  Server   : {args.host}")
    print(f"  Models   : {GPT_OSS}  |  {QWEN}  |  {GEMMA}")
    print(f"  Data dir : {args.data_dir}")

    if not check_server(args.host):
        sys.exit(1)

    client = ollama.Client(host=args.host)

    if "gpt" not in skip:
        demo_gpt_oss(client)

    if "qwen" not in skip:
        demo_qwen(client)

    if "gemma" not in skip:
        demo_gemma(client, args.data_dir)

    separator()
    print("  All demos complete.")


if __name__ == "__main__":
    main()
