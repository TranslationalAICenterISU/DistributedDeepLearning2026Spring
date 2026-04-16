"""
Module 06 — LLM Inference with Ollama
--------------------------------------
Ollama is a tool for running large language models locally (or on a
cluster node) with a simple REST API and Python client.

This script demonstrates six usage patterns:

  1. Check server connectivity and list available models
  2. Basic single-turn generation
  3. Streaming output (token by token)
  4. System prompt (persona / role)
  5. Multi-turn conversation (chat history)
  6. Batch queries (multiple prompts in sequence)

Prerequisites (handled by inference.sh):
  • ollama serve  is running in the background on localhost:11434
  • At least one model has been pulled  (default: llama3.2)

Run manually:
  # In one terminal:  ollama serve
  # In another:       python inference.py --model llama3.2
"""
import argparse
import sys
import time

import ollama


# ── Utilities ─────────────────────────────────────────────────────

def separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print(f"\n{'─'*width}")


def check_server(host: str) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        client = ollama.Client(host=host)
        client.list()
        return True
    except Exception as e:
        print(f"[ERROR] Cannot reach Ollama server at {host}: {e}")
        print("  Make sure 'ollama serve' is running before launching this script.")
        return False


# ── Demo functions ────────────────────────────────────────────────

def demo_list_models(client):
    separator("1. Available Models")
    models = client.list()
    if not models.models:
        print("  No models pulled yet.  Run:  ollama pull llama3.2")
        return
    for m in models.models:
        size_gb = m.size / 1e9 if hasattr(m, "size") else "?"
        print(f"  • {m.model:<35}  {size_gb:.1f} GB" if isinstance(size_gb, float)
              else f"  • {m.model}")


def demo_basic_generation(client, model: str):
    separator("2. Basic Generation")
    prompt = "In one sentence, explain what a GPU is."
    print(f"  Prompt : {prompt}")
    print(f"  Model  : {model}\n")

    t0 = time.time()
    response = client.generate(model=model, prompt=prompt)
    elapsed  = time.time() - t0

    print(f"  Response:\n  {response.response.strip()}")
    print(f"\n  Tokens generated : {response.eval_count}")
    print(f"  Time             : {elapsed:.2f}s")
    print(f"  Tokens / second  : {response.eval_count / elapsed:.1f}")


def demo_streaming(client, model: str):
    separator("3. Streaming Output")
    prompt = "List three reasons why distributed training is useful for deep learning."
    print(f"  Prompt : {prompt}")
    print(f"  Model  : {model}\n")
    print("  Response (streaming):\n")

    for chunk in client.generate(model=model, prompt=prompt, stream=True):
        print(chunk.response, end="", flush=True)
        if chunk.done:
            break
    print()


def demo_system_prompt(client, model: str):
    separator("4. System Prompt (Role / Persona)")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise teaching assistant for a university workshop on "
                "distributed deep learning.  Answer in at most two sentences and "
                "avoid jargon unless you define it."
            ),
        },
        {
            "role": "user",
            "content": "What is the difference between DDP and FSDP in PyTorch?",
        },
    ]
    print(f"  System : {messages[0]['content'][:80]}...")
    print(f"  User   : {messages[1]['content']}\n")

    response = client.chat(model=model, messages=messages)
    print(f"  Response:\n  {response.message.content.strip()}")


def demo_multi_turn(client, model: str):
    separator("5. Multi-Turn Conversation")
    history = []

    turns = [
        "What is gradient descent?",
        "How does a learning rate affect it?",
        "What happens if the learning rate is too large?",
    ]

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})
        print(f"  User      : {user_msg}")

        response = client.chat(model=model, messages=history)
        assistant_msg = response.message.content.strip()
        history.append({"role": "assistant", "content": assistant_msg})

        # Truncate long answers for cleaner demo output
        display = assistant_msg[:200] + "…" if len(assistant_msg) > 200 else assistant_msg
        print(f"  Assistant : {display}\n")


def demo_batch_queries(client, model: str):
    separator("6. Batch Queries")
    prompts = [
        "What is a tensor?",
        "What is backpropagation?",
        "What is the NCCL library used for?",
    ]
    print(f"  Running {len(prompts)} prompts sequentially...\n")

    for i, prompt in enumerate(prompts, 1):
        t0 = time.time()
        resp = client.generate(model=model, prompt=prompt)
        elapsed = time.time() - t0
        answer = resp.response.strip()
        display = answer[:150] + "…" if len(answer) > 150 else answer
        print(f"  Q{i}: {prompt}")
        print(f"  A{i}: {display}")
        print(f"       ({resp.eval_count} tokens, {elapsed:.2f}s)\n")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Module 06 — Ollama LLM Inference")
    parser.add_argument("--model",  default="llama3.2",
                        help="Ollama model name (must be pulled first)")
    parser.add_argument("--host",   default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--demos",  default="all",
                        help="Comma-separated demos to run: list,basic,stream,system,chat,batch "
                             "or 'all'")
    args = parser.parse_args()

    print("=" * 60)
    print("  Module 06 — LLM Inference with Ollama")
    print("=" * 60)
    print(f"  Server : {args.host}")
    print(f"  Model  : {args.model}")

    if not check_server(args.host):
        sys.exit(1)

    client = ollama.Client(host=args.host)

    demos_to_run = (
        {"list", "basic", "stream", "system", "chat", "batch"}
        if args.demos == "all"
        else set(args.demos.split(","))
    )

    if "list"   in demos_to_run: demo_list_models(client)
    if "basic"  in demos_to_run: demo_basic_generation(client, args.model)
    if "stream" in demos_to_run: demo_streaming(client, args.model)
    if "system" in demos_to_run: demo_system_prompt(client, args.model)
    if "chat"   in demos_to_run: demo_multi_turn(client, args.model)
    if "batch"  in demos_to_run: demo_batch_queries(client, args.model)

    separator()
    print("  All demos complete.")


if __name__ == "__main__":
    main()
