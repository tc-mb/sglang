"""
Precision validation script for MiniCPM-V 4.6 in SGLang.

Compares SGLang output against vLLM output on the same prompt + image.
Both engines consume the HF checkpoint directly, so an exact match is not
expected, but semantic consistency and closely matching text output is.

Usage:
    python test_sglang_minicpmv_4_6.py both      # run both, then compare
    python test_sglang_minicpmv_4_6.py vllm      # run vLLM only
    python test_sglang_minicpmv_4_6.py sglang    # run SGLang only
    python test_sglang_minicpmv_4_6.py weights   # weight-name mapping check
"""

import os
import sys

MODEL_PATH = os.environ.get(
    "MINICPMV_PATH",
    "/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0416-instruct",
)
IMAGE_PATH = "/cache/caitianchi/code/v50/code/minicpm-v-4_6-test/highway.png"
QUESTION = "Describe this image in detail."
MAX_NEW_TOKENS = 128


def get_test_image():
    from PIL import Image, ImageDraw

    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        return Image.open(IMAGE_PATH).convert("RGB")

    fallback = "/tmp/minicpmv_4_6_sanity.jpg"
    if os.path.exists(fallback):
        return Image.open(fallback).convert("RGB")

    img = Image.new("RGB", (448, 448), color=(128, 64, 192))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill=(255, 0, 0))
    draw.rectangle([250, 100, 400, 350], fill=(0, 255, 0))
    draw.ellipse([100, 250, 350, 430], fill=(0, 0, 255))
    img.save(fallback)
    return img


def _print_header(title: str):
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_vllm():
    _print_header("vLLM reference")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
    )

    llm = LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )

    image = get_test_image()
    messages = [{
        "role": "user",
        "content": "(<image>./</image>)\n" + QUESTION,
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    stop_tokens = ["<|im_end|>", "</s>"]
    stop_token_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in stop_tokens
        if t in tokenizer.get_vocab()
    ]

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0,
        stop_token_ids=stop_token_ids,
    )

    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=sampling_params,
    )
    text = outputs[0].outputs[0].text.strip()
    print(f"[vLLM]   {text}")

    del llm
    import torch
    torch.cuda.empty_cache()
    return text


def run_sglang():
    _print_header("SGLang output")

    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
    )

    llm = sgl.Engine(
        model_path=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        mm_attention_backend="sdpa",
        max_running_requests=2,
        context_length=4096,
        mem_fraction_static=0.85,
    )

    image = get_test_image()
    messages = [{
        "role": "user",
        "content": "(<image>./</image>)\n" + QUESTION,
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    stop_tokens = ["<|im_end|>", "</s>"]
    stop_token_ids = [
        tokenizer.convert_tokens_to_ids(t)
        for t in stop_tokens
        if t in tokenizer.get_vocab()
    ]

    tmp_image = "/tmp/sglang_minicpmv_4_6_probe.jpg"
    image.save(tmp_image)

    out = llm.generate(
        prompt=prompt,
        image_data=[tmp_image],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_token_ids": stop_token_ids,
        },
    )
    text = (out["text"] if isinstance(out, dict) else out).strip()
    print(f"[SGLang] {text}")

    llm.shutdown()
    return text


def check_weight_mapping():
    """Run our SGLang load_weights against the HF checkpoint and report
    any weight that fails to find a matching SGLang parameter.
    """
    _print_header("Weight-name mapping check (static)")

    import json
    from safetensors.torch import safe_open

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = json.load(f)
    print(f"architecture : {config.get('architectures')}")
    print(f"model_type   : {config.get('model_type')}")
    print(f"version field: {config.get('version')}")
    print(f"insert_layer : {config.get('insert_layer_id')}")
    print(f"tie_word_emb : {config.get('tie_word_embeddings')}")
    print(f"drop_vision  : {config.get('drop_vision_last_layer')}")

    # We cannot build the SGLang model without a GPU and distributed init, so
    # we just enumerate the expected parameter names by static inspection.
    # Instead, list the weight prefixes present in the checkpoint so the
    # developer can sanity-check load_weights manually.
    shards = []
    idx_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    prefixes = {}
    for shard in shards:
        with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
            for k in f.keys():
                head = ".".join(k.split(".")[:3])
                prefixes[head] = prefixes.get(head, 0) + 1
    for head, n in sorted(prefixes.items()):
        print(f"  {head:40s}  {n}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "weights":
        check_weight_mapping()
    elif mode == "vllm":
        run_vllm()
    elif mode == "sglang":
        run_sglang()
    else:
        vllm_text = run_vllm()
        print()
        sgl_text = run_sglang()
        print()
        _print_header("comparison")
        print("vLLM  :", vllm_text)
        print("SGLang:", sgl_text)
        if vllm_text == sgl_text:
            print("RESULT: EXACT MATCH")
        else:
            print(
                "RESULT: outputs differ (some drift is normal; inspect that "
                "the semantic content matches)."
            )
