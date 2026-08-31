"""Measure bound-output dispatch inside real Hugging Face generation on a free T4.

The benchmark loads exact public model revisions, replaces each model RMSNorm
with one shared TileLang kernel per shape, and compares the callee-allocated and
caller-owned APIs during deterministic multi-turn chat generation.  Stock
PyTorch RMSNorm remains a correctness and integration control.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
from collections.abc import Callable
import gc
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.request


REPOSITORY = "nya-a-cat/tilelang"
SOURCE_SHA_ENV = "TILELANG_PYTHON_OVERLAY_SOURCE_SHA"
HELPER_PATH = "benchmark/research/benchmark_bound_outputs_t4.py"
HELPER_SHA256 = "05860ead31f9709b8d1941a29b374e1e8f6e952b9efb2e7830b8f8e41643d50d"
RESULT_PATH = Path("/content/tilelang-hf-bound-output-t4.json")
RESULT_MARKER = "TILELANG_HF_BOUND_OUTPUT_RESULT="
TRANSFORMERS_VERSION = "4.57.1"
ACCELERATE_VERSION = "1.11.0"
MODEL_FILTER_ENV = "TILELANG_HF_MODELS"
NEW_TOKENS_ENV = "TILELANG_HF_NEW_TOKENS"
CYCLES_ENV = "TILELANG_HF_CYCLES"
NEW_TOKENS = int(os.environ.get(NEW_TOKENS_ENV, "32"))
CYCLES = int(os.environ.get(CYCLES_ENV, "4"))
ORDER = ("stock", "callee", "bound", "bound", "callee", "stock")
MODES = ("stock", "callee", "bound")

MODEL_IDS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

MESSAGES = (
    {
        "role": "system",
        "content": "You are a concise technical assistant. Answer accurately and directly.",
    },
    {
        "role": "user",
        "content": "What does matrix multiplication compute? Give a short intuitive explanation.",
    },
    {
        "role": "assistant",
        "content": "It combines rows from one matrix with columns from another using dot products.",
    },
    {
        "role": "user",
        "content": "Now give one practical use in machine learning and explain it in two sentences.",
    },
)


def run(command: list[str], *, capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError(f"geometric mean requires positive values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_exact_setup_helper(source_sha: str) -> tuple[Any, dict[str, Any]]:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{source_sha}/{HELPER_PATH}"
    path = Path("/tmp/tilelang_colab_setup_helper.py")
    started = time.perf_counter()
    urllib.request.urlretrieve(url, path)
    actual_hash = sha256_file(path)
    if actual_hash != HELPER_SHA256:
        raise RuntimeError(f"setup helper hash mismatch: {actual_hash}")
    spec = importlib.util.spec_from_file_location("tilelang_colab_setup_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the exact setup helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {
        "repository_path": HELPER_PATH,
        "source_sha": source_sha,
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
        "download_seconds": time.perf_counter() - started,
    }


class RMSNormKernelRegistry:
    def __init__(self, torch: Any, tilelang: Any, language: Any):
        self.torch = torch
        self.tilelang = tilelang
        self.T = language
        self.kernels: dict[tuple[int, int, str, float], Any] = {}
        self.compile_records: list[dict[str, Any]] = []

    def _make_program(self, rows: int, width: int, dtype: str, epsilon: float, symbol: str) -> Any:
        T = self.T

        @T.prim_func
        def main(
            A: T.Tensor((rows, width), dtype),
            Weight: T.Tensor((width,), dtype),
            B: T.Tensor((rows, width), dtype),
        ):
            with T.Kernel(rows, threads=128) as bx:
                values = T.alloc_fragment((width,), dtype)
                squares = T.alloc_fragment((width,), "float32")
                total = T.alloc_fragment((1,), "float32")
                T.copy(A[bx, 0], values)
                for index in T.Parallel(width):
                    value = T.cast(values[index], "float32")
                    squares[index] = value * value
                T.reduce_sum(squares, total, dim=0)
                scale = T.rsqrt(total[0] / width + epsilon)
                for index in T.Parallel(width):
                    B[bx, index] = T.cast(
                        T.cast(values[index], "float32") * scale * T.cast(Weight[index], "float32"),
                        dtype,
                    )

        return main.with_attr("global_symbol", symbol).with_attr("tilelang_out_idx", [-1])

    def get(self, rows: int, width: int, dtype: Any, epsilon: float) -> Any:
        dtype_name = {
            self.torch.float16: "float16",
            self.torch.float32: "float32",
        }.get(dtype)
        if dtype_name is None:
            raise TypeError(f"unsupported RMSNorm dtype: {dtype}")
        key = (rows, width, dtype_name, float(epsilon))
        cached = self.kernels.get(key)
        if cached is not None:
            return cached
        symbol_epsilon = f"{epsilon:.12g}".replace("-", "m").replace(".", "p")
        symbol = f"hf_rmsnorm_{dtype_name}_{rows}x{width}_{symbol_epsilon}"
        program = self._make_program(rows, width, dtype_name, epsilon, symbol)
        started = time.perf_counter()
        kernel = self.tilelang.compile(
            program,
            target="cuda",
            execution_backend="tvm_ffi",
            verbose=False,
        )
        compile_seconds = time.perf_counter() - started
        expected_out_idx = [len(kernel.params) - 1]
        if kernel.out_idx != expected_out_idx:
            raise RuntimeError(f"unexpected RMSNorm output indices: {kernel.out_idx}")
        self.kernels[key] = kernel
        self.compile_records.append(
            {
                "rows": rows,
                "width": width,
                "dtype": dtype_name,
                "epsilon": epsilon,
                "symbol": symbol,
                "callee_compile_seconds": compile_seconds,
                "callee_cache_key": getattr(kernel, "_tilelang_cache_key", None),
            }
        )
        return kernel

    def record_caller_kernel(self, kernel: Any) -> None:
        companion = kernel._caller_allocated_kernel
        if companion is None or companion.out_idx:
            raise RuntimeError("RMSNorm caller-owned companion ABI was not prepared")
        caller_key = getattr(companion, "_tilelang_cache_key", None)
        callee_key = getattr(kernel, "_tilelang_cache_key", None)
        for record in self.compile_records:
            if record["callee_cache_key"] == callee_key:
                record["caller_cache_key"] = caller_key
                record["distinct_cache_keys"] = caller_key != callee_key
                return
        raise RuntimeError("caller-owned kernel has no compile record")


class TileLangRMSNorm:
    def __init__(self, torch: Any, reference: Any, registry: RMSNormKernelRegistry, module_name: str):
        class Wrapper(torch.nn.Module):
            pass

        self.wrapper = Wrapper()
        self.wrapper.reference = reference
        self.wrapper.forward = self.forward
        self.torch = torch
        self.registry = registry
        self.module_name = module_name
        self.mode = "stock"
        self.bound_calls: dict[tuple[int, int, Any, float], tuple[Any, Callable[..., Any]]] = {}
        self.call_counts: Counter[str] = Counter()
        self.shape_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.noncontiguous_inputs = 0

        epsilon = getattr(reference, "variance_epsilon", None)
        if epsilon is None:
            epsilon = getattr(reference, "eps", None)
        if epsilon is None:
            raise TypeError(f"RMSNorm module {module_name} exposes no epsilon")
        self.epsilon = float(epsilon)

    def forward(self, hidden_states: Any) -> Any:
        mode = self.mode
        self.call_counts[mode] += 1
        shape_key = "x".join(str(value) for value in hidden_states.shape)
        self.shape_counts[mode][shape_key] += 1
        if mode == "stock":
            return self.wrapper.reference(hidden_states)

        width = int(hidden_states.shape[-1])
        rows = int(hidden_states.numel() // width)
        flat = hidden_states.reshape(rows, width)
        if not flat.is_contiguous():
            flat = flat.contiguous()
            self.noncontiguous_inputs += 1
        weight = self.wrapper.reference.weight
        kernel = self.registry.get(rows, width, flat.dtype, self.epsilon)
        if mode == "callee":
            return kernel(flat, weight).view_as(hidden_states)
        if mode != "bound":
            raise ValueError(f"unknown RMSNorm mode: {mode}")

        key = (rows, width, flat.dtype, self.epsilon)
        prepared = self.bound_calls.get(key)
        if prepared is None:
            output = self.torch.empty_like(flat)
            started = time.perf_counter()
            bound = kernel.bind_outputs(output)
            bind_seconds = time.perf_counter() - started
            self.registry.record_caller_kernel(kernel)
            prepared = (output, bound)
            self.bound_calls[key] = prepared
            bound._tilelang_first_bind_seconds = bind_seconds
        output, bound = prepared
        result = bound(flat, weight)
        if result is not output:
            raise RuntimeError("bound RMSNorm did not return its caller-owned output")
        return output.view_as(hidden_states)


def replace_rmsnorm_modules(torch: Any, model: Any, registry: RMSNormKernelRegistry) -> list[TileLangRMSNorm]:
    wrappers: list[TileLangRMSNorm] = []

    def visit(parent: Any, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            module_name = f"{prefix}.{name}" if prefix else name
            if child.__class__.__name__.lower().endswith("rmsnorm") and hasattr(child, "weight"):
                state = TileLangRMSNorm(torch, child, registry, module_name)
                setattr(parent, name, state.wrapper)
                wrappers.append(state)
            else:
                visit(child, module_name)

    visit(model, "")
    if not wrappers:
        raise RuntimeError("model contains no replaceable RMSNorm modules")
    return wrappers


def set_mode(wrappers: list[TileLangRMSNorm], mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    for wrapper in wrappers:
        wrapper.mode = mode


def render_prompt(tokenizer: Any) -> tuple[str, str]:
    try:
        prompt = tokenizer.apply_chat_template(
            list(MESSAGES),
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, "tokenizer_chat_template"
    except (AttributeError, ValueError, TypeError):
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in MESSAGES)
        return prompt + "\nassistant:", "explicit_role_fallback"


def generation_call(
    torch: Any,
    model: Any,
    tokenizer: Any,
    model_inputs: dict[str, Any],
) -> dict[str, Any]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            do_sample=False,
            use_cache=True,
            min_new_tokens=NEW_TOKENS,
            max_new_tokens=NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    prompt_tokens = int(model_inputs["input_ids"].shape[1])
    generated = output[0, prompt_tokens:].detach().cpu().tolist()
    if len(generated) != NEW_TOKENS:
        raise RuntimeError(f"expected {NEW_TOKENS} generated tokens, got {len(generated)}")
    return {
        "seconds": seconds,
        "tokens_per_second": len(generated) / seconds,
        "generated_tokens": len(generated),
        "generated_token_ids": generated,
        "generated_token_sha256": sha256_json(generated),
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }


def summarize_mode(samples: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = [float(sample["seconds"]) for sample in samples]
    rates = [float(sample["tokens_per_second"]) for sample in samples]
    return {
        "samples": len(samples),
        "seconds_p50": statistics.median(seconds),
        "seconds_p90": percentile(seconds, 0.90),
        "tokens_per_second_p50": statistics.median(rates),
        "tokens_per_second_p10": percentile(rates, 0.10),
        "tokens_per_second_p90": percentile(rates, 0.90),
    }


def wrapper_stats(wrappers: list[TileLangRMSNorm]) -> dict[str, Any]:
    bound_buffer_bytes = 0
    bind_seconds: list[float] = []
    for wrapper in wrappers:
        for output, bound in wrapper.bound_calls.values():
            bound_buffer_bytes += int(output.numel() * output.element_size())
            bind_seconds.append(float(getattr(bound, "_tilelang_first_bind_seconds", 0.0)))
    return {
        "modules": len(wrappers),
        "calls": {mode: sum(wrapper.call_counts[mode] for wrapper in wrappers) for mode in MODES},
        "shape_calls": {mode: dict(sorted(sum((wrapper.shape_counts[mode] for wrapper in wrappers), Counter()).items())) for mode in MODES},
        "bound_buffers": sum(len(wrapper.bound_calls) for wrapper in wrappers),
        "bound_buffer_bytes": bound_buffer_bytes,
        "first_bind_seconds_total": sum(bind_seconds),
        "noncontiguous_inputs": sum(wrapper.noncontiguous_inputs for wrapper in wrappers),
        "module_names": [wrapper.module_name for wrapper in wrappers],
    }


def benchmark_model(
    torch: Any,
    tilelang: Any,
    language: Any,
    transformers: Any,
    huggingface_hub: Any,
    model_id: str,
) -> dict[str, Any]:
    resolved = huggingface_hub.model_info(model_id)
    if not resolved.sha or len(resolved.sha) != 40:
        raise RuntimeError(f"Hugging Face returned no exact revision for {model_id}")
    revision = resolved.sha
    load_started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    registry = RMSNormKernelRegistry(torch, tilelang, language)
    wrappers = replace_rmsnorm_modules(torch, model, registry)

    prompt, prompt_source = render_prompt(tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    model_inputs = {name: value.to("cuda") for name, value in encoded.items()}
    prompt_tokens = int(model_inputs["input_ids"].shape[1])

    warmups: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        set_mode(wrappers, mode)
        warmups[mode] = generation_call(torch, model, tokenizer, model_inputs)
    if warmups["callee"]["generated_token_ids"] != warmups["bound"]["generated_token_ids"]:
        raise RuntimeError(f"callee and bound warmup generations diverged for {model_id}")

    raw: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for cycle in range(CYCLES):
        for order_index, mode in enumerate(ORDER):
            set_mode(wrappers, mode)
            sample = generation_call(torch, model, tokenizer, model_inputs)
            sample["cycle"] = cycle
            sample["order_index"] = order_index
            raw[mode].append(sample)

    summaries = {mode: summarize_mode(samples) for mode, samples in raw.items()}
    token_sequences = {mode: sorted({sample["generated_token_sha256"] for sample in samples}) for mode, samples in raw.items()}
    if len(token_sequences["callee"]) != 1 or token_sequences["callee"] != token_sequences["bound"]:
        raise RuntimeError(f"callee and bound timed generations diverged for {model_id}")

    speedup = {
        "bound_vs_callee_seconds_p50": summaries["callee"]["seconds_p50"] / summaries["bound"]["seconds_p50"],
        "bound_vs_callee_tokens_per_second_p50": (
            summaries["bound"]["tokens_per_second_p50"] / summaries["callee"]["tokens_per_second_p50"]
        ),
        "bound_vs_stock_seconds_p50": summaries["stock"]["seconds_p50"] / summaries["bound"]["seconds_p50"],
    }
    result = {
        "model_id": model_id,
        "revision": revision,
        "model_class": model.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
        "dtype": str(next(model.parameters()).dtype),
        "prompt_source": prompt_source,
        "messages": list(MESSAGES),
        "rendered_prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "new_tokens": NEW_TOKENS,
        "warmups": warmups,
        "order": list(ORDER),
        "cycles": CYCLES,
        "summaries": summaries,
        "speedup": speedup,
        "token_sequence_hashes": token_sequences,
        "callee_bound_tokens_identical": token_sequences["callee"] == token_sequences["bound"],
        "stock_bound_tokens_identical": token_sequences["stock"] == token_sequences["bound"],
        "representative_response": raw["bound"][0]["text"],
        "rmsnorm": {
            "wrapper": wrapper_stats(wrappers),
            "kernel_compiles": registry.compile_records,
        },
        "raw_samples": raw,
    }
    print(
        "HF_BOUND_OUTPUT "
        f"model={model_id} prompt_tokens={prompt_tokens} "
        f"callee={summaries['callee']['tokens_per_second_p50']:.3f}tok/s "
        f"bound={summaries['bound']['tokens_per_second_p50']:.3f}tok/s "
        f"speedup={speedup['bound_vs_callee_seconds_p50']:.4f}x",
        flush=True,
    )

    del wrappers, registry, model_inputs, encoded, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def package_versions(names: list[str]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    overall_started = time.perf_counter()
    source_sha = os.environ.get(SOURCE_SHA_ENV, "")
    if len(source_sha) != 40:
        raise RuntimeError("missing exact overlay source SHA")
    if not 1 <= NEW_TOKENS <= 128:
        raise ValueError(f"{NEW_TOKENS_ENV} must be between 1 and 128")
    if not 1 <= CYCLES <= 20:
        raise ValueError(f"{CYCLES_ENV} must be between 1 and 20")
    requested_models = tuple(value for value in os.environ.get(MODEL_FILTER_ENV, "").split(",") if value)
    selected_model_ids = requested_models or MODEL_IDS
    unknown_models = sorted(set(selected_model_ids) - set(MODEL_IDS))
    if unknown_models:
        raise ValueError(f"unknown model IDs: {unknown_models}")
    os.environ["TILELANG_CACHE_DIR"] = f"/tmp/tilelang-hf-bound-cache-{source_sha[:12]}"
    os.environ["HF_HOME"] = "/tmp/tilelang-hf-cache"
    os.environ["TRANSFORMERS_NO_LIBROSA"] = "1"
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    setup, helper_record = load_exact_setup_helper(source_sha)
    prepared = setup.prepare_overlay()
    base_wheel = setup.install_base_wheel(prepared["base_wheel"])
    installer_log = setup.apply_overlay(prepared)
    dependency_started = time.perf_counter()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"transformers=={TRANSFORMERS_VERSION}",
            f"accelerate=={ACCELERATE_VERSION}",
        ]
    )
    dependency_seconds = time.perf_counter() - dependency_started

    import huggingface_hub
    import torch
    import tilelang
    import tilelang.language as T
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    torch.backends.cuda.matmul.allow_tf32 = False

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "tilelang",
                "torch",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "tokenizers",
                "safetensors",
            ]
        ),
        "dependency_install_seconds": dependency_seconds,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "cache_dir": os.environ["TILELANG_CACHE_DIR"],
        "hf_home": os.environ["HF_HOME"],
        "native_libraries": setup.native_libraries(),
        "nvidia_smi_start": setup.nvidia_snapshot(),
    }
    results = [benchmark_model(torch, tilelang, T, transformers, huggingface_hub, model_id) for model_id in selected_model_ids]
    environment["nvidia_smi_end"] = setup.nvidia_snapshot()
    aggregate = {
        "models": len(results),
        "bound_vs_callee_seconds_p50_gmean": geometric_mean([result["speedup"]["bound_vs_callee_seconds_p50"] for result in results]),
        "bound_vs_stock_seconds_p50_gmean": geometric_mean([result["speedup"]["bound_vs_stock_seconds_p50"] for result in results]),
        "all_callee_bound_tokens_identical": all(result["callee_bound_tokens_identical"] for result in results),
    }
    payload = {
        "schema": "tilelang-hf-bound-output-t4-v1",
        "status": "success",
        "created_unix": time.time(),
        "repository": REPOSITORY,
        "candidate_source_sha": source_sha,
        "native_base_sha": setup.BASE_SOURCE_SHA,
        "setup_helper": helper_record,
        "artifact": {
            "id": prepared["artifact_id"],
            "digest": prepared["artifact_digest"],
            "download": prepared["download"],
            "checksums": prepared["checksums"],
            "manifest": prepared["manifest"],
        },
        "base_wheel": base_wheel,
        "installer_log": installer_log,
        "environment": environment,
        "method": {
            "comparison": "stock Hugging Face RMSNorm vs TileLang callee allocation vs TileLang bound-output direct dispatch",
            "model_ids": list(selected_model_ids),
            "model_revision_resolution": "Hugging Face model_info main revision resolved once and reused by exact 40-character commit SHA",
            "dialogue": list(MESSAGES),
            "generation": {
                "do_sample": False,
                "use_cache": True,
                "new_tokens": NEW_TOKENS,
                "attention_implementation": "sdpa",
            },
            "warmups_per_mode_per_model": 1,
            "order": list(ORDER),
            "cycles": CYCLES,
            "speedup_definition": "median(callee generation seconds) / median(bound generation seconds)",
            "aggregation": "geometric mean across exact model revisions",
        },
        "aggregate": aggregate,
        "results": results,
        "total_seconds": time.perf_counter() - overall_started,
        "evidence_boundary": (
            "One free Colab T4 run of two exact public model revisions with real multi-turn chat prompts. "
            "The candidate is explicitly integrated at every model RMSNorm and callee/bound generations "
            "must emit identical token IDs. This run measures the end-to-end impact of caller-owned RMSNorm "
            "outputs within Hugging Face generation. It does not establish fixed-baseline, multi-GPU, "
            "all-operator, or 1.50x TileLang-wide performance."
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(serialized, compresslevel=9)
    RESULT_PATH.write_bytes(serialized)
    RESULT_PATH.with_suffix(".json.gz").write_bytes(compressed)
    print(f"RESULT_JSON_SHA256={hashlib.sha256(serialized).hexdigest()}")
    print(f"RESULT_JSON_BYTES={len(serialized)}")
    print(f"RESULT_GZIP_BASE64={base64.b64encode(compressed).decode()}")
    print(RESULT_MARKER + json.dumps({"aggregate": aggregate, "status": "success"}, separators=(",", ":")))


if __name__ == "__main__":
    main()
