"""Compile generated CUDA source and record reproducible static code metrics.

This tool deliberately does not launch a kernel.  It is intended for paired
backend experiments on machines without a CUDA device: lower the same kernel
with two compiler revisions, then compare their PTX, CUBIN, SASS, register,
spill, shared-memory, and instruction-mix reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SASS_INSTRUCTION_RE = re.compile(
    r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)",
    re.MULTILINE,
)
REGISTER_RE = re.compile(r"Used\s+(\d+)\s+registers")
STACK_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
SMEM_RE = re.compile(r"(\d+) bytes smem")
CMEM_RE = re.compile(r"(\d+) bytes cmem\[\d+\]")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(f"command failed ({result.returncode}): {rendered}\n{result.stdout}{result.stderr}")
    return result


def _find_pip_cuda() -> dict[str, str]:
    explicit = os.environ.get("CUDA_HOME")
    if explicit:
        root = Path(explicit).resolve()
        return {
            "root": str(root),
            "nvcc": str(root / "bin" / "nvcc"),
            "library_dir": str(root / "lib64"),
        }

    helper = REPO_ROOT / "cmake" / "find_pip_cuda.py"
    result = _run([sys.executable, str(helper)])
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected CUDA discovery result: {payload!r}")
    return {str(key): str(value) for key, value in payload.items()}


def _find_disassembler(cuda_root: Path) -> Path:
    candidates: list[Path] = []
    if path := shutil.which("nvdisasm"):
        candidates.append(Path(path))
    candidates.append(cuda_root / "bin" / "nvdisasm")

    try:
        import triton

        candidates.append(Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "bin" / "nvdisasm")
    except ImportError:
        pass

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("nvdisasm was not found in PATH, CUDA_HOME, or the installed Triton package")


def _opcode_groups(opcodes: Counter[str]) -> dict[str, int]:
    def count_prefixes(*prefixes: str) -> int:
        return sum(count for opcode, count in opcodes.items() if opcode.startswith(prefixes))

    return {
        "global_load": count_prefixes("LDG"),
        "global_store": count_prefixes("STG"),
        "shared_load": count_prefixes("LDS", "LDSM"),
        "shared_store": count_prefixes("STS"),
        "async_copy": count_prefixes("LDGSTS", "CPASYNC"),
        "tma": count_prefixes("UTMA", "TMA"),
        "tensor_core": count_prefixes("HMMA", "IMMA", "MMA", "WGMMA", "UTCHMMA", "UTCIMMA"),
        "barrier": count_prefixes("BAR", "MBAR"),
    }


def _parse_ptxas(text: str) -> dict[str, int | None]:
    registers = REGISTER_RE.search(text)
    stack = STACK_RE.search(text)
    smem = SMEM_RE.search(text)
    cmem_values = [int(value) for value in CMEM_RE.findall(text)]
    return {
        "registers": int(registers.group(1)) if registers else None,
        "stack_frame_bytes": int(stack.group(1)) if stack else None,
        "spill_store_bytes": int(stack.group(2)) if stack else None,
        "spill_load_bytes": int(stack.group(3)) if stack else None,
        "static_shared_bytes": int(smem.group(1)) if smem else 0,
        "constant_bytes": sum(cmem_values),
    }


def analyze(source: Path, arch: str, output_dir: Path, label: str) -> dict[str, object]:
    if not re.fullmatch(r"sm_[A-Za-z0-9]+", arch):
        raise ValueError(f"architecture must be an exact SM token, got {arch!r}")
    if not source.is_file():
        raise FileNotFoundError(source)

    cuda = _find_pip_cuda()
    cuda_root = Path(cuda["root"])
    nvcc = Path(cuda["nvcc"])
    nvdisasm = _find_disassembler(cuda_root)
    cxx = os.environ.get("CXX", shutil.which("g++") or "g++")

    output_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = output_dir / f"{label}.{arch}.cubin"
    ptx_path = output_dir / f"{label}.{arch}.ptx"
    sass_path = output_dir / f"{label}.{arch}.sass"
    log_path = output_dir / f"{label}.{arch}.ptxas.txt"
    source_snapshot = output_dir / f"{label}.{arch}.cu"
    summary_path = output_dir / f"{label}.{arch}.json"

    common = [
        str(nvcc),
        f"-ccbin={cxx}",
        "-O3",
        "-lineinfo",
        "-std=c++20",
        f"-arch={arch}",
        f"-I{REPO_ROOT / 'src'}",
        f"-I{REPO_ROOT / '3rdparty' / 'cutlass' / 'include'}",
        f"-I{cuda_root / 'include'}",
    ]

    cubin_result = _run([*common, "--cubin", "--ptxas-options=-v", str(source), "-o", str(cubin_path)])
    ptx_result = _run([*common, "--ptx", str(source), "-o", str(ptx_path)])
    ptxas_text = cubin_result.stdout + cubin_result.stderr
    if ptx_result.stdout or ptx_result.stderr:
        ptxas_text += "\nPTX compilation:\n" + ptx_result.stdout + ptx_result.stderr
    log_path.write_text(ptxas_text)

    sass = _run([str(nvdisasm), str(cubin_path)]).stdout
    sass_path.write_text(sass)
    if source.resolve() != source_snapshot.resolve():
        shutil.copy2(source, source_snapshot)

    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))
    source_text = source.read_text()
    summary: dict[str, object] = {
        "label": label,
        "arch": arch,
        "source": str(source.resolve()),
        "toolchain": {
            "nvcc": str(nvcc),
            "nvcc_version": _run([str(nvcc), "--version"]).stdout.strip().splitlines()[-1],
            "nvdisasm": str(nvdisasm),
        },
        "artifacts": {
            "source": str(source_snapshot),
            "ptx": str(ptx_path),
            "cubin": str(cubin_path),
            "sass": str(sass_path),
            "ptxas": str(log_path),
        },
        "sizes": {
            "source_bytes": len(source_text.encode()),
            "source_lines": len(source_text.splitlines()),
            "ptx_bytes": ptx_path.stat().st_size,
            "cubin_bytes": cubin_path.stat().st_size,
        },
        "resources": _parse_ptxas(ptxas_text),
        "sass": {
            "instruction_count": sum(opcodes.values()),
            "groups": _opcode_groups(opcodes),
            "opcodes": dict(opcodes.most_common()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Generated CUDA source to compile")
    parser.add_argument("--arch", required=True, help="Exact target token, for example sm_100a")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PTX/CUBIN/SASS reports")
    parser.add_argument("--label", help="Artifact filename label; defaults to the source stem")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze(args.source, args.arch, args.output_dir, args.label or args.source.stem)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
