"""Disassemble a CUBIN and summarize resource and SASS instruction metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from analyze_cuda_codegen import SASS_INSTRUCTION_RE, _find_disassembler, _opcode_groups, _run


REGISTER_PATTERNS = (
    re.compile(r"SHI_REGISTERS=(\d+)"),
    re.compile(r'\.sectioninfo\s+@"SHI_REGISTERS=(\d+)"'),
)
RESOURCE_FUNCTION_RE = re.compile(
    r"Function (?P<name>[^:\n]+):\n\s+REG:(?P<registers>\d+) "
    r"STACK:(?P<stack>\d+) SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)"
)


def _find_cuobjdump(cuda_root: Path) -> Path:
    candidates: list[Path] = []
    if path := shutil.which("cuobjdump"):
        candidates.append(Path(path))
    candidates.append(cuda_root / "bin" / "cuobjdump")
    try:
        import triton

        candidates.append(Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "bin" / "cuobjdump")
    except ImportError:
        pass
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("cuobjdump was not found in PATH, CUDA_HOME, or the installed Triton package")


def analyze(cubin: Path, output_dir: Path, label: str, cuda_root: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    disassembler = _find_disassembler(cuda_root)
    sass = _run([str(disassembler), str(cubin)]).stdout
    resource_usage = _run([str(_find_cuobjdump(cuda_root)), "--dump-resource-usage", str(cubin)]).stdout
    sass_path = output_dir / f"{label}.sass"
    sass_path.write_text(sass)
    opcodes = Counter(match.group("opcode").upper() for match in SASS_INSTRUCTION_RE.finditer(sass))
    registers = None
    for pattern in REGISTER_PATTERNS:
        if match := pattern.search(sass):
            registers = int(match.group(1))
            break
    functions = {
        match.group("name"): {
            "registers": int(match.group("registers")),
            "stack_bytes": int(match.group("stack")),
            "shared_bytes": int(match.group("shared")),
            "local_bytes": int(match.group("local")),
        }
        for match in RESOURCE_FUNCTION_RE.finditer(resource_usage)
    }
    if functions:
        registers = max(int(resources["registers"]) for resources in functions.values())
    summary: dict[str, object] = {
        "label": label,
        "cubin": str(cubin.resolve()),
        "cubin_bytes": cubin.stat().st_size,
        "registers": registers,
        "resources": {"functions": functions},
        "sass": {
            "instruction_count": sum(opcodes.values()),
            "groups": _opcode_groups(opcodes),
            "opcodes": dict(opcodes.most_common()),
        },
    }
    (output_dir / f"{label}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubin", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cubin, args.output_dir, args.label, args.cuda_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
