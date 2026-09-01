"""Collect machine-readable LayoutInference cost features without a GPU.

The 18 PrimFuncs are byte-identity guarded against the runtime A/B matrix used
on T4 and consumer Blackwell.  Each unchanged PrimFunc is inferred under the
selected layout policies for explicit CUDA architecture targets.  This script
performs no device compilation or execution.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any

import tilelang as tl
import tilelang.language as T
from tilelang import tvm


POLICIES = tuple(
    policy.strip()
    for policy in os.environ.get(
        "TILELANG_LAYOUT_TRACE_POLICIES",
        "register-count,io-aware,io-aware-regularized",
    ).split(",")
    if policy.strip()
)
if not POLICIES or len(set(POLICIES)) != len(POLICIES):
    raise ValueError("TILELANG_LAYOUT_TRACE_POLICIES must name one or more distinct policies")
TARGET_ARCHES = tuple(
    arch.strip()
    for arch in os.environ.get(
        "TILELANG_LAYOUT_TRACE_ARCHES",
        "sm_75,sm_80,sm_90a,sm_100a,sm_120a",
    ).split(",")
    if arch.strip()
)
RESULT_PATH = Path(
    os.environ.get(
        "TILELANG_LAYOUT_TRACE_RESULT",
        "tilelang-layout-cost-trace.json",
    )
)
LOG_PATH = Path(
    os.environ.get(
        "TILELANG_LAYOUT_TRACE_LOG",
        "tilelang-layout-cost-trace.log",
    )
)
SOURCE_SHA = os.environ.get("TILELANG_SOURCE_SHA")
TRACE_MARKER = "TILELANG_LAYOUT_COST_TRACE"
CASE_BEGIN = "TILELANG_TRACE_CASE_BEGIN"
CASE_END = "TILELANG_TRACE_CASE_END"
CASE_ERROR = "TILELANG_TRACE_CASE_ERROR"
INTEGER_TRACE_FIELDS = {
    "component",
    "attempt_root",
    "rank",
    "mem",
    "regs",
    "spill",
    "global_mem",
    "global_bw",
    "global_issue",
    "measured",
    "worst_case",
    "unavailable",
    "register_price",
}
EXPECTED_TRACE_FIELDS = {
    "schema",
    "phase",
    "model",
    *INTEGER_TRACE_FIELDS,
}


@dataclass(frozen=True)
class TraceCase:
    name: str
    family: str
    program: Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_symbol(program: Any, symbol: str) -> Any:
    return program.with_attr("global_symbol", symbol)


def make_row_broadcast(rows: int, cols: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(
        S: T.Tensor((rows,), T.float32),
        Out: T.Tensor((rows, cols), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            scalar = T.alloc_fragment((rows,), T.float32)
            T.copy(S, scalar)
            for i, j in T.Parallel(rows, cols):
                Out[i, j] = scalar[i] * 2.0

    return with_symbol(main, symbol)


def make_column_broadcast(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(A: T.Tensor((cols,), dtype), B: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((cols,), dtype)
            T.copy(A, values)
            for i, j in T.Parallel(rows, cols):
                B[i, j] = values[j] * 2.0

    return with_symbol(main, symbol)


def make_mixed_chain(rows: int, cols: int, threads: int, symbol: str) -> Any:
    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), T.float16),
        B: T.Tensor((rows, cols), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            x16 = T.alloc_fragment((rows, cols), T.float16)
            x32 = T.alloc_fragment((rows, cols), T.float32)
            T.copy(A, x16)
            for i, j in T.Parallel(rows, cols):
                x32[i, j] = x16[i, j].astype(T.float32) * 2.0
            T.copy(x32, B)

    return with_symbol(main, symbol)


def make_affine(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(
        A: T.Tensor((rows, cols), dtype),
        Scale: T.Tensor((cols,), dtype),
        Bias: T.Tensor((cols,), dtype),
        B: T.Tensor((rows, cols), dtype),
    ):
        with T.Kernel(1, threads=threads):
            values = T.alloc_fragment((rows, cols), dtype)
            scale = T.alloc_fragment((cols,), dtype)
            bias = T.alloc_fragment((cols,), dtype)
            T.copy(A, values)
            T.copy(Scale, scale)
            T.copy(Bias, bias)
            for i, j in T.Parallel(rows, cols):
                values[i, j] = values[i, j] * scale[j] + bias[j]
            T.copy(values, B)

    return with_symbol(main, symbol)


def make_transpose(
    rows: int,
    cols: int,
    dtype: str,
    threads: int,
    symbol: str,
) -> Any:
    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((cols, rows), dtype)):
        with T.Kernel(1, threads=threads):
            fragment = T.alloc_fragment((rows, cols), dtype)
            T.copy(A, fragment)
            for i, j in T.Parallel(rows, cols):
                B[j, i] = fragment[i, j]

    return with_symbol(main, symbol)


EXPECTED_PRIMFUNC_SHA256 = {
    "row_broadcast_f32_2x2560_t256": "2ac431c9ab2aa58cd2afb497fc5325137bef45a68886eedee6cba589f16a6620",
    "row_broadcast_f32_4x1024_t128": "32677ca1cb2fabda2a011f93807890e319791277e058698281ef8e28813c335e",
    "row_broadcast_f32_8x512_t128": "ee64a30ff886cd6a4947a966e7cf6b83183c6f8ea1df6b4c35061e3e6c114ebd",
    "row_broadcast_f32_16x256_t128": "9bdd34de25af445a48aa2c7b768eb2e2bb3aebc42dba42f4fe27734b8bd8540b",
    "row_broadcast_f32_4x4096_t256": "b16f2ee5c8c232e815905ebc571a971bcf904faa40c665b1be3f2b9ce0b599a4",
    "column_broadcast_float16_64x256_t128": "7c849398e1762d5f2aa436ef956ebdf518854b3d9b755f021604e23f568ae013",
    "column_broadcast_float32_128x128_t128": "9261929bc383de758616832a6441cd52035e6f74418a1a1875af6df6381e5ea4",
    "mixed_f16_f32_64x64_t128": "f7e46a617a848de6d823d1a5fde4b6edec7adfc406d95f71d7aa601fae225d10",
    "mixed_f16_f32_128x128_t128": "b4c7dc15c736413b5a8b3caab8b901d9732676b28031649574083fa298ebe7a9",
    "mixed_f16_f32_64x256_t128": "0b446d61b33708d8ba4b8974c39bf34e96c854c3089683644da4342c77906d65",
    "mixed_f16_f32_128x256_t256": "143992b0d45c1c26f9200b49f9c42021e9b734ee1d0345d2007ec90dbb595613",
    "affine_float16_32x128_t128": "d183f4607e5c799e99d2612b08f7d2f6cd5aaf8d08cd6b3c055bb440dcc25310",
    "affine_float16_64x128_t128": "345a5770ea7626507723d8f5c15facbe87dbf7befc057704ce761181df56fa28",
    "affine_float32_32x128_t128": "341fa28adc147f19bb37d2eb4de823cb158bc141787aaa83eec1bc0389968c64",
    "affine_float32_64x256_t256": "e7f2a204dc8351086b9bc44b1bb7b3fe5236b9fdd88e8887993cbeb85a57928d",
    "transpose_float16_128x128_t128": "f8e874c0c4fc0e8dff234ece07a3577afc6547e50983826eff689551f3b928fb",
    "transpose_float16_128x256_t256": "25495804132b7048898fca78036dbffb037c205d1f1116fdec62f3139f014d48",
    "transpose_float32_128x128_t128": "95f7d98442d491ba3e33e5f63b53af46654c881ed05d2810d02e50393a30d086",
}


def build_cases() -> list[TraceCase]:
    cases: list[TraceCase] = []

    for rows, cols, threads in (
        (2, 2560, 256),
        (4, 1024, 128),
        (8, 512, 128),
        (16, 256, 128),
        (4, 4096, 256),
    ):
        name = f"row_broadcast_f32_{rows}x{cols}_t{threads}"
        cases.append(
            TraceCase(
                name,
                "row_broadcast",
                make_row_broadcast(rows, cols, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (64, 256, "float16", 128),
        (128, 128, "float32", 128),
    ):
        name = f"column_broadcast_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            TraceCase(
                name,
                "column_broadcast",
                make_column_broadcast(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    for rows, cols, threads in (
        (64, 64, 128),
        (128, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
    ):
        name = f"mixed_f16_f32_{rows}x{cols}_t{threads}"
        cases.append(
            TraceCase(
                name,
                "mixed_dtype",
                make_mixed_chain(rows, cols, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (32, 128, "float16", 128),
        (64, 128, "float16", 128),
        (32, 128, "float32", 128),
        (64, 256, "float32", 256),
    ):
        name = f"affine_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            TraceCase(
                name,
                "affine",
                make_affine(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    for rows, cols, dtype, threads in (
        (128, 128, "float16", 128),
        (128, 256, "float16", 256),
        (128, 128, "float32", 128),
    ):
        name = f"transpose_{dtype}_{rows}x{cols}_t{threads}"
        cases.append(
            TraceCase(
                name,
                "transpose",
                make_transpose(rows, cols, dtype, threads, f"scan_{name}"),
            )
        )

    actual_names = tuple(case.name for case in cases)
    if len(cases) != 18 or set(actual_names) != set(EXPECTED_PRIMFUNC_SHA256):
        raise RuntimeError(f"runtime calibration case manifest drifted: {actual_names}")
    return cases


def infer_case(case: TraceCase, arch: str, policy: str, trace_enabled: bool) -> None:
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    config: dict[str, Any] = {"tl.layout_cost_model": policy}
    if trace_enabled:
        config["tl.enable_layout_cost_trace"] = True
    with target, tvm.transform.PassContext(config=config):
        module = tvm.IRModule({"main": case.program})
        module = tvm.tirx.transform.BindTarget(target)(module)
        module = tl.transform.MaterializeKernelLaunch()(module)
        tl.transform.LayoutInference()(module)


def child_main(
    *,
    arch: str,
    policy: str,
    trace_enabled: bool,
    case_name: str | None,
) -> int:
    failures = 0
    cases = build_cases()
    if case_name is not None:
        cases = [case for case in cases if case.name == case_name]
        if len(cases) != 1:
            raise ValueError(f"unknown trace case: {case_name}")

    for case in cases:
        actual_hash = sha256_text(str(case.program))
        print(f"{CASE_BEGIN} name={case.name} hash={actual_hash}", flush=True)
        try:
            expected_hash = EXPECTED_PRIMFUNC_SHA256[case.name]
            if actual_hash != expected_hash:
                raise RuntimeError(f"canonical PrimFunc hash drift: expected {expected_hash}, got {actual_hash}")
            infer_case(case, arch, policy, trace_enabled)
        except Exception as error:  # noqa: BLE001 - preserve the remaining matrix
            failures += 1
            print(
                f"{CASE_ERROR} "
                + json.dumps(
                    {
                        "name": case.name,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(f"{CASE_END} name={case.name}", flush=True)
    return 2 if failures else 0


def parse_trace_record(line: str) -> dict[str, Any]:
    payload = line.split(TRACE_MARKER, 1)[1].strip()
    record: dict[str, Any] = {}
    for token in payload.split():
        key, separator, value = token.partition("=")
        if not separator:
            raise ValueError(f"malformed trace token: {token!r}")
        record[key] = int(value) if key in INTEGER_TRACE_FIELDS else value
    missing = EXPECTED_TRACE_FIELDS - set(record)
    if missing:
        raise ValueError(f"trace record is missing fields: {sorted(missing)}")
    if record["schema"] != "v2" or record["phase"] not in {"attempt", "selected"}:
        raise ValueError(f"unsupported trace record: {record}")
    return record


def parse_child_output(
    output: str,
    *,
    arch: str,
    policy: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    current: dict[str, Any] | None = None

    for line in output.splitlines():
        if line.startswith(f"{CASE_BEGIN} "):
            if current is not None:
                errors.append(f"nested case marker before {current['name']} ended")
            match = re.fullmatch(
                rf"{CASE_BEGIN} name=([^ ]+) hash=([0-9a-f]{{64}})",
                line,
            )
            if match is None:
                errors.append(f"malformed begin marker: {line}")
                current = None
                continue
            current = {
                "name": match.group(1),
                "canonical_primfunc_sha256": match.group(2),
                "arch": arch,
                "policy": policy,
                "trace_records": [],
                "errors": [],
            }
            continue

        if TRACE_MARKER in line:
            if current is None:
                errors.append(f"orphan trace record: {line}")
                continue
            try:
                record = parse_trace_record(line)
                if record["model"] != policy:
                    raise ValueError(f"trace model {record['model']!r} differs from child policy {policy!r}")
                current["trace_records"].append(record)
            except Exception as error:  # noqa: BLE001 - retain all parse failures
                current["errors"].append(f"{type(error).__name__}: {error}")
            continue

        if line.startswith(f"{CASE_ERROR} "):
            if current is None:
                errors.append(f"orphan case error: {line}")
                continue
            try:
                current["errors"].append(json.loads(line.split(" ", 1)[1]))
            except json.JSONDecodeError as error:
                current["errors"].append(f"malformed case error: {error}: {line}")
            continue

        if line.startswith(f"{CASE_END} "):
            match = re.fullmatch(rf"{CASE_END} name=([^ ]+)", line)
            if match is None or current is None:
                errors.append(f"malformed or orphan end marker: {line}")
                current = None
                continue
            if match.group(1) != current["name"]:
                current["errors"].append(f"end marker names {match.group(1)!r}, expected {current['name']!r}")
            parsed.append(current)
            current = None

    if current is not None:
        errors.append(f"unterminated case marker: {current['name']}")
    return parsed, errors


def aggregate_case(case: dict[str, Any], family: str) -> dict[str, Any]:
    records = case.pop("trace_records")
    selected = [record for record in records if record["phase"] == "selected"]
    attempts = [record for record in records if record["phase"] == "attempt"]
    numeric_fields = (
        "rank",
        "mem",
        "regs",
        "spill",
        "global_mem",
        "global_bw",
        "global_issue",
        "measured",
        "worst_case",
        "unavailable",
        "register_price",
    )
    selected_cost = {field: sum(record[field] for record in selected) for field in numeric_fields}
    case.update(
        {
            "family": family,
            "status": "complete" if selected and not case["errors"] else "failed",
            "attempt_count": len(attempts),
            "selected_component_count": len(selected),
            "selected_attempt_roots": [record["attempt_root"] for record in selected],
            "selected_cost": selected_cost,
            "trace_records": records,
        }
    )
    return case


def run_child(
    *,
    arch: str,
    policy: str,
    trace_enabled: bool,
    case_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--arch",
        arch,
        "--policy",
        policy,
        "--trace-enabled",
        "true" if trace_enabled else "false",
    ]
    if case_name is not None:
        command.extend(("--case", case_name))
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=600,
    )


def write_json(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def runtime_identity() -> dict[str, Any] | None:
    package = Path(tl.__file__).resolve().parent
    identity_path = package / "_python_overlay_identity.json"
    if not identity_path.is_file():
        return None
    return json.loads(identity_path.read_text(encoding="utf-8"))


def parent_main() -> int:
    started = time.time()
    cases = build_cases()
    case_families = {case.name: case.family for case in cases}
    payload: dict[str, Any] = {
        "schema": "tilelang-layout-cost-feature-trace-v2",
        "repository": "nya-a-cat/tilelang",
        "source_sha": SOURCE_SHA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "started_unix": started,
        "target_arches": list(TARGET_ARCHES),
        "policies": list(POLICIES),
        "case_count": len(cases),
        "expected_entry_count": len(cases) * len(TARGET_ARCHES) * len(POLICIES),
        "expected_primfunc_sha256": EXPECTED_PRIMFUNC_SHA256,
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "tilelang": tl.__version__,
            "tilelang_file": str(Path(tl.__file__).resolve()),
            "runtime_identity": runtime_identity(),
        },
        "evidence_boundary": (
            "CPU-only LayoutInference trace over the same 18 unchanged PrimFuncs used by the T4 and consumer "
            "Blackwell runtime A/B. It records decomposed rank, IO, register, and spill features for every "
            "selected policy on explicit CUDA targets and performs no device compilation, resource measurement, "
            "correctness execution, or runtime timing."
        ),
        "default_off_probe": {},
        "results": [],
        "failures": [],
        "status": "running",
    }
    log_chunks: list[str] = []

    probe = run_child(
        arch=TARGET_ARCHES[0],
        policy=POLICIES[0],
        trace_enabled=False,
        case_name=cases[0].name,
    )
    log_chunks.append(f"===== default-off arch={TARGET_ARCHES[0]} policy={POLICIES[0]} =====\n{probe.stdout}")
    probe_records = probe.stdout.count(TRACE_MARKER)
    payload["default_off_probe"] = {
        "arch": TARGET_ARCHES[0],
        "policy": POLICIES[0],
        "case": cases[0].name,
        "returncode": probe.returncode,
        "trace_record_count": probe_records,
        "status": "complete" if probe.returncode == 0 and probe_records == 0 else "failed",
    }
    if payload["default_off_probe"]["status"] != "complete":
        payload["failures"].append("default-off probe emitted a trace or failed")

    for arch in TARGET_ARCHES:
        for policy in POLICIES:
            completed = run_child(
                arch=arch,
                policy=policy,
                trace_enabled=True,
            )
            log_chunks.append(f"===== traced arch={arch} policy={policy} returncode={completed.returncode} =====\n{completed.stdout}")
            parsed, parse_errors = parse_child_output(
                completed.stdout,
                arch=arch,
                policy=policy,
            )
            if completed.returncode != 0:
                payload["failures"].append(f"child failed: arch={arch} policy={policy} returncode={completed.returncode}")
            payload["failures"].extend(f"arch={arch} policy={policy}: {error}" for error in parse_errors)

            seen_names: set[str] = set()
            for parsed_case in parsed:
                name = parsed_case["name"]
                seen_names.add(name)
                if name not in case_families:
                    parsed_case["errors"].append(f"unexpected case: {name}")
                    family = "unknown"
                else:
                    family = case_families[name]
                expected_hash = EXPECTED_PRIMFUNC_SHA256.get(name)
                if parsed_case["canonical_primfunc_sha256"] != expected_hash:
                    parsed_case["errors"].append("canonical PrimFunc hash differs from the runtime A/B manifest")
                result = aggregate_case(parsed_case, family)
                payload["results"].append(result)
                if result["status"] != "complete":
                    payload["failures"].append(f"incomplete trace: arch={arch} policy={policy} case={name}")

            missing_names = set(case_families) - seen_names
            if missing_names:
                payload["failures"].append(f"missing cases: arch={arch} policy={policy} names={sorted(missing_names)}")
            payload["completed_entry_count"] = len(payload["results"])
            payload["duration_seconds"] = time.time() - started
            write_json(payload)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(log_chunks), encoding="utf-8")
    payload["raw_log"] = {
        "path": str(LOG_PATH),
        "sha256": sha256_file(LOG_PATH),
        "bytes": LOG_PATH.stat().st_size,
    }
    payload["finished_unix"] = time.time()
    payload["duration_seconds"] = payload["finished_unix"] - started
    payload["completed_entry_count"] = len(payload["results"])
    if payload["completed_entry_count"] != payload["expected_entry_count"]:
        payload["failures"].append(f"entry count {payload['completed_entry_count']} != {payload['expected_entry_count']}")
    payload["status"] = "failed" if payload["failures"] else "complete"
    write_json(payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "completed_entry_count": payload["completed_entry_count"],
                "expected_entry_count": payload["expected_entry_count"],
                "duration_seconds": payload["duration_seconds"],
                "failure_count": len(payload["failures"]),
                "result": str(RESULT_PATH),
                "log": str(LOG_PATH),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 2 if payload["failures"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--arch", choices=TARGET_ARCHES)
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--trace-enabled", choices=("true", "false"))
    parser.add_argument("--case")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.child:
        return parent_main()
    if args.arch is None or args.policy is None or args.trace_enabled is None:
        raise SystemExit("child mode requires --arch, --policy, and --trace-enabled")
    return child_main(
        arch=args.arch,
        policy=args.policy,
        trace_enabled=args.trace_enabled == "true",
        case_name=args.case,
    )


if __name__ == "__main__":
    raise SystemExit(main())
