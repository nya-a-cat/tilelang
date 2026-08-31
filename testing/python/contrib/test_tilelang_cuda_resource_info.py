"""Tests for structured CUDA ptxas resource reporting."""

from tilelang.contrib.cuda_resource_info import filter_and_record, pop_recorded, reset_recorder
from tilelang.contrib.kernel_resource_info import dump_to_file, load_from_file


def test_cuda_resource_parser_records_multiple_kernels_and_preserves_diagnostics():
    output = """ptxas info    : 64 bytes gmem
ptxas info    : Compiling entry function 'kernel_a' for 'sm_80'
ptxas info    : Function properties for kernel_a
    16 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 32 registers, used 4 barriers, 2048 bytes smem, 352 bytes cmem[0], 24 bytes lmem
ptxas warning : Registers are spilled to local memory in function 'kernel_a'
ptxas info    : Compiling entry function \"kernel_b\" for \"sm_80\"
ptxas info    : Function properties for kernel_b
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, 8 bytes cmem[0], 16 bytes cmem[2]
nvcc warning : unrelated diagnostic
"""

    reset_recorder()
    filtered = filter_and_record(output)
    usage = pop_recorded()

    assert set(usage) == {"kernel_a", "kernel_b"}
    assert usage["kernel_a"].n_regs == 32
    assert usage["kernel_a"].n_spills == 5
    assert usage["kernel_a"].scratch_bytes == 16
    assert usage["kernel_a"].stack_frame_bytes == 16
    assert usage["kernel_a"].spill_store_bytes == 8
    assert usage["kernel_a"].spill_load_bytes == 12
    assert usage["kernel_a"].shared_bytes == 2048
    assert usage["kernel_a"].constant_bytes == 352
    assert usage["kernel_a"].local_bytes == 24
    assert usage["kernel_a"].barrier_count == 4
    assert usage["kernel_a"].arch == "sm_80"
    assert usage["kernel_b"].n_regs == 20
    assert usage["kernel_b"].constant_bytes == 24
    assert "ptxas warning" in filtered
    assert "nvcc warning" in filtered
    assert "Compiling entry function" not in filtered
    assert "bytes stack frame" not in filtered


def test_cuda_resource_parser_qualifies_duplicate_entries_from_fatbins():
    output = """ptxas info : Compiling entry function 'kernel' for 'sm_100a'
ptxas info : Function properties for kernel
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 40 registers, 1024 bytes smem
ptxas info : Compiling entry function 'kernel' for 'sm_103a'
ptxas info : Function properties for kernel
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 44 registers, 2048 bytes smem
"""

    reset_recorder()
    assert filter_and_record(output) == ""
    usage = pop_recorded()

    assert set(usage) == {"kernel@sm_100a", "kernel@sm_103a"}
    assert usage["kernel@sm_100a"].n_regs == 40
    assert usage["kernel@sm_103a"].shared_bytes == 2048


def test_cuda_resource_parser_leaves_unrelated_output_unchanged():
    output = "nvcc warning : a diagnostic without resource statistics\n"

    reset_recorder()

    assert filter_and_record(output) == output
    assert pop_recorded() == {}


def test_resource_usage_json_round_trip_preserves_cuda_fields(tmp_path):
    output = """ptxas info : Compiling entry function 'kernel' for 'sm_90a'
ptxas info : Function properties for kernel
8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads
ptxas info : Used 64 registers, 8192 bytes smem, 512 bytes cmem[0]
"""
    resource_path = tmp_path / "resource_usage.json"

    reset_recorder()
    filter_and_record(output)
    dump_to_file(pop_recorded(), str(resource_path))
    restored = load_from_file(str(resource_path))

    assert restored["kernel"].arch == "sm_90a"
    assert restored["kernel"].n_regs == 64
    assert restored["kernel"].n_spills == 2
    assert restored["kernel"].shared_bytes == 8192
    assert restored["kernel"].constant_bytes == 512
