"""Shared helpers for preparing TVM-FFI callable ABIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tirx import PrimFunc


TILELANG_OUT_IDX_ATTR = "tilelang_out_idx"


def _normalize_output_indices(output_indices: list[int], num_params: int) -> list[int]:
    normalized = []
    for raw_index in output_indices:
        index = int(raw_index)
        if index < 0:
            index += num_params
        if index < 0 or index >= num_params:
            raise ValueError(f"out_idx index {raw_index} is out of range for a function with {num_params} parameters")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"out_idx contains duplicate tensor indices: {output_indices}")
    return normalized


def prepare_tvm_ffi_callee_allocated_outputs(
    func: PrimFunc,
    out_idx: list[int] | int | None,
) -> tuple[PrimFunc, list[int] | None]:
    """Resolve output indices and expose them to TVM-FFI lowering."""
    requested_indices = None if out_idx is None else ([out_idx] if isinstance(out_idx, int) else list(out_idx))
    attr_indices = None
    if func.attrs is not None and TILELANG_OUT_IDX_ATTR in func.attrs:
        attr_indices = [int(index) for index in func.attrs[TILELANG_OUT_IDX_ATTR]]

    if attr_indices is not None:
        if requested_indices is not None:
            num_params = len(func.params)
            if _normalize_output_indices(requested_indices, num_params) != _normalize_output_indices(attr_indices, num_params):
                raise ValueError("out_idx does not match the PrimFunc's tilelang_out_idx attribute")
        return func, attr_indices

    output_indices = requested_indices or []
    if not output_indices:
        return func, None
    _normalize_output_indices(output_indices, len(func.params))
    return func.with_attr(TILELANG_OUT_IDX_ATTR, output_indices), output_indices


def prepare_tvm_ffi_caller_allocated_outputs(func: PrimFunc) -> PrimFunc:
    """Return a derived PrimFunc whose output buffers stay in the packed ABI.

    Eager ``T.empty`` kernels carry ``tilelang_out_idx`` so TVM-FFI can omit
    those buffers and allocate result slots inside the packed entry.  A caller
    that owns reusable output buffers needs the original full parameter list.
    Removing the derived ABI attribute restores that path without mutating the
    frontend PrimFunc.
    """
    if func.attrs is None or TILELANG_OUT_IDX_ATTR not in func.attrs:
        return func
    return func.without_attr(TILELANG_OUT_IDX_ATTR)
