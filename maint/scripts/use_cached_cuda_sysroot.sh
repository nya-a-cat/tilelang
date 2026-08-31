#!/usr/bin/env bash

# Materialize an exact, relocatable CUDA toolchain once and reuse it without
# reinstalling cached RPMs in every manylinux container.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CACHE_ROOT REPORTED_CACHE_HIT" >&2
  exit 2
fi

cache_root="$1"
reported_cache_hit="$2"
case "${cache_root}" in
  /*/.cache/ci-toolchain) ;;
  *)
    echo "Refusing unexpected toolchain cache root: ${cache_root}" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
installer="${script_dir}/install_cached_cuda_toolchain.sh"
bin_dir="${cache_root}/bin"
cuda_root="${cache_root}/cuda-13.0"
marker="${cache_root}/relocatable-toolchain-v1.identity"
expected_identity="cuda=13.0.88;ccache=3.7.7;ninja=1.13.2"

validate_sysroot() {
  [[ -f "${marker}" ]] || return 1
  [[ "$(<"${marker}")" == "${expected_identity}" ]] || return 1
  [[ -x "${cuda_root}/bin/nvcc" ]] || return 1
  [[ -f "${cuda_root}/include/cuda.h" ]] || return 1
  [[ -f "${cuda_root}/lib64/stubs/libcuda.so" ]] || return 1
  [[ -x "${bin_dir}/ccache" ]] || return 1
  [[ -x "${bin_dir}/ninja" ]] || return 1
  "${cuda_root}/bin/nvcc" --version | grep -Fq 'V13.0.88'
  [[ "$("${bin_dir}/ccache" --version | head -n 1)" == "ccache version 3.7.7" ]]
  [[ "$("${bin_dir}/ninja" --version)" == "1.13.2" ]]
  ! ldd "${bin_dir}/ccache" | grep -Fq 'not found'
}

mkdir -p "${cache_root}" "${bin_dir}"
if validate_sysroot; then
  install_mode="relocatable_cache"
else
  bootstrap_cache_hit="${reported_cache_hit}"
  if find "${cache_root}/dnf" -type f -name '*.rpm' -print -quit 2>/dev/null | grep -q .; then
    bootstrap_cache_hit="true"
  fi

  "${installer}" "${cache_root}" "${bootstrap_cache_hit}"

  installed_cuda_root="$(readlink -f /usr/local/cuda-13.0)"
  [[ -d "${installed_cuda_root}" ]]
  staging="${cache_root}/.cuda-13.0.staging.$$"
  rm -rf -- "${staging}"
  mkdir -p "${staging}"
  trap 'rm -rf -- "${staging}"' EXIT
  cp -a "${installed_cuda_root}/." "${staging}/"
  install -m 0755 "$(command -v ccache)" "${bin_dir}/ccache"

  rm -rf -- "${cuda_root}"
  mv -- "${staging}" "${cuda_root}"
  trap - EXIT
  printf '%s\n' "${expected_identity}" > "${marker}.tmp"
  mv -- "${marker}.tmp" "${marker}"

  # The relocatable tree supersedes cached RPM payloads and avoids storing both
  # representations of the same toolchain.
  rm -rf -- "${cache_root}/dnf"
  validate_sysroot
  install_mode="relocatable_fill"
fi

export PATH="${bin_dir}:${cuda_root}/bin:${PATH}"
export CUDAToolkit_ROOT="${cuda_root}"
export CUDA_HOME="${cuda_root}"
export LD_LIBRARY_PATH="${cuda_root}/lib64:${cuda_root}/lib64/stubs:${LD_LIBRARY_PATH:-}"

{
  echo "install_mode=${install_mode}"
  echo "identity=${expected_identity}"
  echo "cuda_root=${cuda_root}"
  echo "cuda_root_bytes=$(du -sb "${cuda_root}" | awk '{ print $1 }')"
  echo "nvcc_sha256=$(sha256sum "${cuda_root}/bin/nvcc" | awk '{ print $1 }')"
  echo "ccache_sha256=$(sha256sum "${bin_dir}/ccache" | awk '{ print $1 }')"
  echo "ninja_sha256=$(sha256sum "${bin_dir}/ninja" | awk '{ print $1 }')"
  nvcc --version
  ccache --version | head -n 1
  ninja --version
} | tee "${cache_root}/toolchain-provenance.txt"
