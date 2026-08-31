#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CACHE_ROOT CACHE_HIT" >&2
  exit 2
fi

cache_root="$1"
cache_hit="$2"
bin_dir="${cache_root}/bin"
download_dir="${cache_root}/downloads"
dnf_cache_dir="${cache_root}/dnf"
python_bin="/opt/python/cp310-cp310/bin/python"

mkdir -p "${bin_dir}" "${download_dir}" "${dnf_cache_dir}"

download_verified() {
  local url="$1"
  local sha256="$2"
  local destination="$3"

  if [[ -f "${destination}" ]] && ! echo "${sha256}  ${destination}" | sha256sum --check --status; then
    rm -f -- "${destination}"
  fi

  if [[ ! -f "${destination}" ]]; then
    local partial="${destination}.part"
    rm -f -- "${partial}"
    curl --fail --location --retry 5 --output "${partial}" "${url}"
    echo "${sha256}  ${partial}" | sha256sum --check --status
    mv -- "${partial}" "${destination}"
  fi

  echo "${sha256}  ${destination}" | sha256sum --check --status
}

ninja_version="1.13.2"
ninja_archive="${download_dir}/ninja-linux-${ninja_version}.zip"
ninja_url="https://github.com/ninja-build/ninja/releases/download/v${ninja_version}/ninja-linux.zip"
ninja_sha256="5749cbc4e668273514150a80e387a957f933c6ed3f5f11e03fb30955e2bbead6"

download_verified "${ninja_url}" "${ninja_sha256}" "${ninja_archive}"

if [[ ! -x "${bin_dir}/ninja" ]] || [[ "$("${bin_dir}/ninja" --version)" != "${ninja_version}" ]]; then
  extract_dir="$(mktemp -d "${cache_root}/ninja-extract.XXXXXX")"
  trap 'rm -rf -- "${extract_dir}"' EXIT
  "${python_bin}" -m zipfile -e "${ninja_archive}" "${extract_dir}"
  install -m 0755 "${extract_dir}/ninja" "${bin_dir}/ninja"
  rm -rf -- "${extract_dir}"
  trap - EXIT
fi

export PATH="${bin_dir}:${PATH}"
ninja --version

dnf config-manager --add-repo https://developer.download.nvidia.cn/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

dnf_args=(
  --setopt="cachedir=${dnf_cache_dir}"
  --setopt=keepcache=True
  --setopt=metadata_expire=-1
)
toolchain_packages=(
  ccache-3.7.7-1.el8
  cuda-minimal-build-13-0-13.0.3-1
  cuda-driver-devel-13-0-13.0.96-1
  cuda-nvrtc-devel-13-0-13.0.88-1
)

if [[ "${cache_hit}" == "true" ]]; then
  dnf "${dnf_args[@]}" --cacheonly install -y "${toolchain_packages[@]}"
  install_mode="cache_only"
else
  dnf "${dnf_args[@]}" install -y "${toolchain_packages[@]}"
  install_mode="network_fill"
fi

export PATH="/usr/local/cuda/bin:${PATH}"
ccache --version | head -n 1

{
  echo "install_mode=${install_mode}"
  echo "ccache_version=$(ccache --version | head -n 1)"
  echo "ninja_version=$(ninja --version)"
  nvcc --version
  rpm -qa | grep -E '^(cuda-|libnv|gcc-c\+\+)' | sort
} | tee "${cache_root}/toolchain-provenance.txt"
