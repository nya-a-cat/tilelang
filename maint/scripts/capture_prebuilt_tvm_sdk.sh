#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CACHE_ROOT SDK_FINGERPRINT WHEEL" >&2
  exit 2
fi

cache_root="$1"
sdk_fingerprint="$2"
wheel="$3"

if [[ ! "${sdk_fingerprint}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid TVM SDK fingerprint: ${sdk_fingerprint}" >&2
  exit 2
fi
test -f "${wheel}"

staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/tilelang-tvm-sdk.XXXXXX")"
trap 'rm -rf -- "${staging_dir}"' EXIT

libraries=(libtvm_runtime.so libtvm_compiler.so)
for library in "${libraries[@]}"; do
  unzip -p "${wheel}" "tilelang/lib/${library}" > "${staging_dir}/${library}"
  test -s "${staging_dir}/${library}"
done

(
  cd "${staging_dir}"
  sha256sum "${libraries[@]}" > SHA256SUMS
)

{
  echo "tvm_sdk_fingerprint=${sdk_fingerprint}"
  echo "wheel_sha256=$(sha256sum "${wheel}" | awk '{ print $1 }')"
  for library in "${libraries[@]}"; do
    echo "library=${library}"
    readelf -d "${staging_dir}/${library}" | grep -E '\((NEEDED|SONAME|RPATH|RUNPATH)\)' || true
  done
} > "${staging_dir}/tvm-sdk-provenance.txt"

mkdir -p "${cache_root}"
for library in "${libraries[@]}"; do
  install -m 0755 "${staging_dir}/${library}" "${cache_root}/${library}"
done
install -m 0644 "${staging_dir}/SHA256SUMS" "${cache_root}/SHA256SUMS"
install -m 0644 "${staging_dir}/tvm-sdk-provenance.txt" \
  "${cache_root}/tvm-sdk-provenance.txt"
printf '%s\n' "tvm_sdk_fingerprint=${sdk_fingerprint}" \
  > "${cache_root}/.tilelang-ci-cache-identity.tmp"
mv -f -- "${cache_root}/.tilelang-ci-cache-identity.tmp" \
  "${cache_root}/.tilelang-ci-cache-identity"

cat "${cache_root}/tvm-sdk-provenance.txt"
