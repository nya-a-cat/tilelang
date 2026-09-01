#!/usr/bin/env bash
# Atomically apply a verified TileLang native/Python/source overlay to an exact base wheel.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <tilelang-native-overlay.tar.gz> <native-overlay-manifest.json>" >&2
    exit 2
fi

overlay_archive="$(realpath "$1")"
overlay_manifest="$(realpath "$2")"
python_bin="${PYTHON:-python}"

for path in "${overlay_archive}" "${overlay_manifest}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required overlay file does not exist: ${path}" >&2
        exit 2
    fi
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "The native development overlay supports Linux x86_64 only." >&2
    exit 2
fi

mapfile -t manifest_values < <(
    "${python_bin}" - "${overlay_manifest}" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("schema") != "tilelang-colab-native-overlay-v1":
    raise SystemExit("Unsupported native overlay manifest schema.")
if manifest.get("repository") != "nya-a-cat/tilelang":
    raise SystemExit("Native overlay does not belong to the authorized fork.")

hex40 = re.compile(r"[0-9a-f]{40}")
hex64 = re.compile(r"[0-9a-f]{64}")
for key in ("source_sha", "native_base_sha"):
    value = manifest.get(key, "")
    if not hex40.fullmatch(value):
        raise SystemExit(f"Invalid {key} in native overlay manifest.")
for key in (
    "base_native_tree_sha256",
    "result_native_tree_sha256",
    "patched_libtilelang_sha256",
    "overlay_sha256",
    "runtime_identity_sha256",
):
    value = manifest.get(key, "")
    if not hex64.fullmatch(value):
        raise SystemExit(f"Invalid {key} in native overlay manifest.")

version = manifest.get("base_distribution_version", "")
if not version or not re.fullmatch(r"[0-9A-Za-z_.+!-]+", version):
    raise SystemExit("Invalid base distribution version in native overlay manifest.")

for key in (
    "source_sha",
    "native_base_sha",
    "base_distribution_version",
    "base_native_tree_sha256",
    "result_native_tree_sha256",
    "patched_libtilelang_sha256",
    "overlay_sha256",
    "runtime_identity_sha256",
):
    print(manifest[key])
PY
)

if [[ "${#manifest_values[@]}" -ne 8 ]]; then
    echo "Native overlay manifest is incomplete." >&2
    exit 2
fi

source_sha="${manifest_values[0]}"
base_sha="${manifest_values[1]}"
base_version="${manifest_values[2]}"
base_native_hash="${manifest_values[3]}"
result_native_hash="${manifest_values[4]}"
patched_lib_hash="${manifest_values[5]}"
expected_archive_hash="${manifest_values[6]}"
runtime_identity_hash="${manifest_values[7]}"

actual_archive_hash="$(sha256sum "${overlay_archive}" | awk '{ print $1 }')"
if [[ "${actual_archive_hash}" != "${expected_archive_hash}" ]]; then
    echo "Native overlay archive hash mismatch." >&2
    exit 2
fi

"${python_bin}" - \
    "${overlay_archive}" \
    "${source_sha}" \
    "${base_sha}" \
    "${runtime_identity_hash}" <<'PY'
import hashlib
import json
import sys
import tarfile
from pathlib import PurePosixPath

archive, source_sha, base_sha, identity_hash = sys.argv[1:]
counts = {"identity": 0, "library": 0, "python": 0, "source": 0}
identity_bytes = None

with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
            raise SystemExit(f"Unsafe native overlay path: {member.name}")
        if not member.isfile():
            raise SystemExit(f"Native overlay contains a non-regular member: {member.name}")

        name = path.as_posix()
        if name == "tilelang/lib/libtilelang.so":
            counts["library"] += 1
        elif name == "tilelang/_python_overlay_identity.json":
            counts["identity"] += 1
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise SystemExit("Unable to read runtime identity.")
            identity_bytes = extracted.read()
        elif name.startswith("tilelang/src/") and len(path.parts) > 2:
            counts["source"] += 1
        elif (
            name.startswith("tilelang/")
            and not name.startswith(("tilelang/lib/", "tilelang/src/", "tilelang/3rdparty/"))
            and path.suffix in (".py", ".pyi", ".md")
        ):
            counts["python"] += 1
        else:
            raise SystemExit(f"Unsupported native overlay member: {member.name}")

if counts["identity"] != 1 or counts["library"] != 1:
    raise SystemExit("Native overlay must contain one runtime identity and one libtilelang.so.")
if counts["python"] == 0 or counts["source"] == 0:
    raise SystemExit("Native overlay is missing the complete Python or source tree.")
if identity_bytes is None or hashlib.sha256(identity_bytes).hexdigest() != identity_hash:
    raise SystemExit("Runtime identity hash mismatch.")

identity = json.loads(identity_bytes)
if identity.get("schema") != "tilelang-python-overlay-identity-v1":
    raise SystemExit("Unsupported runtime identity schema.")
if identity.get("repository") != "nya-a-cat/tilelang":
    raise SystemExit("Runtime identity does not belong to the authorized fork.")
if identity.get("source_sha") != source_sha or identity.get("native_base_sha") != base_sha:
    raise SystemExit("Runtime identity and native overlay manifest disagree.")
PY

mapfile -t install_info < <(
    "${python_bin}" - <<'PY'
from importlib import metadata
from pathlib import Path

dist = metadata.distribution("tilelang")
print(Path(dist.locate_file("tilelang")).resolve())
print(dist.version)
PY
)

if [[ "${#install_info[@]}" -ne 2 ]]; then
    echo "Unable to locate the installed TileLang distribution." >&2
    exit 2
fi

package_dir="$(realpath "${install_info[0]}")"
installed_version="${install_info[1]}"
case "${package_dir}" in
    */site-packages/tilelang|*/dist-packages/tilelang) ;;
    *)
        echo "Refusing to modify an unexpected TileLang package path: ${package_dir}" >&2
        exit 2
        ;;
esac

if [[ "${installed_version}" != "${base_version}" ]]; then
    echo "Installed TileLang version ${installed_version} does not match required base ${base_version}." >&2
    exit 2
fi
if [[ ! -f "${package_dir}/__init__.py" || ! -f "${package_dir}/lib/libtilelang.so" ]]; then
    echo "Installed TileLang wheel is missing required Python or native content." >&2
    exit 2
fi

native_tree_digest() {
    "${python_bin}" - "$1" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
if not root.is_dir():
    raise SystemExit(f"Native library directory does not exist: {root}")

files = []
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit(f"Native library tree contains a symlink: {path}")
    if path.is_file():
        files.append(path)

h = sha256()
for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix().encode()
    data = path.read_bytes()
    h.update(len(relative).to_bytes(8, "big"))
    h.update(relative)
    h.update(len(data).to_bytes(8, "big"))
    h.update(data)
print(h.hexdigest())
PY
}

installed_native_hash="$(native_tree_digest "${package_dir}/lib")"
if [[ "${installed_native_hash}" != "${base_native_hash}" ]]; then
    echo "Installed native tree does not match the exact base wheel." >&2
    echo "Expected: ${base_native_hash}" >&2
    echo "Actual:   ${installed_native_hash}" >&2
    exit 2
fi

package_parent="$(realpath "$(dirname "${package_dir}")")"
extract_root="$(mktemp -d)"
stage_root="$(mktemp -d "${package_parent}/.tilelang-native-overlay-stage.XXXXXX")"
staged_package="${stage_root}/tilelang"
backup_dir="${package_parent}/.tilelang-native-overlay-backup.$$"

cleanup() {
    status=$?
    trap - EXIT
    if [[ "${status}" -ne 0 && -d "${backup_dir}" && ! -e "${package_dir}" ]]; then
        mv -- "${backup_dir}" "${package_dir}"
    fi
    if [[ -d "${extract_root}" ]]; then
        rm -rf -- "${extract_root}"
    fi
    if [[ -d "${stage_root}" ]]; then
        rm -rf -- "${stage_root}"
    fi
    exit "${status}"
}
trap cleanup EXIT

if [[ -e "${backup_dir}" ]]; then
    echo "Refusing to overwrite an existing TileLang overlay backup: ${backup_dir}" >&2
    exit 2
fi

tar -xzf "${overlay_archive}" -C "${extract_root}"
actual_patched_lib_hash="$(sha256sum "${extract_root}/tilelang/lib/libtilelang.so" | awk '{ print $1 }')"
if [[ "${actual_patched_lib_hash}" != "${patched_lib_hash}" ]]; then
    echo "Patched libtilelang.so hash mismatch after extraction." >&2
    exit 2
fi

cp -a -- "${package_dir}" "${staged_package}"

find "${staged_package}" \
    \( -path "${staged_package}/lib" -o -path "${staged_package}/src" -o -path "${staged_package}/3rdparty" \) -prune \
    -o -type f \( -name '*.py' -o -name '*.pyi' -o -name '*.md' -o -name '*.pyc' \) \
    -exec rm -f -- {} +
rm -f -- \
    "${staged_package}/_python_overlay_identity.json" \
    "${staged_package}/_native_overlay_identity.json"
find "${staged_package}" -depth -type d -name __pycache__ \
    ! -path "${staged_package}/lib/*" \
    ! -path "${staged_package}/src/*" \
    ! -path "${staged_package}/3rdparty/*" \
    -exec rm -rf -- {} +

staged_src="$(realpath -m "${staged_package}/src")"
case "${staged_src}" in
    "${stage_root}"/*) ;;
    *)
        echo "Refusing to replace an unexpected staged source path: ${staged_src}" >&2
        exit 2
        ;;
esac
rm -rf -- "${staged_src}"
mkdir -p "${staged_src}"
cp -a -- "${extract_root}/tilelang/src/." "${staged_src}/"

while IFS= read -r -d '' source_file; do
    relative="${source_file#"${extract_root}/tilelang/"}"
    install -D -m 0644 -- "${source_file}" "${staged_package}/${relative}"
done < <(
    find "${extract_root}/tilelang" \
        \( -path "${extract_root}/tilelang/lib" -o -path "${extract_root}/tilelang/src" \) -prune \
        -o -type f \( -name '*.py' -o -name '*.pyi' -o -name '*.md' \) -print0
)
install -D -m 0644 -- \
    "${extract_root}/tilelang/_python_overlay_identity.json" \
    "${staged_package}/_python_overlay_identity.json"
install -D -m 0755 -- \
    "${extract_root}/tilelang/lib/libtilelang.so" \
    "${staged_package}/lib/libtilelang.so"

staged_native_hash="$(native_tree_digest "${staged_package}/lib")"
if [[ "${staged_native_hash}" != "${result_native_hash}" ]]; then
    echo "Staged native tree does not match the verified overlay result." >&2
    exit 2
fi

mv -- "${package_dir}" "${backup_dir}"
if ! mv -- "${staged_package}" "${package_dir}"; then
    mv -- "${backup_dir}" "${package_dir}"
    echo "Unable to activate the staged TileLang overlay." >&2
    exit 2
fi
rm -rf -- "${backup_dir}"

echo "TileLang native overlay active at ${package_dir}"
echo "TileLang base wheel version: ${base_version}"
echo "TileLang native base source: ${base_sha}"
echo "TileLang overlay source: ${source_sha}"
echo "Restart any Python process that imported TileLang before applying this overlay."
