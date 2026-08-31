#!/usr/bin/env bash
# Apply a pure-Python TileLang source overlay on top of an installed native wheel.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <tilelang-python-overlay.tar.gz>" >&2
    exit 2
fi

overlay_archive="$(realpath "$1")"
python_bin="${PYTHON:-python}"

if [[ ! -f "${overlay_archive}" ]]; then
    echo "Overlay archive does not exist: ${overlay_archive}" >&2
    exit 2
fi

package_dir="$(${python_bin} - <<'PY'
from pathlib import Path
import tilelang

print(Path(tilelang.__file__).resolve().parent)
PY
)"
package_dir="$(realpath "${package_dir}")"

case "${package_dir}" in
    */site-packages/tilelang|*/dist-packages/tilelang) ;;
    *)
        echo "Refusing to modify an unexpected TileLang package path: ${package_dir}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${package_dir}/__init__.py" || ! -d "${package_dir}/lib" ]]; then
    echo "Installed TileLang wheel is missing Python or native-library content: ${package_dir}" >&2
    exit 2
fi

while IFS= read -r member; do
    case "${member}" in
        tilelang/*.py|tilelang/*.pyi|tilelang/*.md) ;;
        *)
            echo "Unsafe or unsupported overlay member: ${member}" >&2
            exit 2
            ;;
    esac
    if [[ "${member}" == /* || "${member}" == *"../"* ]]; then
        echo "Unsafe overlay path: ${member}" >&2
        exit 2
    fi
done < <(tar -tzf "${overlay_archive}")

native_hash_before="$(
    find "${package_dir}/lib" -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{ print $1 }'
)"

find "${package_dir}" \
    \( -path "${package_dir}/lib" -o -path "${package_dir}/src" -o -path "${package_dir}/3rdparty" \) -prune \
    -o -type f \( -name '*.py' -o -name '*.pyi' -o -name '*.md' -o -name '*.pyc' \) -delete
find "${package_dir}" -depth -type d -name __pycache__ -empty \
    ! -path "${package_dir}/lib/*" \
    ! -path "${package_dir}/src/*" \
    ! -path "${package_dir}/3rdparty/*" \
    -delete
tar -xzf "${overlay_archive}" -C "$(dirname "${package_dir}")"

native_hash_after="$(
    find "${package_dir}/lib" -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{ print $1 }'
)"
test "${native_hash_before}" = "${native_hash_after}"

${python_bin} - <<'PY'
from pathlib import Path
import tilelang

package_dir = Path(tilelang.__file__).resolve().parent
print(f"TileLang Python overlay active at {package_dir}")
print(f"TileLang version from native base wheel: {tilelang.__version__}")
PY
