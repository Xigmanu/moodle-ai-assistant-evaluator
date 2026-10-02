#!/bin/bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

PROFILE="prod"
case "${1:-}" in
    --dev) PROFILE="dev" ;;
    "") ;;
    *)
        printf 'unknown argument %s\n' "$1" >&2
        exit 2
        ;;
esac

os_ws_dir() {
    case "$(uname -s)" in
        Darwin)               printf '%s' "$HOME/Library/Application Support/VoltEval" ;;
        MINGW*|MSYS*|CYGWIN*) printf '%s' "${LOCALAPPDATA:-$HOME/AppData/Local}/.volteval" ;;
        *)                    printf '%s' "$HOME/.volteval" ;;
    esac
}

seed_file() {
    local src="$1" dst="$2"
    if [[ "$PROFILE" == "dev" || ! -e "$dst" ]]; then
        cp "$src" "$dst"
    fi
}

if [[ "$PROFILE" == "dev" ]]; then
    WORKSPACE="$ROOT/.volteval"
else
    WORKSPACE="$(os_ws_dir)"
fi

mkdir -p "$WORKSPACE/experiments"

while IFS= read -r -d '' file; do
    seed_file "$file" "$WORKSPACE/experiments/$(basename "$file")"
done < <(find "$ROOT/config/experiments" -maxdepth 1 -type f \
    \( -name '*.yaml' -o -name '*.yml' \) -print0 2>/dev/null)

seed_file "$ROOT/config/config.example.toml" "$WORKSPACE/config.toml"
seed_file "$ROOT/.env.example" "$WORKSPACE/.env"

if [[ "$PROFILE" == "dev" ]]; then
    VENV="$ROOT/.venv"
    [[ -x "$VENV/bin/python" ]] || "$PYTHON" -m venv "$VENV"

    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -e "$ROOT[dev]"
else
    "$PYTHON" -m pip install --user --upgrade "$ROOT"
fi
