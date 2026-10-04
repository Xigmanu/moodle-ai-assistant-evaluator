#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT/.venv/bin"
TARGET=("$ROOT/src/evaluator")

"$BIN/autoflake" --in-place --recursive --remove-all-unused-imports --exclude '__init__.py' "$TARGET"
"$BIN/isort" --profile black "$TARGET"
"$BIN/black" "$TARGET"
