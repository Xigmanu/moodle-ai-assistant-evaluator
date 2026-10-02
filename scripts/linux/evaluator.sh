#!/bin/bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

export VOLT_HOME="$ROOT/.volteval"
exec "$ROOT/.venv/bin/evaluator" "$@"
