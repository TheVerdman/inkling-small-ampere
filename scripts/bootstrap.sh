#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bootstrap_environment="${repository_root}/.tools/uv-bootstrap"
uv_version="0.11.29"

python3 -m venv "${bootstrap_environment}"
"${bootstrap_environment}/bin/python" -m pip install --disable-pip-version-check "uv==${uv_version}"

cd "${repository_root}"
"${bootstrap_environment}/bin/uv" sync --frozen

echo "Developer environment ready at ${repository_root}/.venv"
echo "Activate it with: . .venv/bin/activate"

