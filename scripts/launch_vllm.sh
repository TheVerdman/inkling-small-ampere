#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_profile="${repository_root}/configs/serving/responses-2k-bringup-v1.json"
profile="${INKLING_SERVING_PROFILE:-${default_profile}}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  profile="${1}"
  shift
fi

export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m inkling_ampere.serving.launch --profile "${profile}" "$@"
