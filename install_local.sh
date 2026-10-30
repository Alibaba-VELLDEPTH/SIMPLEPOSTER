#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "${repo_dir}/requirements.txt"

echo "Third-party Python dependencies are installed."
