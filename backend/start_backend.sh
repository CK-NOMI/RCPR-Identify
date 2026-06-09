#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "未找到 venv，请先创建虚拟环境并安装 requirements.txt。"
  exit 1
fi

source venv/bin/activate
export PYTHONUNBUFFERED=1
python cosod_backend.py --host 0.0.0.0 --port "${PORT:-8080}"
