#!/usr/bin/env bash
# Test tiers (§4.17 小收口).  Usage: scripts/run_tests.sh [core|module|full]
#
#   core    核心不变量，约 30 秒：账本、门禁、协议、策略、转换检查
#   module  模块测试（不含会话级），分块跑约 1.5–2 分钟
#   full    全景回归（含会话级），分块跑约 3.5 分钟
#
# 分块是必须的：本机内存受限，一次加载全部图像会被 OOM 杀掉（已发生两次）。
set -euo pipefail
tier="${1:-core}"
cd "$(dirname "$0")/.."

chunked() {           # chunked <pytest args...>
  # Small chunks on purpose: this box's shared cgroup kills a run that loads
  # too many images at once (seen three times), and a killed chunk reports
  # nothing about what passed.
  local size=8
  mapfile -t files < <(ls tests/test_*.py)
  local index=0
  while (( index < ${#files[@]} )); do
    python -m pytest "${files[@]:index:size}" -q "$@"
    index=$(( index + size ))
  done
}

case "$tier" in
  core)   echo "== 核心不变量（约 30 秒）"; python -m pytest tests -q -m core ;;
  module) echo "== 模块测试（不含 slow，分块）"; chunked -m "not slow" ;;
  full)   echo "== 全景回归（分块）"; chunked ;;
  *) echo "unknown tier: $tier (core|module|full)"; exit 2;;
esac
