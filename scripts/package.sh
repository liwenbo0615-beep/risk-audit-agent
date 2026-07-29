#!/usr/bin/env bash
# 生成干净的交付包：基于当前工作区，排除密钥 / 日志 / 缓存 / IDE / git 内部 / 个人素材。
# 用法： bash scripts/package.sh
# 产物： 上级目录 risk-audit-agent-delivery.tar.gz
#
# 为什么不直接 zip 文件夹：会把 .env（含真实 API Key）、audit_log.jsonl、result.json、
# __pycache__、.git 等一起打进去——既泄露密钥又不专业。本脚本显式排除这些。
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="../risk-audit-agent-delivery.tar.gz"

tar --no-xattrs \
  --exclude='./.git' \
  --exclude='./.env' \
  --exclude='*.jsonl' \
  --exclude='./result*.json' \
  --exclude='./review_queue.json' \
  --exclude='./PROJECT_AUDIT_REPORT.md' \
  --exclude='./INTERVIEW_PREP.md' \
  --exclude='./lx.txt' \
  --exclude='./.claude' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='./.vscode' \
  --exclude='./.pytest_cache' \
  --exclude='./简历项目描述.md' \
  -czf "$OUT" .

echo "已生成交付包: $OUT"
echo "--- 自检：包内绝不应出现 .env / *.jsonl / __pycache__ ---"
if tar -tzf "$OUT" | grep -E '(^|/)\.env$|\.jsonl$|__pycache__|\.git/'; then
  echo "❌ 警告：交付包仍含敏感/垃圾文件，请检查排除规则！" >&2
  exit 1
fi
echo "✅ 自检通过，包内无敏感文件。"
