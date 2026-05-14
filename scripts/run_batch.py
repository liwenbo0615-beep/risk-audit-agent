"""CLI batch runner.

Usage examples:
    python scripts/run_batch.py -c "内容1" "内容2" "内容3"
    python scripts/run_batch.py -f comments.txt
    python scripts/run_batch.py -f comments.txt -d approve -o results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from audit import batch_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UGC 内容批量审核工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", metavar="PATH", help="输入文件（每行一条评论）")
    group.add_argument("--comments", "-c", nargs="+", metavar="TEXT", help="直接传入评论内容")
    parser.add_argument(
        "--auto-decision", "-d",
        choices=["approve", "reject", "skip"],
        default="skip",
        dest="auto_decision",
        help="人工复核自动决策 (默认: skip)",
    )
    parser.add_argument("--output", "-o", metavar="PATH", help="输出 JSON 结果到文件")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"错误: 文件不存在 — {path}", file=sys.stderr)
            sys.exit(1)
        comments = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        comments = args.comments

    if not comments:
        print("错误: 没有可处理的评论内容", file=sys.stderr)
        sys.exit(1)

    results = batch_audit(comments, args.auto_decision)

    if args.output:
        out = [json.loads(r["final_report"]) for r in results if r["final_report"]]
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入: {args.output}")


if __name__ == "__main__":
    main()
