# -*- coding: utf-8 -*-
"""
数据抽取：读取 knowledgeBase.xlsx，抽取 (id, question, answer) 三元组，
导出为 JSONL 供后续向量化使用。

用法：
    python scripts/extract.py                 # 默认抽取全部（--limit 0 全量）
    python scripts/extract.py --limit 200     # 只抽取前 200 条（采样）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import openpyxl  # noqa: E402


def extract(xlsx_path: str, limit: int = 0):
    """从 xlsx 抽取问答对，返回 list[dict]。

    limit > 0 时只取前 limit 条；limit == 0 表示全量。
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Sheet1"]  # 数据只有一个 sheet

    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:  # 表头 (序号, question, answer)
            continue
        qid, question, answer = row[0], row[1], row[2]
        # 去空值 / 统一文本
        if question is None or answer is None:
            continue
        question = str(question).strip()
        answer = str(answer).strip()
        if not question or not answer:
            continue
        rows.append({
            "id": int(qid) if qid is not None else i,
            "question": question,
            "answer": answer,
        })
        if limit > 0 and len(rows) >= limit:
            break

    wb.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="抽取政策问答知识库为 JSONL")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT,
                        help="抽取条数，0 表示全量（默认 %d）" % config.DEFAULT_LIMIT)
    parser.add_argument("--xlsx", type=str, default=config.DATA_XLSX, help="源 xlsx 路径")
    parser.add_argument("--out", type=str, default=config.DATA_JSONL, help="输出 jsonl 路径")
    args = parser.parse_args()

    rows = extract(args.xlsx, limit=args.limit)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[extract] 完成：共抽取 {len(rows)} 条 → {args.out}")


if __name__ == "__main__":
    main()
