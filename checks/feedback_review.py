# -*- coding: utf-8 -*-
"""误检反馈复核：把界面里一条条「识别不对」汇总成能指导补数据的结论。

用法：
    python checks/feedback_review.py                    # 只统计
    python checks/feedback_review.py --prepare          # 顺便把图片归拢到待标注目录
    python checks/feedback_review.py --dir <目录>        # 指定程序数据目录（默认脚本同级）

输出：
    屏幕：总量、来源分布、模型判断 × 用户判断 的混淆计数、漏检里用户报的类别
    文件：<目录>/feedback/summary.csv（每对计数）、<目录>/feedback/to_label/（--prepare）
"""
import os
import sys
import csv
import shutil
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_rows(feedback_dir):
    csv_path = feedback_dir / "feedback.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("predicted")]


def main():
    ap = argparse.ArgumentParser(description="误检反馈复核")
    ap.add_argument("--dir", default=str(ROOT), help="程序数据目录（含 feedback/）")
    ap.add_argument("--prepare", action="store_true", help="把图片归拢到 feedback/to_label/")
    args = ap.parse_args()

    feedback_dir = Path(args.dir).resolve() / "feedback"
    rows = load_rows(feedback_dir)
    print("=" * 60)
    print("反馈目录：%s" % feedback_dir)
    print("反馈条数：%d" % len(rows))
    if not rows:
        print("\n还没有反馈。在程序里对识别结果点「识别不对」，或在图片页点「报告漏检」。")
        return 0

    modes = Counter(r.get("mode", "") for r in rows)
    print("来源分布：%s" % "，".join("%s %d 条" % (k or "未知", v) for k, v in modes.items()))

    pairs = Counter((r.get("predicted", ""), r.get("user_choice", "")) for r in rows)
    print("\n模型判断 → 用户判断（前 10）")
    for (pred, truth), n in pairs.most_common(10):
        print("  %-12s → %-14s %d" % (pred, truth, n))

    misses = [r for r in rows if r.get("mode") == "miss" or r.get("predicted") == "(无检出)"]
    if misses:
        print("\n漏检（模型没检出）里用户报的类别")
        for name, n in Counter(r.get("user_choice", "") for r in misses).most_common():
            print("  %-14s %d" % (name, n))
        print("  ⇒ 漏检样本优先补：这是项目里最危险的错误类型")

    same = sum(n for (p, u), n in pairs.items() if p == u)
    if same:
        print("\n注意：有 %d 条「判断和模型一致」的反馈——这类通常是阈值/定位问题，不是分类问题。" % same)

    summary = feedback_dir / "summary.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model_predicted", "user_choice", "count"])
        for (pred, truth), n in pairs.most_common():
            w.writerow([pred, truth, n])
    print("\n已写出汇总：%s" % summary)

    if args.prepare:
        dst = feedback_dir / "to_label"
        dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for r in rows:
            name = r.get("image") or ""
            src = feedback_dir / name
            if name and src.exists() and not (dst / name).exists():
                shutil.copy2(src, dst / name)
                copied += 1
        # LabelImg 的类别顺序必须与训练一致（见 .trae/skills 里的硬约定）
        classes = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]
        (dst / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        print("已准备待标注目录：%s（新增 %d 张）" % (dst, copied))
        print("标注命令示例：labelImg \"%s\" \"%s\"" % (dst, dst / "classes.txt"))
        print("提示：反馈表里只记了「实际是什么」，框位置仍需人工画；标完再并入数据集。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
