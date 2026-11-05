"""审计本地防治库的证据完整度（不导入主程序，避免启动 GUI）。

从 rice_dp1_fluent.py 里按 AST 取出防治库及其增强表，复现运行时合并逻辑，
输出 Markdown 表格。用法：

    python scripts/audit_treatment_evidence.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "rice_dp1_fluent.py"

NAMES = {
    "TREATMENT_DB",
    "TREATMENT_ENRICH",
    "TREATMENT_CLASS_META",
    "TREATMENT_THRESHOLDS",
    "TREATMENT_ADD",
    "TREATMENT_ADD_MORE",
    "DRUG_SAFE_DAYS",
}


def load_literals(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns = {"THRESHOLD_SOURCE_NATIONAL": "《2025 年粮食作物重大病虫害防控技术方案》"}
    values = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in NAMES:
            continue
        values[target.id] = eval(compile(ast.Expression(node.value), "<ast>", "eval"), ns)
    return values


def build_db(values: dict) -> dict:
    db = values["TREATMENT_DB"]
    enrich = values["TREATMENT_ENRICH"]
    keys = sorted(enrich, key=len, reverse=True)

    for tr in db.values():
        for drug in tr.get("drugs", []):
            name = drug.get("name", "") or ""
            extra = enrich.get(name)
            if extra is None:
                for key in keys:
                    if key in name:
                        extra = enrich[key]
                        break
            if extra:
                for key, value in extra.items():
                    drug.setdefault(key, value)

    for cls, meta in values["TREATMENT_CLASS_META"].items():
        if cls in db:
            db[cls].setdefault("meta", {}).update(meta)
    for cls, threshold in values.get("TREATMENT_THRESHOLDS", {}).items():
        if cls in db:
            db[cls].setdefault("meta", {}).update(threshold)

    merged = dict(values["TREATMENT_ADD"])
    for cls, drugs in values["TREATMENT_ADD_MORE"].items():
        merged[cls] = list(merged.get(cls, [])) + list(drugs)
    for cls, drugs in merged.items():
        tr = db.get(cls)
        if not tr:
            continue
        have = {d.get("name") for d in tr.setdefault("drugs", [])}
        for drug in drugs:
            if drug.get("name") not in have:
                tr["drugs"].append(dict(drug))
                have.add(drug.get("name"))

    for key, days, source in sorted(values["DRUG_SAFE_DAYS"], key=lambda x: -len(x[0])):
        for tr in db.values():
            for drug in tr.get("drugs", []):
                if key in (drug.get("name") or "") and not drug.get("safe_days"):
                    drug["safe_days"] = days
                    drug["safe_source"] = source
    return db


def evidence_status(drug: dict) -> str:
    name = drug.get("name") or ""
    reg_raw = (drug.get("reg_no") or "").strip()
    # “（政府用药指导，未逐一核对登记证）”是占位说明，不是可核验的登记证号
    reg = bool(reg_raw) and "未逐一核对" not in reg_raw and not reg_raw.startswith("（")
    days = drug.get("safe_days")
    source = drug.get("source") or drug.get("safe_source")
    bio = drug.get("type") == "生物"
    # 单剂条目引用了复配剂证号，或同时列了多个证号：需要回到单剂标签核对
    reg_nos = set(re.findall(r"PD\d+", reg_raw + " " + (source or "")))
    mixed_note = "复配" in reg_raw or "复配" in (source or "")
    is_mixture_name = "·" in name
    if reg and not is_mixture_name and (mixed_note or len(reg_nos) > 1):
        return "B 证号待核对"
    if reg and days and source:
        return "A 完整"
    if reg and not days and bio:
        return "B 生物制剂未标间隔期"
    if reg and not days:
        return "B 缺间隔期"
    if days and source and not reg:
        return "B 缺登记证"
    if reg and not source:
        return "B 缺来源"
    return "C 待补"


def cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    values = load_literals(SOURCE)
    db = build_db(values)
    rows = []
    counts = {}
    for cls, tr in db.items():
        for drug in tr.get("drugs", []):
            status = evidence_status(drug)
            counts[status] = counts.get(status, 0) + 1
            rows.append({
                "类别": cls,
                "药剂": drug.get("name", ""),
                "规格": drug.get("spec", ""),
                "亩用量": drug.get("dose", ""),
                "间隔期": drug.get("safe_days", ""),
                "登记证号": drug.get("reg_no", ""),
                "依据": drug.get("source") or drug.get("safe_source") or "",
                "状态": status,
            })

    print("## 汇总\n")
    print("| 项 | 数量 |")
    print("|---|---|")
    print(f"| 类别 | {len(db)} |")
    print(f"| 药剂条目 | {len(rows)} |")
    for key in sorted(counts):
        print(f"| {key} | {counts[key]} |")
    print()
    print("## 明细\n")
    print("| 类别 | 药剂 | 规格 | 亩用量 | 安全间隔期(天) | 登记证号 | 来源/依据 | 证据状态 |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(cell(row[k]) for k in
              ("类别", "药剂", "规格", "亩用量", "间隔期", "登记证号", "依据", "状态")) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
