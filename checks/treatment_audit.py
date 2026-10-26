# -*- coding: utf-8 -*-
"""本地防治库体检：列出缺口（可在补完数据后再跑一次对照）

用法：python checks\treatment_audit.py
"""
import os
import sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("RICE_NO_WEBENGINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rice_dp1_fluent as m   # noqa: E402

a = m.treatments_audit()
print("=" * 66)
print("本地防治库体检")
print("=" * 66)
print("类别 %d 个 · 药剂条目 %d 条（其中生物农药 %d 条）"
      % (a["类别数"], a["药剂条目"], a["生物农药"]))
print("有登记证号 %d · 有安全间隔期 %d · 有防治适期 %d"
      % (a["有登记证号"], a["有安全间隔期"], a["有防治适期"]))
for key in ("缺登记证号", "缺安全间隔期", "缺防治适期"):
    items = a[key]
    print("\n%s（%d 条）：" % (key, len(items)))
    for it in items:
        print("   · %s" % it)
print("\n类别缺防治要点：%s" % ("、".join(a["类别缺防治要点"]) or "无"))
print("类别缺依据：%s" % ("、".join(a["类别缺依据"]) or "无"))
print("\n完整度：登记证号 %.0f%% · 安全间隔期 %.0f%% · 防治适期 %.0f%%"
      % (a["有登记证号"] * 100.0 / a["药剂条目"],
         a["有安全间隔期"] * 100.0 / a["药剂条目"],
         a["有防治适期"] * 100.0 / a["药剂条目"]))
print("\n提示：缺的条目一律显示「待补，以标签为准」，程序不会用猜测值替代。")
