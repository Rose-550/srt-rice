# -*- coding: utf-8 -*-
"""真机验收：Agent 工具调用（真实 DeepSeek API + 真实天气接口）。

⚠️ 会真实计费（一次问答约几分钱）。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["RICE_APP_DIR"] = str(ROOT / "checks" / "_real_out")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop, QTimer
import rice_dp1_fluent as m

cfg = m._migrate_config(json.loads((ROOT / "config.json").read_text(encoding="utf-8")))
m.CONFIG.update({"llm_provider": "deepseek",
                 "deepseek_api_key": cfg.get("deepseek_api_key", ""),
                 "deepseek_model": cfg.get("deepseek_model", "deepseek-flash"),
                 "enable_llm": True, "llm_thinking": False, "vision_override": "auto"})
print("模型：", m.CONFIG["deepseek_model"])

# 先单独验一下免 Key 的天气工具
w = m.run_agent_tool("get_weather", {"city": "南京"})
print("天气工具：", str(w)[:160].replace("\n", " "))

app = QApplication(sys.argv)
Q = ("稻瘟病破口期怎么防治？另外帮我查一下南京未来三天天气适不适合打药，"
     "再确认一下我准备用的三环唑在不在禁用名单里。")
th = m.LLMAnalysisThread("稻瘟病", severity={"level": "中", "boxes": 2,
                                            "area_ratio": 0.03, "max_ratio": 0.02,
                                            "mean_conf": 0.8, "basis": "按检测框占画面面积估计"},
                         prompt=Q)
tools, chunks, final = [], [], []
th.tool_used.connect(lambda n, l: tools.append(l))
th.chunk.connect(lambda s: chunks.append(s))
th.result_ready.connect(lambda s: final.append(s))
th.failed.connect(lambda s: final.append("ERR: " + s))
t0 = time.time()
th.start()
loop = QEventLoop()
timer = QTimer(); timer.setInterval(100)
timer.timeout.connect(lambda: loop.quit() if final else None)
timer.start()
QTimer.singleShot(120000, loop.quit)
loop.exec_()
th.wait(5000)

print("调用过的工具：", tools)
print("分片数：%d | 耗时：%.1f 秒" % (len(chunks), time.time() - t0))
if final and not final[0].startswith("ERR:"):
    text = final[0]
    print("回答 %d 字，前 300 字：\n%s" % (len(text), text[:300].replace("\n", " ")))
    print("禁用农药扫描：", m.pesticide_hits(text) or "无")
else:
    print("失败：", final)
# 留档：回答正文 + 调用轨迹（可作参赛佐证材料）
outdir = ROOT / "checks" / "_real_out"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "live_agent_answer.txt").write_text(
    "问题：%s\n\n调用工具：%s\n\n回答：\n%s\n" % (Q, "、".join(tools), final[0] if final else ""),
    encoding="utf-8")
print("审计日志：", (outdir / "logs" / "llm_log.jsonl").exists())
print("留档：", outdir / "live_agent_answer.txt")

# ── 追加验收：用量记账 / 视觉复核 / 诊断卡 ──────────────────────
print("\n--- 用量与成本 ---")
print("usage:", th.usage)
if th.usage:
    print("估算成本(元):", m.estimate_cost(m.get_model_name(), th.usage))

print("\n--- 视觉复核工具（裁框交给视觉模型二次确认）---")
import cv2 as _cv2
_img = _cv2.imread(str(ROOT / "手机端可跑版" / "samples" / "bacterial_blight_0.jpg"))
_boxes = [{"name": "白叶枯病", "conf": 0.69, "box": [0, 0, _img.shape[1], _img.shape[0]]}]
print("复核结果:", m.run_agent_tool("verify_with_vision", {"box_index": 0}, _img, _boxes)[:300])

print("\n--- 结构化诊断卡（含本地库逐条核验）---")
card_thread = m.DiagnosisCardThread("稻瘟病", final[0] if final else "", {"level": "中"})
cards = []
card_thread.card_ready.connect(lambda c: cards.append(c))
card_thread.failed.connect(lambda e: cards.append({"err": e}))
card_thread.start()
_loop2 = QEventLoop()
_t2 = QTimer(); _t2.setInterval(100)
_t2.timeout.connect(lambda: _loop2.quit() if cards else None)
_t2.start(); QTimer.singleShot(90000, _loop2.quit); _loop2.exec_()
card_thread.wait(5000)
if cards and "err" not in cards[0]:
    c = cards[0]
    print("诊断:", c.get("diagnosis"), "| 严重度:", c.get("severity_level"))
    for chem in c.get("chemicals", []):
        print("  药剂:", chem.get("name"), chem.get("dose"), "→", chem.get("source"))
    print("警告:", c.get("_warnings"))
else:
    print("诊断卡失败:", cards)

print("\n--- 记忆：会话已落盘 ---")
print("会话数:", len(m.load_session_index()))
print("工具 list_sessions:", m.run_agent_tool("list_sessions", {"days": 1})[:160])

# ── 追加验收：视觉复核级联（真实样本自动触发）────────────────────
print("\n--- 视觉复核级联（真实样本）---")
_img2 = _cv2.imread(str(ROOT / "手机端可跑版" / "samples" / "bacterial_blight_0.jpg"))
_model, _dev = m.get_model()
_res = _model(_img2, conf=m.CONF_THRESHOLD, imgsz=640, verbose=False)
_dets = [{"name": m.CLASS_NAMES[int(b.cls.item())], "conf": float(b.conf.item()),
          "box": b.xyxy[0].tolist()} for b in _res[0].boxes]
_best = max(_dets, key=lambda x: x["conf"]) if _dets else None
if _best:
    _need, _why = m.should_auto_verify(_best["name"], _best["conf"])
    print("检测：%s %.2f | 是否触发复核：%s（%s）" % (_best["name"], _best["conf"], _need, _why))
    if _need:
        _v = m.vision_recheck(_img2, _best["box"], None, 0, source="cascade_live")
        print("复核结果：", _v.get("verdict") or _v.get("error"), "| 原文:", (_v.get("raw") or "")[:80])
        print("是否一致：", _v.get("verdict") == _best["name"])
        print("⟶ 不一致时界面上会弹出「是 A / 是 B / 都不是」的人工裁决按钮，"
              "裁决结果写入 feedback 队列")
else:
    print("该样本未检出目标，跳过")

# ── 追加验收：无人值守流水线（真实模型 + 真实视觉复核）──────────
print("\n--- 无人值守巡检（把样例目录当收件箱）---")
_sample_dir = ROOT / "手机端可跑版" / "samples"
_t0 = time.time()
_rows = []
for _p in sorted(_sample_dir.glob("*.jpg")):
    _rec = m.process_image_file(_p, do_cascade=True)
    _rows.append(_rec)
    print("  %-28s → %-8s conf=%.2f 强度=%-2s 复核=%-8s %s"
          % (_rec["file"], _rec["cls"] or "未检出", _rec["conf"], _rec["severity"],
             _rec["verify"] or "—", ("⚠ " + m.alert_level(_rec)) if m.alert_level(_rec) else ""))
print("共 %d 张，耗时 %.1f 秒；台账：%s" % (len(_rows), time.time() - _t0, m.WATCH_RESULTS))
print("提醒条数：", sum(1 for r in _rows if m.alert_level(r)))

print("\n--- 主动简报（不用人问）---")
_dg = json.loads(m.run_agent_tool("daily_digest", {"days": 7}))
for _k in ("统计天数", "期间检测次数", "期间类别分布", "视觉复核次数", "视觉复核一致率%",
           "人工裁决次数", "反馈条数", "实训答题数"):
    if _k in _dg:
        print("  %s：%s" % (_k, _dg[_k]))

# ── 追加验收：任务规划与执行（真实 API）──────────────────────────
print("\n--- 任务规划与执行（真实 API）---")
_goal = "把样例目录里的图片处理一遍，再给我一份最近 7 天的数据简报"
_tp = m.TaskPlanThread(_goal)
_plans = []
_tp.planned.connect(lambda p: _plans.append(p))
_tp.failed.connect(lambda e: _plans.append({"err": e}))
_tp.start()
_l3 = QEventLoop(); _t3 = QTimer(); _t3.setInterval(100)
_t3.timeout.connect(lambda: _l3.quit() if _plans else None)
_t3.start(); QTimer.singleShot(90000, _l3.quit); _l3.exec_(); _tp.wait(5000)
if _plans and "err" not in _plans[0]:
    _plan = _plans[0]
    print("目标：", _plan.get("goal"))
    for _i, _s in enumerate(_plan.get("steps") or [], 1):
        print("  %d. %s %s  ← %s" % (_i, _s["tool"], json.dumps(_s["args"], ensure_ascii=False),
                                     _s.get("why", "")[:40]))
    print("提示：", _plan.get("note", "")[:80], "| 校验警告：", _plan.get("_warnings"))
    print("→ 实际执行：")
    for _i, _s in enumerate(_plan.get("steps") or [], 1):
        _ok, _out = m.run_task_step(_s, None, None)
        print("  %d. %s %s → %s" % (_i, _s["tool"], "成功" if _ok else "失败",
                                    str(_out)[:100].replace("\n", " ")))
else:
    print("规划失败：", _plans)

print("\n--- 定时任务（不等人问）---")
for _it in m.load_schedules():
    print("  %s · %s · %s · %s" % (_it["name"], _it["freq"], _it["time"],
                                   "启用" if _it.get("enabled") else "停用"))
print("补数据提醒：\n" + m.run_schedule_action("gap_check"))
print("当前到点任务：", [x[0]["name"] for x in m.due_schedules()] or "无（都已跑过或未到点）")

# ── 追加验收：失败自动重规划（真实 API）────────────────────────
print("\n--- 失败自动重规划（真实 API）---")
_goal2 = "把 D:\\不存在的目录_xyz 里的水稻图片处理一遍"
_tp2 = m.TaskPlanThread(_goal2)
_p2 = []
_tp2.planned.connect(lambda p: _p2.append(p))
_tp2.failed.connect(lambda e: _p2.append({"err": e}))
_tp2.start()
_l4 = QEventLoop(); _t4 = QTimer(); _t4.setInterval(100)
_t4.timeout.connect(lambda: _l4.quit() if _p2 else None)
_t4.start(); QTimer.singleShot(90000, _l4.quit); _l4.exec_(); _tp2.wait(5000)
if _p2 and "err" not in _p2[0]:
    _plan2 = _p2[0]
    print("首版计划：", [(s["tool"], s["args"]) for s in _plan2.get("steps", [])])
    _tr = m.TaskRunThread(_plan2)
    _ev = []
    _tr.step_done.connect(lambda i, s, ok, o: _ev.append(("step", s["tool"], ok, str(o)[:70])))
    _tr.replanning.connect(lambda n, why: _ev.append(("replan", n, why)))
    _done = []
    _tr.task_done.connect(lambda p, r: _done.append(r))
    _tr.failed.connect(lambda e: _done.append([("ERR", e)]))
    _tr.start()
    _l5 = QEventLoop(); _t5 = QTimer(); _t5.setInterval(100)
    _t5.timeout.connect(lambda: _l5.quit() if _done else None)
    _t5.start(); QTimer.singleShot(180000, _l5.quit); _l5.exec_(); _tr.wait(8000)
    for _e in _ev:
        if _e[0] == "step":
            print("  步骤 %-14s %s  %s" % (_e[1], "✅" if _e[2] else "⚠", _e[3]))
        else:
            print("  ↻ 第 %d 次重规划：%s" % (_e[1], _e[2][:60]))
    if _done:
        _rs = _done[0]
        print("最终：共 %d 步，成功 %d 步" % (len(_rs), sum(1 for _, ok, _ in _rs if ok)))
else:
    print("规划失败：", _p2)

# ── 追加验收：从官方公开资料导入教学题库（真实 API）──────────────
print("\n--- 教学资料导入：官方公开资料 → 题库/案例（真实 API）---")
_src_url = "http://nyncj.zunyi.gov.cn/nyncgz/jcyj/202605/t20260518_90183060.html"
print("来源：", _src_url)
_res_t = m.import_teaching_file(url=_src_url, with_llm=True, limit=6)
print("导入结果：", {k: v for k, v in _res_t.items() if k != "source"})
_bank_t = m.load_question_bank()
print("题库总量：%d 题，案例：%d 条" % (len(_bank_t), len(m.load_case_library())))
for _q in _bank_t[-4:]:
    print("  Q: %s" % _q["question"][:60])
    if _q["options"]:
        print("     选项：%s" % " / ".join(_q["options"])[:80])
    print("     答案：%s" % str(_q["answer"])[:50])
    print("     来源：%s" % _q["source"][:70])
_cases_t = m.load_case_library()
for _c in _cases_t[-2:]:
    print("  案例：%s" % _c["title"][:50])
    print("     处置：%s" % (_c.get("action") or "")[:80])
print("教学统计：", m.teaching_stats())

# ── 追加验收：地块地图 Agent（真实 API）────────────────────────
print("\n--- 地块地图 Agent：让模型自己调地图工具（真实 API）---")
import shutil as _sh2
_GEO2 = os.path.join(os.path.dirname(m.APP_DIR), "map_geo")
_sh2.rmtree(_GEO2, ignore_errors=True)
os.environ["RICE_GEO_DIR"] = _GEO2
m._GEO_STATE.update({"tried": False, "store": None})
_st2 = m.geo_store()
_now2 = m.datetime.now()
for _p, _d, _c, _cf, _dlon, _dlat in [
        ("东片1号田", 1, "稻瘟病", 0.88, 0.0, 0.0),
        ("东片1号田", 4, "稻瘟病", 0.79, 0.0004, 0.0002),
        ("东片1号田", 11, "稻瘟病", 0.74, 0.0002, 0.0005),
        ("西片3号田", 2, "稻纵卷叶螟", 0.68, 0.0030, 0.0018),
        ("西片3号田", 25, "稻纵卷叶螟", 0.60, 0.0032, 0.0016),
        ("北片7号田", 45, "胡麻斑病", 0.52, 0.0060, 0.0040)]:
    _st2.add_record(shot_at=(_now2 - m.timedelta(days=_d)).strftime("%Y-%m-%d %H:%M:%S"),
                    lon=118.8620 + _dlon, lat=32.0510 + _dlat, coord_src="exif",
                    cls=_c, conf=_cf, plot=_p, note="演示数据")
print("演示地块库：%d 条记录 / %d 个地块" % (_st2.count(), len(_st2.plots())))
_chunks2, _tools2, _done2 = [], [], []
_th2 = m.LLMAnalysisThread("地块地图", None, None, prompt=m.MAP_AGENT_PROMPT)
_th2.chunk.connect(lambda t: _chunks2.append(t))
_th2.tool_used.connect(lambda n, l: (_tools2.append(n), print("   🔧 %s（%s）" % (l, n))))
_th2.result_ready.connect(lambda t: _done2.append(t))
_th2.failed.connect(lambda e: _done2.append("ERR: " + str(e)))
_th2.start()
_l6 = QEventLoop(); _t6 = QTimer(); _t6.setInterval(100)
_t6.timeout.connect(lambda: _l6.quit() if _done2 else None)
_t6.start(); QTimer.singleShot(180000, _l6.quit); _l6.exec_(); _th2.wait(8000)
print("Agent 实际调用的工具：", [t for t in _tools2 if t.startswith(("map_", "plot_", "patrol", "nearby"))] or _tools2)
print("--- 结论 ---")
print((_done2[0] if _done2 else "（无结果）")[:900])

# ── 追加验收：天气与施药窗口（真实 API + 真实天气）────────────────
print("\n--- 天气模块：Agent 自主判断能不能打药（真实 API）---")
_wx = m.fetch_weather(city="南京", days=3)
print("天气来源：%s | 地点：%s %s" % (_wx.get("来源"), _wx.get("地点"), _wx.get("坐标")))
print(m.weather_summary_text(_wx))
print()
_sw = json.loads(m.run_agent_tool("spray_window", {"city": "南京", "hours": 48}))
print("施药窗口：", json.dumps(_sw.get("窗口"), ensure_ascii=False)[:200] or "无")
print("受限原因：", json.dumps(_sw.get("受限原因Top"), ensure_ascii=False))
print()
_q = "南京明天适合打药吗？如果适合，说清楚几点到几点、要注意什么；不适合就说原因。"
_ch7, _tl7, _dn7 = [], [], []
_th7 = m.LLMAnalysisThread("水稻病虫害", None, None, prompt=_q)
_th7.chunk.connect(lambda t: _ch7.append(t))
_th7.tool_used.connect(lambda n, l: (_tl7.append(n), print("   🔧 %s（%s）" % (l, n))))
_th7.result_ready.connect(lambda t: _dn7.append(t))
_th7.failed.connect(lambda e: _dn7.append("ERR: " + str(e)))
_th7.start()
_l7 = QEventLoop(); _t7 = QTimer(); _t7.setInterval(100)
_t7.timeout.connect(lambda: _l7.quit() if _dn7 else None)
_t7.start(); QTimer.singleShot(180000, _l7.quit); _l7.exec_(); _th7.wait(8000)
print("Agent 调用的工具：", _tl7)
print("--- 结论 ---")
print((_dn7[0] if _dn7 else "（无结果）")[:800])

# ── 追加验收：极端天气 + 施药记录与补施（真实 API）──────────────
print("\n--- 极端天气预警（真实预报）---")
_wx2 = m.fetch_weather(city="南京", days=3)
for _a in m.extreme_weather_alerts(_wx2):
    print("  ⚠ %s（%s）：%s → %s" % (_a["类型"], _a["等级"], _a["依据"], _a["应对"]))
else:
    print("  （未来 3 天无极端天气预警）" if not m.extreme_weather_alerts(_wx2) else "")
print("历史小时数据：%d 条（用于药后雨量核算）" % len(_wx2.get("过去逐小时") or []))
print()
print("--- 施药记录 → 雨后冲刷判断（真实降水）---")
m.add_spray_record(plot="东片1号田", drug="吡虫啉", cls="稻飞虱",
                   when=m.datetime.now() - m.timedelta(days=2), note="真机验收")
_checks = m.spray_rain_check(days=7)
for _c in _checks[-3:]:
    print("  %s %s（%s）：药后6h %s mm / 24h %s mm → %s"
          % (_c["地块"], _c["药剂"], (_c["施药时间"] or "")[5:16],
             _c["药后6小时雨量mm"], _c["药后24小时雨量mm"], _c["结论"]))
    print("     建议：%s" % _c["建议"][:110])
print()
print("--- Agent：要不要补施（真实 API）---")
_q8 = "我两天前在东片1号田打了吡虫啉防稻飞虱，这两天有没有被雨冲掉？需不需要补施？"
_ch8, _tl8, _dn8 = [], [], []
_th8 = m.LLMAnalysisThread("水稻病虫害", None, None, prompt=_q8)
_th8.chunk.connect(lambda t: _ch8.append(t))
_th8.tool_used.connect(lambda n, l: (_tl8.append(n), print("   🔧 %s（%s）" % (l, n))))
_th8.result_ready.connect(lambda t: _dn8.append(t))
_th8.failed.connect(lambda e: _dn8.append("ERR: " + str(e)))
_th8.start()
_l8 = QEventLoop(); _t8 = QTimer(); _t8.setInterval(100)
_t8.timeout.connect(lambda: _l8.quit() if _dn8 else None)
_t8.start(); QTimer.singleShot(180000, _l8.quit); _l8.exec_(); _th8.wait(8000)
print("Agent 调用的工具：", _tl8)
print("--- 结论 ---")
print((_dn8[0] if _dn8 else "（无结果）")[:700])

# ── 追加验收：识别 → 按方案施药并登记（闭环，真实 API）────────
print("\n--- 闭环：方案 → 校验 → 登记 → 可追溯 ---")
_plan = m.treatment_plan_rows("稻瘟病")
print("本地库方案药剂：", "、".join(r["名称"] for r in _plan["推荐药剂"])[:80])
print("可选的第一条：%s（%s，%s，间隔期 %s 天）"
      % (_plan["推荐药剂"][0]["名称"], _plan["推荐药剂"][0]["规格"],
         _plan["推荐药剂"][0]["亩用量"], _plan["推荐药剂"][0]["安全间隔期"]))
_pc = m.spray_precheck(cls_name="稻瘟病", plot="东片1号田", drug="三环唑")
print("施药前四条校验 →", _pc["结论"])
for _k in ("农药合规", "天气窗口", "重复施药", "安全间隔期"):
    print("   %s：%s" % (_k, json.dumps(_pc[_k], ensure_ascii=False)[:120]))
_rec = m.add_spray_record(plot="东片1号田", drug="三环唑", cls="稻瘟病", dose="60g/亩",
                          source="detection", safe_days=_pc["安全间隔期"]["天数"],
                          detection={"cls": "稻瘟病", "conf": 0.87, "boxes": 3},
                          plan_from="本地防治库 TREATMENT_DB", note="真机闭环验收")
print("已登记：%s %s（来源 %s，识别依据 %s 置信度 %s）"
      % (_rec["time"], _rec["drug"], _rec["source"],
         _rec["detection"]["cls"], _rec["detection"]["conf"]))
_tx = json.loads(m.run_agent_tool("get_treatment", {"cls_name": "稻瘟病"}))
print("方案工具自动带出近期施药：%s 次 → %s"
      % (_tx.get("近期施药", {}).get("次数"), _tx.get("近期施药", {}).get("提示", "")[:50]))
_sl2 = json.loads(m.run_agent_tool("spray_log", {"days": 7}))
print("施药台账可追溯：", json.dumps(_sl2["雨后冲刷判断"][-1], ensure_ascii=False)[:180])
print()
print("--- Agent：这套方案还要不要再打（真实 API）---")
_q9 = "东片1号田的稻瘟病，我按方案打过一次三环唑了，现在还要不要补打？说清依据。"
_ch9, _tl9, _dn9 = [], [], []
_th9 = m.LLMAnalysisThread("稻瘟病", None, None, prompt=_q9)
_th9.chunk.connect(lambda t: _ch9.append(t))
_th9.tool_used.connect(lambda n, l: (_tl9.append(n), print("   🔧 %s（%s）" % (l, n))))
_th9.result_ready.connect(lambda t: _dn9.append(t))
_th9.failed.connect(lambda e: _dn9.append("ERR: " + str(e)))
_th9.start()
_l9 = QEventLoop(); _t9 = QTimer(); _t9.setInterval(100)
_t9.timeout.connect(lambda: _l9.quit() if _dn9 else None)
_t9.start(); QTimer.singleShot(180000, _l9.quit); _l9.exec_(); _th9.wait(8000)
print("Agent 调用的工具：", _tl9)
print("--- 结论 ---")
print((_dn9[0] if _dn9 else "（无结果）")[:800])

# ── 追加验收：安全间隔期与收获期（真实 API）────────────────────
print("\n--- 安全间隔期与「最早可收日期」---")
print("带间隔期的药剂：%d/19；缺的：%s"
      % (19 - len(m.drug_safety_gaps()), "、".join(g["药剂"] for g in m.drug_safety_gaps())))
for _d in ("三环唑", "吡虫啉", "稻瘟灵", "阿维菌素"):
    print("   %s → %s 天" % (_d, m.drug_safe_days(_d)))
_hv = (m.datetime.now() + m.timedelta(days=18)).strftime("%Y-%m-%d")
m.set_harvest_date("东片1号田", _hv, "真机验收：计划收获期")
print("已登记收获期：东片1号田 %s" % _hv)
for _h in m.earliest_harvest():
    print("   %s｜末次施药 %s（%s）｜间隔期 %s 天｜最早可收 %s｜登记 %s"
          % (_h["地块"], _h.get("末次施药"), _h.get("末次药剂"), _h.get("安全间隔期"),
             _h.get("最早可收日期"), _h.get("登记收获期")))
    print("      结论：%s" % _h["结论"])
print()
print("--- Agent：什么时候能收（真实 API）---")
_q10 = "东片1号田我打了三环唑防治稻瘟病，计划%s收获，能收吗？如果不能，最早什么时候能收？" % _hv
_ch10, _tl10, _dn10 = [], [], []
_th10 = m.LLMAnalysisThread("稻瘟病", None, None, prompt=_q10)
_th10.chunk.connect(lambda t: _ch10.append(t))
_th10.tool_used.connect(lambda n, l: (_tl10.append(n), print("   🔧 %s（%s）" % (l, n))))
_th10.result_ready.connect(lambda t: _dn10.append(t))
_th10.failed.connect(lambda e: _dn10.append("ERR: " + str(e)))
_th10.start()
_l10 = QEventLoop(); _t10 = QTimer(); _t10.setInterval(100)
_t10.timeout.connect(lambda: _l10.quit() if _dn10 else None)
_t10.start(); QTimer.singleShot(180000, _l10.quit); _l10.exec_(); _th10.wait(8000)
print("Agent 调用的工具：", _tl10)
print("--- 结论 ---")
print((_dn10[0] if _dn10 else "（无结果）")[:700])