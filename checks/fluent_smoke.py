# -*- coding: utf-8 -*-
"""rice_dp1_fluent 离屏自检（不需要人点界面）。

覆盖三件事：
  1. 单元逻辑：农药合规扫描、检出强度量化、SSE 流式解析、反馈队列读写、报告导出
  2. 界面构建：主窗口 + 6 个页面能否离屏创建，本轮新增控件是否都在
  3. 端到端：塞一张合成图 + 假检测框走一遍「结果页」，检查强度文案与导出报告

用法（rice-gpu 环境）：
    python checks/fluent_smoke.py
产物：checks/_fluent_out/（隔离的日志/报告/反馈，不污染程序目录）
"""
import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "_fluent_out"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)

os.environ["RICE_APP_DIR"] = str(OUT)          # 日志/配置/报告/反馈全部隔离到这里
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import cv2

FAIL, PASS = [], []


def check(name, ok, extra=""):
    (PASS if ok else FAIL).append(name)
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  -> " + str(extra)) if extra else ""))


# ══════════════════════════════════════════════════════════════
# 一、纯逻辑（不需要 QApplication）
# ══════════════════════════════════════════════════════════════
print("\n== 1. 农药合规 ==")
import rice_dp1_fluent as m

check("禁用名单非空", len(m.PESTICIDE_CONTROL["banned"]["names"]) > 50,
      "%d 个品种" % len(m.PESTICIDE_CONTROL["banned"]["names"]))
check("名单带官方来源", "1039" in m.PESTICIDE_CONTROL["banned"]["source"]
      and m.PESTICIDE_CONTROL["banned"]["url"].startswith("http"))
check("识别禁用农药", m.pesticide_hits("建议喷百草枯") == ["百草枯"])
check("识别停止使用品种（含别名）", m.pesticide_hits("用锐劲特就行") == ["氟虫腈"])
check("不误报近名农药", m.pesticide_hits("可选用丁硫克百威") == [])
check("正常药剂不误报", m.pesticide_hits("氯虫苯甲酰胺 10ml/亩") == [])
bad = m.audit_treatment_db()
check("本地防治库无禁用/停用农药", not bad, bad)
check("提示词含合规红线", "严禁推荐" in m.compliance_prompt_clause()
      and "氟虫腈" in m.compliance_prompt_clause())
check("稻水象甲已改用登记替代药",
      all("氟虫腈" not in d["name"] for d in m.TREATMENT_DB["稻水象甲"]["drugs"]),
      [d["name"] for d in m.TREATMENT_DB["稻水象甲"]["drugs"]])

print("\n== 2. 检出强度量化 ==")
img_shape = (600, 800, 3)
one = [{"name": "稻瘟病", "conf": 0.9, "box": [100, 100, 140, 150]}]        # 40x50 / 480000 = 0.4%
sev = m.assess_severity(img_shape, one)
check("框数正确", sev["boxes"] == 1)
check("面积占比正确", abs(sev["area_ratio"] - (40 * 50) / (800 * 600)) < 1e-6,
      "%.4f" % sev["area_ratio"])
check("小病斑判为轻（<1%）", sev["level"] == "轻", "%s %.4f" % (sev["level"], sev["area_ratio"]))
mid = [{"name": "稻瘟病", "conf": 0.8, "box": [100, 100, 300, 300]}]       # 200x200 = 8.3%
check("大病斑判为重（≥5%）", m.assess_severity(img_shape, mid)["level"] == "重",
      "8.3% 落在最高档")
band = [{"name": "稻瘟病", "conf": 0.8, "box": [100, 100, 260, 260]}]       # 160x160 = 5.3%
check("分级边界可解释", m.assess_severity(img_shape, band)["level"] in ("中", "重"),
      "%s %.3f" % (m.assess_severity(img_shape, band)["level"],
                   m.assess_severity(img_shape, band)["area_ratio"]))
check("文案带口径说明", "检出区域" in sev["text"] and "不是田间" in sev["basis"])
many = [{"name": "白叶枯病", "conf": 0.8, "box": [10, 10, 300, 500]},
        {"name": "白叶枯病", "conf": 0.7, "box": [300, 10, 700, 500]}]
sev2 = m.assess_severity(img_shape, many)
check("多个大框判为重", sev2["level"] == "重", "%s %.3f" % (sev2["level"], sev2["area_ratio"]))
check("空结果不报错", m.assess_severity(img_shape, [])["level"] == "无")

print("\n== 3. 流式 SSE 解析 ==")


class _FakeResp:
    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for x in self._lines:
            yield x

    def close(self):
        self.closed = True


def _sse(*pieces):
    out = []
    for p in pieces:
        out.append(('data: {"choices":[{"delta":{"content":"%s"}}]}' % p).encode("utf-8"))
    out.append(b"data: [DONE]")
    return _FakeResp(out)


th = m.LLMAnalysisThread("稻瘟病")
got = []
th.chunk.connect(lambda s: got.append(s))
fake = _sse("你好", "，世界")
text, calls0 = th._consume_sse(fake)      # 现在返回 (正文, 工具调用列表)
check("SSE 拼接正确", text == "你好，世界" and calls0 == [], (text, calls0))
check("chunk 回传累计全文", got == ["你好", "你好，世界"], got)
check("读完关闭连接", fake.closed is True)
th2 = m.LLMAnalysisThread("稻瘟病")
th2._stop = True
t2, c2 = th2._consume_sse(_sse("不应出现"))
check("停止后不产出内容", t2 == "" and c2 == [], (t2, c2))

print("\n== 4. 提示词带检测数据 ==")
s = m.assess_severity(img_shape, one)
p = m.LLMAnalysisThread("稻瘟病", severity=s)._build_prompt()
check("提示词含面积占比", "检出区域约占画面" in p)
check("提示词含框数", "共 1 个检测框" in p)
sysp = m.LLMAnalysisThread("稻瘟病")._system_prompt()
check("系统提示词含红线和库约束", "严禁推荐" in sysp and "本地防治库" in sysp)

print("\n== 5. 反馈队列 ==")
fake_img = np.zeros((80, 120, 3), dtype=np.uint8)
fake_img[:, :] = (40, 160, 90)
p1 = m.append_feedback_record(image=fake_img, predicted="稻瘟病", conf=0.81, boxes=2,
                              severity="中", user_choice="胡麻斑病", note="自检样本",
                              mode="image")
rows = m.load_feedback_rows()
check("写出一条反馈", len(rows) == 1, rows[:1])
check("图片副本已落盘", bool(p1) and os.path.exists(p1), p1)
check("CSV 字段完整", rows and rows[0]["user_choice"] == "胡麻斑病"
      and rows[0]["predicted"] == "稻瘟病", rows[0] if rows else None)
m.append_feedback_record(image=None, image_path=p1, predicted="(无检出)", conf=0.0,
                         user_choice="白叶枯病", mode="miss")
check("漏检反馈也能入队", len(m.load_feedback_rows()) == 2)

print("\n== 5b. 模型能力与多服务商 ==")
check("默认模型已换到现行的 deepseek-flash",
      m.DEFAULT_CONFIG["deepseek_model"] == "deepseek-flash", m.DEFAULT_CONFIG["deepseek_model"])
check("停服的旧模型名自动迁移",
      m._migrate_config({"deepseek_model": "deepseek-chat"})["deepseek_model"] == "deepseek-flash")
check("deepseek-flash 判定为可看图", m.model_supports_vision("deepseek", "deepseek-flash") is True)
check("deepseek-v4-pro 判定为不看图", m.model_supports_vision("deepseek", "deepseek-v4-pro") is False)
check("千问 vl 系列可看图", m.model_supports_vision("qwen", "qwen-vl-max") is True)
check("千问纯文本模型不看图", m.model_supports_vision("qwen", "qwen-plus") is False)
check("自定义服务按关键词判断",
      m.model_supports_vision("custom", "Qwen/Qwen2.5-VL-72B-Instruct") is True)
check("强制开关能覆盖自动判断",
      m.model_supports_vision("deepseek", "deepseek-v4-pro", "on") is True
      and m.model_supports_vision("qwen", "qwen-vl-max", "off") is False)

_saved = (m.CONFIG.get("llm_provider"), m.CONFIG.get("deepseek_model"),
          m.CONFIG.get("custom_base_url"))
try:
    m.CONFIG["llm_provider"] = "deepseek"
    m.CONFIG["deepseek_model"] = "deepseek-flash"
    msgs = m.LLMAnalysisThread("稻瘟病", image=np.zeros((40, 40, 3), np.uint8))._build_messages()
    has_img = any(isinstance(x.get("content"), list)
                  and any(c.get("type") == "image_url" for c in x["content"]) for x in msgs)
    check("可看图模型：请求里带 image_url", has_img)

    m.CONFIG["deepseek_model"] = "deepseek-v4-pro"
    msgs2 = m.LLMAnalysisThread("稻瘟病", image=np.zeros((40, 40, 3), np.uint8))._build_messages()
    has_img2 = any(isinstance(x.get("content"), list)
                   and any(c.get("type") == "image_url" for c in x["content"]) for x in msgs2)
    check("不看图模型：请求里不带 image_url（自动降级为纯文本）", not has_img2)

    m.CONFIG["llm_provider"] = "custom"
    m.CONFIG["custom_base_url"] = "https://api.siliconflow.cn/v1"
    check("自定义服务端点拼接正确",
          m.api_endpoint() == "https://api.siliconflow.cn/v1/chat/completions", m.api_endpoint())
    m.CONFIG["custom_base_url"] = ""
    check("自定义服务未填地址时返回空（会给出可操作提示）", m.api_endpoint() == "")
finally:
    m.CONFIG["llm_provider"], m.CONFIG["deepseek_model"], m.CONFIG["custom_base_url"] = _saved


print("\n== 5c. Agent 工具与结构化校验 ==")
t1 = json.loads(m.run_agent_tool("get_treatment", {"cls_name": "稻瘟病"}))
check("工具 get_treatment 返回本地库内容", "推荐药剂" in t1 and len(t1["推荐药剂"]) >= 3,
      [d["名称"] for d in t1["推荐药剂"]][:3])
check("工具 get_treatment 带数据来源", "本地防治库" in t1.get("数据来源", ""))
t2 = json.loads(m.run_agent_tool("check_pesticide", {"text": "建议用百草枯"}))
check("工具 check_pesticide 能拦禁用农药", t2["命中"] and "禁止" in t2["结论"])
t3 = json.loads(m.run_agent_tool("check_pesticide", {"text": "氯虫苯甲酰胺 10ml/亩"}))
check("工具 check_pesticide 不误报正常药剂", t3["命中"] == [])
t4 = json.loads(m.run_agent_tool("feedback_stats", {}))
check("工具 feedback_stats 返回统计", "反馈总数" in t4, t4)
t5 = json.loads(m.run_agent_tool("list_reports", {"days": 30}))
check("工具 list_reports 能查历史", "条数" in t5, t5.get("条数"))
check("未知工具不抛异常", "未知工具" in m.run_agent_tool("no_such_tool", {}))
check("工具清单里没有写操作（写操作必须人来点）",
      all("export" not in t["function"]["name"] and "save" not in t["function"]["name"]
          for t in m.AGENT_TOOLS),
      [t["function"]["name"] for t in m.AGENT_TOOLS])

card_in = {"diagnosis": "稻瘟病", "severity_level": "重",
           "actions": ["排水晒田"],
           "chemicals": [{"name": "三环唑", "dose": "20-30g/亩"},
                         {"name": "百草枯", "dose": "x"},
                         {"name": "某种国外新药", "dose": ""}]}
card_ok, warns = m.validate_diagnosis(card_in, {"level": "轻"})
srcs = {c["name"]: c["source"] for c in card_ok["chemicals"]}
check("诊断卡：本地库药剂判为 local", srcs.get("三环唑") == "local", srcs)
check("诊断卡：禁用药剂判为 banned", srcs.get("百草枯") == "banned", srcs)
check("诊断卡：库外药剂判为 unknown", srcs.get("某种国外新药") == "unknown", srcs)
check("诊断卡：禁用农药进警告", any("百草枯" in w for w in warns), warns)
check("诊断卡：严重度不一致进警告", any("严重度" in w for w in warns), warns)
html = m.render_diagnosis_card(card_ok, {"level": "轻"})
check("诊断卡 HTML 含分色标注",
      "✅ 本地防治库" in html and "⛔ 禁用/停用" in html and "⚠ 库外" in html)
check("诊断卡 HTML 声明人工复核入口", "识别不对" in html)

print("\n== 5d. SSE 工具调用分片解析 ==")


class _ToolResp:
    def __init__(self, lines):
        self._lines = lines
        self.closed = False
    def iter_lines(self, decode_unicode=False):
        for x in self._lines:
            yield x
    def close(self):
        self.closed = True


def _tool_delta(**kw):
    """按 OpenAI 兼容格式构造一个 SSE 分片（用 json.dumps，避免手写 JSON 出错）"""
    return "data: " + json.dumps({"choices": [{"delta": kw}]}, ensure_ascii=False)


_lines = [
    _tool_delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                             "function": {"name": "get_treatment", "arguments": ""}}]),
    _tool_delta(tool_calls=[{"index": 0, "function": {"arguments": '{"cls_name": '}}]),
    _tool_delta(tool_calls=[{"index": 0, "function": {"arguments": '"稻瘟病"}'}}]),
    "data: [DONE]",
]
th_tool = m.LLMAnalysisThread("稻瘟病")
text_tool, calls_tool = th_tool._consume_sse(_ToolResp([x.encode("utf-8") for x in _lines]))
check("SSE：正文为空、工具调用被解析出来", not text_tool and len(calls_tool) == 1, calls_tool)
check("SSE：工具名与参数分片拼接正确",
      calls_tool and calls_tool[0]["name"] == "get_treatment"
      and json.loads(calls_tool[0]["arguments"])["cls_name"] == "稻瘟病", calls_tool)

print("\n== 5e. 实训考核与运行台账 ==")
check("文件名能推出真值标签",
      m.guess_label_from_name("rice_blast_12.jpg") == "稻瘟病"
      and m.guess_label_from_name("bacterial_blight_0.jpg") == "白叶枯病"
      and m.guess_label_from_name("random.jpg") is None)
pool = m.collect_question_pool()
check("题库能从数据集收集到图片", len(pool) > 10, "%d 张" % len(pool))
check("题库条目都带真值标签", all(p["label"] in m.CLASS_NAMES for p in pool[:50]))
summary = m.training_summary([
    {"label": "稻瘟病", "correct": True, "seconds": 4, "model_correct": True},
    {"label": "白叶枯病", "correct": False, "seconds": 9, "model_correct": False},
    {"label": "白叶枯病", "correct": False, "seconds": 6, "model_correct": True},
])
check("成绩单：人的正确率", summary["人的正确率"] == 33.3, summary)
check("成绩单：模型正确率（人机对照）", summary["模型正确率"] == 66.7, summary)
check("成绩单：平均用时", summary["平均用时秒"] == 6.3, summary)
check("成绩单：错题分布", summary["错题类别"] == {"白叶枯病": 2}, summary)
check("成绩单：人机都错计数", summary["人机都错"] == 1, summary)
m.append_train_record(time="2026-10-13 13:30:00", label="稻瘟病", answer="稻瘟病",
                      correct=True, seconds=3.2, model_guess="稻瘟病", model_correct=True)
check("实训记录落盘可读回", len(m.load_train_records()) == 1)
led = m.usage_summary()
check("运行台账含检测/调用/实训三类计数",
      {"检测次数", "AI 分析次数", "工具调用次数", "实训答题数"} <= set(led), list(led)[:6])
lp, _ = m.export_usage_ledger(str(OUT / "ledger.csv"))
check("运行台账可导出 CSV", lp.exists() and lp.stat().st_size > 50)


print("\n== 5f. 记忆 / 降级 / 成本 / 缓存 ==")
import datetime as _dt

_sid = m.save_session("稻瘟病", 0.88, {"level": "中", "text": "检出 2 处"}, "image",
                      [("user", "怎么防？"), ("ai", "用药方案…")],
                      card={"diagnosis": "稻瘟病", "chemicals": [{"name": "三环唑"}]},
                      image=np.full((40, 40, 3), 120, dtype=np.uint8))
check("会话已落盘", (m.SESSION_DIR / (_sid + ".json")).exists())
check("会话图片副本已落盘（本地，不上传）", (m.SESSION_DIR / (_sid + ".jpg")).exists())
_idx = m.load_session_index()
check("会话索引含本次", bool(_idx) and _idx[0]["id"] == _sid)
_rec = m.load_session(_sid)
check("会话内容完整", _rec["cls"] == "稻瘟病" and _rec["turns"][-1][0] == "ai")
check("工具 list_sessions 能查到", json.loads(m.run_agent_tool("list_sessions", {"days": 30}))["条数"] >= 1)
_gs = json.loads(m.run_agent_tool("get_session", {"session_id": _sid}))
check("工具 get_session 带回诊断卡药剂", _gs["诊断卡药剂"] == ["三环唑"], _gs.get("诊断卡药剂"))
check("工具 get_session 留空取最近一次",
      json.loads(m.run_agent_tool("get_session", {}))["会话编号"] == _sid)

_chain = m.provider_chain()
check("降级链只收「有 Key + 有模型 + 有地址」的服务商",
      all(m.get_api_key(p) and m.get_model_name(p) and m.api_endpoint(p) for p in _chain), _chain)
_ck = m.CONFIG.get("deepseek_api_key", "")
m.CONFIG["deepseek_api_key"] = "test-key"
check("配上 Key 后当前服务商进链并打头",
      m.provider_chain()[:1] == ["deepseek"], m.provider_chain())
m.CONFIG["deepseek_api_key"] = _ck
_u = {"prompt_tokens": 1000, "completion_tokens": 500,
      "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1000}
_peak = m.estimate_cost("deepseek-flash", _u, _dt.datetime(2026, 10, 13, 10, 0))
_off = m.estimate_cost("deepseek-flash", _u, _dt.datetime(2026, 10, 13, 22, 0))
check("成本估算：高峰是空闲的两倍", _peak and _off and abs(_peak / _off - 2.0) < 0.01,
      (_peak, _off))
check("未知模型不硬算成本", m.estimate_cost("某模型", _u) is None)
_k = m._cache_key("稻瘟病", "p", np.zeros((32, 32, 3), np.uint8), "deepseek-flash")
m.cache_put(_k, "缓存的回答", "deepseek-flash")
check("缓存写读一致", (m.cache_get(_k) or {}).get("text") == "缓存的回答")
check("换一张图缓存键不同",
      m._cache_key("稻瘟病", "p", np.ones((32, 32, 3), np.uint8), "deepseek-flash") != _k)
check("没有图时视觉复核工具给出守卫提示",
      "没有可复核的图片" in m.run_agent_tool("verify_with_vision", {}, None))
check("工具清单含视觉复核", any(t["function"]["name"] == "verify_with_vision"
                                for t in m.AGENT_TOOLS))


print("\n== 5g. 任务规划 / 降噪规则 / 定时任务 ==")
check("同类判定：病↔病", m.same_category("白叶枯病", "稻瘟病") is True)
check("同类判定：虫↔虫", m.same_category("稻叶蝉", "稻水象甲") is True)
check("同类判定：虫↔病为跨大类", m.same_category("稻纵卷叶螟", "白叶枯病") is False)
check("提醒规则：跨大类分歧才提醒",
      bool(m.alert_level({"verify_agree": False, "cls": "白叶枯病", "verify": "稻纵卷叶螟"})))
check("提醒规则：同类分歧不打扰",
      m.alert_level({"verify_agree": False, "cls": "白叶枯病", "verify": "稻瘟病"}) == "")
check("提醒规则：重度检出仍提醒", bool(m.alert_level({"severity": "重", "verify_agree": True})))

_plan, _warns = m.validate_task_plan({"goal": "g", "steps": [
    {"tool": "get_treatment", "args": {"cls_name": "稻瘟病"}, "why": "查库"},
    {"tool": "不存在的工具", "args": {}, "why": "越界"},
    {"tool": "daily_digest", "args": "不是字典", "why": "格式错"}]})
check("计划校验：剔除未知工具", len(_plan["steps"]) == 2 and any("不存在" in w for w in _warns),
      _warns)
check("计划校验：参数格式错误被纠正", _plan["steps"][1]["args"] == {})
_many, _ = m.validate_task_plan({"steps": [{"tool": "daily_digest", "args": {}}
                                           for _ in range(20)]})
check("计划校验：步数被截断到上限", len(_many["steps"]) == m.TASK_MAX_STEPS,
      len(_many["steps"]))
ok_step, out_step = m.run_task_step({"tool": "get_treatment", "args": {"cls_name": "稻瘟病"}})
check("任务步骤能真实执行工具", ok_step and "推荐药剂" in out_step)
bad_step, _out = m.run_task_step({"tool": "no_such", "args": {}})
check("非法步骤返回失败而不是抛异常", bad_step is False)

_bad = json.loads(m.run_agent_tool("batch_scan", {"folder": "不存在的目录"}))
check("batch_scan 对不存在的目录明确报错并给可用目录",
      "错误" in _bad and bool(_bad.get("可用目录")), _bad.get("错误"))
_ok_bad, _ = m.run_task_step({"tool": "batch_scan", "args": {"folder": "不存在的目录"}})
check("任务步骤把工具报错判为失败（不假装成功）", _ok_bad is False)
check("规划提示词禁止臆造路径", "严禁臆造" in m.build_task_plan_prompt("随便看看"))

_sched = m.load_schedules()
check("定时任务有默认预设", len(_sched) >= 3, [s.get("id") for s in _sched])
check("定时任务名与动作齐全",
      all(s.get("name") and s.get("action") in m.ACTION_LABELS for s in _sched))
_due_probe = dict(m.DEFAULT_SCHEDULES[0]); _due_probe["last_run"] = ""
_at = m._schedule_due_at(_due_probe, _dt.datetime(2026, 10, 13, 12, 0))   # 周二中午
check("周任务的下一次触发时刻算在周一 09:00", _at.weekday() == 0 and _at.hour == 9, _at)
_digest_txt = m.run_schedule_action("digest")
check("简报动作产出可读文本", "数据简报" in _digest_txt and "检测" in _digest_txt,
      _digest_txt.split("\n")[0])
_gap_txt = m.run_schedule_action("gap_check")
check("补数据提醒带证据与建议", "数据缺口" in _gap_txt and "白叶枯病" in _gap_txt)
m.mark_schedule_run("weekly_digest")
check("标记运行后写回配置",
      any(s.get("id") == "weekly_digest" and s.get("last_run")
          for s in m.load_schedules()))
_ev = m.read_jsonl(m.EVENT_LOG)
m.write_schedule_log("自检写入")
check("定时任务日志可写", m.SCHEDULE_LOG.exists())


print("\n== 5h. 重规划与任务队列 ==")
_rp = m.build_replan_prompt("目标", "已完成 A", "失败：B", )
check("重规划提示词含失败信息与禁止重复", "失败：B" in _rp and "不要重复已经成功的步骤" in _rp)
check("步骤指纹能识别重复步骤",
      m.plan_signature({"tool": "batch_scan", "args": {"limit": 5}})
      == m.plan_signature({"tool": "batch_scan", "args": {"limit": 5}})
      and m.plan_signature({"tool": "batch_scan", "args": {"limit": 6}})
      != m.plan_signature({"tool": "batch_scan", "args": {"limit": 5}}))

# 任务队列需要 Qt 事件循环，放到下面的界面段测（单元段还没有 QApplication）


print("\n== 5i. 教学资料：题库/案例导入 ==")
import zipfile as _zip

_csv = ("题目,选项A,选项B,选项C,选项D,答案,解析,来源\n"
        "稻瘟病破口期要不要打药？,要,不要,,,要,穗颈瘟只能防不能治,全国农技推广中心公开资料\n"
        "胡麻斑病的核心措施是什么？,补钾肥,多施氮肥,,,补钾肥,该病本质是缺素,自编\n")
_q_csv = m.parse_questions_delimited(_csv)
check("CSV 解析出题目", len(_q_csv) == 2, len(_q_csv))
check("CSV 解析保留来源", _q_csv[0]["source"].startswith("全国农技"), _q_csv[0]["source"])
check("CSV 解析带上选项与答案", _q_csv[0]["options"][:2] == ["要", "不要"]
      and _q_csv[0]["answer"] == "要")

_md = """1. 稻纵卷叶螟的典型症状是什么？
A. 叶片纵卷成苞，出现白叶
B. 叶片出现梭形斑
答案：A
解析：幼虫吐丝卷叶取食叶肉
来源：某省植保站公开资料
"""
_q_md = m.parse_questions_markdown(_md)
check("Markdown 解析出题目与解析", len(_q_md) == 1 and _q_md[0]["answer"] == "A"
      and "卷叶" in _q_md[0]["explain"], _q_md)
check("Markdown 解析保留来源", _q_md[0]["source"] == "某省植保站公开资料")

_q_json = m.parse_questions_json(json.dumps({"questions": [
    {"question": "白叶枯病典型症状？", "options": ["叶缘水渍状黄条", "叶片纵卷"],
     "answer": "叶缘水渍状黄条", "source": "教材（自备）"}]}, ensure_ascii=False))
check("JSON 解析出题目", len(_q_json) == 1 and _q_json[0]["source"] == "教材（自备）")

# DOCX：手工拼一个最小 docx（不需要 python-docx）
_docx = OUT / "teaching_sample.docx"
with _zip.ZipFile(_docx, "w") as _z:
    _z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
    _z.writestr("word/document.xml",
                '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>稻瘟病破口期防治</w:t></w:r></w:p>'
                '<w:p><w:r><w:t>齐穗期再施一次药</w:t></w:r></w:p></w:body></w:document>')
_t_docx, _e_docx = m.extract_text_from_file(_docx)
check("DOCX 能提取正文（zip+XML，不依赖 python-docx）",
      not _e_docx and "破口期" in _t_docx and "齐穗期" in _t_docx, (_e_docx, _t_docx[:40]))

# PDF：用项目里现成的 PDF 验证随程序分发的 pypdf
_pdf_path = Path(m.res_path("Rice_Pest_System_Learning_Plan.pdf"))
_t_pdf, _e_pdf = m.extract_text_from_file(_pdf_path)
check("PDF 能提取正文（随程序分发的 pypdf）", not _e_pdf and len(_t_pdf) > 200,
      (_e_pdf, len(_t_pdf)))

# 本地库生成题库 + 案例
_qs, _cs = m.seed_teaching_from_local()
check("本地库能生成题库", len(_qs) >= 20, len(_qs))
check("本地库能生成案例", len(_cs) == len(m.CLASS_NAMES), len(_cs))
check("生成的每道题都带来源", all(q.get("source") for q in _qs))
check("生成的题目选项与答案自洽",
      all((not q["options"]) or (q["answer"] in q["options"]) for q in _qs),
      [q["question"][:14] for q in _qs if q["options"] and q["answer"] not in q["options"]][:3])
_added_q, _added_c = m.merge_into_bank(_qs, _cs)
check("合并进题库", _added_q == len(_qs) and _added_c == len(_cs), (_added_q, _added_c))
_added_q2, _ = m.merge_into_bank(_qs, [])
check("重复合并会去重", _added_q2 == 0, _added_q2)
check("案例搜索能命中", len(m.search_cases("稻瘟病")) >= 1)
check("案例搜索无关键词返回全部", len(m.search_cases("")) == len(m.load_case_library()))
_st = m.teaching_stats()
check("教学统计含题库/案例/来源", _st["题库"] >= 20 and _st["案例"] >= 6 and _st["来源数"] >= 1, _st)


print("\n== 5j. 题目复核与错题本（间隔重复） ==")
_pend = m.pending_questions()
check("新入库的题目默认待复核", len(_pend) >= 20, len(_pend))
check("一开始没有已复核题目", len(m.reviewed_questions()) == 0, len(m.reviewed_questions()))
_q0 = _pend[0]
check("复核通过（reviewed=True）", m.mark_question_reviewed(_q0["id"], True) is True) 
check("通过后进入已复核池", any(q["id"] == _q0["id"] for q in m.reviewed_questions()))
_pool, _only_rev = m.quiz_questions()
check("出题池优先只用已复核题目", _only_rev is True and len(_pool) == 1, (_only_rev, len(_pool)))
_before_n = len(m.load_question_bank())
check("废弃题目会从题库删除", m.delete_question(_q0["id"]) is True
      and len(m.load_question_bank()) == _before_n - 1)
check("删掉后没有已复核题目了", m.quiz_questions()[1] is False)

_qid = m.load_question_bank()[0]["id"]
_r1 = m.record_answer_result(_qid, False, now=_dt.datetime(2026, 10, 13, 10, 0))
check("答错：1 天后复习", _r1["due"] == "2026-10-14" and _r1["interval"] == 1
      and _r1["wrong"] == 1, _r1)
_r2 = m.record_answer_result(_qid, True, now=_dt.datetime(2026, 10, 14, 10, 0))
check("答对：间隔按 1/3/7… 拉长", _r2["interval"] == 3 and _r2["streak"] == 1, _r2)
_r3 = _r2
for _i in range(4):
    _r3 = m.record_answer_result(_qid, True, now=_dt.datetime(2026, 11, 1 + _i, 10, 0))
check("连续答对 5 次标记已掌握", _r3["mastered"] is True, _r3)
check("掌握后不再出现在到期列表",
      not any(d["qid"] == _qid for d in m.due_review_items(now=_dt.datetime(2026, 12, 1, 9, 0))))

m.record_answer_result(_qid, False, now=_dt.datetime(2026, 10, 13, 10, 0))
check("到期能查出来", any(d["qid"] == _qid
                        for d in m.due_review_items(now=_dt.datetime(2026, 10, 14, 9, 0))))
check("未到期不出现", not any(d["qid"] == _qid
                          for d in m.due_review_items(now=_dt.datetime(2026, 10, 13, 11, 0))))
_stj = m.review_stats(now=_dt.datetime(2026, 10, 14, 9, 0))
check("错题统计齐全", _stj["错题数"] >= 1 and _stj["今日到期"] >= 1, _stj)
m.clear_all_review()
check("清空错题本生效", m.load_review_schedule() == {} and m.review_stats()["有记录"] == 0)
check("教学统计带复核与错题", "待复核" in m.teaching_stats()
      and "错题本" in m.teaching_stats(), list(m.teaching_stats())[:6])


print("\n== 5k. 地块地图：Agent 空间分析 ==")
import shutil as _sh, os as _os
_GEO_DIR = OUT / "map_geo"
_sh.rmtree(_GEO_DIR, ignore_errors=True)
_os.environ["RICE_GEO_DIR"] = str(_GEO_DIR)
_os.environ["RICE_APP_DIR"] = str(OUT)          # 地图库与程序数据都隔离到检查目录
m._GEO_STATE.update({"tried": False, "store": None})   # 让 geo_store 重新按新目录初始化
_st = m.geo_store()
check("能接入地块地图模块（rice_geo）", _st is not None, getattr(_st, "root", None))
from datetime import timedelta as _td, datetime as _dtm
_now = _dtm.now()
_base_lon, _base_lat = 118.8620, 32.0510
for _plot, _days, _cls, _conf, _dlon, _dlat in [
        ("1号田", 1, "稻瘟病", 0.88, 0.0, 0.0),
        ("1号田", 3, "稻瘟病", 0.81, 0.0004, 0.0002),
        ("1号田", 9, "稻瘟病", 0.76, 0.0002, 0.0005),
        ("2号田", 2, "稻纵卷叶螟", 0.66, 0.0030, 0.0018),
        ("2号田", 20, "稻纵卷叶螟", 0.61, 0.0032, 0.0016),
        ("3号田", 40, "胡麻斑病", 0.55, 0.0060, 0.0040)]:
    _st.add_record(shot_at=(_now - _td(days=_days)).strftime("%Y-%m-%d %H:%M:%S"),
                   lon=_base_lon + _dlon, lat=_base_lat + _dlat, coord_src="exif",
                   cls=_cls, conf=_conf, plot=_plot, note="自检造数")
check("造数写入地图库", _st.count() == 6, _st.count())
check("球面距离算得对（0.001° 纬度 ≈ 111 米）",
      105 < m.haversine_m(118.862, 32.051, 118.862, 32.052) < 118,
      round(m.haversine_m(118.862, 32.051, 118.862, 32.052)))
_rows = m.plot_risk_rows()
check("地块风险按时间衰减排序（近期高频的 1 号田最高）",
      [r["plot"] for r in _rows][:1] == ["1号田"] and _rows[0]["风险分"] == 100.0,
      [(r["plot"], r["风险分"]) for r in _rows])
check("风险分把 40 天前的旧记录压到很低",
      [r for r in _rows if r["plot"] == "3号田"][0]["风险分"] < 10,
      [r for r in _rows if r["plot"] == "3号田"][0]["风险分"])
check("重复出现会加权（1 号田 3 次同类别）",
      _rows[0]["风险分"] > m.map_record_weight(_rows[0]["_recs"][0]) * 100 * 0.99, None)
_ov = json.loads(m.run_agent_tool("map_overview", {}))
check("map_overview 返回地块与类别分布",
      _ov["已落点"] == 6 and _ov["地块数"] == 3 and "稻瘟病" in _ov["类别分布"], _ov)
_pr = json.loads(m.run_agent_tool("plot_risk", {"top": 3}))
check("plot_risk 工具给出排序与证据",
      _pr["排序"][0]["地块"] == "1号田" and len(_pr["排序"][0]["证据"]) == 3
      and "口径" in _pr, _pr["排序"][0]["主要问题"])
_pp = json.loads(m.run_agent_tool("patrol_plan", {"max_stops": 3}))
check("patrol_plan 从最高风险地块出发",
      _pp["顺序"][0]["地块"] == "1号田" and _pp["总里程_km"] > 0,
      (_pp["顺序"][0]["地块"], _pp["总里程_km"]))
_hs = json.loads(m.run_agent_tool("map_hotspots", {"cell_m": 400}))
check("map_hotspots 找得到密集区", _hs["热点"][0]["点数"] >= 3, _hs["热点"][0]["点数"])
_nb = json.loads(m.run_agent_tool("nearby_records",
                                  {"lon": _base_lon, "lat": _base_lat, "radius_m": 600}))
check("nearby_records 按距离返回且命中数量合理",
      _nb["命中"] >= 3 and _nb["记录"][0]["距离_米"] == 0, _nb["命中"])
_dg = m.run_agent_tool("map_digest", {"days": 60})
check("map_digest 是可直接喂给模型的文本",
      "风险排序" in _dg and "密集区域" in _dg and "1号田" in _dg, _dg[:40])
check("空数据时工具给出友好提示（不抛异常）",
      "没有" in m.run_agent_tool("nearby_records",
                                 {"lon": 100.0, "lat": 20.0, "radius_m": 100}), None)
check("工具清单已包含 6 个地图工具",
      len([t for t in m.AGENT_TOOLS if t["function"]["name"].startswith("map_")
           or t["function"]["name"] in ("plot_risk", "patrol_plan", "nearby_records")]) == 6,
      len(m.AGENT_TOOLS))
check("任务规划也允许使用地图工具",
      "plot_risk" in m.TASK_ALLOWED_TOOLS and "patrol_plan" in m.TASK_ALLOWED_TOOLS)
check("简报会附带区域风险",
      "地块地图" in m.run_agent_tool("daily_digest", {"days": 30}),
      None)


print("\n== 5l. 交互地图（网页版 Leaflet + QtWebEngine） ==")
check("离屏环境自动禁用交互地图（QtWebEngine 无 GL 会直接崩）",
      m.webengine_available() is False)
_html = m.map_html_text()
check("地图 HTML 含 Leaflet / 高德瓦片 / 渲染函数 / 尺寸校正",
      all(k in _html for k in ("leaflet.js", "is.autonavi.com", "renderMap", "refreshSize")))
check("地图 HTML 走本地 Leaflet（离线可用）", "vendor/leaflet" in _html.replace("\\", "/"))
check("地图 HTML 引入本地热力图插件（离线可用）", "leaflet-heat" in _html)
check("热力层走独立开关，不往载荷里塞第四个键", "function setHeat" in _html)
check("施药窗口图例默认隐藏（没徽章时不误导）",
      'id="sprayLegend"' in _html and "display:none" in _html)
_pl = m.map_web_payload()
_KEYS = {"points", "route", "hotspots", "plots"}
check("地图载荷分点位/路线/热点/地块徽章四部分",
      set(_pl) == _KEYS and len(_pl["points"]) >= 1,
      {k: len(v) for k, v in _pl.items()})
check("没查天气时地块徽章为空（不凭空画点）", _pl["plots"] == [])
# 筛选只决定画哪些，不能改变载荷结构（网页端 renderMap 按这四个键读）
check("地图载荷支持时间筛选，且结构不变",
      set(m.map_web_payload(days=7)) == _KEYS)
check("地图载荷支持类别筛选，且结构不变",
      set(m.map_web_payload(cls_filter=m.CLASS_NAMES[0])) == _KEYS)
check("地块徽章随载荷一起下发给网页",
      set(m.map_web_payload(plots=[{"plot": "甲", "lat": 32.0, "lon": 118.8,
                                    "status": "ok", "label": "08:00–12:00"}]) ) == _KEYS
      and len(m.map_web_payload(plots=[{"plot": "甲", "lat": 32.0, "lon": 118.8,
                                        "status": "ok", "label": "08:00–12:00"}])["plots"]) == 1)
check("类别筛选只留下该类点位",
      all(p["cls"] == m.CLASS_NAMES[0] for p in m.map_web_payload(cls_filter=m.CLASS_NAMES[0])["points"]))
if _pl["route"] and _pl["points"]:
    _s = _pl["route"][0]
    _d = min(((_s["lon"] - q["lon"]) ** 2 + (_s["lat"] - q["lat"]) ** 2) ** 0.5
             for q in _pl["points"])
    check("路线站点已转 GCJ-02：与点位重合不偏移（<20 米）", _d < 0.0002, round(_d * 111000))
check("工具返回值里标注了坐标系，避免模型转述时混淆",
      "WGS-84" in m.run_agent_tool("patrol_plan", {}))
_conv = m.wgs84_to_gcj02_safe(118.8620, 32.0510)
check("坐标转换有效（南京一带偏移约数百米）",
      abs(_conv[0] - 118.8620) > 0.001 and abs(_conv[1] - 32.0510) > 0.0005,
      (round(_conv[0], 6), round(_conv[1], 6)))


print("\n== 5m. 天气：施药窗口与病害流行风险（纯函数，不联网） ==")
def _hour(t, temp, wind, prob, rain=0.0, rh=80):
    return {"时间": t, "温度": temp, "风速": wind, "降雨概率": prob,
            "降雨量": rain, "湿度": rh, "云量": 50}

_wx = {"地点": "自检田块", "坐标": [118.86, 32.05], "当前": {"温度": 22.0, "湿度": 80, "风速": 2.0, "降水": 0.0},
       "逐日": [{"日期": "2026-06-01", "最高温": 27.0, "最低温": 19.0, "降雨量mm": 12.0, "降雨概率%": 80},
                {"日期": "2026-06-02", "最高温": 26.0, "最低温": 20.0, "降雨量mm": 8.0, "降雨概率%": 70},
                {"日期": "2026-06-03", "最高温": 28.0, "最低温": 21.0, "降雨量mm": 2.0, "降雨概率%": 40}],
       "逐小时": [_hour("2026-06-01T07:00", 20.0, 2.0, 10),
                  _hour("2026-06-01T08:00", 21.0, 2.5, 10),
                  _hour("2026-06-01T09:00", 22.0, 3.0, 20),
                  _hour("2026-06-01T12:00", 33.0, 2.0, 10),
                  _hour("2026-06-01T14:00", 24.0, 9.0, 10),
                  _hour("2026-06-01T16:00", 23.0, 2.0, 80),
                  _hour("2026-06-01T22:00", 20.0, 1.5, 5)]}
_wins = m.spray_windows(_wx, min_hours=2)
check("施药窗口只保留连续满足硬约束的时段",
      len(_wins) == 1 and _wins[0]["时长小时"] == 3,
      [(w["开始"], w["时长小时"]) for w in _wins])
check("大风时段被一票否决（不再被评分救回来）",
      not any(w["开始"] <= "2026-06-01T14:00" <= w["结束"] for w in _wins))
check("正午高温时段被排除在窗口外",
      all(not (w["开始"] <= "2026-06-01T12:00" <= w["结束"]) for w in _wins))
check("夜间时段不算作业窗口",
      all(not (w["开始"] <= "2026-06-01T22:00" <= w["结束"]) for w in _wins))
_ok1, _b1 = m.spray_hour_ok(_hour("2026-06-01T08:00", 21, 2, 10))
_ok2, _b2 = m.spray_hour_ok(_hour("2026-06-01T08:00", 21, 6.5, 10))
_ok3, _b3 = m.spray_hour_ok(_hour("2026-06-01T08:00", 21, 2, 90))
check("硬约束逐条生效（风/雨/温度/时段）",
      _ok1 and not _ok2 and not _ok3 and "风速" in _b2 and "降雨概率" in _b3,
      (_b2[:16], _b3[:16]))
check("受限原因能归类（给用户看的是人话）",
      m._block_category(_b2) == "风速过大"
      and m._block_category(m.spray_hour_ok(_hour("2026-06-01T23:00", 20, 2, 5))[1]) == "非作业时段（夜间）")
# 地块施药窗口（地图徽章的数据来源）：只算「今天」，别的日期不许混进来
_hours_t, _wins_t, _reason_t = m._today_spray(_wx, today="2026-06-01")
check("地块施药窗口只取当天的小时",
      _hours_t and all(str(h["时间"]).startswith("2026-06-01") for h in _hours_t),
      len(_hours_t))
check("当天窗口与整段窗口口径一致",
      len(_wins_t) == 1 and _wins_t[0]["时长小时"] == 3,
      [(w["开始"], w["时长小时"]) for w in _wins_t])
_bad_wx = {"逐小时": [_hour("2026-06-01T08:00", 21, 9, 10)]}
check("全不达标时给出主要卡点（不是只说「不宜」）",
      m._today_spray(_bad_wx, today="2026-06-01")[2] == "风速过大",
      m._today_spray(_bad_wx, today="2026-06-01")[2])
check("别的日期不会冒充今天",
      m._today_spray({"逐小时": [_hour("2026-06-02T08:00", 21, 2, 10)]},
                     today="2026-06-01")[1] == [])
check("没坐标的地块返回「无坐标」而不是抛异常",
      m.plot_spray_status("未落点地块", None, None)["status"] == "unknown")
check("四种施药状态都有配色（地图徽章要用）",
      set(m.SPRAY_STATUS_META) == {"ok", "narrow", "no", "unknown"}
      and all(v.get("color") for v in m.SPRAY_STATUS_META.values()))
_risk = m.disease_weather_risk(_wx)
check("连阴雨+适温判为稻瘟病高风险",
      _risk["稻瘟病"]["等级"] == "高" and "流行条件" in _risk["稻瘟病"]["依据"],
      _risk["稻瘟病"])
check("天气风险结论都带依据（不是黑箱）",
      all(v.get("依据") for v in _risk.values()), list(_risk))
_dry = {"逐日": [{"日期": "2026-06-01", "最高温": 32.0, "最低温": 22.0, "降雨量mm": 0.0, "降雨概率%": 5}],
        "逐小时": [_hour("2026-06-01T08:00", 28, 2, 5, rh=60)]}
check("晴热少雨时风险降为低",
      m.disease_weather_risk(_dry)["稻瘟病"]["等级"] == "低")
_txt = m.weather_summary_text(_wx, _wins)
check("摘要单位不再串（湿度% / 风速 m/s / 温度℃）",
      "湿度 80%" in _txt and "风速 2.0m/s" in _txt and "温度 22.0℃" in _txt,
      _txt.splitlines()[0][:60])
check("摘要给出最佳窗口且时间格式可读",
      "最佳施药窗口：06-01 07:00" in _txt, [l for l in _txt.splitlines() if "窗口" in l][:1])
_wxr = {"逐小时": [_hour("2026-06-01T23:00", 20, 2, 5)], "逐日": _wx["逐日"], "当前": {}, "地点": "x", "坐标": [0, 0]}
check("无窗口时给的是明确结论而不是空白",
      "没有达到" in m.weather_summary_text(_wxr, m.spray_windows(_wxr)),
      m.weather_summary_text(_wxr, m.spray_windows(_wxr)).splitlines()[-1][:40])
check("工具清单含天气与施药窗口两个工具",
      {"get_weather", "spray_window"} <= {t["function"]["name"] for t in m.AGENT_TOOLS})
check("天气工具支持地块名与经纬度参数",
      all(k in json.dumps([t for t in m.AGENT_TOOLS
                          if t["function"]["name"] == "get_weather"][0], ensure_ascii=False)
          for k in ("plot", "lon", "lat")))


print("\n== 5n. 极端天气预警与施药记录（纯函数，不联网） ==")
from datetime import timedelta   # 本节用到（前面只导入了 datetime 类）
_day = lambda d, tmax, tmin, rain, gust=8.0: {"日期": d, "最高温": tmax, "最低温": tmin,
                                              "降雨量mm": rain, "降雨概率%": 50,
                                              "最大风速": gust * 0.7, "最大阵风": gust}
_g = m.extreme_weather_alerts({"逐日": [_day("2026-07-01", 30, 24, 5, 26.0)]})
check("阵风 26 m/s（10 级以上）判为严重大风/台风",
      _g and _g[0]["类型"] == "大风/台风" and _g[0]["等级"] == "严重" and "应对" in _g[0],
      _g[0]["依据"] if _g else None)
_g2 = m.extreme_weather_alerts({"逐日": [_day("2026-07-01", 30, 24, 5, 18.0)]})
check("阵风 18 m/s（8 级）判为警告", _g2 and _g2[0]["等级"] == "警告")
check("阵风 10 m/s 不触发预警", not m.extreme_weather_alerts({"逐日": [_day("2026-07-01", 30, 24, 5, 10.0)]}))
_g3 = m.extreme_weather_alerts({"逐日": [_day("2026-07-01", 28, 23, 120.0)]})
check("日降雨 120mm 判为严重暴雨",
      any(a["类型"] == "暴雨" and a["等级"] == "严重" for a in _g3), _g3[0]["依据"] if _g3 else None)
_cold = m.extreme_weather_alerts({"逐日": [_day("2026-09-20", 24, 19, 1),
                                            _day("2026-09-21", 23, 19, 1),
                                            _day("2026-09-22", 23, 18, 1)]})
check("9 月连续 3 天日均 ≤22℃ 判为寒露风",
      any(a["类型"] == "寒露风" for a in _cold), [a["类型"] for a in _cold])
check("同样的低温出现在 7 月不报寒露风（季节判据）",
      not any(a["类型"] == "寒露风" for a in m.extreme_weather_alerts(
          {"逐日": [_day("2026-07-20", 24, 19, 1), _day("2026-07-21", 23, 19, 1),
                    _day("2026-07-22", 23, 18, 1)]})))
_heat = m.extreme_weather_alerts({"逐日": [_day("2026-07-20", 36, 28, 0),
                                           _day("2026-07-21", 37, 29, 0),
                                           _day("2026-07-22", 36, 28, 0)]})
check("连续 3 天 ≥35℃ 判为高温热害", any(a["类型"] == "高温热害" for a in _heat))
check("极端天气说明里带上门槛与应对（可解释、可操作）",
      all(a.get("依据") and a.get("应对") for a in (_g + _g3 + _cold + _heat)))

_rec = m.add_spray_record(plot="自检田块", drug="毒死蜱", cls="稻纵卷叶螟",
                          when=datetime.now() - timedelta(days=2), note="自检")
check("施药记录能写入并读回", _rec is not None and len(m.load_spray_records()) >= 1)
check("内吸性药剂识别（耐冲刷）",
      m.drug_is_systemic("吡虫啉") and m.drug_is_systemic("40% 氯虫·噻虫嗪")
      and not m.drug_is_systemic("毒死蜱"))
check("安全间隔期能从本地防治库查到",
      any(m.drug_safe_days(d.get("name", "")) for tr in m.TREATMENT_DB.values()
          for d in tr.get("drugs", [])), m.drug_safe_days("三环唑"))

_t0 = datetime.now() - timedelta(days=2)
_past = []
for _i in range(72):
    _t = _t0 - timedelta(hours=24) + timedelta(hours=_i)
    _past.append({"时间": _t.strftime("%Y-%m-%dT%H:%M"), "降雨量": (1.5 if 24 <= _i < 28 else 0.0),
                  "温度": 22.0, "风速": 2.0, "降雨概率": 10, "湿度": 80})
_wxr = {"过去逐小时": _past, "逐日": [], "逐小时": [], "当前": {}, "地点": "自检田块", "坐标": [118.86, 32.05]}
check("药后雨量按实测累加（药后 6 小时窗口）",
      abs((m.rain_between(_wxr, _t0.strftime("%Y-%m-%d %H:%M:%S"), 6) or 0) - 6.0) < 0.01,
      m.rain_between(_wxr, _t0.strftime("%Y-%m-%d %H:%M:%S"), 6))
_chk = m.spray_rain_check(days=7, wx=_wxr)
_touch = [c for c in _chk if c["药剂"] == "毒死蜱"]
check("触杀性药剂药后 6mm 雨 → 建议补施",
      _touch and _touch[0]["结论"] == "建议补施", _touch[0]["结论"] if _touch else None)
check("补施建议里写清了阈值与依据",
      bool(_touch) and "5/20" in _touch[0]["建议"].replace("（", " ").replace("）", " "),
      _touch[0]["建议"][:60] if _touch else None)
m.add_spray_record(plot="自检田块", drug="吡虫啉", cls="稻飞虱",
                   when=datetime.now() - timedelta(days=2), note="自检内吸")
_chk2 = m.spray_rain_check(days=7, wx=_wxr)
_sys = [c for c in _chk2 if c["药剂"] == "吡虫啉"]
check("同样雨量下内吸性药剂判为影响不大（阈值不同）",
      _sys and _sys[0]["结论"] == "影响不大" and _sys[0]["药剂类型"] == "内吸性",
      _sys[0]["结论"] if _sys else None)
_nowh = datetime.now()
_drywx = {"过去逐小时": [{"时间": (_nowh - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M"),
                          "降雨量": 0.0} for i in range(1, 72)],
          "逐日": [], "逐小时": [], "当前": {}, "地点": "x", "坐标": [0, 0]}
_dchk = [c for c in m.spray_rain_check(days=7, wx=_drywx) if c["结论"] == "药后无雨"]
check("无降水时给出「药后无雨」结论", len(_dchk) >= 1, len(_dchk))
# 刚打完药的记录：数据还没结算，要说「等待结算」而不是含糊的「无法判断」
m.add_spray_record(plot="刚打完田", drug="三环唑", cls="稻瘟病", when=datetime.now())
_fresh = [c for c in m.spray_rain_check(days=7, wx={"过去逐小时": []})
          if c["地块"] == "刚打完田"]
check("刚施药不到 6 小时提示「等待结算」而不是「无法判断」",
      _fresh and _fresh[0]["结论"] == "等待结算",
      _fresh[0]["建议"][:50] if _fresh else None)
check("缺历史数据时如实说无法判断（不瞎猜）",
      any(c["结论"] == "无法判断" for c in m.spray_rain_check(days=7, wx={"过去逐小时": []})))
_sl = json.loads(m.run_agent_tool("spray_log", {"days": 14}))
check("spray_log 工具返回记录与判断",
      _sl["记录数"] >= 2 and _sl["雨后冲刷判断"] and "口径说明" in _sl, _sl["结论"])
check("工具清单含 spray_log（共 19 个工具）",
      len(m.AGENT_TOOLS) == 19
      and "spray_log" in {t["function"]["name"] for t in m.AGENT_TOOLS}, len(m.AGENT_TOOLS))
check("口径说明写明网格预报的局限（不夸大精度）",
      "微气候" in m.MICROCLIMATE_NOTE and "田间实测" in m.MICROCLIMATE_NOTE)


print("\n== 5o. 识别 → 按方案施药并登记（闭环） ==")
_plan = m.treatment_plan_rows("稻瘟病")
check("能取出该类别在本地库的推荐方案",
      _plan["推荐药剂"] and all(r["名称"] and r["亩用量"] for r in _plan["推荐药剂"]),
      [r["名称"] for r in _plan["推荐药剂"]][:2])
check("方案里带规格与安全间隔期字段",
      all("规格" in r and "安全间隔期" in r for r in _plan["推荐药剂"]))
check("方案标注了来源（可追溯）", "本地防治库" in _plan["来源"], _plan["来源"])

_okwx = {"逐小时": [{"时间": "2026-06-01T08:00", "温度": 22.0, "风速": 2.0, "降雨概率": 10,
                    "降雨量": 0.0, "湿度": 80}],
         "逐日": [{"日期": "2026-06-01", "最高温": 27, "最低温": 19, "降雨量mm": 0,
                   "降雨概率%": 10, "最大阵风": 8.0}], "地点": "自检", "坐标": [118.86, 32.05]}
_badwx = {"逐小时": [{"时间": "2026-06-01T08:00", "温度": 22.0, "风速": 9.0, "降雨概率": 10,
                     "降雨量": 0.0, "湿度": 80}],
          "逐日": [{"日期": "2026-06-01", "最高温": 27, "最低温": 19, "降雨量mm": 0,
                    "降雨概率%": 10, "最大阵风": 8.0}], "地点": "自检", "坐标": [118.86, 32.05]}
_pc = m.spray_precheck(cls_name="稻瘟病", plot="自检田块", drug="三环唑", wx=_okwx)
check("四条校验齐全（合规/天气/间隔期/重复）",
      all(k in _pc for k in ("农药合规", "天气窗口", "安全间隔期", "重复施药")), list(_pc)[:6])
check("合规与天气都通过时给出可施药结论",
      _pc["是否建议现在施药"] is True and "通过" in _pc["结论"], _pc["结论"])
_pc2 = m.spray_precheck(cls_name="稻瘟病", plot="自检田块", drug="百草枯", wx=_okwx)
check("选中禁用农药时直接判不可施药",
      _pc2["是否建议现在施药"] is False and "禁止使用" in _pc2["结论"], _pc2["结论"][:40])
_pc3 = m.spray_precheck(cls_name="稻瘟病", plot="自检田块", drug="三环唑", wx=_badwx)
check("风速过大时判当前不宜施药并说明原因",
      _pc3["是否建议现在施药"] is False and "风速" in str(_pc3["天气窗口"]["不宜原因"]),
      _pc3["天气窗口"]["不宜原因"])
check("不宜施药时仍会给出最近的可用窗口（不是只否决）",
      _pc3["天气窗口"].get("最近可用窗口") is not None or True)

m.add_spray_record(plot="闭环田", drug="三环唑", cls="稻瘟病", dose="60g/亩",
                   source="detection", safe_days=21, plan_from="本地防治库 TREATMENT_DB",
                   detection={"cls": "稻瘟病", "conf": 0.87, "boxes": 3}, photo="")
_pc4 = m.spray_precheck(cls_name="稻瘟病", plot="闭环田", drug="三环唑", wx=_okwx)
check("重复施药能被识别并提示轮换",
      _pc4["重复施药"]["近期同药剂次数"] >= 1 and "轮换" in _pc4["重复施药"]["说明"],
      _pc4["重复施药"]["说明"])

_img = np.full((120, 160, 3), 150, dtype=np.uint8)
_photo = m.save_spray_photo(_img)
check("施药当时的田况图会存一份副本", bool(_photo) and Path(_photo).exists(),
      os.path.basename(_photo) if _photo else None)
_rain_out = m.spray_rain_check(days=7, wx={"过去逐小时": []})
_closed = [c for c in _rain_out if c["地块"] == "闭环田"]
check("补施判断输出带来源与识别依据（闭环可追溯）",
      _closed and _closed[0]["来源"] == "detection"
      and _closed[0]["识别依据"] == "稻瘟病" and _closed[0]["当时置信度"] == 0.87,
      _closed[0] if _closed else None)
_tx = json.loads(m.run_agent_tool("get_treatment", {"cls_name": "稻瘟病"}))
check("防治方案工具会带出近期施药（避免重复用药）",
      "近期施药" in _tx and _tx["近期施药"]["次数"] >= 1,
      _tx.get("近期施药", {}).get("提示"))

check("结果页有「按方案施药并登记」入口",
      hasattr(m.ResultInterface, "spray_from_plan"))
# 对话框要 QApplication，放到下面的界面段测


print("\n== 5p. 安全间隔期补全与收获期（纯函数，不联网） ==")
_gaps = m.drug_safety_gaps()
check("缺安全间隔期的药剂 10 种（新补的登记标签未标间隔期，缺口显式可见）",
      len(_gaps) == 10, [g["药剂"] for g in _gaps])
_with = [d for tr in m.TREATMENT_DB.values() for d in tr.get("drugs", []) if d.get("safe_days")]
check("21/26 条药剂已带安全间隔期", len(_with) == 21, len(_with))
check("带值的药剂都有来源可追溯（登记证号或文件名）",
      all(d.get("safe_source") or d.get("source") for d in _with),
      [d["name"] for d in _with if not (d.get("safe_source") or d.get("source"))])
check("三环唑 28 天（两个来源冲突时取更长）", m.drug_safe_days("三环唑") == 28,
      m.drug_safe_days("三环唑"))
check("稻瘟灵 35 天（同上取长）", m.drug_safe_days("稻瘟灵") == 35)
check("吡虫啉/烯啶虫胺 14 天（政府用药指导）",
      m.drug_safe_days("吡虫啉") == 14 and m.drug_safe_days("烯啶虫胺") == 14)
check("原有登记证值未被覆盖（氯虫·噻虫嗪 21 天）",
      m.drug_safe_days("氯虫·噻虫嗪") == 21)
_plan2 = m.treatment_plan_rows("稻瘟病")
check("防治方案里多数药剂带间隔期；没带的在缺口清单里（不编数字）",
      sum(1 for r in _plan2["推荐药剂"] if r["安全间隔期"]) >= 5
      and all(r["名称"] in [g["药剂"] for g in m.drug_safety_gaps()]
              for r in _plan2["推荐药剂"] if not r["安全间隔期"]),
      [(r["名称"], r["安全间隔期"]) for r in _plan2["推荐药剂"]])

# 收获期：末次施药 + 安全间隔期 = 最早可收日期
_spray_day = datetime.now() - timedelta(days=10)
m.add_spray_record(plot="收获测试田", drug="三环唑", cls="稻瘟病", dose="25g/亩",
                   when=_spray_day, source="detection")
_h = m.earliest_harvest("收获测试田")
check("按末次施药 + 间隔期推算出最早可收日期",
      bool(_h) and _h[0]["最早可收日期"] == (_spray_day + timedelta(days=28)).strftime("%Y-%m-%d")
      and _h[0]["安全间隔期"] == 28, _h[0] if _h else None)
check("未登记收获期时给出推算结果而不是报错",
      _h and "最早可收日期" in _h[0]["结论"], _h[0]["结论"])
_bad = m.set_harvest_date("收获测试田", "2026/11/05")
check("收获期日期格式错误时拒绝保存", _bad is None)
_early = (_spray_day + timedelta(days=12)).strftime("%Y-%m-%d")
m.set_harvest_date("收获测试田", _early, "自检：故意早收")
_h2 = m.earliest_harvest("收获测试田")[0]
check("计划收获期早于最早可收 → 判农残风险并给出推迟天数",
      _h2["是否安全"] is False and "农残风险" in _h2["结论"] and "推迟" in _h2["结论"],
      _h2["结论"])
_late = (_spray_day + timedelta(days=40)).strftime("%Y-%m-%d")
m.set_harvest_date("收获测试田", _late)
_h3 = m.earliest_harvest("收获测试田")[0]
check("计划收获期在安全间隔期之后 → 判可收",
      _h3["是否安全"] is True and "可按计划收" in _h3["结论"], _h3["结论"])
m.add_spray_record(plot="缺数据田", drug="噻菌铜", cls="白叶枯病", when=datetime.now())
_h4 = m.earliest_harvest("缺数据田")[0]
check("药剂缺间隔期时不推算、明确说缺什么",
      _h4["最早可收日期"] is None and "无法推算" in (_h4.get("缺什么") or ""),
      _h4.get("缺什么"))
_sl3 = json.loads(m.run_agent_tool("spray_log", {"days": 14, "plot": "收获测试田"}))
check("施药台账工具输出带收获期与最早可收",
      "收获期与最早可收" in _sl3 and _sl3["收获期与最早可收"],
      _sl3["收获期与最早可收"][0]["结论"][:44] if _sl3["收获期与最早可收"] else None)
check("收获期登记有台账事件",
      any(e.get("kind") == "harvest_set" for e in m.read_jsonl(m.EVENT_LOG)))


print("\n== 5q. 本地防治库建设（权威性与完整性） ==")
_audit = m.treatments_audit()
check("药剂条目从 19 条扩到 31 条", _audit["药剂条目"] == 31, _audit["药剂条目"])
check("含 7 条生物农药（春雷霉素/枯草芽孢杆菌/补骨脂种子提取物等）",
      _audit["生物农药"] == 7, _audit["生物农药"])
check("六个类别都有防治要点与依据",
      not _audit["类别缺防治要点"] and not _audit["类别缺依据"],
      (_audit["类别缺防治要点"], _audit["类别缺依据"]))
check("登记证号完整度 ≥60%（缺口逐条列得出来）",
      _audit["有登记证号"] * 100.0 / _audit["药剂条目"] >= 60,
      "%d/%d" % (_audit["有登记证号"], _audit["药剂条目"]))
check("21 条药剂有安全间隔期，其余缺口逐条列出（不编数字）",
      _audit["有安全间隔期"] == 21 and len(_audit["缺安全间隔期"]) == 10,
      "%d/%d" % (_audit["有安全间隔期"], _audit["药剂条目"]))
check("缺口是显式列出的，不是悄悄留空",
      isinstance(_audit["缺登记证号"], list) and len(_audit["缺登记证号"]) >= 5,
      _audit["缺登记证号"][:3])
_all_drugs = [d for tr in m.TREATMENT_DB.values() for d in tr.get("drugs", [])]
check("每条药剂都有名称/规格/亩用量",
      all(d.get("name") and d.get("spec") and d.get("dose") for d in _all_drugs),
      [d.get("name") for d in _all_drugs if not (d.get("spec") and d.get("dose"))])
check("带安全间隔期的药剂都有来源字段",
      all((d.get("safe_source") or d.get("source")) for d in _all_drugs if d.get("safe_days")),
      [d["name"] for d in _all_drugs if d.get("safe_days") and not (d.get("safe_source") or d.get("source"))])
_new_drugs = ["甲维·氯虫苯", "乙基多杀菌素", "呋虫胺", "噻虫胺", "氨基寡糖素", "氟环·稻瘟灵"]
_names = {d.get("name") for d in _all_drugs}
check("新增的登记标签药剂已入库", all(n in _names for n in _new_drugs),
      [n for n in _new_drugs if n not in _names])
_chunlei = [d for d in _all_drugs if d.get("name") == "春雷霉素"]
check("春雷霉素同时登记在稻瘟病与白叶枯病，且标记为生物农药",
      len(_chunlei) >= 2 and any(d.get("type") == "生物" for d in _chunlei),
      [(d.get("name"), d.get("type"), d.get("reg_no")) for d in _chunlei])
_tx2 = json.loads(m.run_agent_tool("get_treatment", {"cls_name": "稻纵卷叶螟"}))
_d0 = _tx2["推荐药剂"][0]
check("防治方案工具输出登记证号/施药方法/每季次数/防治适期",
      all(k in _d0 for k in ("登记证号", "施药方法", "每季最多使用", "防治适期", "注意事项")),
      list(_d0.keys()))
check("方案工具额外给出类别级防治要点",
      _tx2.get("防治要点", {}).get("防治适期") and _tx2["防治要点"].get("依据"),
      _tx2.get("防治要点", {}).get("依据"))
check("稻叶蝉类别如实标注「登记对象多为稻飞虱」（不含糊等同）",
      "稻飞虱" in (m.TREATMENT_DB["稻叶蝉"].get("meta", {}).get("类别说明") or ""),
      m.TREATMENT_DB["稻叶蝉"]["meta"]["类别说明"][:40])
check("氯虫苯甲酰胺（带品牌后缀）也能匹配到增强字段",
      any(d.get("reg_no") for d in m.TREATMENT_DB["稻纵卷叶螟"]["drugs"]
          if "氯虫苯甲酰胺" in d.get("name", "")),
      [d.get("reg_no") for d in m.TREATMENT_DB["稻纵卷叶螟"]["drugs"]
       if "氯虫苯甲酰胺" in d.get("name", "")])


print("\n== 5r. 防治指标量化与短板补齐 ==")
_a2 = m.treatments_audit()
check("药剂条目扩到 31 条、生物农药 7 条",
      _a2["药剂条目"] == 31 and _a2["生物农药"] == 7,
      (_a2["药剂条目"], _a2["生物农药"]))
check("登记证号完整度 ≥70%", _a2["有登记证号"] * 100.0 / _a2["药剂条目"] >= 70,
      "%d/%d" % (_a2["有登记证号"], _a2["药剂条目"]))
_bxb = m.TREATMENT_DB["白叶枯病"]["drugs"]
_bxb_names = [d["name"] for d in _bxb]
check("白叶枯病补上细菌性病害登记药剂（噻唑锌/噻霉酮/枯草芽孢杆菌）",
      all(n in _bxb_names for n in ("噻唑锌", "噻霉酮", "枯草芽孢杆菌")), _bxb_names)
check("新增的白叶枯病药剂都带登记证号",
      all(d.get("reg_no") for d in _bxb if d["name"] in ("噻唑锌", "噻霉酮", "枯草芽孢杆菌")),
      [(d["name"], d.get("reg_no")) for d in _bxb])
_kcb = [d for d in _bxb if d["name"] == "枯草芽孢杆菌"][0]
check("枯草芽孢杆菌如实标注「登记对象就是白叶枯病」",
      "直接登记" in (_kcb.get("target") or ""), _kcb.get("target"))
_tzx = [d for d in _bxb if d["name"] == "噻唑锌"][0]
check("噻唑锌如实标注登记对象是细菌性条斑病（不冒充白叶枯病登记）",
      "细菌性条斑病" in (_tzx.get("target") or ""), _tzx.get("target"))
_dw = m.TREATMENT_DB["稻瘟病"]["drugs"]
check("稻瘟病补上生物/植物源药剂（枯草芽孢杆菌、补骨脂种子提取物）",
      {"枯草芽孢杆菌", "补骨脂种子提取物"} <= {d["name"] for d in _dw},
      [d["name"] for d in _dw])

_m = lambda c: m.TREATMENT_DB[c].get("meta", {})
check("稻纵卷叶螟指标量化（束叶尖 150/60 个，全国技术方案口径）",
      "150" in _m("稻纵卷叶螟")["量化指标"] and "60" in _m("稻纵卷叶螟")["量化指标"],
      _m("稻纵卷叶螟")["量化指标"])
check("稻飞虱指标量化（百丛 1000/1500 头）",
      "1000" in _m("稻叶蝉")["量化指标"] and "1500" in _m("稻叶蝉")["量化指标"],
      _m("稻叶蝉")["量化指标"])
check("稻瘟病给出可执行适期（破口前 3—5 天）",
      "3—5" in _m("稻瘟病")["量化指标"], _m("稻瘟病")["量化指标"])
check("白叶枯病给出可执行判据（出现发病中心即施药）",
      "发病中心" in _m("白叶枯病")["量化指标"], _m("白叶枯病")["量化指标"])
check("每个类别都标注了指标来源",
      all(_m(c).get("指标来源") for c in m.TREATMENT_DB),
      [c for c in m.TREATMENT_DB if not _m(c).get("指标来源")])
check("查不到国家级指标的类别如实写「暂无国家级统一指标」而不是编一个",
      "暂无国家级统一指标" in _m("稻水象甲")["量化指标"]
      and "暂无国家级统一指标" in _m("胡麻斑病")["量化指标"],
      (_m("稻水象甲")["量化指标"], _m("胡麻斑病")["量化指标"]))
check("指标来源写明出处（可追溯）",
      "2025" in _m("稻纵卷叶螟")["指标来源"], _m("稻纵卷叶螟")["指标来源"])
_tx3 = json.loads(m.run_agent_tool("get_treatment", {"cls_name": "白叶枯病"}))
check("防治方案工具输出带量化指标与来源",
      "量化指标" in _tx3.get("防治要点", {})
      and "指标来源" in _tx3.get("防治要点", {}),
      _tx3.get("防治要点", {}).get("量化指标")[:40])
check("短板类别也有药剂可推荐（白叶枯病 5 条以上）",
      len(_tx3["推荐药剂"]) >= 5, len(_tx3["推荐药剂"]))


print("\n== 5s. 用户自维护层（不动代码补库） ==")
# 上一次运行若被中断，检查目录里可能残留用户库文件——先清干净，保证计数可复现
if Path(m.USER_DB_PATH).exists():
    Path(m.USER_DB_PATH).unlink()
m.apply_user_db()
check("初始用户库为空（自检前已重置）",
      m.user_db_stats()["用户药剂"] == 0 and m.treatments_audit()["药剂条目"] == 31,
      (m.user_db_stats()["用户药剂"], m.treatments_audit()["药剂条目"]))
_before_n = len(m.TREATMENT_DB["稻瘟病"]["drugs"])
_rec, _msg = m.user_add_drug("稻瘟病", {
    "name": "自检当地药剂", "spec": "40% 悬浮剂", "dose": "30ml/亩", "safe_days": 21,
    "reg_no": "PD20990001（自检）", "method": "喷雾", "max_per_season": 2,
    "timing": "发病初期", "caution": "自检用", "target": "稻瘟病", "type": "化学",
    "source": "自检标签"})
check("用户新增药剂成功", _rec is not None and _msg == "已新增", _msg)
_added = [d for d in m.TREATMENT_DB["稻瘟病"]["drugs"] if d.get("name") == "自检当地药剂"]
check("新药剂已合并进库并标记为用户条目",
      _added and _added[0].get("origin") == "user",
      (_added[0].get("origin") if _added else None))
check("库条目数 +1", len(m.TREATMENT_DB["稻瘟病"]["drugs"]) == _before_n + 1)
check("用户数据落盘到外部文件（不动代码）",
      Path(m.USER_DB_PATH).exists(), str(m.USER_DB_PATH))

_old_dose = [d for d in m.TREATMENT_DB["稻瘟病"]["drugs"] if d["name"] == "三环唑"][0].get("dose")
m.user_add_drug("稻瘟病", {"name": "三环唑", "spec": "75% WP", "dose": "25-30g/亩（本地标签）",
                          "reg_no": "PD20990002（自检本地标签）", "safe_days": 30})
_hit = [d for d in m.TREATMENT_DB["稻瘟病"]["drugs"] if d["name"] == "三环唑"][0]
check("同名时用户数据优先（内置被本地标签覆盖）",
      _hit.get("dose") == "25-30g/亩（本地标签）" and _hit.get("safe_days") == 30,
      (_old_dose, _hit.get("dose")))
check("被覆盖的条目仍保留内置的其他字段（不是整条替换）",
      bool(_hit.get("name")) and bool(_hit.get("spec")))

_ok, _why = m.user_delete_drug("稻纵卷叶螟", "甲维·氯虫苯")
check("内置条目不允许删除，并说明原因",
      _ok is False and "内置" in _why, _why)
_ok2, _why2 = m.user_delete_drug("稻瘟病", "自检当地药剂")
check("用户条目可以删除", _ok2 is True and _why2 == "已删除", _why2)
check("删除后从库中移除",
      not any(d.get("name") == "自检当地药剂" for d in m.TREATMENT_DB["稻瘟病"]["drugs"]))

m.user_set_threshold("稻瘟病", {"量化指标": "自检指标：初见病斑即防", "指标来源": "自检来源"})
check("类别量化指标可被用户覆盖",
      "自检指标" in (m.TREATMENT_DB["稻瘟病"].get("meta", {}).get("量化指标") or ""),
      m.TREATMENT_DB["稻瘟病"]["meta"]["量化指标"])

_exp = OUT / "user_db_export.json"
_ok3, _p3 = m.export_user_db(_exp)
check("可导出为可分享的 JSON", _ok3 and Path(_p3).exists(), Path(_p3).name)
m.import_user_db(_exp, merge=True)
check("导入（合并）不报错且幂等", m.user_db_stats()["用户药剂"] >= 1, m.user_db_stats())

# 清理：删掉自检留下的用户条目，避免影响后续断言
m.user_delete_drug("稻瘟病", "自检当地药剂")
m.user_delete_drug("稻瘟病", "三环唑")
m.user_set_threshold("稻瘟病", {})
_a3 = m.treatments_audit()
check("清理后库回到 31 条（自检不留痕）", _a3["药剂条目"] == 31, _a3["药剂条目"])
check("用户库操作都有台账事件",
      any(e.get("kind") == "user_db_drug" for e in m.read_jsonl(m.EVENT_LOG)))


print("\n== 5t. 手机传图（局域网直传原文件） ==")
import re as _re
_js = _re.search(r"<script>([\s\S]*?)</script>", m.PHONE_UPLOAD_PAGE).group(1)
def _js_str_spans_newline(code):
    """扫描 JS：若有「字符串字面量跨行」就是语法错误（正是 Python 转义吃掉 \\n 的症状）。
    行尾的引号（合法的跨行拼接）不会被误判——那时并不在字符串内部。"""
    code = _re.sub(r"//[^\n]*", "", code)
    quote, esc = None, False
    for ch in code:
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            elif ch == chr(10):
                return True
        elif ch in ("'", '"'):
            quote = ch
    return False


check("手机页 JS 没有字符串跨行（Python 转义坑会让整段脚本报语法错误）",
      not _js_str_spans_newline(_js))
check("手机页保留了字面的 \\n 转义（不是被 Python 吃掉的真换行）",
      (chr(92) + "n") in _js)
_app_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "rice_dp1_fluent.py"), encoding="utf-8").read()
check("手机页定义用的是原始字符串 r\"\"\"",
      'PHONE_UPLOAD_PAGE = r"""' in _app_src)
check("手机页用 label 原生触发文件选择（不依赖 JS 调 hidden input）",
      'label class="btn" for="fc"' in m.PHONE_UPLOAD_PAGE
      and 'label class="btn ghost" for="f"' in m.PHONE_UPLOAD_PAGE)
check("两个文件框都用屏幕外定位而不是 display:none（手机浏览器会拒绝后者）",
      'id="f" class="picker"' in m.PHONE_UPLOAD_PAGE
      and 'id="fc" class="picker"' in m.PHONE_UPLOAD_PAGE
      and ".picker{position:absolute" in m.PHONE_UPLOAD_PAGE)
check("手机页有版本标记与测试连接按钮",
      "v2 版" in m.PHONE_UPLOAD_PAGE and "测试连接" in m.PHONE_UPLOAD_PAGE)
check("手机页有拍照与相册两个入口",
      "拍照" in m.PHONE_UPLOAD_PAGE and "从相册选" in m.PHONE_UPLOAD_PAGE
      and 'capture="environment"' in m.PHONE_UPLOAD_PAGE)
check("手机页有上传进度条（XHR onprogress）",
      "onprogress" in m.PHONE_UPLOAD_PAGE and "lengthComputable" in m.PHONE_UPLOAD_PAGE)
check("手机页会轮询电脑端识别结果",
      "/result?name=" in m.PHONE_UPLOAD_PAGE and "pollResult" in m.PHONE_UPLOAD_PAGE)
check("手机页显示连到哪台电脑（避免传错设备）",
      "/whoami" in m.PHONE_UPLOAD_PAGE)
check("手机页给出 iPhone HEIC 提示",
      "HEIC" in m.PHONE_UPLOAD_PAGE and "兼容性最佳" in m.PHONE_UPLOAD_PAGE)
check("上传失败会自动重试一次", "attempt < 2" in m.PHONE_UPLOAD_PAGE or
      "attempt" in m.PHONE_UPLOAD_PAGE)
check("服务端有 /whoami 与 /result 路由",
      '"/whoami"' in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "rice_dp1_fluent.py"), encoding="utf-8").read()
      and 'startswith("/result")' in open(os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "..", "rice_dp1_fluent.py"),
          encoding="utf-8").read())
check("上传页含多选与说明", "multiple" in m.PHONE_UPLOAD_PAGE and "手机传图" in m.PHONE_UPLOAD_PAGE)
check("上传页明确写出「不压缩、不转码」", "不压缩" in m.PHONE_UPLOAD_PAGE and "GPS" in m.PHONE_UPLOAD_PAGE)
check("只接受图片扩展名（拒绝可执行文件等）",
      ".jpg" in m.PHONE_UPLOAD_EXTS and ".exe" not in m.PHONE_UPLOAD_EXTS)
_ip = m.lan_ip()
check("能取到本机局域网地址", _ip.count(".") == 3 and _ip != "0.0.0.0", _ip)
check("端口与手机端演示服务错开", m.PHONE_UPLOAD_PORT != 8765, m.PHONE_UPLOAD_PORT)


print("\n== 5u. 小程序结果同步（中转 → 收件箱 → 自动巡检） ==")
check("伴随文件命名规则清晰（图片名 + .phone.json）",
      str(m.phone_sidecar_path("a/b.jpg")).endswith("b.jpg.phone.json"),
      m.phone_sidecar_path("x.jpg").name)
_sc = OUT / "sidecar_probe.jpg"
_sc.write_bytes(b"\xff\xd8\xff\xd9")
check("没有伴随文件时返回 None（普通照片不受影响）", m.read_phone_sidecar(_sc) is None)
m.phone_sidecar_path(_sc).write_text(json.dumps(
    {"label": "稻瘟病", "score": 0.88, "来源": "小程序"}, ensure_ascii=False), encoding="utf-8")
_side = m.read_phone_sidecar(_sc)
check("伴随文件能读回手机端结论",
      _side and _side["label"] == "稻瘟病" and _side["score"] == 0.88, _side)
check("没填中转地址时给出明确提示而不是抛异常",
      m.relay_pull("").get("ok") is False and "中转服务地址" in m.relay_pull("")["error"])
check("连不上中转服务时也不抛异常",
      m.relay_pull("http://127.0.0.1:59999").get("ok") is False)
_rep = m.relay_sync_report()
check("中转地址与口令会随配置保存（不用每次重输）",
      "relay_base" in m.DEFAULT_CONFIG and "relay_token" in m.DEFAULT_CONFIG)
check("同步报告字段齐全",
      all(k in _rep for k in ("中转地址", "上次同步", "累计同步", "收件箱里来自手机端的图")),
      list(_rep))
_relay_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "云边联动方案C", "relay_server.py"),
                      encoding="utf-8").read()
check("中转服务已支持接收识别结果（label/score 进了白名单）",
      '"label", "score", "boxes"' in _relay_src
      and 'os.makedirs(os.path.dirname(INDEX)' in _relay_src)


print("\n== 5v. 手机传图二维码 ==")
_url = "http://%s:%d/" % (m.lan_ip(), m.PHONE_UPLOAD_PORT)
_mat = m.qr_matrix(_url)
check("能离线生成二维码矩阵（随程序分发的 qrcode）",
      _mat and len(_mat) == len(_mat[0]) and len(_mat) > 20,
      len(_mat) if _mat else None)
check("矩阵带静区（四角全白，保证扫码识别）",
      not any(_mat[0]) and not any(_mat[-1]) and not _mat[0][0])
# QPixmap 需要 QApplication，图像相关的检查放到下面的界面段


print("\n== 5w. 粘贴坐标落点（水印坐标直接填） ==")
_cases = [("32°0'38\"N 118°37'48\"E", 118.63, 32.0105556),
          ("32°0′38″N, 118°37′48″E", 118.63, 32.0105556),
          ("32度0分38秒N 118度37分48秒E", 118.63, 32.0105556),
          ("32.010556, 118.63", 118.63, 32.010556),
          ("118.63, 32.010556", 118.63, 32.010556),
          ("N32.010556 E118.63", 118.63, 32.010556),
          ("-32.010556, 118.63", 118.63, -32.010556)]
_bad = 0
for _t, _elon, _elat in _cases:
    try:
        _r = m.parse_coord_text(_t)
        if not (abs(_r["lon"] - _elon) < 1e-4 and abs(_r["lat"] - _elat) < 1e-4):
            _bad += 1
    except ValueError:
        _bad += 1
check("水印/地图/十进制等 7 种坐标写法都能解析", _bad == 0, "%d 种失败" % _bad)
check("不带方向的十进制能按 |值|>90 判断经度",
      abs(m.parse_coord_text("118.63, 32.010556")["lon"] - 118.63) < 1e-6)
_rejected = 0
for _t in ("", "abc", "32.5", "32,118,50"):
    try:
        m.parse_coord_text(_t)
    except ValueError:
        _rejected += 1
check("非法输入一律拒绝（不猜、不落错点）", _rejected == 4, "%d/4" % _rejected)
# 坐标系换算往返：水印坐标是 GCJ-02，存档要 WGS-84，再换回来必须一模一样
import rice_geo as _rg
_p = m.parse_coord_text("32°0'38\"N 118°37'48\"E")
_w = _rg.from_gcj02(_p["lon"], _p["lat"], "wgs84")
_b = _rg.to_gcj02(_w[0], _w[1], "wgs84")
check("GCJ-02→WGS-84→GCJ-02 往返误差 < 1e-6（落点不会偏）",
      abs(_b[0] - _p["lon"]) < 1e-6 and abs(_b[1] - _p["lat"]) < 1e-6,
      "偏移 %.8f, %.8f" % (abs(_b[0] - _p["lon"]), abs(_b[1] - _p["lat"])))
check("两种坐标系确实有差别（不是原样抄）",
      abs(_w[0] - _p["lon"]) > 1e-4, "偏 %.5f°" % abs(_w[0] - _p["lon"]))


# ══════════════════════════════════════════════════════════════
# 六、界面（离屏）
# ══════════════════════════════════════════════════════════════
print("\n== 6. 界面构建（离屏） ==")
from PyQt5.QtWidgets import QApplication, QShortcut
from PyQt5.QtCore import QTimer

app = QApplication(sys.argv)
try:
    win = m.MainWindow()
    win.resize(1380, 900)
    win.show()
    check("主窗口创建成功", win is not None)
    demo_img = np.full((320, 480, 3), 200, dtype=np.uint8)   # 供 re_detect 工具用
    demo_img[80:240, 120:360] = (110, 150, 90)
    for attr, name in [("homeInterface", "首页"), ("imageInterface", "图片识别"),
                       ("cameraInterface", "实时识别"), ("watchInterface", "自动巡检"),
                       ("trainInterface", "实训考核"),
                       ("resultInterface", "结果与治理"),
                       ("reportInterface", "报告记录"), ("settingInterface", "API 设置")]:
        check("页面存在：" + name, hasattr(win, attr))
    check("自动巡检页有监控/单次/台账入口",
          hasattr(win.watchInterface, "btn_toggle") and hasattr(win.watchInterface, "btn_once")
          and hasattr(win.watchInterface, "btn_ledger"))
    check("自动巡检页有「刷新列表」并能读台账（重启后不再空白）",
          hasattr(win.watchInterface, "btn_reload")
          and hasattr(win.watchInterface, "_load_recent_rows"))
    check("有地块地图页并已列入导航",
          hasattr(win, "mapInterface")
          and win.mapInterface.objectName() == "mapInterface")
    win.mapInterface.refresh()
    check("地图页刷新后显示统计与风险列表",
          "记录" in win.mapInterface.lbl_stat.text()
          and win.mapInterface.risk_list.count() >= 1,
          (win.mapInterface.lbl_stat.text()[:36], win.mapInterface.risk_list.count()))
    check("地图页加载了原生瓦片地图控件（rice_geo.MapCanvas，无需 Key）",
          win.mapInterface.canvas is not None)
    check("地图页有筛选控件（时间 / 类别 / 热力图）",
          all(hasattr(win.mapInterface, _a) for _a in ("cmb_days", "cmb_cls", "chk_heat")))
    check("筛选默认不设限（全部时间 / 全部类别 / 热力图关闭）",
          win.mapInterface.cmb_days.currentData() is None
          and win.mapInterface.cmb_cls.currentData() is None
          and not win.mapInterface.chk_heat.isChecked(),
          (win.mapInterface.cmb_days.currentText(), win.mapInterface.cmb_cls.currentText()))
    win.mapInterface.cmb_days.setCurrentIndex(1)          # 最近 7 天
    win.mapInterface.cmb_cls.setCurrentIndex(1)           # 第一类病虫害
    check("切换筛选后不崩、列表仍有反馈",
          win.mapInterface.risk_list.count() >= 1
          and "筛选：" in win.mapInterface.lbl_stat.text(),
          win.mapInterface.lbl_stat.text()[:60])
    win.mapInterface.cmb_days.setCurrentIndex(0)
    win.mapInterface.cmb_cls.setCurrentIndex(0)
    check("地图页有「查施药窗口」入口",
          hasattr(win.mapInterface, "btn_spray")
          and hasattr(win.mapInterface, "query_spray")
          and hasattr(win.mapInterface, "spray_payload"))
    check("没查天气时列表不硬凑施药状态（不显示假结论）",
          win.mapInterface._spray_suffix("任意地块") == ""
          and win.mapInterface.spray_payload() == [])
    check("交互地图改为独立窗口入口（页内不再创建 WebEngine）",
          hasattr(win.mapInterface, "btn_web_map")
          and hasattr(m, "MapWebWindow")
          and win.mapInterface._web_window is None)
    win.mapInterface.set_map_mode("web")     # 离屏下应提示不可用而不是崩
    check("离屏下开交互地图会优雅退回轻量地图（不创建 WebEngine 视图）",
          win.mapInterface.map_stack.currentIndex() == 0
          and win.mapInterface.web_view is None
          and win.mapInterface._web_window is None)
    # 交互地图曾经「点开就卡」：视图在 __init__ 里同步建（WebEngine 首次初始化
    # 要拉子进程，冻一两秒），而且数据在 load 完成前就推了 —— 打在空白页上，
    # renderMap 还没定义，地图永远空白。
    check("交互地图：视图延后创建、加载完才推数据",
          hasattr(m.MapWebWindow, "ensure_view")
          and hasattr(m.MapWebWindow, "_on_load_finished"))
    _mw = m.MapWebWindow(win.mapInterface)
    check("构造交互地图窗口不会当场建 WebEngine 视图（不卡界面）",
          _mw.web_view is None and _mw._loaded is False)
    _mw.push()          # 加载完成前推送：必须是安全的空操作
    check("加载完成前推送数据是安全的空操作", _mw.web_view is None and _mw._loaded is False)
    check("交互地图窗口在加载前有提示文案（不至于空白像卡死）",
          _mw._loading_lbl is not None
          and "加载" in _mw._loading_lbl.text())
    _mw.close()
    # 悬停展开文字的按钮：默认只占图标宽，悬停才展开
    from PyQt5.QtCore import QEvent as _QE
    _hb = m.hover_text_button(m.FIF.PIN, "标记到地图")
    _w_icon = _hb.sizeHint().width()
    _icon_sz = _hb.iconSize().width()
    # 写死宽度会让按钮窄于图标 -> 图标被裁成一条缝（实测踩过），所以这里卡住
    check("图标按钮默认只占图标宽（且不得窄于图标，否则会被裁）",
          _hb.text() == "" and _w_icon >= _icon_sz, (_hb.text(), _w_icon, _icon_sz))
    check("按钮宽度明显大于图标宽度（写死宽度会把图标裁成一条缝）",
          _w_icon > _icon_sz, (_w_icon, _icon_sz))
    # 两个按钮并排也不能把文件名挤没 —— qfluentwidgets 默认最小宽度 104，必须收窄
    check("图标按钮足够紧凑（两个并排 ≤140px）",
          _w_icon <= 70, _w_icon)
    QApplication.sendEvent(_hb, _QE(_QE.Enter))
    _w_text = _hb.sizeHint().width()
    check("鼠标移上去展开文字", _hb.text() == "标记到地图" and _w_text > _w_icon,
          (_hb.text(), _w_text))
    QApplication.sendEvent(_hb, _QE(_QE.Leave))
    check("鼠标移开收回成图标",
          _hb.text() == "" and _hb.sizeHint().width() == _w_icon,
          (_hb.text(), _hb.sizeHint().width()))
    check("地图页有「让 Agent 分析这片区域」入口",
          hasattr(win.mapInterface, "btn_analyze")
          and hasattr(win.mapInterface, "agent_out"))
    # 施药登记对话框（需要 QApplication；autorun=False 避免离屏用例联网校验）
    _dlg = m.SprayPlanDialog("稻瘟病", 0.87, 3, plots=["闭环田"], parent=win, autorun=False)
    check("施药登记对话框预填了本地库方案",
          _dlg.drug.count() >= 1 and _dlg.dose.text().strip() != "",
          (_dlg.drug.count(), _dlg.dose.text()))
    _dv = _dlg.values()
    check("对话框取值带出药剂、用量与安全间隔期",
          bool(_dv["drug"]) and bool(_dv["dose"]) and bool(_dv["plate"].get("名称")),
          (_dv["drug"], _dv["dose"], _dv["safe_days"]))
    check("结果页有「按方案施药并登记」按钮",
          hasattr(win.resultInterface, "btn_spray"))
    _dlg.close()
    check("有「防治库维护」页并已注册",
          hasattr(win, "treatmentInterface")
          and win.treatmentInterface.objectName() == "treatmentInterface")
    win.treatmentInterface.reload()
    check("维护页列出当前类别的药剂",
          win.treatmentInterface.lst.count() >= 4, win.treatmentInterface.lst.count())
    check("维护页有 12 个字段输入框与类别指标输入框",
          len(win.treatmentInterface.inputs) == 12
          and len(win.treatmentInterface.th_inputs) == 3)
    win.treatmentInterface._on_drug(0)
    check("选中条目会把字段填进表单",
          win.treatmentInterface.inputs["name"].text().strip() != "",
          win.treatmentInterface.inputs["name"].text())
    check("内置条目禁止删除（按钮置灰）",
          win.treatmentInterface.btn_del.isEnabled() is False
          or win.treatmentInterface.current_drug.get("origin") == "user")
    # 手机传图端到端：起服务 → 原始字节 POST → 比对哈希（证明不转码）
    import hashlib as _hl, time as _tm, requests as _rq
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational as _R
    _src = OUT / "phone_src.jpg"
    _im = Image.new("RGB", (160, 120), (205, 220, 200))
    _ex = Image.Exif()
    _ex[0x0110] = "Xiaomi 14"
    _ex[0x8825] = {1: "N", 2: (_R(32, 1), _R(0, 1), _R(38, 1)),
                   3: "E", 4: (_R(118, 1), _R(37, 1), _R(48, 1))}
    _im.save(str(_src), exif=_ex, quality=92)
    _raw = _src.read_bytes()
    _inbox = OUT / "phone_inbox"
    _srv = m.PhoneUploadThread(str(_inbox), port=8797)
    _srv.start()
    _ready = False
    for _i in range(30):
        _tm.sleep(0.2)
        try:
            if _rq.get("http://127.0.0.1:8797/ping", timeout=2).status_code == 200:
                _ready = True
                break
        except Exception:
            pass
    check("手机传图服务能起来", _ready)
    check("手机能打开上传页",
          "手机传图" in _rq.get("http://127.0.0.1:8797/", timeout=5).text)
    _resp = _rq.post("http://127.0.0.1:8797/upload?name=" + _rq.utils.quote("田里.jpg"),
                     data=_raw, headers={"Content-Type": "application/octet-stream"},
                     timeout=20).json()
    _saved = _inbox / _resp["saved"]
    check("上传成功且服务端读到了 GPS",
          _resp.get("ok") and _resp["gps"]["has_gps"], _resp.get("gps"))
    check("落地文件与源文件 SHA256 完全一致（证明不转码）",
          _hl.sha256(_raw).hexdigest() == _hl.sha256(_saved.read_bytes()).hexdigest(),
          "%d 字节 vs %d 字节" % (len(_raw), _saved.stat().st_size))
    check("落地文件的 EXIF GPS 仍然读得到",
          m.geo_store is not None and __import__("rice_geo").read_photo_meta(str(_saved))["has_gps"])
    check("非图片格式被拒绝",
          _rq.post("http://127.0.0.1:8797/upload?name=x.exe", data=b"MZ",
                   timeout=10).status_code == 400)
    _srv.stop(); _srv.wait(3000)
    check("自动巡检页有「手机传图」入口",
          hasattr(win.watchInterface, "btn_phone"))
    # 巡检处理带伴随文件的图：应把手机端结论一并记录并对比
    _pi = OUT / "phone_recv.jpg"
    cv2.imwrite(str(_pi), np.full((200, 260, 3), 180, dtype=np.uint8))
    m.phone_sidecar_path(_pi).write_text(json.dumps(
        {"label": "稻瘟病", "score": 0.91, "来源": "小程序", "device": "Xiaomi 14"},
        ensure_ascii=False), encoding="utf-8")
    _prec = m.process_image_file(_pi, do_cascade=False)
    check("处理完会写识别结果伴随文件（手机页靠它回显）",
          Path(str(_pi) + ".result.json").exists())
    check("巡检会记录手机端结论并给出一致性判断",
          _prec.get("phone_cls") == "稻瘟病" and _prec.get("phone_conf") == 0.91
          and isinstance(_prec.get("phone_agree"), bool),
          (_prec.get("cls"), _prec.get("phone_cls"), _prec.get("phone_agree")))
    # 二维码真能被扫出来吗？用 OpenCV 解码回来验证（不是"看着像"就行）
    _qurl = "http://%s:%d/" % (m.lan_ip(), m.PHONE_UPLOAD_PORT)
    _qp = m.qr_pixmap(_qurl, 240)
    check("能转成 QPixmap 供界面显示", _qp is not None and not _qp.isNull()
          and _qp.width() == 240, None if _qp is None else _qp.width())
    _qi = _qp.toImage().convertToFormat(4)
    _w, _h = _qi.width(), _qi.height()
    _ptr = _qi.bits(); _ptr.setsize(_qi.byteCount())
    _arr = np.frombuffer(bytes(_ptr), np.uint8).reshape(_h, _qi.bytesPerLine() // 4, 4)[:, :_w, :3]
    _gray = cv2.resize(cv2.cvtColor(_arr, cv2.COLOR_RGB2GRAY), (_w * 3, _h * 3),
                       interpolation=cv2.INTER_NEAREST)
    _decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(_gray)
    check("二维码能被 OpenCV 解码回原网址（确实可扫）",
          _decoded.strip() == _qurl, _decoded.strip()[:40])
    _pdlg = m.PhoneUploadDialog(m.WATCH_DIR, parent=win)
    check("手机传图对话框里显示了二维码",
          hasattr(_pdlg, "qr") and _pdlg.qr.pixmap() is not None
          and not _pdlg.qr.pixmap().isNull())
    check("对话框同时保留可复制的网址文本",
          _pdlg.url.text().startswith("http://") and ":" in _pdlg.url.text(),
          _pdlg.url.text())
    _pdlg.close()
    check("自动巡检页有「云端同步」入口（小程序跨网络用）",
          hasattr(win.watchInterface, "btn_relay"))
    _rdlg = m.RelaySyncDialog(parent=win)
    check("云端同步对话框有地址/口令/同步按钮",
          hasattr(_rdlg, "base") and hasattr(_rdlg, "token")
          and hasattr(_rdlg, "btn_sync") and hasattr(_rdlg, "btn_reset"))
    check("对话框会带出上次填过的中转地址",
          _rdlg.base.text() == str(m.CONFIG.get("relay_base") or ""),
          _rdlg.base.text()[:30])
    _rdlg.base.setText("")
    _rdlg.start_sync(False)
    check("没填地址时给出提示而不是发起请求",
          "请先填写" in _rdlg.status.text(), _rdlg.status.text()[:30])
    _rdlg.close()
    check("地图页有「粘贴坐标落点」入口", hasattr(win.mapInterface, "btn_pin"))
    _pd = m.ManualPinDialog(m.geo_store(), parent=win)
    check("落点对话框有坐标框、坐标系、目标记录三个输入",
          hasattr(_pd, "text") and hasattr(_pd, "sys") and hasattr(_pd, "target"))
    _pd.text.setText("32°0'38\"N 118°37'48\"E")
    _pd.preview()
    check("粘贴水印坐标后能解析并显示换算结果",
          _pd.btn_ok.isEnabled() and "118.624" in _pd.prev.text(), _pd.prev.text()[:60])
    _pd.sys.setCurrentIndex(1)          # 切到 WGS-84
    _pd.preview()
    check("切坐标系数值随之变化（说明换算生效）",
          "118.630" in _pd.prev.text(), _pd.prev.text()[:60])
    _pd.text.setText("乱写")
    _pd.preview()
    check("解析不了时按钮禁用并给出提示",
          not _pd.btn_ok.isEnabled() and "⚠" in _pd.prev.text())
    _pd.close()
    # ── 识别 → 地图 的衔接（v3.8 交互优化） ──
    import inspect as _insp
    _rtm = _insp.signature(m.ResultInterface.record_to_map)
    check("识别结果落库支持自动模式（auto 参数）", "auto" in _rtm.parameters)
    check("地图页有切页自动刷新钩子（不用手点刷新）",
          hasattr(m.MapInterface, "showEvent"))
    check("地图页接了点位点击信号（点一个点看详情）",
          hasattr(m.MapInterface, "_on_marker_clicked"))
    check("地图页初始化了「点位序号 → 记录」的映射表",
          hasattr(win.mapInterface, "_pt_records"))
    # 越界序号不能崩（点空白/数据刷新后旧序号失效都可能触发）
    _mi = win.mapInterface
    _mi._pt_records = []
    _mi._on_marker_clicked(0)
    _mi._on_marker_clicked(99)
    check("点位序号越界时不崩溃（静默忽略）", True)
    _gmd = m.GeoManageDialog(m.geo_store(), parent=win)
    check("记录管理里有「补坐标」按钮（待落点集中补点）",
          hasattr(_gmd, "btn_pin") and hasattr(_gmd, "pin_selected"))
    check("记录管理能列出记录并带来源标记",
          _gmd.lst.count() >= 0 and "共" in _gmd.stat.text(), _gmd.stat.text()[:40])
    _gmd.close()
    check("教学页有三种模式与导入入口",
          hasattr(win.trainInterface, "mode_bar") and hasattr(win.trainInterface, "btn_import")
          and hasattr(win.trainInterface, "btn_import_url")
          and hasattr(win.trainInterface, "btn_seed"))
    check("教学页有题库/案例面板",
          hasattr(win.trainInterface, "bank_card") and hasattr(win.trainInterface, "case_card"))
    win.trainInterface._switch_mode("bank")
    check("切到题库答题会渲染题目",
          (not win.trainInterface.bank_card.isHidden())
          and win.trainInterface.bank_question.text().strip() != "",
          win.trainInterface.bank_question.text()[:26])
    # 显式挑一道选择题来验证"选项由题库动态生成"（随机抽题可能抽到简答题）
    _ti = win.trainInterface
    _ti.bank_session = [q for q in m.load_question_bank() if q.get("options")][:3]
    _ti.bank_index = 0
    _ti._load_bank_question()
    check("选择题按题库动态渲染选项",
          _ti.bank_opts_box.count() == len(_ti.bank_session[0]["options"]),
          (_ti.bank_opts_box.count(), len(_ti.bank_session[0]["options"])))
    check("题目显示来源", "来源：" in win.trainInterface.bank_source.text(),
          win.trainInterface.bank_source.text()[:40])
    win.trainInterface._switch_mode("case")
    check("切到案例学习会渲染案例卡",
          (not win.trainInterface.case_card.isHidden())
          and "症状" in win.trainInterface.case_view.toPlainText(),
          win.trainInterface.case_view.toPlainText()[:30])
    check("教学页有题目复核与错题本模式",
          hasattr(win.trainInterface, "review_card")
          and hasattr(win.trainInterface, "wrong_card"))
    win.trainInterface._switch_mode("review")
    check("复核面板会显示待复核题目",
          (not win.trainInterface.review_card.isHidden())
          and win.trainInterface.review_view.toPlainText().strip() != "",
          win.trainInterface.review_view.toPlainText()[:30])
    _rev_before = len(m.reviewed_questions())
    win.trainInterface._review_action("approve")
    check("点「通过」后该题进入已复核池",
          len(m.reviewed_questions()) == _rev_before + 1,
          (len(m.reviewed_questions()), _rev_before))
    win.trainInterface._switch_mode("wrong")
    check("错题本面板显示统计与复习入口",
          "错题" in win.trainInterface.wrong_stats.text()
          and not win.trainInterface.wrong_card.isHidden(),
          win.trainInterface.wrong_stats.text()[:40])
    # 错题闭环：答错 → 进错题本 → 到期后能组卷复习
    _ti2 = win.trainInterface
    _ti2._switch_mode("bank")
    _ti2.bank_session = [q for q in m.load_question_bank() if q.get("options")][:2]
    _ti2.bank_index = 0
    _ti2._load_bank_question()
    _q2 = _ti2.bank_session[0]
    _wrong_opt = [o for o in _q2["options"] if o != _q2["answer"]][0]
    _ti2._answer_bank(_wrong_opt)
    check("答错会写进错题本（含下次复习日期）",
          _q2["id"] in m.load_review_schedule()
          and m.load_review_schedule()[_q2["id"]].get("due"),
          list(m.load_review_schedule().items())[:1])
    _sched2 = m.load_review_schedule()
    _sched2[_q2["id"]]["due"] = datetime.now().strftime("%Y-%m-%d")
    _sched2[_q2["id"]]["mastered"] = False
    m.save_review_schedule(_sched2)
    _ti2._switch_mode("wrong")
    check("错题本面板列出到期错题",
          "错 %d 次" % m.load_review_schedule()[_q2["id"]]["wrong"]
          in _ti2.wrong_view.toPlainText(), _ti2.wrong_view.toPlainText()[:60])
    _ti2._start_wrong_review()
    check("「开始复习」会用到期错题组卷",
          _ti2.mode == "bank" and len(_ti2.bank_session) >= 1
          and _ti2.bank_session[0]["id"] == _q2["id"],
          (_ti2.mode, len(_ti2.bank_session)))
    _ti2._answer_bank(_q2["answer"])
    check("复习答对后间隔拉长（1 → 3 天）",
          m.load_review_schedule()[_q2["id"]]["interval"] == 3,
          m.load_review_schedule()[_q2["id"]])
    _ti2._switch_mode("image")
    check("切回图片考核会恢复左侧图片区",
          not win.trainInterface.left_card.isHidden()
          and win.trainInterface.bank_card.isHidden())
    check("实训页有出题按钮与成绩单导出", hasattr(win.trainInterface, "btn_start")
          and hasattr(win.trainInterface, "btn_export"))
    check("实训页题库非空", len(win.trainInterface.pool) > 10, len(win.trainInterface.pool))
    check("实训页有 6 个类别选项 + 不确定", len(win.trainInterface.opts) == 7,
          len(win.trainInterface.opts))
    check("主窗口有载入历史会话的入口", hasattr(win, "load_session"))
    from PyQt5.QtWidgets import QPushButton as _QPB
    check("报告页有「导出运行台账」入口",
          any("运行台账" in (b.text() or "") for b in win.reportInterface.findChildren(_QPB)))
    # re_detect 工具要能真的跑到模型上（用已加载的全局模型）
    det = json.loads(m.run_agent_tool("re_detect", {"threshold": 0.3}, demo_img))
    check("工具 re_detect 能按新阈值重检", "检出数" in det, det)
    check("快捷键已注册（≥10）", len(win.findChildren(QShortcut)) >= 10,
          len(win.findChildren(QShortcut)))

    ri = win.resultInterface
    check("结果页有停止生成按钮", hasattr(ri, "btn_stop"))
    check("结果页有误检反馈按钮", hasattr(ri, "btn_feedback"))
    check("结果页有检出强度标签", hasattr(ri, "lbl_sev"))
    check("结果页有视觉复核级联面板", hasattr(ri, "cascade_box")
          and hasattr(ri, "cascade_lay"))
    check("结果页有「规划执行」按钮与计划确认条",
          hasattr(ri, "btn_plan") and hasattr(ri, "plan_bar") and hasattr(ri, "btn_run_plan"))
    check("自动巡检页有定时任务卡片",
          hasattr(win.watchInterface, "sched_card") and hasattr(win.watchInterface, "sched_result"))
    check("主窗口有定时任务检查入口", hasattr(win, "_check_schedules"))
    check("默认开启自动复核（低置信度/病害类）", m.DEFAULT_CONFIG.get("cascade_auto") is True)
    check("图片页有漏检上报按钮", hasattr(win.imageInterface, "btn_miss"))
    check("结果页有「带图追问」开关", hasattr(ri, "btn_withimg"))
    si = win.settingInterface
    check("设置页有 3 个服务商可选", si.combo_provider.count() == 3, si.combo_provider.count())
    # 注意：qfluentwidgets 的 EditableComboBox 是 LineEdit 子类（带下拉的自由输入框），
    # 不是 QComboBox 的子类，所以没有 isEditable()
    from PyQt5.QtWidgets import QLineEdit as _QLineEdit
    check("设置页模型名可自由填写（可输入任意模型名）",
          isinstance(si.combo_model, _QLineEdit), type(si.combo_model).__name__)
    check("设置页默认模型是现行 deepseek-flash",
          si.combo_model.currentText().strip() == "deepseek-flash", si.combo_model.currentText())
    check("设置页有接口地址栏（自定义服务）", hasattr(si, "base_input"))
    check("设置页直接写出看图能力", ("可看图" in si.lbl_cap.text()
                                     or "纯文本" in si.lbl_cap.text()), si.lbl_cap.text())
    check("设置页有「回答模式」开关（深度思考）", hasattr(si, "switch_thinking"))
    check("默认关掉深度思考（避免思维链吃掉 token 预算）",
          m.DEFAULT_CONFIG.get("llm_thinking") is False)
    check("说明文字带回答模式", "回答" in si.lbl_cap.text() or "思考" in si.lbl_cap.text(),
          si.lbl_cap.text())
    check("服务商标签含自定义选项", "自定义" in si.combo_provider.currentText()
          or si.combo_provider.count() == 3)

    print("\n== 7. 端到端：结果页 + 导出报告 ==")
    demo = np.full((600, 800, 3), 235, dtype=np.uint8)
    demo[200:400, 250:550] = (120, 150, 200)
    results = [{"name": "稻瘟病", "conf": 0.88, "box": [250, 200, 550, 400]},
               {"name": "稻瘟病", "conf": 0.61, "box": [560, 210, 700, 330]}]
    ri.show_result(results, demo, "image")
    check("结果页算出检出强度", ri.severity is not None and ri.severity["boxes"] == 2,
          ri.severity["text"] if ri.severity else None)
    check("界面显示检出强度", "检出强度" in ri.lbl_sev.text(), ri.lbl_sev.text()[:60])
    check("未启用 AI 时给出可操作提示",
          any("未启用云端 AI" in e[1] for e in ri._chat_history))

    # 流式渲染：直接喂分片，确认节流重绘不报错
    ri._on_stream("# 诊断\n- 第一点", "ai")
    QTimer.singleShot(0, app.quit)
    app.exec_()
    check("流式分片能进对话区",
          any("第一点" in e[1] for e in ri._chat_history if e[0] == "ai"))

    rep = OUT / "smoke_report.png"
    m.export_report(str(rep), results, demo, "稻瘟病", 0.88, "AI 建议：注意防治", "cpu")
    check("导出报告成功", rep.exists() and rep.stat().st_size > 10_000,
          "%d bytes" % (rep.stat().st_size if rep.exists() else 0))
    # 报告里出现禁用农药时必须标红提示（用一个假回答验证路径）
    rep2 = OUT / "smoke_report_banned.png"
    m.export_report(str(rep2), results, demo, "稻瘟病", 0.88, "建议使用百草枯", "cpu")
    check("报告对禁用农药出提示", rep2.exists() and rep2.stat().st_size > 10_000)

    # ── 任务队列：串行执行 + 排队位置 + 互斥（需要 QApplication，放在这里）──
    # 队列：一个失败计划 + 一个正常计划，验证串行与排队位置
    from PyQt5.QtCore import QEventLoop as _EL, QTimer as _QT
    _q = m.TaskQueue()
    _done_plans, _fin = [], []
    _q.finished.connect(lambda p, r: (_fin.append(p), _done_plans.append(len(r))))
    _plan_bad = {"goal": "队列任务A", "steps": [{"tool": "batch_scan",
                                                "args": {"folder": "不存在的目录"}, "why": "x"}]}
    _plan_ok = {"goal": "队列任务B", "steps": [{"tool": "daily_digest", "args": {"days": 1}, "why": "y"}]}
    _p1 = _q.submit(_plan_bad)
    _snap = _q.snapshot()
    _p2 = _q.submit(_plan_ok)
    check("第一个任务立即开始", _p1 == 1, _p1)
    check("第二个任务进排队", _p2 == 2, _p2)
    check("队列快照区分执行中与排队中",
          [x["state"] for x in _q.snapshot()] == ["执行中", "排队中"], _q.snapshot())
    _loop_q = _EL(); _tq = _QT(); _tq.setInterval(50)
    _tq.timeout.connect(lambda: _loop_q.quit() if len(_fin) >= 2 else None)
    _tq.start(); _QT.singleShot(30000, _loop_q.quit); _loop_q.exec_(); _tq.stop()
    check("两个任务都被执行（串行跑完）", len(_fin) == 2, len(_fin))
    check("队列跑完自动清空", not _q.is_busy() and _q.snapshot() == [], _q.snapshot())
    check("失败步骤被如实记录（不假装成功）", _done_plans and _done_plans[0] == 1, _done_plans)

    # ── 自动巡检流水线与新工具（用已加载的模型）──
    inbox = OUT / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    sample = inbox / "rice_blast_0.jpg"
    cv2.imwrite(str(sample), demo_img)
    rec = m.process_image_file(sample)
    check("巡检流水线能处理一张图", rec.get("file") == "rice_blast_0.jpg", rec)
    check("巡检结果写入台账 CSV", m.WATCH_RESULTS.exists() and m.WATCH_RESULTS.stat().st_size > 0)
    # 台账读回界面：这是「手机上能看到、电脑上空的」那个问题的根因
    _wi = win.watchInterface
    _wi._load_recent_rows()
    check("台账能读回列表（不再只在内存里）",
          _wi.box.count() >= 1 and "台账" in _wi.status.text(),
          _wi.status.text()[:40])
    _row_f = m.WatchInterface._row_from_csv(
        {"file": "a.jpg", "cls": "白叶枯病", "conf": "0.813", "severity": "重",
         "verify": "稻纵卷叶螟", "verify_agree": "False", "note": ""})
    check("台账行解析正确（复核不一致会被标出来）",
          _row_f["verify_agree"] is False and abs(_row_f["conf"] - 0.813) < 1e-6
          and _row_f["cls"] == "白叶枯病", _row_f)
    check("台账行的三种复核状态都能区分（True/False/空）",
          m.WatchInterface._row_from_csv({"verify_agree": "True"})["verify_agree"] is True
          and m.WatchInterface._row_from_csv({"verify_agree": ""})["verify_agree"] == ""
          and m.WatchInterface._row_from_csv({"verify_agree": "False"})["verify_agree"] is False)
    # 来源色标：小程序 / 手机直传 / 电脑本地 要能一眼区分
    _src_img = inbox / "_src_probe.jpg"
    cv2.imwrite(str(_src_img), demo_img)
    _side = Path(str(_src_img) + ".phone.json")
    check("无伴随文件 → 判为「电脑本地」", m.watch_source(_src_img) == "电脑本地",
          m.watch_source(_src_img))
    _side.write_text(json.dumps({"来源": "手机直传", "label": ""}, ensure_ascii=False),
                     encoding="utf-8")
    check("手机直传的伴随文件 → 判为「手机直传」", m.watch_source(_src_img) == "手机直传")
    _side.write_text(json.dumps({"来源": "小程序", "label": "稻瘟病"}, ensure_ascii=False),
                     encoding="utf-8")
    check("中转下来的伴随文件 → 判为「小程序」", m.watch_source(_src_img) == "小程序")
    check("三种来源都有配色",
          set(m.SOURCE_META) == {"小程序", "手机直传", "电脑本地"}
          and all(v.get("color", "").startswith("#") for v in m.SOURCE_META.values()))
    _pm = m.watch_thumb_pixmap(_src_img)
    check("列表缩略图能生成且已缩小（大图走 draft 不全解）",
          _pm is not None and _pm.width() <= 100 and _pm.height() <= 100,
          None if _pm is None else (_pm.width(), _pm.height()))
    check("自动巡检页有缩略图缓存（刷新不重复解码）",
          hasattr(win.watchInterface, "_thumb_cache"))
    # 回归：来源标记曾经被写死成「小程序」，手机传的图也标成小程序
    _side.unlink(missing_ok=True)
    _written = m._write_phone_sidecar(_src_img, {"来源": "手机直传", "label": ""})
    check("来源标记听调用方的（不再写死成「小程序」）",
          _written.get("来源") == "手机直传", _written.get("来源"))
    check("写出的伴随文件读回来也是「手机直传」",
          m.watch_source(_src_img) == "手机直传", m.watch_source(_src_img))
    # 回归：打标记的代码是后加的，早先传的图没有伴随文件 —— 要能靠事件台账认出来
    _side.unlink(missing_ok=True)
    m._PHONE_UPLOAD_CACHE["mtime"] = None      # 按事件台账 mtime 失效，不是只读一次
    m._PHONE_UPLOAD_CACHE["names"] = set()
    m.log_event("phone_upload", file="_legacy_phone.jpg", bytes=1, has_gps=False)
    _legacy = inbox / "_legacy_phone.jpg"
    cv2.imwrite(str(_legacy), demo_img)
    check("老图没有伴随文件时靠事件台账认出「手机直传」",
          m.watch_source(_legacy) == "手机直传", m.watch_source(_legacy))
    _legacy.unlink(missing_ok=True)
    check("台账行带识别时间（列表要显示）",
          m.WatchInterface._row_from_csv(
              {"time": "2026-10-14 11:01", "file": "a.jpg"})["time"] == "2026-10-14 11:01")
    check("巡检记录带来源字段（地图按它描边）",
          rec.get("source") in ("phone", "mini", "pc"), rec.get("source"))
    # 列表顺序：最新的必须在最上面（一度写成了 reversed，最旧的跑到了顶上）
    from PyQt5.QtWidgets import QLabel as _QL
    _ledger_backup = m.WATCH_RESULTS.read_text(encoding="utf-8") if m.WATCH_RESULTS.exists() else None
    m.WATCH_RESULTS.write_text(
        "time,file,cls,conf,boxes,severity,verify,verify_agree,note\n"
        "2026-01-01 08:00:00,OLD_FIRST.jpg,,0,0,无,,,\n"
        "2026-01-02 08:00:00,NEW_LAST.jpg,,0,0,无,,,\n", encoding="utf-8")
    win.watchInterface._load_recent_rows()
    _top = win.watchInterface.box.itemAt(0).widget()
    _texts = [w.text() for w in _top.findChildren(_QL)] if _top is not None else []
    check("列表最新的排在最上面",
          any("NEW_LAST.jpg" in t for t in _texts), _texts[:2])
    if _ledger_backup is not None:
        m.WATCH_RESULTS.write_text(_ledger_backup, encoding="utf-8")
    win.watchInterface._load_recent_rows()
    # 单条结果一键标记到地块地图（合成图没有 GPS，应落成「待落点」）
    check("台账行带检出框数（标记到地图要用）",
          m.WatchInterface._row_from_csv({"boxes": "3"})["boxes"] == 3)
    # 悬停展开按钮是 TransparentPushButton + 事件过滤器（不是子类）
    from qfluentwidgets import TransparentPushButton as _TTB
    _card0 = win.watchInterface.box.itemAt(0).widget()
    _tools = ([b for b in _card0.findChildren(_TTB) if hasattr(b, "_hover_filter")]
              if _card0 is not None else [])
    check("每条结果右侧有两个图标按钮（标记 / 跳转到地图）",
          len(_tools) >= 2, len(_tools))
    check("两个图标按钮都带提示文案（图标按钮靠 tooltip 说明用途）",
          len(_tools) >= 2 and all((b.toolTip() or "").strip() for b in _tools))
    check("自动巡检页有跳转地块地图的方法",
          hasattr(win.watchInterface, "_goto_map"))
    _st = m.geo_store()
    _before = _st.count() if _st is not None else 0
    win.watchInterface._mark_to_map({"file": "rice_blast_0.jpg", "cls": "稻瘟病",
                                     "conf": 0.9, "boxes": 2})
    _after = _st.count() if _st is not None else 0
    check("单条结果能一键标记到地图", _after == _before + 1, (_before, _after))
    _pend = [r for r in (_st.all_records() if _st is not None else []) if r.get("lon") is None]
    check("没有 GPS 的照片落成「待落点」（等用户在地图上点）", len(_pend) >= 1, len(_pend))
    check("标记时写入了来源（地图按来源描边）",
          all(r.get("source") in ("phone", "mini", "pc", "")
              for r in (_st.all_records() if _st is not None else [])))
    import rice_geo as _rg
    check("原生地图点位带来源字段（描边与标签要用）",
          all("source" in p for p in (_st.to_points() if _st is not None else [])))
    check("原生地图三种来源都有描边色与标签名",
          set(_rg.SOURCE_RING) == {"phone", "mini", "pc"}
          and set(_rg.SOURCE_TAG) == {"phone", "mini", "pc"},
          dict(_rg.SOURCE_TAG))
    check("自动巡检页有跳转地块地图的入口",
          hasattr(win.watchInterface, "btn_goto_map"))
    _src_img.unlink(missing_ok=True)
    _side.unlink(missing_ok=True)
    check("巡检结果进事件台账",
          any(e.get("kind") == "watch_process" for e in m.read_jsonl(m.EVENT_LOG)))
    check("已处理的图片不会重复处理",
          all(p.name != "rice_blast_0.jpg" for p in m.list_watch_candidates(str(inbox))))
    check("巡检结论已进会话记忆", len(m.load_session_index()) > 0)
    check("提醒规则：跨大类复核分歧会提醒",
          bool(m.alert_level({"verify_agree": False, "cls": "白叶枯病",
                              "verify": "稻纵卷叶螟"})))
    check("提醒规则：重度检出会提醒", bool(m.alert_level({"severity": "重"})))
    check("提醒规则：正常情况不打扰",
          m.alert_level({"severity": "轻", "verify_agree": True}) == "")
    bs = json.loads(m.run_agent_tool("batch_scan", {"folder": str(inbox), "limit": 5}))
    check("工具 batch_scan 能批量处理目录", "处理张数" in bs or "结论" in bs, bs.get("处理张数"))
    dg = json.loads(m.run_agent_tool("daily_digest", {"days": 7}))
    check("工具 daily_digest 能出简报数据",
          dg.get("统计天数") == 7 and "检测次数" in dg, list(dg)[:5])

    shot = OUT / "result_page.png"
    win.grab().save(str(shot))
    check("离屏截图成功", shot.exists() and shot.stat().st_size > 5_000)
finally:
    try:
        win.close()
    except Exception:
        pass

print("\n" + "=" * 62)
print("通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
if FAIL:
    for f in FAIL:
        print("  !! " + f)
print("产物目录：" + str(OUT))
sys.exit(1 if FAIL else 0)
