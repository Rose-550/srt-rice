# -*- coding: utf-8 -*-
"""流式输出端到端自检：起一个本地假服务当「大模型」，把 A2 流式改造整条链路跑通。

为什么要这个自检：流式涉及 requests 的 iter_lines、SSE 解析、QThread 信号、界面节流重绘四层，
只测其中一层说明不了问题。这里用本地 HTTP 服务伪造 OpenAI 兼容的 SSE，全程走真实网络栈，
不需要 API Key，也不花一分钱。

用法：python checks/llm_stream_smoke.py
"""
import os
import sys
import json
import time
import shutil
import threading
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "_stream_out"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RICE_APP_DIR"] = str(OUT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

FAIL, PASS = [], []


def check(name, ok, extra=""):
    (PASS if ok else FAIL).append(name)
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  -> " + str(extra)) if extra else ""))


# ── 假模型服务：支持 SSE 流式 / 一次性两种返回 ──────────────────
PIECES = ["## 诊断\n", "这是", "稻瘟病", "的典型", "症状。\n", "- 建议一\n", "- 建议二\n"]
THINK_PIECES = ["先看症状…", "叶瘟是梭形斑…", "还要问破口期…"]     # 思考模式的思维链分片
MODE = {"stream": True, "delay": 0.03, "tools": False,   # tools=True 时先回一个工具调用
        "fail_models": [],                              # 命中这些子串的模型直接返回 500（测降级）
        "usage": False,                                 # True 时末尾分片带 usage
        "vision_text": "",                              # 视觉复核（带图请求）返回的文本
        "json_text": "",                                # 规划/抽取（response_format=json）返回的文本
        "replan_text": ""}                              # 重规划请求（带失败观察）返回的修正计划
REQS = []                                                          # 记录收到的请求体，供断言用


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # 静音
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        REQS.append(body)
        if any(k in str(body.get("model", "")) for k in MODE["fail_models"]):
            err = json.dumps({"error": {"message": "模拟服务商故障"}},
                             ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        want_stream = bool(body.get("stream")) and MODE["stream"]
        thinking = (body.get("thinking") or {}).get("type") == "enabled"
        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            first_round = not any(m.get("role") == "tool"
                                  for m in (body.get("messages") or []))
            deltas = []
            if MODE["tools"] and body.get("tools") and first_round:
                # 先回一个工具调用（分两片下发，模拟真实 API 的 arguments 增量）
                deltas.append({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                               "function": {"name": "get_treatment",
                                                            "arguments": ""}}]})
                deltas.append({"tool_calls": [{"index": 0, "function": {
                    "arguments": '{"cls_name": "稻瘟病"}'}}]})
            else:
                if thinking:
                    deltas += [{"reasoning_content": x} for x in THINK_PIECES]
                deltas += [{"content": x} for x in PIECES]
            for d in deltas:
                payload = json.dumps({"choices": [{"delta": d}]}, ensure_ascii=False)
                data = ("data: " + payload + "\n\n").encode("utf-8")
                chunk = ("%X\r\n" % len(data)).encode() + data + b"\r\n"
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(MODE["delay"])
            if MODE["usage"]:
                u = json.dumps({"choices": [], "usage": {
                    "prompt_tokens": 1200, "completion_tokens": 300,
                    "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1200}},
                    ensure_ascii=False)
                ub = ("data: " + u + "\n\n").encode("utf-8")
                self.wfile.write(("%X\r\n" % len(ub)).encode() + ub + b"\r\n")
                self.wfile.flush()
            done = b"data: [DONE]\n\n"
            self.wfile.write(("%X\r\n" % len(done)).encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            # 不支持流式：返回一次性 JSON。按用途分三种人设：
            #   ① 要 JSON（规划/抽取）→ 返回 MODE["json_text"]
            #   ② 带图（视觉复核）→ 返回 MODE["vision_text"]
            #   ③ 其它 → 返回固定诊断文本
            has_image = any(isinstance(msg.get("content"), list)
                            for msg in (body.get("messages") or []))
            wants_json = (body.get("response_format") or {}).get("type") == "json_object"
            if wants_json:
                user_text = " ".join(str(msg.get("content", ""))
                                     for msg in (body.get("messages") or []))
                # 重规划请求里会带上"失败了"的观察结果，据此返回修正计划
                if "失败了" in user_text and MODE.get("replan_text"):
                    content = MODE["replan_text"]
                else:
                    content = MODE.get("json_text") or "{}"
            elif has_image:
                content = MODE.get("vision_text") or "".join(PIECES)
            else:
                content = "".join(PIECES)
            payload = json.dumps({"choices": [{"message": {"content": content}}],
                                  "usage": {"prompt_tokens": 900, "completion_tokens": 40,
                                            "prompt_cache_hit_tokens": 0,
                                            "prompt_cache_miss_tokens": 900}},
                                 ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
print("假模型服务已启动：%s" % URL)

# ── 导入被测程序并改指向本地服务 ──────────────────────────────
import rice_dp1_fluent as m

m.API_ENDPOINTS["deepseek"] = URL
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["enable_llm"] = True
m.CONFIG["deepseek_api_key"] = "test-key"
m.CONFIG["llm_max_tokens"] = 512
m.CONFIG["enable_llm_cache"] = False   # 界面/停止/思考这些小节要的是"每次真发请求"，缓存留到第 7 节再单独验

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop, QTimer

app = QApplication(sys.argv)


def wait_until(pred, timeout_ms=15000):
    """跑事件循环直到条件成立（界面更新靠事件循环驱动）"""
    loop = QEventLoop()
    deadline = time.time() + timeout_ms / 1000.0
    timer = QTimer()
    timer.setInterval(50)

    def tick():
        if pred() or time.time() > deadline:
            loop.quit()

    timer.timeout.connect(tick)
    timer.start()
    loop.exec_()
    timer.stop()
    return pred()


print("\n== 1. 纯线程层：流式拼接 ==")
th = m.LLMAnalysisThread("稻瘟病")
got = []
th.chunk.connect(lambda s: got.append(s))
final = []
th.result_ready.connect(lambda s: final.append(s))
th.failed.connect(lambda s: final.append("ERR:" + s))
th.start()
ok = wait_until(lambda: bool(final), 20000)
th.wait(3000)
check("线程拿到最终结果", ok and final and not final[0].startswith("ERR:"), final[:1])
check("最终内容完整", final and final[0] == "".join(PIECES))
check("分片多次到达（真的在流式）", len(got) >= 3, "%d 个分片" % len(got))
check("分片是累计全文", got and got[-1] == "".join(PIECES))

print("\n== 2. 界面层：边收边显示 ==")
from rice_dp1_fluent import ResultInterface


class _StubMain:
    device = "cpu"
    class _Rep:
        def reload(self):
            pass

    reportInterface = _Rep()


ri = ResultInterface(_StubMain())
img = np.full((480, 640, 3), 230, dtype=np.uint8)
img[150:300, 200:420] = (110, 140, 190)
ri.show_result([{"name": "稻瘟病", "conf": 0.9, "box": [200, 150, 420, 300]}], img, "image")
seen_streaming = {"mid": False}


def _snapshot_mid():
    if ri._ai_busy and any("稻瘟病" in e[1] for e in ri._chat_history if e[0] == "ai_initial"):
        seen_streaming["mid"] = True
    return False


wait_until(lambda: (ri._ai_busy and _snapshot_mid()) or not ri._ai_busy, 20000)
done = wait_until(lambda: not ri._ai_busy, 20000)
check("界面收到流式内容", any("稻瘟病" in e[1] for e in ri._chat_history if e[0] == "ai_initial"),
      [e[0] for e in ri._chat_history])
check("生成中处于 busy 状态（按钮已切换）", not ri._ai_busy and not ri.btn_stop.isVisible())
check("结束后内容与假服务一致",
      any(e[0] == "ai_initial" and e[1] == "".join(PIECES) for e in ri._chat_history))
check("生成中确实出现过中间态", seen_streaming["mid"])

print("\n== 3. 停止生成 ==")
MODE["delay"] = 0.25                      # 放慢，好在中途喊停
ri.show_result([{"name": "稻瘟病", "conf": 0.9, "box": [200, 150, 420, 300]}], img, "image")
wait_until(lambda: ri._ai_busy and len(ri._chat_history) and
           len(ri._chat_history[-1][1]) > 2, 10000)
ri._stop_generate()
stopped = wait_until(lambda: not ri._ai_busy, 10000)
check("停止后回到可输入状态", stopped and not ri._ai_busy)
partial = [e[1] for e in ri._chat_history if e[0] == "ai_initial"]
check("停止保留已收到内容（不整段丢失）", bool(partial) and len(partial[0]) > 0,
      "已保留 %d 字" % (len(partial[0]) if partial else 0))
check("停止后标注是部分内容",
      any(e[0] == "system" and "已停止" in e[1] for e in ri._chat_history))
MODE["delay"] = 0.02

print("\n== 3b. 思考模式（reasoning_content） ==")
m.CONFIG["llm_thinking"] = False
REQS.clear()
th3 = m.LLMAnalysisThread("稻瘟病")
think3, out3 = [], []
th3.reasoning.connect(lambda s: think3.append(s))
th3.result_ready.connect(lambda s: out3.append(s))
th3.start()
wait_until(lambda: bool(out3), 20000)
th3.wait(3000)
req3 = REQS[-1] if REQS else {}
check("默认关思考：请求里显式 thinking=disabled",
      (req3.get("thinking") or {}).get("type") == "disabled", req3.get("thinking"))
check("关闭思考时保留 temperature（该模式支持）", "temperature" in req3, req3.get("temperature"))
check("关思考：没有思维链信号", not think3, len(think3))

m.CONFIG["llm_thinking"] = True
REQS.clear()
th4 = m.LLMAnalysisThread("稻瘟病")
think4, out4 = [], []
th4.reasoning.connect(lambda s: think4.append(s))
th4.result_ready.connect(lambda s: out4.append(s))
th4.start()
wait_until(lambda: bool(out4), 30000)
th4.wait(3000)
req4 = REQS[-1] if REQS else {}
check("开思考：请求里 thinking=enabled",
      (req4.get("thinking") or {}).get("type") == "enabled", req4.get("thinking"))
check("开思考：token 预算自动抬到 4000",
      int(req4.get("max_tokens", 0)) >= 4000, req4.get("max_tokens"))
check("开思考：不发送 temperature（该模式下不生效）", "temperature" not in req4)
check("思维链分片能收到（用于显示「在思考」）", len(think4) >= 2, "%d 个分片" % len(think4))
check("思考模式下正文仍然完整", out4 and out4[0] == "".join(PIECES), (out4 or [""])[0][:30])

# 界面：思考阶段不能是「卡住不动」的样子
# 把分片间隔拉长，保证「思考中」这个瞬时状态稳定可观测（否则 50ms 轮询会偶发错过）
MODE["delay"] = 0.12
ri2 = ResultInterface(_StubMain())
ri2.show_result([{"name": "稻瘟病", "conf": 0.9, "box": [200, 150, 420, 300]}], img, "image")
saw_thinking = {"hit": False}


def _watch_thinking():
    if any(e[0] == "system" and "深度思考中" in e[1] for e in ri2._chat_history):
        saw_thinking["hit"] = True
    return False


wait_until(lambda: (_watch_thinking(), not ri2._ai_busy)[1], 30000)
check("思考阶段界面显示「深度思考中…」", saw_thinking["hit"],
      [e[1][:16] for e in ri2._chat_history][:3])
MODE["delay"] = 0.02
m.CONFIG["llm_thinking"] = False

print("\n== 5. Agent 工具调用闭环（真实网络栈） ==")
MODE["tools"] = True
th5 = m.LLMAnalysisThread("稻瘟病")
used, out5 = [], []
th5.tool_used.connect(lambda n, l: used.append((n, l)))
th5.result_ready.connect(lambda s: out5.append(s))
th5.failed.connect(lambda s: out5.append("ERR:" + s))
th5.start()
wait_until(lambda: bool(out5), 30000)
th5.wait(3000)
check("工具调用信号到达界面（可显示「正在查本地防治库…」）",
      used and used[0][0] == "get_treatment", used)
check("工具调用后仍能拿到最终回答", out5 and out5[0] == "".join(PIECES), (out5 or [""])[0][:40])
second = REQS[-1] if REQS else {}
tool_msgs = [msg for msg in (second.get("messages") or []) if msg.get("role") == "tool"]
check("第二轮请求带上了工具执行结果", bool(tool_msgs), len(tool_msgs))
check("工具结果来自本地防治库",
      tool_msgs and "本地防治库" in (tool_msgs[0].get("content") or ""),
      (tool_msgs[0].get("content") or "")[:60] if tool_msgs else "")
asst = [msg for msg in (second.get("messages") or []) if msg.get("tool_calls")]
check("助手轮次的 tool_calls 已回灌上下文", bool(asst), len(asst))
check("审计日志落盘（llm_log.jsonl）", (OUT / "logs" / "llm_log.jsonl").exists())
MODE["tools"] = False

print("\n== 6. 服务商降级（主服务商挂了自动换备用） ==")
_save = (m.CONFIG.get("llm_provider"), m.CONFIG.get("deepseek_model"), m.CONFIG.get("qwen_model"),
         m.CONFIG.get("dashscope_api_key"), m.CONFIG.get("enable_llm_cache"))
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["deepseek_model"] = "deepseek-flash"
m.CONFIG["qwen_model"] = "qwen-vl-max"
m.CONFIG["dashscope_api_key"] = "test-backup"
m.API_ENDPOINTS["qwen"] = URL
m.API_ENDPOINTS["deepseek"] = URL
MODE["fail_models"] = ["deepseek"]
REQS.clear()
th6 = m.LLMAnalysisThread("稻瘟病")
th6.use_tools = False
fb, out6 = [], []
th6.fallback.connect(lambda s: fb.append(s))
th6.result_ready.connect(lambda s: out6.append(s))
th6.failed.connect(lambda s: out6.append("ERR:" + s))
th6.start()
wait_until(lambda: bool(out6), 30000)
th6.wait(3000)
check("主服务商失败后自动降级", bool(fb) and "备用" in fb[0], fb)
check("降级后仍拿到回答", out6 and out6[0] == "".join(PIECES), (out6 or [""])[0][:30])
_models = [r.get("model") for r in REQS]
# 注意：HTTP 重试会在降级前重试主服务商（urllib3 Retry 配置），所以只断言"先主后备"
check("先试主服务商，失败后试备用",
      _models and _models[0] == "deepseek-flash" and _models[-1] == "qwen-vl-max", _models[:4])
MODE["fail_models"] = []

print("\n== 7. 回答缓存与用量记账 ==")
MODE["usage"] = True
_img = np.zeros((64, 64, 3), np.uint8)
_sev = {"level": "中", "boxes": 1, "area_ratio": 0.05, "max_ratio": 0.05,
        "mean_conf": 0.8, "basis": "测试"}
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["deepseek_model"] = "deepseek-flash"
m.CONFIG["enable_llm_cache"] = True
REQS.clear()
th7 = m.LLMAnalysisThread("稻瘟病", _img.copy(), severity=dict(_sev))
out7 = []
th7.result_ready.connect(lambda s: out7.append(s))
th7.start()
wait_until(lambda: bool(out7), 30000)
th7.wait(3000)
_n1 = len(REQS)
check("用量被记录下来（成本记账的基础）",
      isinstance(th7.usage, dict) and th7.usage.get("prompt_tokens") == 1200, th7.usage)
th8 = m.LLMAnalysisThread("稻瘟病", _img.copy(), severity=dict(_sev))
out8 = []
th8.result_ready.connect(lambda s: out8.append(s))
th8.start()
wait_until(lambda: bool(out8), 20000)
th8.wait(3000)
check("同一张图第二次命中缓存（不再请求服务端）", len(REQS) == _n1, (_n1, len(REQS)))
check("缓存回放内容一致", out8 and out8[0] == out7[0])
MODE["usage"] = False
(m.CONFIG["llm_provider"], m.CONFIG["deepseek_model"], m.CONFIG["qwen_model"],
 m.CONFIG["dashscope_api_key"], m.CONFIG["enable_llm_cache"]) = _save

print("\n== 8. 视觉复核级联（触发规则 + 冲突裁决） ==")
_imgc = np.full((200, 260, 3), 180, dtype=np.uint8)
_imgc[40:160, 60:200] = (90, 130, 170)
_boxc = [{"name": "白叶枯病", "conf": 0.44, "box": [60, 40, 200, 160]}]

_sv = (m.CONFIG.get("llm_provider"), m.CONFIG.get("deepseek_model"),
       m.CONFIG.get("deepseek_api_key"), m.CONFIG.get("enable_llm"))
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["deepseek_model"] = "deepseek-flash"
m.CONFIG["deepseek_api_key"] = "test-key"
m.CONFIG["enable_llm"] = True
_need, _why = m.should_auto_verify("稻水象甲", 0.92)
check("高置信度虫害不触发复核", _need is False, _why)
_need2, _why2 = m.should_auto_verify("稻水象甲", 0.42)
check("低置信度触发复核", _need2 is True and "置信度" in _why2, _why2)
_need3, _why3 = m.should_auto_verify("白叶枯病", 0.95)
check("病害三类即使高置信度也复核", _need3 is True and "病害" in _why3, _why3)
m.CONFIG["cascade_auto"] = False
check("开关能关掉自动复核", m.should_auto_verify("白叶枯病", 0.3)[0] is False)
m.CONFIG["cascade_auto"] = True
check("级别名解析：从自由文本抓类别",
      m.parse_class_from_text("稻纵卷叶螟。依据：叶片卷曲…") == "稻纵卷叶螟"
      and m.parse_class_from_text("看不出来") == "")

MODE["vision_text"] = "胡麻斑病。依据：叶片有椭圆形褐色斑点。"
_re = m.vision_recheck(_imgc, _boxc[0]["box"], None, 0, source="test")
check("视觉复核走通并归类", _re.get("ok") and _re.get("verdict") == "胡麻斑病", _re)
check("视觉复核记了审计（含成本字段）",
      any(r.get("event") == "vision_recheck" for r in m.read_jsonl(m.AUDIT_LOG)),
      len(m.read_jsonl(m.AUDIT_LOG)))

# 冲突裁决：面板出现按钮 → 点裁决 → 写进 feedback
from rice_dp1_fluent import ResultInterface as _RI


class _StubMain2:
    device = "cpu"
    class _Rep:
        def reload(self):
            pass
    reportInterface = _Rep()


_ri = _RI(_StubMain2())
_ri.show_result(_boxc, _imgc, "image")
# 同类分歧（病↔病）：处置方向一致，只记台账、不打扰
_ri._on_cascade({"ok": True, "verdict": "胡麻斑病", "raw": "胡麻斑病。依据…",
                 "model": "deepseek-flash", "cls_name": "白叶枯病", "conf": 0.44,
                 "reason": "置信度 44% 低于 60%", "agrees": False})
check("同类分歧不打扰（一行提示、不给按钮）", _ri.cascade_lay.count() == 1,
      _ri.cascade_lay.count())
# 跨大类分歧（病↔虫）：处置完全不同，必须让人裁决
# 注意：父窗口没 show()，子控件 isVisible() 恒为 False，这里用 isHidden() 判断显隐意图
_ri._on_cascade({"ok": True, "verdict": "稻纵卷叶螟", "raw": "稻纵卷叶螟。依据…",
                 "model": "deepseek-flash", "cls_name": "白叶枯病", "conf": 0.44,
                 "reason": "置信度 44% 低于 60%", "agrees": False})
check("跨大类分歧出现裁决按钮组",
      (not _ri.cascade_box.isHidden()) and _ri.cascade_lay.count() >= 3,
      (_ri.cascade_lay.count(), _ri.cascade_box.isHidden()))
_ri._resolve_cascade("稻纵卷叶螟")
_rows = m.load_feedback_rows()
check("裁决写入反馈队列（可用于补数据）",
      any(r.get("mode") == "cascade" and r.get("user_choice") == "稻纵卷叶螟" for r in _rows),
      _rows[-1] if _rows else None)
check("裁决后面板给出确认文案",
      any("已记录" in (w.text() or "") for w in
          [_ri.cascade_lay.itemAt(i).widget() for i in range(_ri.cascade_lay.count())
           if _ri.cascade_lay.itemAt(i).widget()]))

# 一致时不弹裁决
_ri._on_cascade({"ok": True, "verdict": "白叶枯病", "raw": "白叶枯病", "agrees": True,
                 "cls_name": "白叶枯病", "conf": 0.9, "reason": "病害三类"})
check("一致时只报平安、不出按钮", _ri.cascade_lay.count() == 1, _ri.cascade_lay.count())
MODE["vision_text"] = ""
(m.CONFIG["llm_provider"], m.CONFIG["deepseek_model"], m.CONFIG["deepseek_api_key"],
 m.CONFIG["enable_llm"]) = _sv

print("\n== 9. 自我修正（反思：发现问题自己返工） ==")
_rsv = (m.CONFIG.get("llm_provider"), m.CONFIG.get("deepseek_api_key"),
        m.CONFIG.get("enable_llm"), m.CONFIG.get("enable_llm_cache"))
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["deepseek_api_key"] = "test-key"
m.CONFIG["enable_llm"] = True
m.CONFIG["enable_llm_cache"] = False
ri3 = ResultInterface(_StubMain())
img3 = np.full((240, 320, 3), 200, dtype=np.uint8)
ri3.show_result([{"name": "稻瘟病", "conf": 0.9, "box": [40, 40, 200, 200]}], img3, "image")
wait_until(lambda: not ri3._ai_busy, 25000)          # 等首次分析结束
# 再等诊断卡线程收尾：否则它和后面的手动注入会交叉，检查结果不稳定
wait_until(lambda: getattr(ri3, "card_thread", None) is None
           or not ri3.card_thread.isRunning(), 20000)
ri3._on_ai_result("建议使用百草枯，每亩 100ml 兑水喷雾。")   # 塞一版含禁用农药的回答
ok9 = wait_until(lambda: ri3._revised and not ri3._ai_busy, 25000)
check("含禁用农药的回答触发自动返工", ri3._revised, ri3._revised)
check("返工后出现修正版内容",
      any(e[0] in ("ai", "ai_initial") and len(e[1]) > 5 for e in ri3._chat_history),
      [e[0] for e in ri3._chat_history])
check("返工过程留有说明",
      any(e[0] == "system" and "修正" in e[1] for e in ri3._chat_history),
      [(e[0], str(e[1])[:14]) for e in ri3._chat_history])
check("返工事件进台账",
      any(e.get("kind") == "self_revision_start" for e in m.read_jsonl(m.EVENT_LOG)))
ri3._on_ai_result("正常回答：建议使用三环唑。")           # 再来一次不应重复返工
check("同一轮不重复返工", ri3._revised is True and not ri3._ai_busy)
(m.CONFIG["llm_provider"], m.CONFIG["deepseek_api_key"], m.CONFIG["enable_llm"],
 m.CONFIG["enable_llm_cache"]) = _rsv

print("\n== 10. 任务规划与执行（端到端：计划 → 确认 → 执行 → 汇报） ==")
_rsv2 = (m.CONFIG.get("llm_provider"), m.CONFIG.get("deepseek_api_key"),
         m.CONFIG.get("enable_llm"), m.CONFIG.get("enable_llm_cache"))
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["deepseek_api_key"] = "test-key"
m.CONFIG["enable_llm"] = True
m.CONFIG["enable_llm_cache"] = False
MODE["json_text"] = json.dumps({
    "goal": "把收件箱跑一遍并出简报",
    "steps": [{"tool": "batch_scan", "args": {"limit": 3}, "why": "处理收件箱里的新图"},
              {"tool": "daily_digest", "args": {"days": 7}, "why": "汇总最近运行数据"}],
    "note": "两步都是只读操作"}, ensure_ascii=False)
ri4 = ResultInterface(_StubMain())
img4 = np.full((200, 260, 3), 190, dtype=np.uint8)
ri4.show_result([{"name": "稻瘟病", "conf": 0.9, "box": [30, 30, 200, 180]}], img4, "image")
wait_until(lambda: not ri4._ai_busy, 25000)
ri4.ai_input.setText("把收件箱里的图跑一遍，再给我一份简报")
ri4._plan_task()
wait_until(lambda: ri4._pending_plan is not None or not ri4._ai_busy, 25000)
_plan4 = ri4._pending_plan or {}
check("规划产出可审阅的计划", len(_plan4.get("steps") or []) == 2, _plan4.get("steps"))
check("计划里出现了真实工具名",
      [s["tool"] for s in _plan4.get("steps", [])] == ["batch_scan", "daily_digest"],
      [s.get("tool") for s in _plan4.get("steps", [])])
check("计划未执行前先等人确认（确认条可见）", not ri4.plan_bar.isHidden())
ri4._run_plan()
_done4 = wait_until(lambda: (not ri4._ai_busy
                             and any(e[0] == "system" and "执行完毕" in e[1]
                                     for e in ri4._chat_history)), 40000)
check("按计划逐步执行并给出汇报", _done4,
      [e[1][:24] for e in ri4._chat_history[-4:]])
check("计划确认条执行后收起", ri4.plan_bar.isHidden())
check("任务事件进台账",
      any(e.get("kind") in ("task_planned", "task_run_done", "task_step")
          for e in m.read_jsonl(m.EVENT_LOG)))
MODE["json_text"] = ""
(m.CONFIG["llm_provider"], m.CONFIG["deepseek_api_key"], m.CONFIG["enable_llm"],
 m.CONFIG["enable_llm_cache"]) = _rsv2

print("\n== 11. 计划失败自动重规划（plan → act → observe → replan） ==")
_rq = (m.CONFIG.get("llm_provider"), m.CONFIG.get("deepseek_api_key"),
       m.CONFIG.get("enable_llm"), m.CONFIG.get("enable_llm_cache"))
m.CONFIG["llm_provider"] = "deepseek"
m.CONFIG["deepseek_api_key"] = "test-key"
m.CONFIG["enable_llm"] = True
m.CONFIG["enable_llm_cache"] = False
MODE["json_text"] = json.dumps({
    "goal": "把收件箱跑一遍",
    "steps": [{"tool": "batch_scan", "args": {"folder": "不存在的目录"}, "why": "故意给错路径"}],
    "note": ""}, ensure_ascii=False)
MODE["replan_text"] = json.dumps({
    "steps": [{"tool": "daily_digest", "args": {"days": 3}, "why": "改成能跑的汇总步骤"}],
    "note": "原路径不存在，改为汇总运行数据"}, ensure_ascii=False)
ri5 = ResultInterface(_StubMain())
img5 = np.full((200, 260, 3), 185, dtype=np.uint8)
ri5.show_result([{"name": "稻瘟病", "conf": 0.9, "box": [30, 30, 200, 180]}], img5, "image")
wait_until(lambda: not ri5._ai_busy, 25000)
ri5.ai_input.setText("把收件箱跑一遍")
ri5._plan_task()
wait_until(lambda: ri5._pending_plan is not None or not ri5._ai_busy, 25000)
check("首个计划里带了一个会失败的步骤（测试前提）",
      (ri5._pending_plan or {}).get("steps", [{}])[0].get("tool") == "batch_scan",
      (ri5._pending_plan or {}).get("steps"))
ri5._run_plan()
_ok11 = wait_until(lambda: (any(e[0] == "system" and "执行完毕" in e[1]
                               for e in ri5._chat_history)) and not ri5._ai_busy, 60000)
check("任务在失败后自动重规划并跑完", _ok11,
      [e[1][:26] for e in ri5._chat_history if e[0] == "system"][-4:])
check("对话里说明了这次重规划", any(e[0] == "system" and "自动重规划" in e[1]
                                  for e in ri5._chat_history),
      [e[1][:20] for e in ri5._chat_history if e[0] == "system"][-3:])
check("重规划事件进台账",
      any(e.get("kind") == "task_replan" for e in m.read_jsonl(m.EVENT_LOG)))
check("重规划请求确实带上了失败观察",
      any("失败了" in " ".join(str(mm.get("content", "")) for mm in (r.get("messages") or []))
          for r in REQS))
MODE["json_text"] = ""
MODE["replan_text"] = ""
(m.CONFIG["llm_provider"], m.CONFIG["deepseek_api_key"], m.CONFIG["enable_llm"],
 m.CONFIG["enable_llm_cache"]) = _rq

print("\n== 12. 教学资料用模型结构化（导入 → 出题） ==")
MODE["json_text"] = json.dumps({"questions": [
    {"question": "稻瘟病破口期为什么要施药？",
     "options": ["防穗颈瘟", "防稻飞虱", "补钾肥", "除草"],
     "answer": "防穗颈瘟", "explain": "原文：穗颈瘟只能防不能治", "tags": ["稻瘟病"]}]},
    ensure_ascii=False)
_mat = OUT / "material_sample.txt"
_mat.write_text("稻瘟病破口期必须施药：穗颈瘟只能防不能治，破口初期与齐穗期各施一次。",
                encoding="utf-8")
_res12 = m.import_teaching_file(path=str(_mat), with_llm=True, limit=5)
check("模型把资料结构化成题目", bool(_res12.get("ok")) and _res12.get("questions", 0) >= 1, _res12)
_bank12 = m.load_question_bank()
check("导入的题目带来源标注",
      any("material_sample.txt" in (q.get("source") or "") for q in _bank12),
      [q.get("source") for q in _bank12][-2:])
check("导入题目自洽（答案在选项内）",
      all((not q["options"]) or (q["answer"] in q["options"]) for q in _bank12))
check("导入记录已落盘", (m.TEACHING_DIR / "imports.jsonl").exists())
check("教学统计反映了导入",
      m.teaching_stats()["导入次数"] >= 1, m.teaching_stats())
MODE["json_text"] = ""

print("\n== 4. 服务端不支持流式时自动退回 ==")
MODE["stream"] = False
th2 = m.LLMAnalysisThread("稻瘟病")
out2 = []
th2.result_ready.connect(lambda s: out2.append(s))
th2.chunk.connect(lambda s: out2.append("CHUNK"))
th2.start()
ok2 = wait_until(lambda: any(not x.startswith("CHUNK") for x in out2), 20000)
th2.wait(3000)
check("退回一次性请求成功", ok2 and "".join(PIECES) in out2, out2[:2])
check("退回路径不发分片", "CHUNK" not in out2)
MODE["stream"] = True

# 收尾：把还在跑的 Qt 线程停掉再退出，否则解释器关闭时会崩（0xC0000409）
for _ri in [locals().get("ri"), locals().get("ri2"), locals().get("ri3"), locals().get("ri4")]:
    if _ri is not None:
        try:
            _ri._stop_threads()
            for _t in ("cascade_thread", "card_thread", "summary_thread", "task_thread"):
                _th = getattr(_ri, _t, None)
                if _th is not None and _th.isRunning():
                    _th.wait(2000)
        except Exception:
            pass
m._wait_active_threads(3000)
srv.shutdown()
print("\n" + "=" * 62)
print("通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  !! " + f)
print("产物目录：" + str(OUT))
sys.stdout.flush()
sys.stderr.flush()
# 这个用例反复创建/销毁带 QThread 的控件，解释器正常关闭时偶尔会在 Qt 析构阶段崩
# （0xC0000409）。判定已完成，这里用确定的状态码退出，避免影响结果判读。
os._exit(1 if FAIL else 0)
