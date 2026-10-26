# -*- coding: utf-8 -*-
"""真实 API 诊断（走用户 config.json 里的 Key，不打印 Key）。

用途：换模型 / 换服务商 / 怀疑「AI 不回话」时，先用它把链路和参数确认一遍，再看界面。
    python checks/live_api_probe.py

会依次验证：
  1. 当前配置的模型名是否可用（含已停服的旧名会怎样）
  2. 思考模式开关的真实差别（关闭=直接回答；开启=先出思维链，token 消耗成倍）
  3. 图片能不能真的发进去（多模态链路）
"""
import base64
import io
import json
import sys
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config.json"
QUESTION = "用一句话说明稻瘟病穗颈瘟为什么要抓破口期防治。"


def load_key():
    if not CFG.exists():
        print("没有 config.json，请先在程序「API 设置」里填 Key")
        sys.exit(1)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    key = cfg.get("deepseek_api_key") or cfg.get("custom_api_key") or ""
    model = cfg.get("deepseek_model") or cfg.get("custom_model") or "deepseek-flash"
    print("配置文件：%s" % CFG)
    print("Key：%s | 配置的模型：%s" % ("已读到" if key else "缺失", model))
    return key, model


def call(url, key, payload, label):
    try:
        r = requests.post(url, headers={"Authorization": "Bearer " + key,
                                        "Content-Type": "application/json"},
                          json=payload, timeout=(10, 180))
    except Exception as e:
        print("%s 请求异常：%s %s" % (label, type(e).__name__, str(e)[:150]))
        return None
    if r.status_code != 200:
        print("%s HTTP %s：%s" % (label, r.status_code, str(r.text)[:200]))
        return None
    d = r.json()
    msg = (d.get("choices") or [{}])[0].get("message", {})
    print("%s content=%r" % (label, (msg.get("content") or "")[:90]))
    print("%s reasoning 长度=%d | completion_tokens=%s"
          % (label, len(msg.get("reasoning_content") or ""),
             (d.get("usage") or {}).get("completion_tokens")))
    return d


def main():
    key, model = load_key()
    url = "https://api.deepseek.com/v1/chat/completions"
    base = {"model": model, "stream": False, "max_tokens": 600,
            "messages": [{"role": "user", "content": QUESTION}]}

    print("\n[1] 当前配置的模型")
    call(url, key, dict(base, thinking={"type": "disabled"}), "  关闭思考 →")

    print("\n[2] 思考模式对比")
    call(url, key, dict(base, thinking={"type": "enabled"}), "  开启思考 →")

    if "deepseek" in url:
        print("\n[3] 旧模型名（deepseek-chat，官方公告称 2026-07-24 停止服务）")
        call(url, key, dict(base, model="deepseek-chat", thinking={"type": "disabled"}),
             "  旧名调用 →")

    print("\n[4] 图片输入（64x64 纯红图）")
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 30, 30)).save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    call(url, key, {"model": model, "max_tokens": 200, "stream": False,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": "data:image/jpeg;base64," + b64}},
                        {"type": "text", "text": "这张图主要是什么颜色？只答颜色。"}]}]},
         "  看图 →")
    print("\n结论怎么看：content 为空但 reasoning 很长 ⇒ 思考模式把预算吃光了，"
          "请在设置里选「快速回答」或调大 max_tokens。")


if __name__ == "__main__":
    main()
