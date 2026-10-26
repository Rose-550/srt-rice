# -*- coding: utf-8 -*-
"""真实链路验收：用真 Key 走程序自己的 LLMAnalysisThread 调一次真实 API。

⚠️ 会真实计费（一次问答约几厘钱）。换模型 / 换服务商 / 怀疑「AI 不回话」时跑它，
   比在界面里肉眼试更快定位问题。前置条件：config.json 里已有可用的 Key。

    python checks/live_app_check.py
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

import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop, QTimer

import rice_dp1_fluent as m


def main():
    cfg_path = ROOT / "config.json"
    if not cfg_path.exists():
        print("没有 config.json，先在程序里配置 API Key")
        return 1
    cfg = m._migrate_config(json.loads(cfg_path.read_text(encoding="utf-8")))
    m.CONFIG.update({"llm_provider": cfg.get("llm_provider", "deepseek"),
                     "deepseek_api_key": cfg.get("deepseek_api_key", ""),
                     "deepseek_model": cfg.get("deepseek_model", "deepseek-flash"),
                     "enable_llm": True, "llm_thinking": False, "vision_override": "auto"})
    print("模型：%s | 能力：%s" % (m.CONFIG["deepseek_model"], m.provider_label()))

    app = QApplication(sys.argv)
    img = cv2.imread(str(ROOT / "手机端可跑版" / "samples" / "rice_blast_0.jpg"))
    res = [{"name": "胡麻斑病", "conf": 0.74, "box": [40, 40, 260, 240]}]
    sev = m.assess_severity(img.shape, res)
    print("检出强度：%s | %s" % (sev["level"], sev["text"]))

    th = m.LLMAnalysisThread("胡麻斑病", img, severity=sev)
    chunks, final, reasons = [], [], []
    t0 = time.time()
    first = {"t": None}

    def on_chunk(s):
        if first["t"] is None:
            first["t"] = time.time() - t0
        chunks.append(s)

    th.chunk.connect(on_chunk)
    th.reasoning.connect(lambda s: reasons.append(s))
    th.result_ready.connect(lambda s: final.append(s))
    th.failed.connect(lambda s: final.append("ERR: " + s))
    th.start()

    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(lambda: loop.quit() if final else None)
    timer.start()
    QTimer.singleShot(90000, loop.quit)
    loop.exec_()
    th.wait(5000)

    print("首字延迟：%s 秒" % (round(first["t"], 2) if first["t"] else "无"))
    print("分片：%d | 思维链信号：%d | 总耗时：%.1f 秒"
          % (len(chunks), len(reasons), time.time() - t0))
    if final and not final[0].startswith("ERR:"):
        print("回答 %d 字，前 200 字：\n%s" % (len(final[0]), final[0][:200].replace("\n", " ")))
        print("禁用农药扫描：%s" % (m.pesticide_hits(final[0]) or "无"))
        return 0
    print("失败：", final)
    return 1


if __name__ == "__main__":
    sys.exit(main())
