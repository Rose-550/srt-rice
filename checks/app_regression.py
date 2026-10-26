# -*- coding: utf-8 -*-
"""回归自检：接入地图功能后，主程序原有窗口是否都还能正常创建。

用法：python checks/app_regression.py
"""
import os, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "_reg_out"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RICE_GEO_DIR"] = str(OUT)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import rice_dp1
import rice_geo
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFontDatabase, QFont

FAIL = []
def check(name, ok, extra=""):
    print(("  [OK] " if ok else "  [FAIL] ") + name + (("  -> " + str(extra)) if extra else ""))
    if not ok:
        FAIL.append(name)

app = QApplication(sys.argv)
from PyQt5.QtWidgets import QMessageBox as MB
MB.warning = staticmethod(lambda *a, **k: None); MB.information = staticmethod(lambda *a, **k: None)
MB.question = staticmethod(lambda *a, **k: MB.Yes); MB.critical = staticmethod(lambda *a, **k: None)
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
           "C:/Windows/Fonts/seguiemj.ttf"):
    if os.path.exists(_f):
        QFontDatabase.addApplicationFont(_f); break
app.setFont(QFont("Microsoft YaHei", 10))

rice_dp1.CONFIG["enable_llm"] = False     # 自检不调用云端 AI

print("== 主程序原有窗口 ==")
home = rice_dp1.HomeWindow()
check("HomeWindow", home is not None)
labels = [b.text() for b in home.findChildren(rice_dp1.AnimatedButton)]
check("首页含「地块地图 · 溯源记录」按钮", any("地块地图" in t for t in labels), labels)
check("首页原有按钮未丢", any("拍照上传" in t for t in labels) and any("摄像头" in t for t in labels)
      and any("退出" in t for t in labels))

img_win = rice_dp1.ImageRecogWindow(home.model, home.device)
check("ImageRecogWindow", img_win is not None)
check("新增 current_path 属性", hasattr(img_win, "current_path"))

res = [{"name": "稻瘟病", "conf": 0.91, "box": [10.0, 10.0, 200.0, 200.0]}]
img = np.zeros((240, 320, 3), dtype=np.uint8)
res_win = rice_dp1.ResultWindow(home.model, home.device, res, img, "image", None)
check("ResultWindow(6参数) 可创建", res_win is not None)
check("ResultWindow 兼容旧 5 参数调用",
      rice_dp1.ResultWindow(home.model, home.device, res, img, "camera") is not None)

api = rice_dp1.ApiSettingsWindow()
check("ApiSettingsWindow", api is not None)
check("设置页有高德 Key 输入框", hasattr(api, "amap_key_input") and hasattr(api, "amap_js_input"))
api.amap_key_input.setText("TEST_WEB_KEY"); api.amap_js_input.setText("TEST_JS_KEY")
saved = {}
rice_dp1.save_config = lambda cfg: saved.update(cfg)      # 不写真实 config.json
api._save()
check("高德 Key 可保存到配置", saved.get("amap_web_key") == "TEST_WEB_KEY"
      and saved.get("amap_js_key") == "TEST_JS_KEY", (saved.get("amap_web_key"), saved.get("amap_js_key")))
check("原有 AI 配置字段仍在", "deepseek_api_key" in saved and "llm_provider" in saved)

print("== 首页截图（含新按钮） ==")
home.show()
def shot():
    home.grab().save(str(OUT / "home_with_map_button.png"))
    app.quit()
QTimer.singleShot(1500, shot)
app.exec_()
check("首页截图已生成", (OUT / "home_with_map_button.png").exists())

print()
print("=" * 60)
print("失败项：", FAIL if FAIL else "无，全部通过 ✅")
sys.exit(1 if FAIL else 0)
