# -*- coding: utf-8 -*-
"""端到端自检：主程序 + 地块地图模块联调（离屏截图）

检查：主程序能否导入地图模块、首页按钮是否可用、识别结果能否一键记录到地图、
      GPS 是否自动落点、窗口风格是否与主程序一致。
"""
import os, sys, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "_app_out"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RICE_GEO_DIR"] = str(OUT)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PIL import Image
import numpy as np

# 造一张带 GPS/时间的「手机照片」
photo = OUT / "field_photo.jpg"
exif = Image.Exif()
exif[306] = "2026:06:18 15:40:12"
exif[271] = "HUAWEI"; exif[272] = "Mate 60"
exif[36867] = "2026:06:18 15:40:12"
exif[34853] = {1: "N", 2: (32.0, 2.0, 30.0), 3: "E", 4: (118.0, 47.0, 40.0)}
Image.new("RGB", (800, 600), (40, 130, 70)).save(photo, exif=exif)

import rice_dp1
import rice_geo
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFontDatabase, QFont

FAIL = []
def check(name, ok, extra=""):
    print(("  [OK] " if ok else "  [FAIL] ") + name + (("  -> " + str(extra)) if extra else ""))
    if not ok:
        FAIL.append(name)

app = QApplication(sys.argv)
# 离屏自检：模态弹窗会卡住/崩溃，统一屏蔽
from PyQt5.QtWidgets import QMessageBox
QMessageBox.warning = staticmethod(lambda *a, **k: None)
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.critical = staticmethod(lambda *a, **k: None)
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
           "C:/Windows/Fonts/seguiemj.ttf"):
    if os.path.exists(_f):
        QFontDatabase.addApplicationFont(_f)
        break
app.setFont(QFont("Microsoft YaHei", 10))

print("== A. 主程序接入 ==")
check("rice_geo 已导入", rice_dp1.GEO_AVAILABLE)
check("类别已同步", rice_geo.CLASS_NAMES == rice_dp1.CLASS_NAMES, rice_geo.CLASS_NAMES[:3])
check("阈值已同步", rice_geo.CONF_THRESHOLD == rice_dp1.CONF_THRESHOLD)
check("窗口装饰已注入", rice_geo._CHROME["round"] is not None and rice_geo._CHROME["home_factory"] is not None)

print("== B. 首页 → 地块地图 ==")
home = rice_dp1.HomeWindow()
home.show()
btns = [b.text() for b in home.findChildren(type(home).__mro__[0].__dict__.get('x', object)) ] if False else []
home.go_map()
win = getattr(home, "_next", None)
check("首页可跳转到地图页", win is not None and isinstance(win, rice_geo.GeoMapWindow))
win.show(); win.resize(1280, 850)
print("   地图窗口标题栏按钮:", hasattr(win, "_btn_fs"), "| 数据目录:", win.store.root)

print("== C. 识别结果 → 记录到地图 ==")
rice_dp1.CONFIG["enable_llm"] = False          # 自检不调用云端 AI
img = np.zeros((480, 640, 3), dtype=np.uint8); img[:, :] = (30, 120, 60)
results = [{"name": "稻瘟病", "conf": 0.93, "box": [120.0, 90.0, 420.0, 380.0]}]
rw = rice_dp1.ResultWindow(None, "cpu", results, img, "image", str(photo))
rw.show()
check("识别结果页有「记录到地图」按钮", hasattr(rw, "btn_map"))
check("识别结果页记录了来源照片", rw.source_path == str(photo))
rw._save_to_map()
win2 = getattr(rw, "_next", None)
check("跳转到地图页并带入结果", win2 is not None and isinstance(win2, rice_geo.GeoMapWindow))
win2.show(); win2.resize(1280, 850)
win = win2
print("   [dbg] 两个地图窗口已就绪，进入事件循环")
_beat = {"n": 0}
def _hb():
    _beat["n"] += 1
    print("   [dbg] 事件循环心跳 %d" % _beat["n"])
_hbt = QTimer(); _hbt.timeout.connect(_hb); _hbt.start(1000)

state = {"paints": 0}
_pump = QTimer(); _pump.timeout.connect(lambda: (win.canvas.update(),
                                                 state.__setitem__("paints", state["paints"] + 1)))
_pump.start(200)

def finish():
    recs = win.store.all_records()
    print("   记录数:", len(recs))
    if recs:
        r = recs[0]
        print("   记录:", {k: r[k] for k in ("cls", "conf", "shot_at", "lat", "lon", "glon", "glat", "coord_src", "device")})
    check("已写入记录", len(recs) == 1, len(recs))
    check("识别类别/置信度已保存", bool(recs) and recs[0]["cls"] == "稻瘟病" and abs(recs[0]["conf"] - 0.93) < 1e-6)
    check("拍摄时间来自 EXIF", bool(recs) and (recs[0]["shot_at"] or "").startswith("2026-06-18 15:40"))
    check("GPS 自动落点", bool(recs) and recs[0]["lat"] is not None and abs(recs[0]["lat"] - 32.0417) < 0.002)
    check("坐标来源=照片EXIF", bool(recs) and recs[0]["coord_src"] == "exif")
    check("已换算高德坐标", bool(recs) and recs[0]["glon"] is not None and abs(recs[0]["glon"] - recs[0]["lon"]) > 0.001)
    check("原图已归档", bool(recs) and os.path.exists(recs[0]["orig_path"]))
    win.grab().save(str(OUT / "app_geo_map.png"))
    win.canvas.zoom = 16
    win.canvas.set_center(recs[0]["glon"] if recs else 118.8, recs[0]["glat"] if recs else 32.04)
    win.canvas.update()
    QTimer.singleShot(2500, lambda: (win.grab().save(str(OUT / "app_geo_zoom.png")), app.quit()))

QTimer.singleShot(11000, finish)
app.exec_()

print()
print("=" * 60)
print("失败项：", FAIL if FAIL else "无，全部通过 ✅")
print("产物：", OUT)
sys.exit(1 if FAIL else 0)
