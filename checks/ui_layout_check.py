# -*- coding: utf-8 -*-
"""结果与治理页布局核对（离屏）：验证 35/65 主从、右栏标签页、图片自适应。

用法（rice-gpu 环境）：
    python checks/ui_layout_check.py
产物：checks/_ui_layout/result_plan.png / result_ai.png（离屏截图，文字可能不渲染）
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "_ui_layout"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RICE_APP_DIR"] = str(OUT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

import rice_dp1_fluent as m

FAIL = []
PASS = []


def check(name, ok, extra=""):
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  -> " + str(extra)) if extra else ""))
    (PASS if ok else FAIL).append(name)


app = QApplication(sys.argv)
win = m.MainWindow()
win.resize(1380, 900)
win.show()
win.switchTo(win.resultInterface)   # 切到结果页，几何才会展开
QTest.qWait(400)

ri = win.resultInterface

demo = np.full((600, 800, 3), 235, dtype=np.uint8)
demo[200:400, 250:550] = (120, 150, 200)
results = [{"name": "稻瘟病", "conf": 0.88, "box": [250, 200, 550, 400]},
           {"name": "稻瘟病", "conf": 0.61, "box": [560, 210, 700, 330]}]
ri.show_result(results, demo, "image")
for _ in range(3):
    app.processEvents()

# 宽度比例（1380 窗口）
lw, rw = ri.left_card.width(), ri.right_col.width()
ratio_l = lw / max(1, lw + rw)
check("1380 宽下左栏约占 35%", 0.30 <= ratio_l <= 0.40,
      "left=%d right=%d ratio=%.1f%%" % (lw, rw, ratio_l * 100))

# 标签页
check("右栏默认在「治理方案」页", ri.right_stack.currentIndex() == 0,
      ri.right_stack.currentIndex())
ri._switch_right_pane("ai")
app.processEvents()
check("可切到「AI 诊断与问答」页", ri.right_stack.currentIndex() == 1,
      ri.right_stack.currentIndex())
ri.grab().save(str(OUT / "result_ai.png"))
ri._switch_right_pane("plan")
app.processEvents()
ri.grab().save(str(OUT / "result_plan.png"))

# 结果图自适应
pix = ri.img_view.pixmap()
check("结果图已按左栏宽度缩放", pix is not None and not pix.isNull()
      and pix.width() <= max(180, ri.img_view.width()),
      (pix.width() if pix else None, ri.img_view.width()))

# 最小窗口下比例仍成立
win.resize(1120, 720)
QTest.qWait(300)
lw, rw = ri.left_card.width(), ri.right_col.width()
ratio_l = lw / max(1, lw + rw)
check("1120 最小窗口下左栏仍约占 35%", 0.30 <= ratio_l <= 0.40,
      "left=%d right=%d ratio=%.1f%%" % (lw, rw, ratio_l * 100))

# ── 首页布局 ──
win.switchTo(win.homeInterface)
QTest.qWait(400)
home = win.homeInterface
hero = home.findChild(m.BrandHero)
check("首页横幅高度 156", hero is not None and hero.height() == 156,
      hero.height() if hero else None)
cards = home.findChildren(m.HomeActionCard)
check("首页有两个行动卡", len(cards) == 2, len(cards))
wc, rc = home.weather_card.width(), home.right_col.width()
check("首页右栏固定约 376", 360 <= rc <= 392, rc)
check("首页天气卡占剩余宽度", wc > rc, (wc, rc))
check("今日概况已填充", home.today_detect.text().endswith("次"), home.today_detect.text())
check("1120 下首页无横向溢出",
      home.scroll.widget().minimumSizeHint().width() <= home.scroll.viewport().width(),
      (home.scroll.widget().minimumSizeHint().width(), home.scroll.viewport().width()))

# 注入一份合成天气，验证「有数据时」长文本也不把页面撑宽
fake_wx = {"地点": "南京", "坐标": [118.78, 32.06],
           "当前": {"温度": 20.6, "湿度": 62, "风速": 6.6, "降水": 0.0},
           "逐日": [{"日期": "2026-11-02", "最低温": 14.0, "最高温": 25.9,
                     "降雨概率%": 0}],
           "逐小时": [{"时间": "2026-11-02T15:00", "温度": 21,
                       "降雨概率": 0, "云量": 10}],
           "生成时间": "2026-11-02 15:35", "来源": "fresh"}
home._on_weather(fake_wx, "", [{"开始": "2026-11-03T06:00", "结束": "2026-11-03T09:00",
                                "时长小时": 3, "平均评分": 95}])
QTest.qWait(150)
check("加载天气数据后 1120 仍无横向溢出",
      home.scroll.widget().minimumSizeHint().width() <= home.scroll.viewport().width(),
      (home.scroll.widget().minimumSizeHint().width(), home.scroll.viewport().width()))
home.grab().save(str(OUT / "home.png"))

win.close()
print("\n通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
print("产物：" + str(OUT))
sys.exit(1 if FAIL else 0)
