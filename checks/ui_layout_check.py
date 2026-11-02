# -*- coding: utf-8 -*-
"""结果与治理页布局核对（离屏）：验证 35/65 主从、右栏标签页、图片自适应。

用法（rice-gpu 环境）：
    python checks/ui_layout_check.py
产物：checks/_ui_layout/result_plan.png / result_ai.png（离屏截图，文字可能不渲染）
"""
import os
import sys
import json
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

# ── 自动巡检布局 ──
win.resize(1380, 900)
win.switchTo(win.watchInterface)
QTest.qWait(350)
wi = win.watchInterface

# 造 10 条台账：5 小程序 / 2 手机直传 / 3 电脑本地
inbox = OUT / "inbox"
inbox.mkdir(parents=True, exist_ok=True)
_lines = ["time,file,cls,conf,boxes,severity,verify,verify_agree,note"]
for i in range(10):
    name = "phone_%02d.jpg" % i
    m.cv2.imwrite(str(inbox / name), demo)
    side = Path(str(inbox / name) + ".phone.json")
    if i % 2 == 0:
        side.write_text('{"来源": "小程序", "label": ""}', encoding="utf-8")
    elif i in (3, 9):
        side.write_text('{"来源": "手机直传", "label": ""}', encoding="utf-8")
    _lines.append("2026-11-02 1%d:%02d:00,%s,白叶枯病,0.8%d,2,重,,,"
                  % (i % 4 + 1, i * 7 % 60, name, i))
m.WATCH_RESULTS.parent.mkdir(parents=True, exist_ok=True)
m.WATCH_RESULTS.write_text("\n".join(_lines) + "\n", encoding="utf-8")
wi._load_recent_rows()
QTest.qWait(250)

check("自动巡检表格最多显示 8 行", wi.box.count() == 8, wi.box.count())
check("来源筛选计数正确（小程序 5）",
      wi._filter_btns["小程序"].text() == "小程序 5",
      wi._filter_btns["小程序"].text())
check("强度长文本归一为「未评估」（不再裁切）",
      m.WatchInterface._severity_display("未评估（手机端）") == ("未评估", "grey")
      and m.WatchInterface._severity_display("重") == ("重", "red")
      and m.WatchInterface._severity_display("") == ("未评估", "grey"),
      m.WatchInterface._severity_display("未评估（手机端）"))
wi._set_filter("小程序")
QTest.qWait(150)
check("点「小程序」后只显示该来源", wi.box.count() == 5, wi.box.count())
wi._set_filter("全部")
QTest.qWait(150)
check("台账状态行带总数", "台账" in wi.status.text() and "共 10 条" in wi.status.text(),
      wi.status.text())
check("监控状态默认「未在监控」",
      wi._monitor_state == "idle" and "未在监控" in wi.mon_chip.text(),
      wi.mon_chip.text())
check("定时任务默认折叠且显示下次运行",
      not wi.sched_detail.isVisible() and "下次运行" in wi.sched_next.text(),
      wi.sched_next.text())
content = wi.scroll.widget()
check("1380 下自动巡检无横向溢出",
      content.minimumSizeHint().width() <= wi.scroll.viewport().width(),
      (content.minimumSizeHint().width(), wi.scroll.viewport().width()))
wi.grab().save(str(OUT / "watch.png"))

win.resize(1120, 720)
QTest.qWait(300)
check("1120 下自动巡检无横向溢出",
      content.minimumSizeHint().width() <= wi.scroll.viewport().width(),
      (content.minimumSizeHint().width(), wi.scroll.viewport().width()))
wi.grab().save(str(OUT / "watch_1120.png"))

# ── 教学与考核（图片考核工作台）──
win.switchTo(win.trainInterface)
QTest.qWait(300)
ti = win.trainInterface
check("1120 教学页无横向溢出",
      ti.scroll.widget().minimumSizeHint().width() <= ti.scroll.viewport().width(),
      (ti.scroll.widget().minimumSizeHint().width(), ti.scroll.viewport().width()))
win.resize(1380, 900)
QTest.qWait(300)
lw, rw = ti.left_card.width(), ti.right_card.width()
check("1380 教学页为 45/55 两栏", 0.40 <= lw / max(1, lw + rw) <= 0.50,
      "left=%d right=%d ratio=%.1f%%" % (lw, rw, lw / max(1, lw + rw) * 100))
a, b, c = ti.opts[0], ti.opts[1], ti.opts[2]
check("答题选项为 2×3 字母网格",
      len(ti.opts) == 7 and len(ti._opt_views) == 6
      and a.y() == b.y() and c.y() > a.y() and abs(a.x() - c.x()) < 4,
      (len(ti.opts), a.x(), a.y(), b.x(), b.y(), c.x(), c.y()))
check("「不确定」占独立一行",
      ti.btn_unsure.y() > ti.opts[2].y(), (ti.btn_unsure.y(), ti.opts[2].y()))
check("讲评卡为三段（结论 / 症状 / 防治）",
      ti.fb_symptom_title.text() == "典型症状"
      and ti.fb_prevent_title.text() == "防治要点"
      and not ti.feedback_card.isHidden())
ti.grab().save(str(OUT / "train_1380.png"))

# ── 图片识别（画布 + 信息栏）──
win.switchTo(win.imageInterface)
QTest.qWait(300)
ii = win.imageInterface
check("图片页空态：开始识别与漏检禁用",
      not ii.btn_run.isEnabled() and not ii.btn_miss.isEnabled()
      and not ii.btn_clear.isVisible(),
      (ii.btn_run.isEnabled(), ii.btn_miss.isEnabled(), ii.btn_clear.isVisible()))
sample = ROOT / "data" / "raw" / "JPEGImages" / "rice_leaf_roller_439.jpg"
ii.load_image(str(sample))
QTest.qWait(300)
check("载入图片后主操作切换",
      ii.btn_run.isEnabled() and ii.btn_miss.isEnabled() and ii.btn_clear.isVisible()
      and ii.file_meta.text().startswith("4608"),
      (ii.btn_run.isEnabled(), ii.btn_clear.isVisible(), ii.file_meta.text()))
check("识别流程步骤 1 完成 / 2 进行中",
      ii.step_rows[0]["sub"].text().startswith("已载入")
      and "点击「开始识别」" in ii.step_rows[1]["sub"].text(),
      [r["sub"].text() for r in ii.step_rows])
check("图片页最近识别有记录",
      "条识别记录" in ii.recent_count.text() and ii.recent_box.count() >= 1,
      (ii.recent_count.text(), ii.recent_box.count()))
check("1380 图片页画布 + 右栏 372",
      ii.right_col.width() == 372 and ii.canvas_card.width() > ii.right_col.width(),
      (ii.canvas_card.width(), ii.right_col.width()))
ii.grab().save(str(OUT / "image_1380.png"))

win.resize(1120, 720)
QTest.qWait(300)
check("1120 图片页无横向溢出",
      ii.scroll.widget().minimumSizeHint().width() <= ii.scroll.viewport().width(),
      (ii.scroll.widget().minimumSizeHint().width(), ii.scroll.viewport().width()))

# ── 报告记录（双标签页）──
win.switchTo(win.reportInterface)
QTest.qWait(300)
rp = win.reportInterface
rep_dir = OUT / "reports"
rep_dir.mkdir(parents=True, exist_ok=True)
sample_path = ROOT / "data" / "raw" / "JPEGImages" / "rice_leaf_roller_439.jpg"
(rep_dir / "index.json").write_text(json.dumps([
    {"path": str(sample_path), "thumb": str(sample_path), "cls_name": "白叶枯病",
     "conf": 0.91, "device": 0, "time": "2026-11-02 12:00"}
], ensure_ascii=False), encoding="utf-8")
rp.reload()
QTest.qWait(200)
check("报告页两个标签页与会话表",
      rp.tabs.widget("sessions").text().startswith("历史会话")
      and rp.tabs.widget("reports").text().startswith("报告文件")
      and rp.sessions_rows.count() >= 1,
      (rp.tabs.widget("sessions").text(), rp.tabs.widget("reports").text(),
       rp.sessions_rows.count()))
rp._switch_tab("reports")
rp.tabs.setCurrentItem("reports")
QTest.qWait(200)
check("报告页能渲染报告文件行", rp.reports_rows.count() >= 1, rp.reports_rows.count())
check("1120 报告页无横向溢出",
      rp.scroll.widget().minimumSizeHint().width() <= rp.scroll.viewport().width(),
      (rp.scroll.widget().minimumSizeHint().width(), rp.scroll.viewport().width()))
win.resize(1380, 900)
QTest.qWait(300)
rp._switch_tab("sessions")
rp.tabs.setCurrentItem("sessions")
QTest.qWait(200)
rp.grab().save(str(OUT / "report_1380.png"))

# ── 地块地图 ──
win.switchTo(win.mapInterface)
QTest.qWait(600)
mi = win.mapInterface
check("地图页地图卡 + 右栏 396",
      mi.right_col.width() == 396 and mi.map_card.width() > mi.right_col.width(),
      (mi.map_card.width(), mi.right_col.width()))
check("地图页风险清单已生成", mi.risk_list.count() >= 1,
      (mi.risk_list.count(), mi.risk_count_chip.text()))
mi.grab().save(str(OUT / "map_1380.png"))
win.resize(1120, 720)
QTest.qWait(300)
check("1120 地图页无横向溢出",
      mi.scroll.widget().minimumSizeHint().width() <= mi.scroll.viewport().width(),
      (mi.scroll.widget().minimumSizeHint().width(), mi.scroll.viewport().width()))

# ── 病害科普 ──
win.resize(1380, 900)
win.switchTo(win.scienceInterface)
QTest.qWait(500)
si = win.scienceInterface
check("科普页 3D 画布 + 右栏 372",
      si.right_col.width() == 372 and si.stage_card.width() > si.right_col.width(),
      (si.stage_card.width(), si.right_col.width()))
btns = [t[0] for t in si.btns.values()]
rows = sorted({b.y() for b in btns})
check("科普页病害选择为 2×3 网格",
      len(btns) == 6 and len(rows) == 3
      and sum(1 for b in btns if b.y() == rows[0]) == 2,
      (len(btns), len(rows)))
check("科普页默认选中稻纵卷叶螟",
      si.btns["稻纵卷叶螟"][0].isChecked()
      and sum(1 for b in btns if b.isChecked()) == 1,
      [n for n, t in si.btns.items() if t[0].isChecked()])
check("科普页详情卡已填充",
      si.lbl_name.text() == "稻纵卷叶螟"
      and si.chip_type.text() == "虫害"
      and si.chip_drugs.text().startswith("本地库·")
      and bool(si.lbl_sym.text()) and all(r.text() for r in si.f_rows)
      and "百丛" in si.lbl_thresh.text(),
      (si.lbl_name.text(), si.chip_type.text(), si.chip_drugs.text(),
       si.lbl_thresh.text()[:18]))
check("科普页 3D 控件在画布卡头",
      si.btn_hd.parentWidget() is si.stage_head
      and si.btn_spin.parentWidget() is si.stage_head
      and si.btn_wire.parentWidget() is si.stage_head
      and si.link_reset.parentWidget() is si.stage_head
      and si.stage_card.isAncestorOf(si.btn_hd)
      and si.view3d is not None,
      (si.btn_hd.text(), si.btn_spin.text(), si.btn_wire.text()))
si.btn_spin.setChecked(False)
si.btn_wire.setChecked(True)
check("科普页自动旋转 / 线框可切换",
      si.view3d.auto_spin is False and si.view3d.wireframe is True,
      (si.view3d.auto_spin, si.view3d.wireframe))
si.btn_wire.setChecked(False)
si.btn_spin.setChecked(True)
si._select("稻瘟病")
QTest.qWait(100)
check("科普页切换病害同步详情与选中态",
      si.lbl_name.text() == "稻瘟病" and si.chip_type.text() == "病害"
      and si.btns["稻瘟病"][0].isChecked()
      and not si.btns["稻纵卷叶螟"][0].isChecked(),
      (si.lbl_name.text(), si.chip_type.text()))
session_dir = OUT / "sessions"
session_dir.mkdir(parents=True, exist_ok=True)
(session_dir / "index.json").write_text(json.dumps([
    {"id": "science_probe", "time": m.datetime.now().strftime("%Y-%m-%d %H:%M"),
     "cls": "稻瘟病", "conf": 0.9, "severity": "中", "turns": 1, "mode": "image"}
], ensure_ascii=False), encoding="utf-8")
si._refresh_recent()
check("科普页近期检出来自识别会话", "1 条记录" in si.lbl_recent.text(),
      si.lbl_recent.text())
si._select("稻纵卷叶螟")
si.grab().save(str(OUT / "science_1380.png"))
win.resize(1120, 720)
QTest.qWait(300)
check("1120 科普页无横向溢出",
      si.scroll.widget().minimumSizeHint().width() <= si.scroll.viewport().width(),
      (si.scroll.widget().minimumSizeHint().width(), si.scroll.viewport().width()))

# ── 1920 全屏（决赛演示基线）──
win.resize(1920, 1080)
QTest.qWait(400)
win.switchTo(win.homeInterface)
QTest.qWait(300)
check("1920 首页右栏约 420", 400 <= home.right_col.width() <= 440,
      home.right_col.width())
check("1920 首页内容限宽居中", home.root.contentsMargins().left() >= 200,
      home.root.contentsMargins().left())
check("1920 首页无横向溢出",
      home.scroll.widget().minimumSizeHint().width() <= home.scroll.viewport().width(),
      (home.scroll.widget().minimumSizeHint().width(), home.scroll.viewport().width()))
home.grab().save(str(OUT / "home_1920.png"))

win.switchTo(win.resultInterface)
QTest.qWait(300)
lw, rw = ri.left_card.width(), ri.right_col.width()
ratio_l = lw / max(1, lw + rw)
check("1920 结果页仍为 35% 主从", 0.30 <= ratio_l <= 0.40,
      "left=%d right=%d ratio=%.1f%%" % (lw, rw, ratio_l * 100))
check("1920 结果页内容限宽居中", ri.root.contentsMargins().left() >= 200,
      ri.root.contentsMargins().left())
ri.grab().save(str(OUT / "result_1920.png"))

win.switchTo(win.watchInterface)
QTest.qWait(300)
check("1920 巡检台账按高度显示 10 条",
      wi.box.count() == 10 and wi.recent_rows == 12,
      (wi.box.count(), wi.recent_rows))
check("1920 巡检台账卡限宽约 1280", wi.ledger_card.width() <= 1300,
      wi.ledger_card.width())
check("1920 巡检页内容限宽居中", wi.root.contentsMargins().left() >= 200,
      wi.root.contentsMargins().left())
check("1920 巡检页无横向溢出",
      wi.scroll.widget().minimumSizeHint().width() <= wi.scroll.viewport().width(),
      (wi.scroll.widget().minimumSizeHint().width(), wi.scroll.viewport().width()))
wi.grab().save(str(OUT / "watch_1920.png"))

win.switchTo(win.trainInterface)
QTest.qWait(300)
check("1920 教学页内容限宽居中", ti.root.contentsMargins().left() >= 200,
      ti.root.contentsMargins().left())
check("1920 教学页无横向溢出",
      ti.scroll.widget().minimumSizeHint().width() <= ti.scroll.viewport().width(),
      (ti.scroll.widget().minimumSizeHint().width(), ti.scroll.viewport().width()))
ti.grab().save(str(OUT / "train_1920.png"))

win.switchTo(win.imageInterface)
QTest.qWait(300)
check("1920 图片页右栏 372 + 内容限宽",
      ii.right_col.width() == 372 and ii.root.contentsMargins().left() >= 200,
      (ii.right_col.width(), ii.root.contentsMargins().left()))
check("1920 图片页无横向溢出",
      ii.scroll.widget().minimumSizeHint().width() <= ii.scroll.viewport().width(),
      (ii.scroll.widget().minimumSizeHint().width(), ii.scroll.viewport().width()))
ii.grab().save(str(OUT / "image_1920.png"))

win.switchTo(win.reportInterface)
QTest.qWait(300)
check("1920 报告页内容限宽居中", rp.root.contentsMargins().left() >= 200,
      rp.root.contentsMargins().left())
check("1920 报告页无横向溢出",
      rp.scroll.widget().minimumSizeHint().width() <= rp.scroll.viewport().width(),
      (rp.scroll.widget().minimumSizeHint().width(), rp.scroll.viewport().width()))
rp.grab().save(str(OUT / "report_1920.png"))

win.switchTo(win.mapInterface)
QTest.qWait(300)
check("1920 地图页右栏 396 + 内容限宽",
      mi.right_col.width() == 396 and mi.root.contentsMargins().left() >= 200,
      (mi.right_col.width(), mi.root.contentsMargins().left()))
check("1920 地图页无横向溢出",
      mi.scroll.widget().minimumSizeHint().width() <= mi.scroll.viewport().width(),
      (mi.scroll.widget().minimumSizeHint().width(), mi.scroll.viewport().width()))
mi.grab().save(str(OUT / "map_1920.png"))

win.switchTo(win.scienceInterface)
QTest.qWait(300)
check("1920 科普页右栏 372 + 内容限宽",
      si.right_col.width() == 372 and si.root.contentsMargins().left() >= 200,
      (si.right_col.width(), si.root.contentsMargins().left()))
check("1920 科普页无横向溢出",
      si.scroll.widget().minimumSizeHint().width() <= si.scroll.viewport().width(),
      (si.scroll.widget().minimumSizeHint().width(), si.scroll.viewport().width()))
si.grab().save(str(OUT / "science_1920.png"))

win.close()
print("\n通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
print("产物：" + str(OUT))
sys.exit(1 if FAIL else 0)
