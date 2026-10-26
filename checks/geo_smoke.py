# -*- coding: utf-8 -*-
"""rice_geo 离线自检：EXIF 读写 / 坐标换算 / 记录库 / 地图窗口渲染截图

用法（conda 环境）：
    python checks/geo_smoke.py
会在 checks/_geo_out/ 下生成截图与临时数据库。
"""
import os, sys, tempfile, shutil
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "_geo_out"
if OUT.exists():
    shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["RICE_GEO_DIR"] = str(OUT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

import rice_geo
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

FAIL = []
def check(name, ok, extra=""):
    print(("  [OK] " if ok else "  [FAIL] ") + name + (("  -> " + str(extra)) if extra else ""))
    if not ok:
        FAIL.append(name)

print("== 1. 坐标换算 ==")
gcj = rice_geo.wgs84_to_gcj02(118.803, 32.045)
back = rice_geo.gcj02_to_wgs84(*gcj)
print("   WGS(118.803,32.045) -> GCJ", gcj, "-> WGS", back)
check("WGS->GCJ 有明显偏移(百米级)", 0.001 < abs(gcj[0] - 118.803) < 0.01, gcj)
check("GCJ->WGS 往返误差 < 1e-5 度", abs(back[0] - 118.803) < 1e-5 and abs(back[1] - 32.045) < 1e-5, back)
check("境外坐标不偏移", rice_geo.wgs84_to_gcj02(139.69, 35.68) == (139.69, 35.68))
mp = rice_geo.meters_per_pixel(32.0, 15)
check("z15 比例尺约 4.3 m/px", 3.5 < mp < 5.0, mp)

print("== 2. 墨卡托投影往返 ==")
for lon, lat, z in ((118.803, 32.045, 12), (116.397, 39.909, 15), (121.47, 31.23, 8)):
    x, y = rice_geo.lonlat_to_world(lon, lat, z)
    lon2, lat2 = rice_geo.world_to_lonlat(x, y, z)
    check("投影往返 z%d" % z, abs(lon2 - lon) < 1e-9 and abs(lat2 - lat) < 1e-9, (lon2, lat2))

print("== 3. EXIF 写入/读取（模拟手机照片） ==")
photo = OUT / "fake_phone_photo.jpg"
exif = Image.Exif()
exif[306] = "2026:05:04 09:12:33"
exif[271] = "Xiaomi"
exif[272] = "13 Pro"
exif[36867] = "2026:05:04 09:12:33"
try:
    exif[34853] = {1: "N", 2: (32.0, 2.0, 42.0), 3: "E", 4: (118.0, 48.0, 10.0)}
except Exception as e:
    print("   写 GPS IFD 失败:", e)
Image.new("RGB", (640, 480), (30, 120, 60)).save(photo, exif=exif)
meta = rice_geo.read_photo_meta(str(photo))
print("   ", meta)
check("读到拍摄时间", meta["taken_at"] is not None and meta["taken_at"].year == 2026, meta["taken_at"])
check("读到 GPS 纬度≈32.045", meta["lat"] is not None and abs(meta["lat"] - 32.045) < 0.001, meta["lat"])
check("读到 GPS 经度≈118.8028", meta["lon"] is not None and abs(meta["lon"] - 118.8028) < 0.001, meta["lon"])
check("识别出设备", "Xiaomi" in (meta["device"] or ""), meta["device"])

print("== 4. 无 GPS 照片（微信传输/截图 常见） ==")
plain = OUT / "no_gps.jpg"
Image.new("RGB", (320, 240), (200, 200, 180)).save(plain)
m2 = rice_geo.read_photo_meta(str(plain))
check("没有 GPS 时不报错且 lat 为空", m2["lat"] is None and m2["lon"] is None)
check("退回文件修改时间", m2["taken_at"] is not None)

print("== 5. 记录库 CRUD + 自动换算 ==")
store = rice_geo.GeoStore()
rid = store.add_record(shot_at=datetime(2026, 5, 4, 9, 12, 33), lon=118.803, lat=32.045,
                       coord_src="exif", cls="稻瘟病", conf=0.91,
                       source_path=str(photo), note="田埂边发现", plot="3号田")
rec = store.get(rid)
check("落库并自动换算 GCJ-02", rec["glon"] is not None and abs(rec["glon"] - gcj[0]) < 1e-6, rec["glon"])
check("原图已归档到库内", os.path.exists(rec["orig_path"]) and str(store.root) in rec["orig_path"], rec["orig_path"])
check("缩略图已生成", os.path.exists(rec["thumb_path"] or ""), rec["thumb_path"])
check("sha1 去重可用", store.has_sha1(rice_geo.sha1_of_file(str(photo))))
store.update(rid, note="已撒药")
check("更新备注", store.get(rid)["note"] == "已撒药")
store.set_location(rid, 118.80, 32.04, coord_src="map-pick")
check("手动落点", store.get(rid)["coord_src"] == "map-pick")
check("点位转换", len(store.to_points()) == 1 and store.to_points()[0]["kind"] == "disease")

print("== 6. 导出 CSV / KML / 网页地图 ==")
csv_p = rice_geo.export_csv(store, OUT / "rec.csv")
kml_p = rice_geo.export_kml(store, OUT / "rec.kml")
web_p = rice_geo.export_web_map(store, OUT / "webmap", key="TESTKEY")
check("CSV 有数据行", len(open(csv_p, encoding="utf-8-sig").read().splitlines()) >= 2)
check("KML 坐标正确", "<coordinates>" in open(kml_p, encoding="utf-8").read())
html = open(web_p, encoding="utf-8").read()
check("网页地图已注入 Key 与点位", "TESTKEY" in html and '"lng"' in html and "AMap" in html)

print("== 7. 地图窗口渲染（离屏截图） ==")
app = QApplication(sys.argv)
# 离屏平台默认找不到字体目录，显式加载系统中文字体，保证截图能看清文字
from PyQt5.QtGui import QFontDatabase, QFont
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
           "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/seguiemj.ttf"):
    if os.path.exists(_f):
        QFontDatabase.addApplicationFont(_f)
        print("   已加载字体:", _f, "->", QFontDatabase.applicationFontFamilies(
            QFontDatabase.addApplicationFont(_f)) if False else "")
        break
app.setFont(QFont("Microsoft YaHei", 10))
rice_geo.install_chrome()
base_lon, base_lat = 118.803, 32.045
names = list(rice_geo.DISEASES) + list(rice_geo.PESTS)
for i in range(6):
    store.add_record(shot_at=datetime.now() - timedelta(days=30 - i * 5),
                     lon=base_lon + (i - 3) * 0.004, lat=base_lat + (i % 3 - 1) * 0.003,
                     coord_src="exif", cls=names[i], conf=0.7 + 0.04 * i,
                     note="演示", plot="%d号田" % (i % 3 + 1))
win = rice_geo.GeoMapWindow()
win.resize(1280, 850)
win.show()
win.canvas.set_center(base_lon + 0.01, base_lat, 13)

result = {"tiles": 0, "paints": 0}
win.canvas.loader.tileReady.connect(lambda *a: result.__setitem__("tiles", result["tiles"] + 1))

# 离屏模式下窗口不会收到系统的曝光事件，用定时器手动触发重绘（真实 GUI 中不需要）
_pump = QTimer()
_pump.timeout.connect(lambda: (win.canvas.update(),
                               result.__setitem__("paints", result["paints"] + 1)))
_pump.start(200)

def finish():
    print("   重绘次数:", result["paints"], "已下载瓦片数:", result["tiles"],
          "内存瓦片:", len(win.canvas._mem))
    win.grab().save(str(OUT / "geo_map.png"))
    win.canvas.zoom = 15
    win.canvas.set_center(base_lon, base_lat)
    win.canvas.update()
    QTimer.singleShot(2500, lambda: (win.grab().save(str(OUT / "geo_map_zoom.png")), app.quit()))

QTimer.singleShot(12000, finish)
app.exec_()
shutil.rmtree(OUT / "tiles", ignore_errors=True)
print("   已下载瓦片数:", result["tiles"])
check("窗口渲染出截图", (OUT / "geo_map.png").exists() and (OUT / "geo_map.png").stat().st_size > 20000)
check("地图瓦片可下载(需联网)", result["tiles"] > 0, result["tiles"])
check("记录列表已填充", win.listw.count() >= 6, win.listw.count())

print()
print("=" * 60)
print("失败项：", FAIL if FAIL else "无，全部通过 ✅")
print("产物目录：", OUT)
sys.exit(1 if FAIL else 0)
