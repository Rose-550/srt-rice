# -*- coding: utf-8 -*-
"""
稻视界 · 地块地图与溯源记录模块（rice_geo.py）
================================================
把「农户上传的照片」变成「地图上的点位 + 可回溯的时间轴」。

核心能力
--------
1. 自动读取照片 EXIF：拍摄时间（DateTimeOriginal）+ GPS 经纬度
2. 坐标自动换算 WGS-84 → GCJ-02（高德「火星坐标」），避免落点偏移几百米
3. 本地 SQLite 记录库：时间 / 坐标 / 地址 / 地块 / 病害 / 置信度 / 备注 / 照片副本
4. 原生 Qt 高德瓦片地图：缩放、平移、点选落点 —— 不需要 Key，也不需要 QtWebEngine
5. 时间回溯：拖动时间轴或一键回放，看病害在时间上的扩散过程
6. 没有 GPS 的照片 → 在地图上点一下即可落点
7. 导出 CSV / KML / 可交互网页地图（高德 JS API，需要 Key）

依赖：PyQt5、Pillow、requests、numpy（均为项目已有依赖），可选 opencv（用于标注图）
"""

import os
import sys
import csv
import json
import math
import time
import shutil
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty

import requests
from PIL import Image, ImageOps

from PyQt5.QtWidgets import (QWidget, QMainWindow, QLabel, QPushButton, QVBoxLayout,
                             QHBoxLayout, QListWidget, QListWidgetItem, QFileDialog,
                             QMessageBox, QComboBox, QCheckBox, QSlider, QLineEdit,
                             QScrollArea, QFrame, QSizePolicy, QApplication)
from PyQt5.QtGui import (QPainter, QPixmap, QColor, QPen, QBrush, QFont, QImage,
                         QIcon, QPolygonF)
from PyQt5.QtCore import (Qt, QTimer, QThread, pyqtSignal, QRect, QRectF, QPointF,
                          QSize, QUrl)
from PyQt5.QtGui import QDesktopServices

# ── 路径（与 rice_dp1.py 保持一致：兼容 PyInstaller 打包与源码运行） ──
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parent

log = __import__("logging").getLogger("rice_geo")

# ══════════════════════════════════════════════════════════════════
# 1. 坐标：WGS-84 / GCJ-02（高德） / BD-09（百度） 互转 + 墨卡托投影
# ══════════════════════════════════════════════════════════════════
_A = 6378245.0                 # 克拉索夫斯基椭球长半轴
_EE = 0.00669342162296594323   # 偏心率平方


def _out_of_china(lon, lat):
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _transform_lat(lon, lat):
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + \
        0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon, lat):
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + \
        0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 * math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * math.pi) + 40.0 * math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 * math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon, lat):
    """GPS 原始坐标 → 高德/腾讯坐标（国内偏移 +300~600m）"""
    if lon is None or lat is None:
        return None, None
    if _out_of_china(lon, lat):
        return lon, lat
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - _EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (_A / sqrt_magic * math.cos(rad_lat) * math.pi)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon, lat):
    """高德坐标 → GPS 原始坐标（迭代逼近，误差 < 0.5m）"""
    if lon is None or lat is None:
        return None, None
    if _out_of_china(lon, lat):
        return lon, lat
    wlon, wlat = lon, lat
    for _ in range(6):
        tlon, tlat = wgs84_to_gcj02(wlon, wlat)
        wlon += lon - tlon
        wlat += lat - tlat
    return wlon, wlat


def gcj02_to_bd09(lon, lat):
    x, y = lon, lat
    z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
    return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006


def bd09_to_gcj02(lon, lat):
    x, y = lon - 0.0065, lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
    return z * math.cos(theta), z * math.sin(theta)


def to_gcj02(lon, lat, src="wgs84"):
    """任意来源坐标 → 高德坐标"""
    if lon is None or lat is None:
        return None, None
    src = (src or "wgs84").lower()
    if src in ("gcj02", "amap", "gaode"):
        return lon, lat
    if src in ("bd09", "baidu"):
        return bd09_to_gcj02(lon, lat)
    return wgs84_to_gcj02(lon, lat)


def from_gcj02(lon, lat, dst="wgs84"):
    if lon is None or lat is None:
        return None, None
    dst = (dst or "wgs84").lower()
    if dst in ("gcj02", "amap", "gaode"):
        return lon, lat
    if dst in ("bd09", "baidu"):
        return gcj02_to_bd09(lon, lat)
    return gcj02_to_wgs84(lon, lat)


# ── Web 墨卡托（高德瓦片即此投影） ──
def lonlat_to_world(lon, lat, zoom):
    scale = 256.0 * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    s = math.sin(math.radians(max(min(lat, 85.05112878), -85.05112878)))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * scale
    return x, y


def world_to_lonlat(x, y, zoom):
    scale = 256.0 * (2 ** zoom)
    lon = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


# ══════════════════════════════════════════════════════════════════
# 2. 照片 EXIF：拍摄时间 + GPS
# ══════════════════════════════════════════════════════════════════
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_DATETIME = 306
_TAG_MAKE = 271
_TAG_MODEL = 272
_IFD_EXIF = 34665
_IFD_GPS = 34853


def _ratio_to_float(v):
    try:
        return float(v)
    except Exception:
        try:
            return v.numerator / float(v.denominator)
        except Exception:
            return None


def _dms_to_deg(value):
    """(度, 分, 秒) → 十进制度"""
    try:
        parts = list(value)
        if len(parts) != 3:
            return None
        d, m, s = (_ratio_to_float(p) for p in parts)
        if d is None:
            return None
        return d + (m or 0) / 60.0 + (s or 0) / 3600.0
    except Exception:
        return None


def _parse_dt(text):
    if not text:
        return None
    text = str(text).strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M",
                "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def read_photo_meta(path):
    """
    读取照片的拍摄时间与 GPS。
    返回 dict: taken_at(datetime|None) / lon / lat（原始坐标系下的值） / device / coord_sys
    读不到 GPS 时 lon=lat=None（照片仍会入库，等用户在地图上点一下落点）。
    """
    meta = {"taken_at": None, "lon": None, "lat": None, "device": "",
            "coord_sys": "wgs84", "has_gps": False, "source": "exif"}
    try:
        with Image.open(path) as im:
            try:
                exif = im.getexif()
            except Exception:
                exif = None
            if exif:
                make = str(exif.get(_TAG_MAKE, "") or "").strip()
                model = str(exif.get(_TAG_MODEL, "") or "").strip()
                meta["device"] = (make + " " + model).strip()
                dt = _parse_dt(exif.get(_TAG_DATETIME_ORIGINAL))
                if dt is None:
                    try:
                        sub = exif.get_ifd(_IFD_EXIF)
                        dt = _parse_dt(sub.get(_TAG_DATETIME_ORIGINAL)) or \
                            _parse_dt(sub.get(_TAG_DATETIME_DIGITIZED))
                    except Exception:
                        pass
                if dt is None:
                    dt = _parse_dt(exif.get(_TAG_DATETIME))
                meta["taken_at"] = dt
                try:
                    gps = exif.get_ifd(_IFD_GPS) or {}
                except Exception:
                    gps = {}
                if not gps:
                    try:
                        gps = exif.get(_IFD_GPS) or {}
                    except Exception:
                        gps = {}
                if gps:
                    lat = _dms_to_deg(gps.get(2))
                    lon = _dms_to_deg(gps.get(4))
                    if lat is not None and str(gps.get(1, "N")).upper().startswith("S"):
                        lat = -lat
                    if lon is not None and str(gps.get(3, "E")).upper().startswith("W"):
                        lon = -lon
                    if lat is not None and lon is not None and (abs(lat) > 0.0001 or abs(lon) > 0.0001):
                        meta["lat"], meta["lon"] = lat, lon
                        meta["has_gps"] = True
    except Exception as e:
        log.debug("读取 EXIF 失败 %s: %s", path, e)
    if meta["taken_at"] is None:
        try:
            meta["taken_at"] = datetime.fromtimestamp(os.path.getmtime(path))
            meta["source"] = "filetime"
        except Exception:
            meta["taken_at"] = datetime.now()
    return meta


def sha1_of_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except Exception:
        return ""
    return h.hexdigest()


def make_thumbnail(src_path, dst_path, size=(320, 240), quality=82):
    """生成缩略图（自动按 EXIF 方向摆正）"""
    try:
        with Image.open(src_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail(size, Image.LANCZOS)
            canvas = Image.new("RGB", size, (240, 244, 240))
            canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
            Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            canvas.save(dst_path, "JPEG", quality=quality)
        return str(dst_path)
    except Exception as e:
        log.warning("缩略图生成失败 %s: %s", src_path, e)
        return ""


# ══════════════════════════════════════════════════════════════════
# 3. 本地记录库（SQLite）
# ══════════════════════════════════════════════════════════════════
DISEASES = ("白叶枯病", "胡麻斑病", "稻瘟病")
PESTS = ("稻纵卷叶螟", "稻水象甲", "稻叶蝉")
KIND_COLOR = {"disease": "#E74C3C", "pest": "#F39C12", "unknown": "#64748B"}

# 来源：描边色 + 标签后缀，用来区分「手机直传 / 小程序 / 电脑本地」。
# 和程序里列表的色标、网页地图的描边用同一套颜色，看哪儿都是同一个含义。
SOURCE_RING = {"phone": "#007A3F", "mini": "#2C6E8F", "pc": "#6E7873"}
SOURCE_TAG = {"phone": "手机直传", "mini": "小程序", "pc": "电脑"}


def disease_kind(cls_name):
    if cls_name in DISEASES:
        return "disease"
    if cls_name in PESTS:
        return "pest"
    return "unknown"


def kind_label(kind):
    return {"disease": "病害", "pest": "虫害"}.get(kind, "未识别")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uid         TEXT UNIQUE,
    shot_at     TEXT,
    created_at  TEXT,
    lon         REAL, lat  REAL,          -- WGS-84（GPS 原始）
    glon        REAL, glat REAL,          -- GCJ-02（高德）
    coord_src   TEXT,                     -- exif / manual / map-pick
    address     TEXT,
    cls         TEXT,
    conf        REAL,
    boxes       TEXT,
    note        TEXT,
    plot        TEXT,
    orig_path   TEXT,
    ann_path    TEXT,
    thumb_path  TEXT,
    sha1        TEXT,
    device      TEXT,
    source      TEXT                      -- 谁识别的：pc（电脑本地）/ phone（手机直传）/ miniapp
);
CREATE INDEX IF NOT EXISTS idx_shot_at ON records(shot_at);
CREATE INDEX IF NOT EXISTS idx_sha1    ON records(sha1);
"""


def writable_root():
    """数据目录：环境变量 RICE_GEO_DIR > 程序目录/geo > 「文档/稻视界_地块地图」"""
    env = os.environ.get("RICE_GEO_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    cand = APP_DIR / "geo"
    try:
        cand.mkdir(parents=True, exist_ok=True)
        probe = cand / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return cand
    except Exception:
        fallback = Path.home() / "Documents" / "稻视界_地块地图"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


class GeoStore:
    """照片 + 识别结果 + 坐标 + 时间 的本地档案库"""

    def __init__(self, root=None, exif_coord_sys="wgs84"):
        self.root = Path(root) if root else writable_root()
        self.photo_dir = self.root / "photos"
        self.tile_dir = self.root / "tiles"
        self.export_dir = self.root / "export"
        for d in (self.photo_dir, self.tile_dir, self.export_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "records.db"
        self.exif_coord_sys = (exif_coord_sys or "wgs84").lower()
        self._init_db()

    # ── 基础 ──
    def _conn(self):
        c = sqlite3.connect(str(self.db_path), timeout=15)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.executescript(_SCHEMA)
            # 老库补列：source 是后加的，已经存在时会抛 OperationalError，忽略即可
            try:
                c.execute("ALTER TABLE records ADD COLUMN source TEXT")
            except sqlite3.OperationalError:
                pass

    def count(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def has_sha1(self, sha1):
        if not sha1:
            return False
        with self._conn() as c:
            return c.execute("SELECT 1 FROM records WHERE sha1=? LIMIT 1", (sha1,)).fetchone() is not None

    def all_records(self, order="shot_at ASC"):
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM records ORDER BY {order}").fetchall()
        return [dict(r) for r in rows]

    def get(self, rec_id):
        with self._conn() as c:
            row = c.execute("SELECT * FROM records WHERE id=?", (rec_id,)).fetchone()
        return dict(row) if row else None

    def plots(self):
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT plot FROM records "
                             "WHERE plot IS NOT NULL AND plot<>'' ORDER BY plot").fetchall()
        return [r[0] for r in rows]

    # ── 写入 ──
    def add_record(self, *, shot_at=None, lon=None, lat=None, coord_src="manual",
                   cls="", conf=0.0, boxes=None, note="", plot="",
                   orig_path=None, ann_path=None, source_path=None,
                   device="", sha1="", thumb_src=None, source=""):
        """
        新增一条记录。
        lon/lat 为 GPS（WGS-84）坐标；lat/lon 为 None 表示尚未落点。
        source_path：原始照片，会被复制进档案库（防止原图被删/移动后无法回溯）。
        """
        shot_at = shot_at or datetime.now()
        if isinstance(shot_at, str):
            shot_at = _parse_dt(shot_at) or datetime.now()
        sha1 = sha1 or (sha1_of_file(source_path) if source_path else "")
        uid = "%s_%s" % (shot_at.strftime("%Y%m%d%H%M%S"), (sha1 or hashlib.md5(
            ("%s%s" % (time.time(), shot_at)).encode()).hexdigest())[:8])

        # 归档原图 + 标注图 + 缩略图
        archived = ""
        if source_path and os.path.exists(source_path):
            ext = os.path.splitext(source_path)[1].lower() or ".jpg"
            archived = str(self.photo_dir / (uid + "_orig" + ext))
            try:
                shutil.copy2(source_path, archived)
            except Exception as e:
                log.warning("归档原图失败: %s", e)
                archived = source_path
        elif orig_path:
            archived = orig_path

        ann_final = ""
        if isinstance(ann_path, str):
            if ann_path and os.path.exists(ann_path):
                ann_final = str(self.photo_dir / (uid + "_ann.jpg"))
                try:
                    shutil.copy2(ann_path, ann_final)
                except Exception:
                    ann_final = ann_path
        elif ann_path is not None:
            # numpy BGR 数组
            ann_final = str(self.photo_dir / (uid + "_ann.jpg"))
            try:
                import cv2
                cv2.imwrite(ann_final, ann_path)
            except Exception as e:
                log.warning("保存标注图失败: %s", e)
                ann_final = ""

        thumb = ""
        thumb_from = thumb_src or archived or ann_final
        if thumb_from and os.path.exists(thumb_from):
            thumb = make_thumbnail(thumb_from, self.photo_dir / (uid + "_thumb.jpg"))

        glon, glat = to_gcj02(lon, lat, self.exif_coord_sys)
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO records(uid,shot_at,created_at,lon,lat,glon,glat,"
                "coord_src,address,cls,conf,boxes,note,plot,orig_path,ann_path,thumb_path,"
                "sha1,device,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (uid, shot_at.strftime("%Y-%m-%d %H:%M:%S"),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 lon, lat, glon, glat, coord_src, "", cls or "", float(conf or 0.0),
                 json.dumps(boxes or [], ensure_ascii=False), note or "", plot or "",
                 archived, ann_final, thumb, sha1, device or "", source or ""))
            return cur.lastrowid

    def update(self, rec_id, **fields):
        allowed = {"shot_at", "lon", "lat", "glon", "glat", "coord_src", "address",
                   "cls", "conf", "boxes", "note", "plot", "orig_path", "ann_path",
                   "thumb_path", "device"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append("%s=?" % k)
                vals.append(json.dumps(v, ensure_ascii=False) if k == "boxes" else v)
        if not sets:
            return
        vals.append(rec_id)
        with self._conn() as c:
            c.execute("UPDATE records SET %s WHERE id=?" % ",".join(sets), vals)

    def set_location(self, rec_id, lon_wgs, lat_wgs, coord_src="manual"):
        glon, glat = to_gcj02(lon_wgs, lat_wgs, "wgs84")
        self.update(rec_id, lon=lon_wgs, lat=lat_wgs, glon=glon, glat=glat,
                    coord_src=coord_src)

    def delete(self, rec_id):
        rec = self.get(rec_id)
        if not rec:
            return
        for key in ("orig_path", "ann_path", "thumb_path"):
            p = rec.get(key) or ""
            if p and os.path.exists(p) and str(p).startswith(str(self.root)):
                try:
                    os.remove(p)
                except Exception:
                    pass
        with self._conn() as c:
            c.execute("DELETE FROM records WHERE id=?", (rec_id,))

    def to_points(self, records=None):
        """转成地图点位（GCJ-02）"""
        pts = []
        for r in (records if records is not None else self.all_records()):
            if r.get("glon") is None or r.get("glat") is None:
                continue
            pts.append({
                "id": r["id"], "lon": r["glon"], "lat": r["glat"],
                "cls": r.get("cls") or "", "conf": r.get("conf") or 0.0,
                "time": (r.get("shot_at") or "")[:16],
                "shot_at": r.get("shot_at") or "",
                "kind": disease_kind(r.get("cls") or ""),
                "plot": r.get("plot") or "",
                "address": r.get("address") or "",
                "thumb": r.get("thumb_path") or "",
                "note": r.get("note") or "",
                # 来源（phone/mini/pc）：地图上用它给点描边，一眼看出哪些是手机传的
                "source": r.get("source") or "",
            })
        return pts


# ══════════════════════════════════════════════════════════════════
# 4. 高德在线服务（REST，需要 Web 服务 Key；不填 Key 也不影响地图）
# ══════════════════════════════════════════════════════════════════
AMAP_REST = "https://restapi.amap.com/v3"


def get_config():
    """读取 config.json（与主程序共用一份配置；每次读取保证与设置页同步）"""
    path = APP_DIR / "config.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def amap_key():
    return (os.environ.get("AMAP_KEY") or get_config().get("amap_web_key") or "").strip()


def amap_js_key():
    return (os.environ.get("AMAP_JS_KEY") or get_config().get("amap_js_key") or
            amap_key()).strip()


def amap_regeo(lon, lat, key=None, timeout=8):
    """逆地理编码：坐标 → 文字地址（需要 Key）。失败返回空串。"""
    key = key or amap_key()
    if not key:
        return ""
    try:
        r = requests.get(AMAP_REST + "/geocode/regeo",
                         params={"key": key, "location": "%.6f,%.6f" % (lon, lat),
                                 "extensions": "base", "radius": 500},
                         timeout=timeout)
        data = r.json()
        if data.get("status") == "1":
            return (data.get("regeocode") or {}).get("formatted_address", "") or ""
        log.info("逆地理编码未成功: %s", data.get("info"))
    except Exception as e:
        log.info("逆地理编码失败: %s", e)
    return ""


def amap_geocode(address, city="", key=None, timeout=8):
    """地址 → 坐标（GCJ-02）"""
    key = key or amap_key()
    if not key:
        return None
    try:
        r = requests.get(AMAP_REST + "/geocode/geo",
                         params={"key": key, "address": address, "city": city},
                         timeout=timeout)
        data = r.json()
        if data.get("status") == "1" and data.get("geocodes"):
            loc = data["geocodes"][0]["location"].split(",")
            return float(loc[0]), float(loc[1])
    except Exception as e:
        log.info("地理编码失败: %s", e)
    return None


def amap_static_map(lon, lat, zoom=15, size="600*360", markers=None,
                    key=None, timeout=10):
    """静态地图（需要 Key），返回 PNG 字节；失败返回 None"""
    key = key or amap_key()
    if not key:
        return None
    params = {"key": key, "location": "%.6f,%.6f" % (lon, lat), "zoom": zoom,
              "size": size, "scale": 1}
    if markers:
        params["markers"] = markers
    try:
        r = requests.get(AMAP_REST + "/staticmap", params=params, timeout=timeout)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content
    except Exception as e:
        log.info("静态地图获取失败: %s", e)
    return None


# ══════════════════════════════════════════════════════════════════
# 5. 高德瓦片（无需 Key 的原生底图）
# ══════════════════════════════════════════════════════════════════
TILE_STYLES = {
    "road": {"name": "高德路网", "sub": ("1", "2", "3", "4"),
             "url": "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"},
    "satellite": {"name": "高德卫星", "sub": ("1", "2", "3", "4"),
                  "url": "https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"},
}


class TileLoader(QThread):
    """后台抓瓦片：磁盘缓存优先，网络失败不阻塞界面"""
    tileReady = pyqtSignal(int, int, int, str, bytes)   # z, x, y, style, data
    tileFailed = pyqtSignal(int, int, int, str)

    def __init__(self, cache_root, workers=3, parent=None):
        super().__init__(parent)
        self.cache_root = Path(cache_root)
        self.n_workers = workers
        self._q = Queue()
        self._stop = threading.Event()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 RiceGeo/1.0"})

    def request(self, z, x, y, style="road"):
        self._q.put((z, x, y, style))

    def set_workers(self, n):
        self.n_workers = max(1, int(n))

    def stop(self):
        self._stop.set()
        try:
            self._q.put(None)
        except Exception:
            pass
        for _ in range(self.n_workers):
            try:
                self._q.put(None)
            except Exception:
                pass

    def _cache_path(self, style, z, x, y):
        return self.cache_root / style / str(z) / str(x) / ("%d.png" % y)

    def run(self):
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = set()
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=0.2)
                except Empty:
                    futures = {f for f in futures if not f.done()}
                    continue
                if item is None:
                    break
                z, x, y, style = item
                futures = {f for f in futures if not f.done()}
                if len(futures) >= self.n_workers * 2:
                    time.sleep(0.05)
                futures.add(pool.submit(self._fetch_one, z, x, y, style))

    def _fetch_one(self, z, x, y, style):
        p = self._cache_path(style, z, x, y)
        if p.exists():
            try:
                self.tileReady.emit(z, x, y, style, p.read_bytes())
                return
            except Exception:
                pass
        cfg = TILE_STYLES.get(style, TILE_STYLES["road"])
        sub = cfg["sub"][(x + y) % len(cfg["sub"])]
        url = cfg["url"].format(s=sub, x=x, y=y, z=z)
        try:
            r = self._session.get(url, timeout=8)
            if r.status_code == 200 and len(r.content) > 100:
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(r.content)
                except Exception:
                    pass
                self.tileReady.emit(z, x, y, style, r.content)
                return
        except Exception:
            pass
        self.tileFailed.emit(z, x, y, style)


class MapCanvas(QWidget):
    """原生高德瓦片地图：滚轮缩放 / 拖动平移 / 点击落点 / 点位标记"""
    markerClicked = pyqtSignal(int)
    mapClicked = pyqtSignal(float, float)     # lon, lat（GCJ-02）

    MIN_ZOOM, MAX_ZOOM = 4, 18

    def __init__(self, tile_root, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 300)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setStyleSheet("background:#eef2ee; border-radius:12px;")

        self.center = [118.803, 32.045]      # 南京农业大学（GCJ-02）
        self.zoom = 12
        self.style = "road"
        self.points = []
        self.selected_id = None
        self.pick_mode = False
        self.show_labels = True
        self.show_track = False      # 按时间顺序连线，展示扩散路径

        self._mem = {}          # (style,z,x,y) -> QPixmap
        self._requested = set()
        self._failed = set()
        self._drag_from = None
        self._drag_center = None
        self._moved = False

        self.loader = TileLoader(tile_root)
        self.loader.tileReady.connect(self._on_tile)
        self.loader.tileFailed.connect(self._on_tile_failed)
        self.loader.start()

    # ── 数据 ──
    def set_points(self, points):
        self.points = list(points or [])
        self.update()

    def set_center(self, lon, lat, zoom=None):
        self.center = [lon, lat]
        if zoom:
            self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, int(zoom)))
        self.update()

    def set_style(self, style):
        if style == self.style:
            return
        self.style = style
        self._mem.clear()
        self._requested.clear()
        self._failed.clear()
        self.update()

    def set_pick_mode(self, on):
        self.pick_mode = bool(on)
        self.setCursor(Qt.CrossCursor if on else Qt.OpenHandCursor)
        self.update()

    def fit_points(self, points=None):
        pts = [p for p in (points if points is not None else self.points) if p.get("lon")]
        if not pts:
            return
        lon_min = min(p["lon"] for p in pts); lon_max = max(p["lon"] for p in pts)
        lat_min = min(p["lat"] for p in pts); lat_max = max(p["lat"] for p in pts)
        self.center = [(lon_min + lon_max) / 2, (lat_min + lat_max) / 2]
        w = max(self.width(), 200); h = max(self.height(), 200)
        best = self.MIN_ZOOM
        for z in range(self.MIN_ZOOM, self.MAX_ZOOM + 1):
            x0, y0 = lonlat_to_world(lon_min, lat_max, z)
            x1, y1 = lonlat_to_world(lon_max, lat_min, z)
            if (x1 - x0) < w * 0.82 and (y1 - y0) < h * 0.82:
                best = z
        self.zoom = best
        self.update()

    def focus_point(self, lon, lat, zoom=16):
        self.set_center(lon, lat, zoom)

    # ── 瓦片回调 ──
    def _on_tile(self, z, x, y, style, data):
        if style != self.style:
            return                      # 切换底图后姗姗来迟的旧瓦片直接丢弃
        img = QImage.fromData(data)
        if not img.isNull():
            self._mem[(style, z, x, y)] = QPixmap.fromImage(img)
            self._requested.discard((z, x, y))
            self.update()

    def _on_tile_failed(self, z, x, y, style):
        if style != self.style:
            return
        self._failed.add((z, x, y))
        self._requested.discard((z, x, y))
        self.update()

    # ── 绘制 ──
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#eef2ee"))

        z = int(self.zoom)
        cx, cy = lonlat_to_world(self.center[0], self.center[1], z)
        ox, oy = cx - w / 2.0, cy - h / 2.0

        x0 = int(math.floor(ox / 256.0)); x1 = int(math.floor((ox + w) / 256.0))
        y0 = int(math.floor(oy / 256.0)); y1 = int(math.floor((oy + h) / 256.0))
        n = 2 ** z
        missing = 0
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                sx, sy = tx * 256 - ox, ty * 256 - oy
                if ty < 0 or ty >= n:
                    p.fillRect(QRectF(sx, sy, 256, 256), QColor("#dfe6e0"))
                    continue
                wx = tx % n
                pix = self._mem.get((self.style, z, wx, ty))
                if pix is not None:
                    p.drawPixmap(QRect(int(sx), int(sy), 256, 256), pix)
                else:
                    p.fillRect(QRectF(sx, sy, 256, 256), QColor("#e6ece6"))
                    key = (wx, ty)
                    if key not in self._requested and key not in self._failed:
                        self._requested.add(key)
                        self.loader.request(z, wx, ty, self.style)
                    missing += 1
        if missing:
            p.setPen(QColor("#9aa79a"))
            p.setFont(QFont("Microsoft YaHei", 10))
            p.drawText(QRect(0, h - 26, w, 20), Qt.AlignCenter,
                       "正在加载高德地图瓦片…" if missing < (x1 - x0 + 1) * (y1 - y0 + 1)
                       else "离线状态：底图暂不可用，点位仍按经纬度绘制")

        # ── 时间轨迹连线（按拍摄时间串起来，回溯扩散路径） ──
        if self.show_track and len(self.points) > 1:
            ordered = sorted(self.points, key=lambda q: q.get("shot_at") or "")
            p.setPen(QPen(QColor(0, 120, 60, 140), 2.5, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            poly = QPolygonF()
            for q in ordered:
                qx, qy = lonlat_to_world(q["lon"], q["lat"], z)
                poly.append(QPointF(qx - ox, qy - oy))
            p.drawPolyline(poly)

        # ── 点位 ──
        for pt in self.points:
            sx, sy = lonlat_to_world(pt["lon"], pt["lat"], z)
            sx -= ox; sy -= oy
            if sx < -30 or sx > w + 30 or sy < -40 or sy > h + 30:
                continue
            self._draw_marker(p, sx, sy, pt, pt["id"] == self.selected_id)

        # ── 比例尺 ──
        mpp = meters_per_pixel(self.center[1], z)
        target_px = 90
        raw_m = mpp * target_px
        nice = 1
        for cand in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
            if cand <= raw_m:
                nice = cand
        bar_px = nice / mpp
        bx, by = 14, h - 22
        p.setPen(QPen(QColor(40, 60, 40, 200), 2))
        p.drawLine(QPointF(bx, by), QPointF(bx + bar_px, by))
        p.drawLine(QPointF(bx, by - 4), QPointF(bx, by + 4))
        p.drawLine(QPointF(bx + bar_px, by - 4), QPointF(bx + bar_px, by + 4))
        txt = ("%d km" % (nice // 1000)) if nice >= 1000 else ("%d m" % nice)
        p.setFont(QFont("Microsoft YaHei", 9))
        p.setPen(QColor(40, 60, 40, 230))
        p.drawText(QPointF(bx + bar_px / 2 - 16, by - 8), txt)

        # ── 版权 / 状态 ──
        p.setPen(QColor(90, 105, 90, 220))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QRect(w - 210, h - 24, 200, 20), Qt.AlignRight,
                   "© 高德地图 · %s" % TILE_STYLES[self.style]["name"])
        hint = ("落点模式：点击地图确定该照片的位置" if self.pick_mode
                else "滚轮缩放 · 拖动平移 · 点击点位查看详情")
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 205)))
        p.drawRoundedRect(QRectF(8, 6, min(w - 16, 290), 24), 8, 8)
        p.setPen(QColor(40, 90, 50) if self.pick_mode else QColor(70, 90, 70))
        p.setFont(QFont("Microsoft YaHei", 9, QFont.Bold if self.pick_mode else QFont.Normal))
        p.drawText(QRectF(16, 6, min(w - 16, 290), 24), Qt.AlignVCenter | Qt.AlignLeft, hint)

        if not self.points:
            p.setPen(QColor(120, 140, 120))
            p.setFont(QFont("Microsoft YaHei", 13))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "还没有记录\n点「导入照片并识别」，照片里的时间与经纬度会自动落在这里")

    def _draw_marker(self, p, sx, sy, pt, selected):
        color = QColor(KIND_COLOR.get(pt.get("kind", "unknown"), "#64748B"))
        r = 11 if selected else 8
        # 阴影
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 45)))
        p.drawEllipse(QPointF(sx, sy + 2), r + 2, r + 2)
        # 选中光环
        if selected:
            p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 60)))
            p.drawEllipse(QPointF(sx, sy), r + 9, r + 9)
        # 描边色 = 来源（没标来源的才用默认白边）：一眼看出哪些点是手机传的
        ring = SOURCE_RING.get(pt.get("source") or "")
        p.setBrush(QBrush(color))
        p.setPen(QPen(QColor(ring) if ring else QColor(255, 255, 255), 2.5))
        p.drawEllipse(QPointF(sx, sy), r, r)
        # 标签
        if self.show_labels and self.zoom >= 14:
            label = pt.get("cls") or "未识别"
            if pt.get("plot"):
                label = "%s · %s" % (pt["plot"], label)
            tag = SOURCE_TAG.get(pt.get("source") or "")
            if tag:
                label = "%s · %s" % (label, tag)
            p.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            fm = p.fontMetrics()
            tw = fm.width(label) + 12
            box = QRectF(sx + r + 3, sy - 11, tw, 20)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 255, 255, 225)))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor(45, 65, 45))
            p.drawText(box, Qt.AlignCenter, label)

    # ── 交互 ──
    def _hit_marker(self, pos):
        z = int(self.zoom)
        cx, cy = lonlat_to_world(self.center[0], self.center[1], z)
        ox, oy = cx - self.width() / 2.0, cy - self.height() / 2.0
        best, best_d = None, 16.0
        for pt in self.points:
            sx, sy = lonlat_to_world(pt["lon"], pt["lat"], z)
            sx -= ox; sy -= oy
            d = math.hypot(sx - pos.x(), sy - pos.y())
            if d < best_d:
                best, best_d = pt, d
        return best

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_from = e.pos()
            self._drag_center = list(self.center)
            self._moved = False
            self.setCursor(Qt.ClosedHandCursor if not self.pick_mode else Qt.CrossCursor)

    def mouseMoveEvent(self, e):
        if self._drag_from is not None and (e.buttons() & Qt.LeftButton):
            dx = e.pos().x() - self._drag_from.x()
            dy = e.pos().y() - self._drag_from.y()
            if abs(dx) + abs(dy) > 3:
                self._moved = True
            z = int(self.zoom)
            ox, oy = lonlat_to_world(self._drag_center[0], self._drag_center[1], z)
            lon, lat = world_to_lonlat(ox - dx, oy - dy, z)
            self.center = [lon, lat]
            self.update()
        else:
            self.setCursor(Qt.PointingHandCursor if self._hit_marker(e.pos())
                           else (Qt.CrossCursor if self.pick_mode else Qt.OpenHandCursor))

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        was_drag = self._moved
        self._drag_from = None
        self.setCursor(Qt.CrossCursor if self.pick_mode else Qt.OpenHandCursor)
        if was_drag:
            return
        if self.pick_mode:
            z = int(self.zoom)
            cx, cy = lonlat_to_world(self.center[0], self.center[1], z)
            ox, oy = cx - self.width() / 2.0, cy - self.height() / 2.0
            lon, lat = world_to_lonlat(ox + e.pos().x(), oy + e.pos().y(), z)
            self.mapClicked.emit(lon, lat)
            return
        hit = self._hit_marker(e.pos())
        if hit:
            self.markerClicked.emit(hit["id"])

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if not delta:
            return
        old_zoom = int(self.zoom)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, old_zoom + (1 if delta > 0 else -1)))
        if new_zoom == old_zoom:
            return
        # 以鼠标位置为锚点缩放
        cx, cy = lonlat_to_world(self.center[0], self.center[1], old_zoom)
        ox, oy = cx - self.width() / 2.0, cy - self.height() / 2.0
        anchor_lon, anchor_lat = world_to_lonlat(ox + e.pos().x(), oy + e.pos().y(), old_zoom)
        nc_x, nc_y = lonlat_to_world(anchor_lon, anchor_lat, new_zoom)
        self.zoom = new_zoom
        self.center = list(world_to_lonlat(nc_x - e.pos().x() + self.width() / 2.0,
                                           nc_y - e.pos().y() + self.height() / 2.0, new_zoom))
        self._requested.clear()
        self.update()

    def closeEvent(self, e):
        try:
            self.loader.stop()
            self.loader.wait(1500)
        except Exception:
            pass
        e.accept()



# ══════════════════════════════════════════════════════════════════
# 6. 结果标注（中文标签）
# ══════════════════════════════════════════════════════════════════
_FONT_CANDIDATES = ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simsun.ttc"]


def _load_font(size):
    from PIL import ImageFont
    for fp in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate_boxes(img_bgr, boxes, font_size=26):
    """在 BGR 图上画检测框 + 中文标签，返回新图（不修改原图）"""
    try:
        import cv2
        import numpy as np
        from PIL import ImageDraw
        out = img_bgr.copy()
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = _load_font(font_size)
        for item in boxes or []:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            text = "%s %.0f%%" % (item.get("name", ""), float(item.get("conf", 0)) * 100)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            tb = draw.textbbox((x1, max(0, y1 - font_size)), text, font=font)
            draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(255, 0, 0))
            draw.text((x1, max(0, y1 - font_size)), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("标注绘制失败: %s", e)
        return img_bgr


# ══════════════════════════════════════════════════════════════════
# 7. 批量导入线程：EXIF → 识别 → 落库
# ══════════════════════════════════════════════════════════════════
CLASS_NAMES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]
CONF_THRESHOLD = 0.4


class ImportThread(QThread):
    progress = pyqtSignal(int, int, str)     # 当前序号 / 总数 / 文件名
    added = pyqtSignal(int, bool)            # rec_id, 是否有 GPS
    done = pyqtSignal(int, int, int)         # 新增 / 跳过重复 / 失败

    def __init__(self, store, files, model=None, conf=CONF_THRESHOLD,
                 class_names=None, auto_regeo=False, parent=None):
        super().__init__(parent)
        self.store = store
        self.files = list(files)
        self.model = model
        self.conf = conf
        self.class_names = class_names or CLASS_NAMES
        self.auto_regeo = auto_regeo

    def run(self):
        ok = skipped = failed = 0
        total = len(self.files)
        for i, fp in enumerate(self.files, 1):
            self.progress.emit(i, total, os.path.basename(fp))
            try:
                sha1 = sha1_of_file(fp)
                if sha1 and self.store.has_sha1(sha1):
                    skipped += 1
                    continue
                meta = read_photo_meta(fp)
                boxes, cls, conf, ann = [], "", 0.0, None
                if self.model is not None:
                    img = None
                    try:
                        import cv2
                        img = cv2.imread(fp)
                    except Exception:
                        img = None
                    if img is not None:
                        res = self.model(img, conf=self.conf, imgsz=640, verbose=False)
                        for b in res[0].boxes:
                            boxes.append({"name": self.class_names[int(b.cls.item())],
                                          "conf": float(b.conf.item()),
                                          "box": b.xyxy[0].tolist()})
                        if boxes:
                            best = max(boxes, key=lambda x: x["conf"])
                            cls, conf = best["name"], best["conf"]
                            ann = annotate_boxes(img, boxes)
                rec_id = self.store.add_record(
                    shot_at=meta["taken_at"], lon=meta["lon"], lat=meta["lat"],
                    coord_src="exif" if meta["has_gps"] else "none",
                    cls=cls, conf=conf, boxes=boxes, source_path=fp,
                    ann_path=ann, device=meta.get("device", ""), sha1=sha1)
                if rec_id:
                    ok += 1
                    if meta["has_gps"] and self.auto_regeo and amap_key():
                        glon, glat = to_gcj02(meta["lon"], meta["lat"])
                        addr = amap_regeo(glon, glat)
                        if addr:
                            self.store.update(rec_id, address=addr)
                    self.added.emit(rec_id, bool(meta["has_gps"]))
                else:
                    skipped += 1
            except Exception as e:
                log.warning("导入失败 %s: %s", fp, e)
                failed += 1
        self.done.emit(ok, skipped, failed)


# ══════════════════════════════════════════════════════════════════
# 8. 导出：CSV / KML / 可交互网页地图
# ══════════════════════════════════════════════════════════════════
CSV_HEADER = ["选用", "图片相对路径", "类别", "置信度", "纬度", "经度", "时间",
              "备注", "地块", "地址", "坐标来源", "高德经度", "高德纬度"]


def export_csv(store, path, records=None):
    rows = records if records is not None else store.all_records()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([
                "是" if r.get("lon") is not None else "",
                r.get("ann_path") or r.get("orig_path") or "",
                r.get("cls") or "", "%.4f" % (r.get("conf") or 0),
                "" if r.get("lat") is None else "%.6f" % r["lat"],
                "" if r.get("lon") is None else "%.6f" % r["lon"],
                r.get("shot_at") or "", r.get("note") or "", r.get("plot") or "",
                r.get("address") or "", r.get("coord_src") or "",
                "" if r.get("glon") is None else "%.6f" % r["glon"],
                "" if r.get("glat") is None else "%.6f" % r["glat"],
            ])
    return str(path)


def _xml_escape(s):
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def export_kml(store, path, records=None):
    """KML：可直接用 Google Earth / 奥维 / LSV 打开（坐标为 WGS-84）"""
    rows = [r for r in (records if records is not None else store.all_records())
            if r.get("lon") is not None]
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             '<name>稻视界 · 田间病虫害记录</name>']
    for color, kid in (("ffe74c3c", "disease"), ("fff39c12", "pest"), ("ff8b9a8b", "unknown")):
        parts.append(
            '<Style id="s_%s"><IconStyle><color>%s</color><scale>1.1</scale>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>'
            '</Icon></IconStyle></Style>' % (kid, color))
    for r in rows:
        desc = ("<![CDATA[类别：%s<br/>置信度：%.1f%%<br/>时间：%s<br/>地块：%s<br/>"
                "地址：%s<br/>备注：%s]]>" % (
                    _xml_escape(r.get("cls") or "未识别"), (r.get("conf") or 0) * 100,
                    _xml_escape(r.get("shot_at")), _xml_escape(r.get("plot")),
                    _xml_escape(r.get("address")), _xml_escape(r.get("note"))))
        parts.append(
            '<Placemark><name>%s · %s</name><description>%s</description>'
            '<styleUrl>#s_%s</styleUrl><Point><coordinates>%.6f,%.6f,0</coordinates>'
            '</Point></Placemark>' % (
                _xml_escape(r.get("cls") or "未识别"),
                _xml_escape((r.get("shot_at") or "")[:16]), desc,
                disease_kind(r.get("cls") or ""), r["lon"], r["lat"]))
    parts.append('</Document></kml>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return str(path)


_WEB_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>稻视界 · 田间病虫害点位图</title>
<style>
 :root{--green:#00783C;--soft:#E6F0EA;--ink:#1A3A28;--disease:#E74C3C;--pest:#F39C12;--line:#DCDCDC}
 *{box-sizing:border-box;margin:0;padding:0}
 html,body{height:100%;font-family:"Microsoft YaHei","PingFang SC",sans-serif;color:var(--ink)}
 #app{height:100%;display:flex;flex-direction:column}
 #top{height:56px;flex:0 0 auto;display:flex;align-items:center;gap:12px;padding:0 16px;
      background:#fff;border-bottom:2px solid var(--green)}
 #top .dot{width:12px;height:12px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px var(--soft)}
 #top .brand{font-size:17px;font-weight:700}
 #top .sub{font-size:12px;color:#647878}
 #top .sp{flex:1}
 #top button{border:1px solid var(--green);background:#fff;color:var(--green);font-size:13px;
      padding:6px 14px;border-radius:16px;cursor:pointer;font-family:inherit}
 #top button:hover,#top button.on{background:var(--green);color:#fff}
 #wrap{flex:1;position:relative}
 #map{position:absolute;inset:0;background:#eef2ee}
 #side{position:absolute;top:14px;right:14px;width:270px;max-height:64%;background:rgba(255,255,255,.96);
      border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.12);display:flex;flex-direction:column;overflow:hidden}
 #side .hd{padding:10px 14px;border-bottom:1px solid var(--line);font-size:13px;font-weight:700;color:var(--green)}
 #side ul{list-style:none;overflow-y:auto;padding:4px 0}
 #side li{padding:7px 14px;font-size:13px;cursor:pointer;display:flex;gap:8px;align-items:center;border-left:3px solid transparent}
 #side li:hover{background:var(--soft);border-left-color:var(--green)}
 #side li .d{width:9px;height:9px;border-radius:50%;flex:0 0 auto}
 #side li .c{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 #side li .t{color:#9ca3af;font-size:11px}
 #legend{position:absolute;left:14px;bottom:14px;background:rgba(255,255,255,.96);border-radius:10px;
      padding:10px 14px;font-size:12.5px;line-height:1.9;box-shadow:0 4px 18px rgba(0,0,0,.12)}
 #legend .row{display:flex;align-items:center;gap:8px}
 #legend .d{width:11px;height:11px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.15)}
 #alert{position:absolute;top:14px;left:50%;transform:translateX(-50%);background:#FEF3C7;border:1px solid #F0A014;
      color:#92400E;padding:10px 16px;border-radius:8px;font-size:13px;display:none;max-width:76%;z-index:9}
 .pop{width:236px;font-family:inherit}
 .pop img{width:100%;height:130px;object-fit:cover;border-radius:6px;display:block;background:#f3f4f6}
 .pop .pt{font-size:14px;font-weight:700;margin:8px 0 2px}
 .pop .tag{font-size:11px;padding:1px 7px;border-radius:8px;color:#fff;font-weight:400;margin-left:6px}
 .pop .tag.disease{background:var(--disease)} .pop .tag.pest{background:var(--pest)}
 .pop .r{font-size:12px;color:#4b5563;line-height:1.7;margin-top:2px}
 .pop .r em{font-style:normal;color:#9ca3af}
</style></head>
<body><div id="app">
 <div id="top"><span class="dot"></span><span class="brand">稻视界</span>
  <span class="sub">田间病虫害点位分布图 · 数据来自桌面端「地块地图」</span><span class="sp"></span>
  <span class="sub" id="cnt"></span>
  <button id="tour">▶ 依次巡览</button></div>
 <div id="wrap"><div id="map"></div><div id="alert"></div>
  <aside id="side"><div class="hd">点位列表</div><ul id="list"></ul></aside>
  <div id="legend"><b>图例</b>
   <div class="row"><span class="d" style="background:#E74C3C"></span>病害（白叶枯病/胡麻斑病/稻瘟病）</div>
   <div class="row"><span class="d" style="background:#F39C12"></span>虫害（稻纵卷叶螟/稻水象甲/稻叶蝉）</div></div>
 </div></div>
<script>const AMAP_KEY="__AMAP_KEY__";const POINTS=__POINTS__;</script>
<script src="https://webapi.amap.com/maps?v=2.0&key=__AMAP_KEY__"></script>
<script>
(function(){
 var $=function(s){return document.querySelector(s)};
 var box=$('#alert');
 function alertMsg(m){box.textContent=m;box.style.display='block'}
 function popupHTML(p){
  var kind=p.kind==='pest'?'虫害':'病害';
  var rows='';
  if(p.conf) rows+='<div class="r"><em>置信度：</em>'+p.conf+'</div>';
  if(p.time) rows+='<div class="r"><em>记录时间：</em>'+p.time+'</div>';
  if(p.plot) rows+='<div class="r"><em>地块：</em>'+p.plot+'</div>';
  if(p.note) rows+='<div class="r"><em>备注：</em>'+p.note+'</div>';
  return '<div class="pop">'+(p.img?'<img src="'+p.img+'" alt="">':'')+
   '<div class="pt">'+(p.cls||'未识别')+'<span class="tag '+p.kind+'">'+kind+'</span></div>'+rows+'</div>';
 }
 var listEl=$('#list');
 $('#cnt').textContent=POINTS.length?('共 '+POINTS.length+' 个点位'):'';
 POINTS.forEach(function(p,i){
  var li=document.createElement('li');
  li.innerHTML='<span class="d" style="background:'+(p.kind==='pest'?'#F39C12':'#E74C3C')+'"></span>'+
   '<span class="c">'+(i+1)+'. '+(p.cls||'未识别')+'</span><span class="t">'+(p.time||'')+'</span>';
  li.onclick=function(){ if(window.__map) tour(p); };
  listEl.appendChild(li);
 });
 if(typeof AMap==='undefined'){
  alertMsg('高德地图脚本未加载：请联网，并在「API 设置」里填好高德 Web端(JS API) Key 后重新导出。下方列表仍可查看全部点位。');
  return;
 }
 if(!AMAP_KEY){ alertMsg('尚未配置高德 Key（「API 设置」→ 高德 JS API Key），地图可能无法显示。'); }
 var map=new AMap.Map('map',{viewMode:'2D',zoom:12,center:[118.803,32.045],mapStyle:'amap://styles/whitesmoke'});
 window.__map=map;
 var info=new AMap.InfoWindow({offset:new AMap.Pixel(0,-16)});
 var markers=[];
 POINTS.forEach(function(p){
  var mk=new AMap.Marker({position:[p.lng,p.lat],
   content:'<div style="width:20px;height:20px;border-radius:50%;border:3px solid #fff;'+
    'background:'+(p.kind==='pest'?'#F39C12':'#E74C3C')+';box-shadow:0 1px 8px rgba(0,0,0,.4)"></div>',
   offset:new AMap.Pixel(-10,-10),map:map,title:p.cls||''});
  mk.on('click',function(){info.setContent(popupHTML(p));info.open(map,[p.lng,p.lat])});
  p.__mk=mk; markers.push(mk);
 });
 function tour(p){ map.panTo([p.lng,p.lat]); info.setContent(popupHTML(p)); info.open(map,[p.lng,p.lat]); }
 window.tour=tour;
 if(markers.length){ map.setFitView(markers,false,[80,80,80,80],16); }
 else { alertMsg('暂无点位数据。'); }
 var timer=null;
 $('#tour').onclick=function(){
  if(timer){clearInterval(timer);timer=null;this.classList.remove('on');this.textContent='▶ 依次巡览';return}
  if(!markers.length) return;
  var k=0; this.classList.add('on'); this.textContent='⏸ 停止巡览';
  tour(POINTS[0]);
  timer=setInterval(function(){k=(k+1)%POINTS.length;tour(POINTS[k])},2200);
 };
})();
</script></body></html>
"""


def export_web_map(store, out_dir=None, key=None, records=None, title_note=""):
    """导出一份可双击打开 / 可上传服务器的交互式网页地图"""
    rows = records if records is not None else store.all_records()
    out_dir = Path(out_dir) if out_dir else (store.export_dir / "webmap")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pts = []
    for r in rows:
        if r.get("glon") is None:
            continue
        img_rel = ""
        src = r.get("thumb_path") or r.get("ann_path") or r.get("orig_path")
        if src and os.path.exists(src):
            name = "%s.jpg" % (r.get("uid") or r["id"])
            try:
                shutil.copy2(src, img_dir / name)
                img_rel = "images/" + name
            except Exception:
                img_rel = ""
        pts.append({
            "lng": round(r["glon"], 6), "lat": round(r["glat"], 6),
            "cls": r.get("cls") or "", "kind": disease_kind(r.get("cls") or ""),
            "conf": ("%.1f%%" % ((r.get("conf") or 0) * 100)) if r.get("conf") else "",
            "time": (r.get("shot_at") or "")[:16], "plot": r.get("plot") or "",
            "note": r.get("note") or "", "img": img_rel,
        })
    html = _WEB_TEMPLATE.replace("__AMAP_KEY__", key if key is not None else amap_js_key()) \
                       .replace("__POINTS__", json.dumps(pts, ensure_ascii=False))
    out_html = out_dir / "map.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out_html)


def open_in_amap(lon, lat, name="水稻病虫害点位"):
    """在高德网页版打开该坐标"""
    url = ("https://uri.amap.com/marker?position=%.6f,%.6f&name=%s&coordinate=gaode&callnative=0"
           % (lon, lat, requests.utils.quote(name)))
    QDesktopServices.openUrl(QUrl(url))
    return url


def reveal_in_explorer(path):
    try:
        p = Path(path)
        if p.is_file():
            import subprocess
            subprocess.Popen(["explorer", "/select,", str(p)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
    except Exception as e:
        log.info("打开目录失败: %s", e)


# ══════════════════════════════════════════════════════════════════
# 9. 主程序风格注入（可选）
# ══════════════════════════════════════════════════════════════════
_CHROME = {"center": None, "round": None, "home_factory": None}

C = {
    "primary": "#00783C", "primary_hover": "#006433", "primary_light": "#E6F0EA",
    "accent_gold": "#F0A014", "bg": "#F0F0F0", "card": "#FFFFFF",
    "dark": "#1A3A28", "text": "#2D3A30", "subtext": "#647878",
    "border": "#DCDCDC", "red": "#C0392B", "warning": "#E67E22",
}


def install_chrome(center_fn=None, round_fn=None, home_factory=None):
    """由主程序注入：窗口居中函数、圆角标题栏函数、返回首页工厂"""
    if center_fn:
        _CHROME["center"] = center_fn
    if round_fn:
        _CHROME["round"] = round_fn
    if home_factory:
        _CHROME["home_factory"] = home_factory


def _apply_chrome(win, w, h, title):
    try:
        if _CHROME["round"]:
            _CHROME["round"](win, 26, title)
        if _CHROME["center"]:
            _CHROME["center"](win, w, h)
        else:
            win.resize(w, h)
            scr = QApplication.primaryScreen()
            if scr:
                g = scr.availableGeometry()
                win.move(g.center().x() - w // 2, g.center().y() - h // 2)
    except Exception as e:
        log.warning("窗口装饰失败（不影响功能）: %s", e)


def card_style():
    return ("background-color: %s; border-radius: 14px; border: 1px solid %s;"
            % (C["card"], C["border"]))


def _btn_style(color=None, small=True):
    bg = color or C["primary"]
    pad = "6px 14px" if small else "10px 22px"
    return ("""
        QPushButton { background: %s; color: white; border: none; border-radius: 16px;
                      padding: %s; font-size: %s; font-weight: bold; }
        QPushButton:hover { background: %s; }
        QPushButton:disabled { background: #B9C6BD; }
    """ % (bg, pad, "12px" if small else "15px", C["primary_hover"]))


def _ghost_btn_style():
    return ("""
        QPushButton { background: %s; color: %s; border: 1px solid %s;
                      border-radius: 16px; padding: 6px 14px; font-size: 12px; }
        QPushButton:hover { border-color: %s; color: %s; }
    """ % (C["bg"], C["text"], C["border"], C["primary"], C["primary"]))


def _fmt_dt(s):
    return (s or "").replace("-", "/")[:16]


# ══════════════════════════════════════════════════════════════════
# 10. 地图主窗口
# ══════════════════════════════════════════════════════════════════
class GeoMapWindow(QMainWindow):
    """地块地图 · 溯源记录：照片 → 点位 → 时间回溯"""

    def __init__(self, model=None, device=None, pending=None, focus_id=None,
                 home_factory=None):
        super().__init__()
        self.model = model
        self.device = device
        self.pending = pending
        self.focus_id = focus_id
        self.home_factory = home_factory or _CHROME.get("home_factory")
        self.store = GeoStore()
        self.records = []
        self.view_records = []
        self.current_id = None
        self.pending_ids = []
        self.import_thread = None
        self.t_min = None
        self.t_max = None
        self._playing = False

        self.setWindowTitle("地块地图 · 溯源记录")
        self._build()
        _apply_chrome(self, 1280, 850, "🗺  地块地图 · 溯源记录")

        self.refresh(fit=True)
        if self.pending:
            QTimer.singleShot(80, self._consume_pending)
        elif self.focus_id:
            QTimer.singleShot(80, lambda: self._select_record(self.focus_id, focus_map=True))
        QTimer.singleShot(300, self._first_hint)

    # ── 界面 ──
    def _build(self):
        central = QWidget()
        central.setStyleSheet("background-color: %s;" % C["bg"])
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(26, 14, 26, 14)
        root.setSpacing(10)

        # 顶部标题
        head = QHBoxLayout()
        title = QLabel("🗺  地块地图 · 溯源记录")
        title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        title.setStyleSheet("color:%s;background:transparent;border:none;" % C["dark"])
        sub = QLabel("照片里的时间戳与经纬度 → 自动落到地图上，随时查看、随时回溯")
        sub.setFont(QFont("Microsoft YaHei", 11))
        sub.setStyleSheet("color:%s;background:transparent;border:none;" % C["subtext"])
        head.addWidget(title)
        head.addSpacing(12)
        head.addWidget(sub)
        head.addStretch()
        self.btn_home = QPushButton("🏠 返回首页")
        self.btn_home.setCursor(Qt.PointingHandCursor)
        self.btn_home.setStyleSheet(_ghost_btn_style())
        self.btn_home.clicked.connect(self.go_home)
        head.addWidget(self.btn_home)
        root.addLayout(head)

        # 工具条
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_import = QPushButton("📥 导入照片并识别")
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.setStyleSheet(_btn_style())
        self.btn_import.clicked.connect(self.import_photos)

        self.btn_locate = QPushButton("📍 手动落点")
        self.btn_locate.setCursor(Qt.PointingHandCursor)
        self.btn_locate.setStyleSheet(_btn_style(C["accent_gold"]))
        self.btn_locate.clicked.connect(self.start_locate)

        self.btn_regeo_all = QPushButton("🔎 批量解析地址")
        self.btn_regeo_all.setCursor(Qt.PointingHandCursor)
        self.btn_regeo_all.setStyleSheet(_ghost_btn_style())
        self.btn_regeo_all.clicked.connect(self.batch_regeo)
        self.btn_regeo_all.setToolTip("用高德逆地理编码把坐标变成文字地址（需要 Web 服务 Key）")

        lbl_style = QLabel("底图")
        lbl_style.setStyleSheet("color:%s;background:transparent;" % C["subtext"])
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["高德路网", "高德卫星"])
        self.cmb_style.setFixedHeight(32)
        self.cmb_style.setStyleSheet("QComboBox{background:#fff;border:1px solid %s;border-radius:16px;padding:2px 12px;}"
                                     % C["border"])
        self.cmb_style.currentIndexChanged.connect(
            lambda i: self.canvas.set_style("satellite" if i == 1 else "road"))

        self.chk_track = QCheckBox("连线轨迹")
        self.chk_track.setStyleSheet("QCheckBox{color:%s;background:transparent;}" % C["text"])
        self.chk_track.toggled.connect(lambda v: (setattr(self.canvas, "show_track", v),
                                                  self.canvas.update()))

        self.btn_csv = QPushButton("📤 导出 CSV")
        self.btn_csv.setCursor(Qt.PointingHandCursor)
        self.btn_csv.setStyleSheet(_ghost_btn_style())
        self.btn_csv.clicked.connect(lambda: self.export("csv"))

        self.btn_kml = QPushButton("🌍 导出 KML")
        self.btn_kml.setCursor(Qt.PointingHandCursor)
        self.btn_kml.setStyleSheet(_ghost_btn_style())
        self.btn_kml.clicked.connect(lambda: self.export("kml"))

        self.btn_web = QPushButton("🌐 网页版地图")
        self.btn_web.setCursor(Qt.PointingHandCursor)
        self.btn_web.setStyleSheet(_ghost_btn_style())
        self.btn_web.clicked.connect(lambda: self.export("web"))

        for w in (self.btn_import, self.btn_locate, self.btn_regeo_all, lbl_style,
                  self.cmb_style, self.chk_track, self.btn_csv, self.btn_kml, self.btn_web):
            bar.addWidget(w)
        bar.addStretch()
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:%s;background:transparent;" % C["subtext"])
        bar.addWidget(self.lbl_status)
        root.addLayout(bar)

        # 主体三栏
        body = QHBoxLayout()
        body.setSpacing(10)

        # 左：记录列表
        left_card = QWidget(); left_card.setStyleSheet(card_style())
        lv = QVBoxLayout(left_card); lv.setContentsMargins(12, 12, 12, 12); lv.setSpacing(8)
        lv.addWidget(self._cap("记录列表"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["全部记录", "只看病害", "只看虫害", "未识别", "待落点", "已落点"])
        self.cmb_filter.setFixedHeight(32)
        self.cmb_filter.setStyleSheet("QComboBox{background:%s;border:1px solid %s;border-radius:8px;padding:4px 10px;}"
                                      % (C["bg"], C["border"]))
        self.cmb_filter.currentIndexChanged.connect(lambda _: self.apply_filter())
        lv.addWidget(self.cmb_filter)

        self.listw = QListWidget()
        self.listw.setIconSize(QSize(60, 45))
        self.listw.setStyleSheet("""
            QListWidget{background:transparent;border:none;font-size:12px;color:%s;}
            QListWidget::item{padding:6px 4px;border-bottom:1px solid #EEF2EE;}
            QListWidget::item:selected{background:%s;color:%s;border-radius:6px;}
        """ % (C["text"], C["primary_light"], C["primary"]))
        self.listw.currentItemChanged.connect(self._on_list_changed)
        lv.addWidget(self.listw, 1)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color:%s;background:transparent;font-size:11px;" % C["subtext"])
        lv.addWidget(self.lbl_count)
        left_card.setFixedWidth(300)
        body.addWidget(left_card)

        # 中：地图 + 时间轴
        mid_card = QWidget(); mid_card.setStyleSheet(card_style())
        mv = QVBoxLayout(mid_card); mv.setContentsMargins(10, 10, 10, 10); mv.setSpacing(8)
        self.canvas = MapCanvas(self.store.tile_dir)
        self.canvas.markerClicked.connect(self._select_record)
        self.canvas.mapClicked.connect(self._on_map_picked)
        mv.addWidget(self.canvas, 1)

        tl = QHBoxLayout(); tl.setSpacing(8)
        tl.addWidget(self._small("时间回溯"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(1000)
        self.slider.valueChanged.connect(self._on_slider)
        tl.addWidget(self.slider, 1)
        self.lbl_cut = QLabel("全部")
        self.lbl_cut.setFixedWidth(120)
        self.lbl_cut.setStyleSheet("color:%s;background:transparent;font-size:12px;" % C["primary"])
        tl.addWidget(self.lbl_cut)
        self.chk_accum = QCheckBox("累积显示")
        self.chk_accum.setChecked(True)
        self.chk_accum.setStyleSheet("QCheckBox{color:%s;background:transparent;font-size:12px;}" % C["text"])
        self.chk_accum.toggled.connect(lambda _: self.apply_filter())
        tl.addWidget(self.chk_accum)
        self.btn_play = QPushButton("▶ 回放")
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.setStyleSheet(_ghost_btn_style())
        self.btn_play.clicked.connect(self.toggle_play)
        tl.addWidget(self.btn_play)
        mv.addLayout(tl)
        body.addWidget(mid_card, 1)

        # 右：详情
        right_card = QWidget(); right_card.setStyleSheet(card_style())
        rv = QVBoxLayout(right_card); rv.setContentsMargins(12, 12, 12, 12); rv.setSpacing(8)
        rv.addWidget(self._cap("记录详情"))
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QFrame.NoFrame)
        self.detail_scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        holder = QWidget(); holder.setStyleSheet("background:transparent;")
        self.dv = QVBoxLayout(holder); self.dv.setContentsMargins(0, 0, 0, 0); self.dv.setSpacing(6)

        self.thumb = QLabel("未选择记录")
        self.thumb.setFixedHeight(170)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background:%s;border-radius:10px;color:%s;"
                                 % (C["bg"], C["subtext"]))
        self.dv.addWidget(self.thumb)

        self.lbl_fields = QLabel("")
        self.lbl_fields.setWordWrap(True)
        self.lbl_fields.setTextFormat(Qt.RichText)
        self.lbl_fields.setStyleSheet("color:%s;background:transparent;font-size:12.5px;" % C["text"])
        self.dv.addWidget(self.lbl_fields)

        self.dv.addWidget(self._small("备注（例如：雨后叶尖发黄，已撒药）"))
        self.edit_note = QLineEdit()
        self.edit_note.setStyleSheet("QLineEdit{background:%s;border:1px solid %s;border-radius:8px;"
                                     "padding:6px 10px;}" % (C["bg"], C["border"]))
        self.dv.addWidget(self.edit_note)

        self.dv.addWidget(self._small("地块名称（例如：3号田 / 东大田）"))
        self.cmb_plot = QComboBox()
        self.cmb_plot.setEditable(True)
        self.cmb_plot.setStyleSheet("QComboBox{background:%s;border:1px solid %s;border-radius:8px;"
                                    "padding:6px 10px;}" % (C["bg"], C["border"]))
        self.dv.addWidget(self.cmb_plot)

        self.btn_save_note = QPushButton("💾 保存备注与地块")
        self.btn_save_note.setCursor(Qt.PointingHandCursor)
        self.btn_save_note.setStyleSheet(_btn_style())
        self.btn_save_note.clicked.connect(self.save_note)
        self.dv.addWidget(self.btn_save_note)

        row1 = QHBoxLayout(); row1.setSpacing(6)
        for text, slot in (("🗺 定位", lambda: self._select_record(self.current_id, focus_map=True)),
                           ("🔎 地址", self.regeo_current),
                           ("🌐 高德打开", self.open_amap_current)):
            b = QPushButton(text); b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_ghost_btn_style()); b.clicked.connect(slot)
            row1.addWidget(b)
        self.dv.addLayout(row1)

        row2 = QHBoxLayout(); row2.setSpacing(6)
        b_folder = QPushButton("📁 打开文件夹"); b_folder.setCursor(Qt.PointingHandCursor)
        b_folder.setStyleSheet(_ghost_btn_style())
        b_folder.clicked.connect(self.open_folder_current)
        b_del = QPushButton("🗑 删除记录"); b_del.setCursor(Qt.PointingHandCursor)
        b_del.setStyleSheet(_ghost_btn_style())
        b_del.clicked.connect(self.delete_current)
        row2.addWidget(b_folder); row2.addWidget(b_del)
        self.dv.addLayout(row2)
        self.dv.addStretch()

        self.detail_scroll.setWidget(holder)
        rv.addWidget(self.detail_scroll, 1)
        right_card.setFixedWidth(300)
        body.addWidget(right_card)

        root.addLayout(body, 1)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color:%s;background:transparent;font-size:12px;" % C["subtext"])
        root.addWidget(self.lbl_stats)

    def _cap(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        lbl.setStyleSheet("color:%s;background:transparent;border:none;" % C["primary"])
        return lbl

    def _small(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:%s;background:transparent;border:none;font-size:11.5px;" % C["subtext"])
        return lbl

    # ── 数据刷新 ──
    def refresh(self, fit=False):
        self.records = self.store.all_records()
        times = [r["shot_at"] for r in self.records if r.get("shot_at")]
        self.t_min = _parse_dt(min(times)) if times else None
        self.t_max = _parse_dt(max(times)) if times else None
        self.apply_filter(fit=fit)
        self._update_stats()
        if self.current_id is None and self.view_records:
            self._select_record(self.view_records[0]["id"])

    def _cutoff(self):
        if not self.t_min or not self.t_max:
            return None
        if self.slider.value() >= 1000:
            return None
        span = (self.t_max - self.t_min).total_seconds()
        return self.t_min + timedelta(seconds=span * self.slider.value() / 1000.0)

    def apply_filter(self, fit=False):
        idx = self.cmb_filter.currentIndex()
        cut = self._cutoff()
        accum = self.chk_accum.isChecked()
        rows = []
        for r in self.records:
            has_geo = r.get("glon") is not None
            if idx == 1 and disease_kind(r.get("cls") or "") != "disease":
                continue
            if idx == 2 and disease_kind(r.get("cls") or "") != "pest":
                continue
            if idx == 3 and (r.get("cls") or ""):
                continue
            if idx == 4 and has_geo:
                continue
            if idx == 5 and not has_geo:
                continue
            if cut is not None:
                t = _parse_dt(r.get("shot_at"))
                if t is not None:
                    if accum and t > cut:
                        continue
                    if not accum and (t > cut or t < cut - timedelta(hours=24)):
                        continue
            rows.append(r)
        self.view_records = rows
        self._fill_list()
        self.canvas.set_points(self.store.to_points([r for r in rows if r.get("glon") is not None]))
        if fit and self.view_records:
            self.canvas.fit_points(self.store.to_points(rows))
        self.lbl_count.setText("当前显示 %d / 共 %d 条" % (len(rows), len(self.records)))

    def _fill_list(self):
        self.listw.blockSignals(True)
        self.listw.clear()
        for i, r in enumerate(self.view_records, 1):
            geo = ""
            if r.get("glon") is not None:
                geo = r.get("plot") or r.get("address") or ("%.5f, %.5f" % (r["lat"], r["lon"]))
            else:
                geo = "⚠ 未落点（点「手动落点」在地图上点一下）"
            text = "%d. %s · %s\n     %s" % (i, r.get("cls") or "未识别",
                                             _fmt_dt(r.get("shot_at")), geo)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, r["id"])
            thumb = r.get("thumb_path") or ""
            if thumb and os.path.exists(thumb):
                item.setIcon(QIcon(thumb))
            if r["id"] == self.current_id:
                self.listw.setCurrentItem(item)
            self.listw.addItem(item)
        self.listw.blockSignals(False)

    def _update_stats(self):
        total = len(self.records)
        geo = sum(1 for r in self.records if r.get("glon") is not None)
        plots = len({r.get("plot") for r in self.records if r.get("plot")})
        span = ""
        if self.t_min and self.t_max:
            span = " · 时间跨度 %s ~ %s" % (self.t_min.strftime("%Y/%m/%d"),
                                            self.t_max.strftime("%Y/%m/%d"))
        self.lbl_stats.setText(
            "共 %d 条记录 · 已落点 %d 条 · 地块 %d 个%s　|　数据目录：%s"
            % (total, geo, plots, span, self.store.root))

    # ── 详情 ──
    def _on_list_changed(self, cur, _prev):
        if cur is None:
            return
        self._select_record(cur.data(Qt.UserRole), focus_map=False)

    def _select_record(self, rec_id, focus_map=False):
        if not rec_id:
            return
        rec = self.store.get(rec_id)
        if not rec:
            return
        self.current_id = rec_id
        self.canvas.selected_id = rec_id
        for i in range(self.listw.count()):
            if self.listw.item(i).data(Qt.UserRole) == rec_id:
                self.listw.blockSignals(True)
                self.listw.setCurrentRow(i)
                self.listw.blockSignals(False)
                break
        # 缩略图
        thumb = rec.get("ann_path") or rec.get("orig_path") or rec.get("thumb_path") or ""
        if thumb and os.path.exists(thumb):
            pix = QPixmap(thumb)
            if not pix.isNull():
                self.thumb.setPixmap(pix.scaled(max(self.thumb.width(), 260), 170, Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))
        else:
            self.thumb.setText("无照片副本")
        coord = ("%.6f, %.6f" % (rec["lat"], rec["lon"])) if rec.get("lat") is not None else "未记录"
        gcoord = ("%.6f, %.6f" % (rec["glon"], rec["glat"])) if rec.get("glon") is not None else "未换算"
        src_cn = {"exif": "照片 EXIF(GPS)", "map-pick": "地图点选",
                  "manual": "手动填写", "none": "无"}.get(rec.get("coord_src") or "", rec.get("coord_src") or "")
        self.lbl_fields.setText(
            "<b>%s</b>　<span style='color:#C0392B'>%.1f%%</span><br/>"
            "<span style='color:#647878'>拍摄时间：</span>%s<br/>"
            "<span style='color:#647878'>GPS(WGS-84)：</span>%s<br/>"
            "<span style='color:#647878'>高德(GCJ-02)：</span>%s<br/>"
            "<span style='color:#647878'>坐标来源：</span>%s<br/>"
            "<span style='color:#647878'>地址：</span>%s<br/>"
            "<span style='color:#647878'>设备：</span>%s<br/>"
            "<span style='color:#647878'>入库时间：</span>%s"
            % (rec.get("cls") or "未识别", (rec.get("conf") or 0) * 100,
               rec.get("shot_at") or "", coord, gcoord, src_cn,
               rec.get("address") or "（点「🔎 地址」解析）",
               rec.get("device") or "—", rec.get("created_at") or ""))
        self.edit_note.setText(rec.get("note") or "")
        self.cmb_plot.clear()
        self.cmb_plot.addItems(self.store.plots())
        self.cmb_plot.setCurrentText(rec.get("plot") or "")
        self.canvas.update()
        if focus_map and rec.get("glon") is not None:
            self.canvas.focus_point(rec["glon"], rec["glat"], max(self.canvas.zoom, 16))

    def save_note(self):
        if not self.current_id:
            QMessageBox.information(self, "提示", "请先在左侧选择一条记录")
            return
        self.store.update(self.current_id, note=self.edit_note.text().strip(),
                          plot=self.cmb_plot.currentText().strip())
        self.refresh()
        self._select_record(self.current_id)
        self._flash("已保存备注与地块")

    def delete_current(self):
        if not self.current_id:
            return
        rec = self.store.get(self.current_id)
        if QMessageBox.question(self, "删除确认",
                                "确定删除这条记录吗？\n%s  %s" % (rec.get("cls") or "未识别",
                                                              rec.get("shot_at") or ""),
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.store.delete(self.current_id)
        self.current_id = None
        self.canvas.selected_id = None
        self.refresh()
        self._flash("已删除")

    def open_folder_current(self):
        rec = self.store.get(self.current_id) if self.current_id else None
        target = (rec or {}).get("orig_path") or str(self.store.photo_dir)
        reveal_in_explorer(target)

    def open_amap_current(self):
        rec = self.store.get(self.current_id) if self.current_id else None
        if not rec or rec.get("glon") is None:
            QMessageBox.information(self, "提示", "这条记录还没有坐标")
            return
        open_in_amap(rec["glon"], rec["glat"], rec.get("cls") or "水稻病虫害点位")

    def regeo_one(self, rec_id):
        rec = self.store.get(rec_id)
        if not rec or rec.get("glon") is None:
            return False
        if not amap_key():
            QMessageBox.information(self, "需要高德 Web 服务 Key",
                                    "逆地理编码需要高德「Web服务」类型的 Key。\n"
                                    "请到 https://lbs.amap.com 申请后，填到「⚙ API 设置」里。")
            return False
        addr = amap_regeo(rec["glon"], rec["glat"])
        if addr:
            self.store.update(rec_id, address=addr)
            return True
        return False

    def regeo_current(self):
        if not self.current_id:
            return
        if self.regeo_one(self.current_id):
            self.refresh()
            self._select_record(self.current_id)
            self._flash("地址已解析")
        else:
            self._flash("地址解析失败（检查 Key / 网络）")

    def batch_regeo(self):
        if not amap_key():
            QMessageBox.information(self, "需要高德 Web 服务 Key",
                                    "请先到 https://lbs.amap.com 申请「Web服务」Key，\n"
                                    "然后填到「⚙ API 设置」→ 高德地图。")
            return
        todo = [r for r in self.store.all_records()
                if r.get("glon") is not None and not r.get("address")]
        if not todo:
            self._flash("所有记录都已有地址")
            return
        n = 0
        for r in todo[:60]:
            if self.regeo_one(r["id"]):
                n += 1
            QApplication.processEvents()
        self.refresh()
        self._flash("已解析 %d 条地址" % n)

    # ── 时间回溯 ──
    def _on_slider(self, _v):
        cut = self._cutoff()
        self.lbl_cut.setText("截至 " + cut.strftime("%m/%d %H:%M") if cut else "全部")
        self.apply_filter()

    def toggle_play(self):
        if self._playing:
            self.play_timer_stop()
            return
        if not self.t_min or not self.t_max:
            self._flash("还没有带时间戳的记录")
            return
        self._playing = True
        self.btn_play.setText("⏸ 停止")
        self.chk_accum.setChecked(True)
        self.slider.setValue(0)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_step)
        self._play_timer.start(90)

    def _play_step(self):
        v = self.slider.value() + 8
        if v >= 1000:
            v = 1000
            self.slider.setValue(v)
            self.play_timer_stop()
            self._flash("回放结束")
            return
        self.slider.setValue(v)

    def play_timer_stop(self):
        self._playing = False
        self.btn_play.setText("▶ 回放")
        t = getattr(self, "_play_timer", None)
        if t:
            t.stop()

    # ── 落点 ──
    def start_locate(self):
        cur = self.store.get(self.current_id) if self.current_id else None
        if cur and cur.get("glon") is None:
            self.pending_ids = [cur["id"]]
        else:
            self.pending_ids = [r["id"] for r in self.records if r.get("glon") is None]
        if not self.pending_ids:
            if cur:
                self.pending_ids = [cur["id"]]      # 允许重设已有坐标的记录
            else:
                QMessageBox.information(self, "无需落点", "所有记录都有坐标了。\n"
                                                            "（也可以先选中一条记录再点「手动落点」来修正位置）")
                return
        self.canvas.set_pick_mode(True)
        self.btn_locate.setText("📍 落点中…（点地图）")
        self.btn_locate.setStyleSheet(_btn_style(C["red"]))
        self._flash("请在右侧地图上点击这 %d 张照片的实际位置（可先缩放/拖动）" % len(self.pending_ids))

    def _on_map_picked(self, lon, lat):
        if not self.pending_ids:
            self.canvas.set_pick_mode(False)
            return
        wlon, wlat = gcj02_to_wgs84(lon, lat)
        for rid in self.pending_ids:
            self.store.set_location(rid, wlon, wlat, coord_src="map-pick")
        n = len(self.pending_ids)
        self.pending_ids = []
        self.canvas.set_pick_mode(False)
        self.btn_locate.setText("📍 手动落点")
        self.btn_locate.setStyleSheet(_btn_style(C["accent_gold"]))
        self.refresh()
        self._flash("已为 %d 条记录落点" % n)

    # ── 导入 ──
    def import_photos(self):
        if self.import_thread is not None and self.import_thread.isRunning():
            QMessageBox.information(self, "提示", "正在导入中，请稍候…")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择田间照片（可多选；带 GPS 的会自动落点）", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not files:
            return
        use_model = self.model is not None
        if use_model:
            ans = QMessageBox.question(
                self, "导入方式",
                "是否对导入的照片同时运行病虫害识别？\n\n"
                "是：识别 + 落点（照片多时较慢）\n否：只读取时间和经纬度（很快）",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                return
            use_model = (ans == QMessageBox.Yes)
        self.btn_import.setEnabled(False)
        self.btn_import.setText("⏳ 导入中…")
        self.import_thread = ImportThread(
            self.store, files, model=self.model if use_model else None,
            conf=CONF_THRESHOLD, class_names=CLASS_NAMES,
            auto_regeo=bool(amap_key()))
        self.import_thread.progress.connect(
            lambda i, n, name: self.lbl_status.setText("导入 %d/%d：%s" % (i, n, name)))
        self.import_thread.added.connect(lambda *_: None)
        self.import_thread.done.connect(self._on_import_done)
        self.import_thread.start()

    def _on_import_done(self, ok, skipped, failed):
        self.btn_import.setEnabled(True)
        self.btn_import.setText("📥 导入照片并识别")
        self.lbl_status.setText("")
        self.import_thread = None
        self.refresh(fit=True)
        no_geo = sum(1 for r in self.store.all_records() if r.get("glon") is None)
        msg = "导入完成：新增 %d 张" % ok
        if skipped:
            msg += "，跳过重复 %d 张" % skipped
        if failed:
            msg += "，失败 %d 张" % failed
        if no_geo:
            msg += "\n\n其中 %d 张照片没有 GPS 信息（微信传输/截图常见），\n" \
                   "点「📍 手动落点」在地图上点一下即可补上。" % no_geo
        QMessageBox.information(self, "导入结果", msg)

    # ── 从识别结果页带过来的记录 ──
    def _consume_pending(self):
        p = self.pending or {}
        self.pending = None
        try:
            meta = {"taken_at": datetime.now(), "lon": None, "lat": None, "device": ""}
            src = p.get("source_path")
            if src and os.path.exists(src):
                meta = read_photo_meta(src)
            rec_id = self.store.add_record(
                shot_at=meta.get("taken_at"), lon=meta.get("lon"), lat=meta.get("lat"),
                coord_src="exif" if meta.get("lon") is not None else "none",
                cls=p.get("cls") or "", conf=p.get("conf") or 0.0,
                boxes=p.get("results") or [], note=p.get("note") or "",
                source_path=src, ann_path=p.get("annotated"),
                device=meta.get("device", ""))
            self.refresh()
            if rec_id:
                self._select_record(rec_id, focus_map=True)
                if meta.get("lon") is None:
                    self.pending_ids = [rec_id]
                    self.canvas.set_pick_mode(True)
                    self.btn_locate.setText("📍 落点中…（点地图）")
                    self._flash("这张照片没有 GPS：请在地图上点击它的实际位置")
                else:
                    self._flash("已记录到地图（位置来自照片 GPS）")
        except Exception as e:
            log.warning("保存识别记录失败: %s", e)
            QMessageBox.warning(self, "保存失败", str(e))

    # ── 导出 ──
    def export(self, kind):
        if not self.view_records and not self.records:
            QMessageBox.information(self, "没有数据", "还没有可导出的记录")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if kind == "csv":
                path = self.store.export_dir / ("田间记录_%s.csv" % stamp)
                export_csv(self.store, path, self.view_records or None)
                reveal_in_explorer(path)
            elif kind == "kml":
                path = self.store.export_dir / ("田间记录_%s.kml" % stamp)
                export_kml(self.store, path, self.view_records or None)
                reveal_in_explorer(path)
            else:
                path = export_web_map(self.store, self.store.export_dir / ("webmap_%s" % stamp),
                                      records=self.view_records or None)
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self._flash("已导出：%s" % os.path.basename(str(path)))
        except Exception as e:
            log.warning("导出失败: %s", e)
            QMessageBox.warning(self, "导出失败", str(e))

    # ── 杂项 ──
    def _flash(self, text):
        self.lbl_status.setText(text)

    def _first_hint(self):
        if self.records:
            return
        self._flash("提示：点「📥 导入照片并识别」，选择手机里拍的田间照片")

    def go_home(self):
        factory = self.home_factory or _CHROME.get("home_factory")
        if factory:
            try:
                win = factory()
                win.show()
                self.close()
                return
            except Exception as e:
                log.warning("返回首页失败: %s", e)
        self.close()

    def closeEvent(self, e):
        try:
            self.play_timer_stop()
            self.canvas.loader.stop()
            self.canvas.loader.wait(1500)
        except Exception:
            pass
        e.accept()


# ══════════════════════════════════════════════════════════════════
# 11. 供主程序调用的一步式接口
# ══════════════════════════════════════════════════════════════════
def record_detection(image_bgr, annotated_bgr=None, results=None, cls="", conf=0.0,
                     source_path=None, note=""):
    """
    把一次识别结果写进地图档案（不需要打开窗口）。
    返回记录 id；照片自带 GPS 时会自动落点。
    """
    store = GeoStore()
    meta = {"taken_at": datetime.now(), "lon": None, "lat": None, "device": ""}
    if source_path and os.path.exists(source_path):
        meta = read_photo_meta(source_path)
    return store.add_record(
        shot_at=meta.get("taken_at"), lon=meta.get("lon"), lat=meta.get("lat"),
        coord_src="exif" if meta.get("lon") is not None else "none",
        cls=cls, conf=conf, boxes=results or [], note=note,
        source_path=source_path, ann_path=annotated_bgr, device=meta.get("device", ""))


if __name__ == "__main__":
    # 单独运行本文件：打开地图窗口（带 6 条演示数据，方便先看效果）
    import random
    app = QApplication(sys.argv)
    if "--demo" in sys.argv:
        st = GeoStore()
        if st.count() == 0:
            base_lon, base_lat = 118.803, 32.045      # 南农大（WGS-84）
            names = list(DISEASES) + list(PESTS)
            for i in range(6):
                lon = base_lon + random.uniform(-0.012, 0.012)
                lat = base_lat + random.uniform(-0.008, 0.008)
                st.add_record(shot_at=datetime.now() - timedelta(days=30 - i * 5),
                              lon=lon, lat=lat, coord_src="exif",
                              cls=names[i % len(names)], conf=0.72 + 0.05 * i,
                              note="演示数据", plot="%d号田" % (i % 3 + 1))
        print("演示库：", st.db_path)
    win = GeoMapWindow()
    win.show()
    sys.exit(app.exec_())

