import sys
import os
import json
import logging
import logging.handlers
import warnings
import base64
import re
import markdown
import math
import cv2
import torch
import numpy as np
import requests
from functools import lru_cache
import subprocess
import csv
import io
import random
import hashlib
import shutil
import socket
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timedelta
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter

try:                       # urllib3 是 requests 的依赖，正常一定存在
    from urllib3.util.retry import Retry
except ImportError:        # pragma: no cover
    Retry = None

from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QMessageBox, QProgressBar, QTextEdit, QInputDialog,
                             QGraphicsDropShadowEffect, QScrollArea, QFrame,
                             QLineEdit, QComboBox, QCheckBox, QShortcut, QSizePolicy,
                             QListWidget, QListWidgetItem, QStackedWidget, QGridLayout)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QFontMetrics, QFontDatabase,
                         QKeySequence, QDragEnterEvent, QDropEvent, QColor, QIcon)
from PyQt5.QtCore import (Qt, QTimer, QThread, QSize, QSettings, pyqtSignal,
                          QVariantAnimation, QEasingCurve, QObject, pyqtSlot,
                          QEvent, QUrl)      # 交互地图要用：QUrl.fromLocalFile 加载本地 map.html

warnings.filterwarnings("ignore")

# ── 路径解析（兼容 PyInstaller 打包 与 源码直接运行） ─────────────
# BASE_DIR：只读资源根目录。打包后指向解压临时目录 _MEIPASS，源码运行时指向脚本所在目录。
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# APP_DIR：可写目录（配置文件、用户后续自行补充的图片）。打包后指向 exe 所在目录。
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parent
# 可用环境变量 RICE_APP_DIR 覆盖（自动化测试把日志/配置/报告隔离到临时目录时用）
if os.environ.get("RICE_APP_DIR"):
    APP_DIR = Path(os.environ["RICE_APP_DIR"]).resolve()

# ── 日志系统 ──────────────────────────────────────────────
log = logging.getLogger("rice_app")


def _setup_logging():
    """控制台 + 滚动文件双通道。

    打包成 --noconsole 后控制台日志会全部丢失，文件日志用于事后排查现场问题。
    """
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    if not log.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        log.addHandler(stream)
    try:
        log_dir = APP_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as e:
        log.warning("无法创建日志文件，仅输出到控制台: %s", e)
    log.propagate = False


_setup_logging()


def _resource_roots():
    """资源搜索根目录：优先打包内置资源，其次程序目录（便于用户替换/补充）"""
    roots = [BASE_DIR]
    if APP_DIR != BASE_DIR:
        roots.append(APP_DIR)
    return roots


def res_path(rel):
    """把相对资源路径解析为绝对路径；找不到时返回首选路径，交由调用方判断存在性"""
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    for root in _resource_roots():
        cand = root / p
        if cand.exists():
            return str(cand)
    return str(BASE_DIR / p)


# ── 字体：优先用随程序分发的开源字体，缺失时回退系统字体 ──────────
FONT_DIR = Path(res_path("resources/fonts"))
# 正文字体候选（前面找不到就往后顺延，最后一个一定存在于 Windows）
UI_FONT_CANDIDATES = ["HarmonyOS Sans SC", "Noto Sans SC", "Source Han Sans SC", "思源黑体",
                      "MiSans", "Microsoft YaHei UI", "Microsoft YaHei"]
# 标题字体候选：把 "Smiley Sans"（得意黑）放到最前面即可自动启用
# 标题族：优先衬线（宋体），笔锋更有分量，也更有出版物气质；
# 都没有时退回黑体，不会报错。
DISPLAY_FONT_CANDIDATES = ["Smiley Sans", "得意黑", "Noto Serif SC", "Source Han Serif SC",
                           "思源宋体", "HarmonyOS Sans SC", "MiSans",
                           "Noto Sans SC", "Microsoft YaHei"]
_font_family_cache = {}


def _register_bundled_fonts():
    """把 resources/fonts 下的字体注册进 Qt（必须在 QApplication 之后调用）"""
    if not FONT_DIR.is_dir():
        return
    for f in sorted(FONT_DIR.glob("*.otf")) + sorted(FONT_DIR.glob("*.ttf")):
        if QFontDatabase.addApplicationFont(str(f)) < 0:
            log.warning("字体注册失败: %s", f.name)


def _pick_family(candidates):
    installed = set(QFontDatabase().families())
    for name in candidates:
        if name in installed:
            return name
    return candidates[-1]


def ui_family():
    """正文字体族名（延迟解析：查字体库必须在 QApplication 之后）"""
    if "ui" not in _font_family_cache:
        _font_family_cache["ui"] = _pick_family(UI_FONT_CANDIDATES)
        log.info("正文字体: %s", _font_family_cache["ui"])
    return _font_family_cache["ui"]


def display_family():
    """标题字体族名"""
    if "display" not in _font_family_cache:
        _font_family_cache["display"] = _pick_family(DISPLAY_FONT_CANDIDATES)
        log.info("标题字体: %s", _font_family_cache["display"])
    return _font_family_cache["display"]


# 西文/数字用 Inter（OFL，观感最接近 iOS 的 SF Pro），中文自动回退。
# Qt 5.13+ 的 QFont.setFamilies 支持按顺序回退，所以一套字体栈就能做中英混排。
LATIN_FONT = "Inter"
_font_stack_cache = {}


def font_stack(primary):
    """字体栈：[西文 Inter, 中文 primary]；没装 Inter 就退化成单字体"""
    if primary not in _font_stack_cache:
        installed = set(QFontDatabase().families())
        stack = [f for f in (LATIN_FONT, primary) if f in installed]
        _font_stack_cache[primary] = stack or [primary]
    return _font_stack_cache[primary]


def _make_font(size, families, weight):
    f = QFont()
    f.setFamilies(families)
    f.setPointSize(size)
    f.setWeight(weight)
    # 中文小字开全 hinting：笔画边缘会锐利很多，不发虚
    f.setHintingPreference(QFont.PreferFullHinting)
    return f


def ui_font(size, bold=False, weight=None):
    """正文/控件字体：西文 Inter + 中文 Noto Sans SC"""
    w = weight if weight is not None else (QFont.Bold if bold else QFont.Normal)
    return _make_font(size, font_stack(ui_family()), w)


def display_font(size, bold=True, weight=None):
    """标题字体：衬线中文不掺西文，保持笔锋统一"""
    w = weight if weight is not None else (QFont.Bold if bold else QFont.Normal)
    return _make_font(size, [display_family()], w)

# ====================== 【一键配置区】修改这里即可 ======================
MODEL_PATH = res_path("best.pt")
CLASS_NAMES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]
# 置信度阈值：0.4（2026-10-09 由 0.6 下调。干净 val 实测：0.6 时白叶枯病召回 0.506、稻瘟病 0.685；
# 0.4 时分别为 0.652 / 0.832，总体 F1 0.921（0.6 时 0.904）。漏检是本项目最危险的错误类型。）
CONF_THRESHOLD = 0.4
ICON_PATH = res_path("resources/icon.ico")
# 说明：原 BG_HOME / BG_RECOG 指向的 resources/bg_home.png、bg_recog.png 在项目中并不存在，
# 且全文件从未引用，已移除。若要做首页背景，把图片放进 resources/ 后再在 _build 里使用。
MODEL_INFO = {
    "FLOPs": "8.1 GFLOPs",
    "Params": "3.0 M",
    "Input Size": "640x640",
    "训练数据集": "NAU水稻病虫害数据集",
    "模型版本": "YOLOv8n"
}
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
# 检测框标注的字号/线宽已改为随图片尺寸自适应（见 annotate_image），不再用固定字号
# ===========================================================================

# ── 治理方案数据库（农户友好版） ──────────────────────────────
# ══════════════════════════════════════════════════════════════
# 安全间隔期（水稻）参考值
# --------------------------------------------------------------
# 数据来源：农业农村部农药登记标签汇编（moa.gov.cn 公告附件）中「水稻」用途标签，
# 以及地方政府植保站的《稻飞虱主要防治药剂用药指导》。每条都带登记证号或文件名，可追溯。
#
# ⚠ 重要：安全间隔期是"按登记证"的——同一有效成分、不同厂家的产品可能不同。
# 这里给的是常见登记值，只作参考；实际用药请以你手上那份产品标签为准（界面会显示这句话）。
# 查不到的项目**故意留空**，不猜数字：界面会显示「待补，以标签为准」。
# ══════════════════════════════════════════════════════════════
DRUG_SAFE_DAYS = [
    # (名称关键字, 安全间隔期天数, 来源)  —— 按关键字长度降序匹配，更具体的优先
    ("氯虫·噻虫嗪", 21, "农药登记证 PD20110596（40% 水分散粒剂，水稻）"),
    ("氯虫苯甲酰胺", 21, "农药登记证 PD20220103（水稻上每季最多 1 次，安全间隔期 21 天）"),
    ("甲氨基阿维菌素", 21, "农药登记证 PD20110861（水稻上每季最多 2 次，安全间隔期 21 天）"),
    ("阿维菌素", 21, "农药登记证 PD20130532（水稻每生长周期最多 2 次，安全间隔期 21 天）"),
    ("噻虫嗪", 21, "农药登记证 PD20160518（水稻稻飞虱，安全间隔期 21 天）"),
    ("烯啶虫胺", 14, "《稻飞虱主要防治药剂用药指导》（地方政府植保站，安全间隔期 14 天）"),
    ("吡虫啉", 14, "《稻飞虱主要防治药剂用药指导》（地方政府植保站，安全间隔期 14 天）"),
    ("春雷霉素", 21, "农药登记证 PD20081484（水稻上每季最多 3 次，安全间隔期 21 天）"),
    ("苯醚甲环唑", 25, "农药登记证 PD20110564 / PD20140932（苯甲·丙环唑，水稻安全间隔期 25 天）"),
    ("肟菌·戊唑醇", 28, "农药登记证 PD20201042（水稻每季最多 2 次，安全间隔期 28 天）"),
    ("戊唑醇", 21, "农药登记证 PD20121848（水稻上每季最多 3 次，安全间隔期 21 天）"),
    # 下面这几项存在来源冲突：不同登记证/资料给的值不一样。
    # 取值规则：**取更长的间隔期**（宁可多等几天，也不要农残超标），并在注释里写明两个来源。
    ("稻瘟灵", 35, "《常用农药安全间隔期表》（zaqsap，40% 稻瘟灵 2 次 35 天）；"
                   "另一登记证 PD20211015（氟环·稻瘟灵）为 30 天——取较长的 35 天"),
    ("三环唑", 28, "《常用农药安全间隔期表》（zaqsap，2 次 28 天）；"
                   "河南省《常用农药使用安全间隔期目录》为 21 天——取较长的 28 天"),
    ("吡唑醚菌酯", 28, "《常用农药安全间隔期表》（zaqsap，2 次 28 天）"),
    ("嘧菌酯", 28, "《常用农药安全间隔期表》（zaqsap，1 次 28 天）"),
]
DRUG_SAFE_NOTE = "安全间隔期按登记证各不相同，此处为常见登记值，实际以你手上产品标签为准"


# ══════════════════════════════════════════════════════════════
# 本地防治库增强数据（逐条带农药登记证号，来源可追溯）
# --------------------------------------------------------------
# 来源：农业农村部农药登记标签汇编（185 张标签全文，已留档 .tools/_std/moa.txt），
#       以及地方政府植保站《稻飞虱主要防治药剂用药指导》。
# 每条都写了「登记对象」：有些药剂登记的是稻飞虱，而模型类别是稻叶蝉，
# 两者常混合发生、新烟碱类药剂兼治——但必须如实标注，不能默认等同。
# ══════════════════════════════════════════════════════════════
TREATMENT_CLASS_META = {
    "稻纵卷叶螟": {
        "防治适期": "卵孵化盛期至 1—2 龄幼虫期（幼虫未卷叶前防效最好）",
        "施药要点": "晴天下午或傍晚施药；兑水均匀喷雾，重点喷上部叶片",
        "抗性管理": "每季限用次数（多为 1—2 次），并与不同作用机制药剂轮换",
        "环境注意": "鱼虾蟹套养稻田禁用，田水不得直排；蚕室桑园附近、赤眼蜂放飞区禁用",
        "依据": "农药登记标签 PD20220103 / PD20210525 / PD20110861 / PD20130532",
    },
    "稻水象甲": {
        "防治适期": "成虫盛发期与幼虫孵化初期；秧田期重点防成虫，本田期防幼虫",
        "施药要点": "毒土法撒施（保水层 4cm、药后一周不排水）或浸秧处理",
        "抗性管理": "与氯虫苯甲酰胺等不同机制药剂轮换，避免连年单一用药",
        "环境注意": "稻田养鱼虾蟹区域慎用；施药后田水不得直接排入水体",
        "依据": "农药登记证 PD20110596 标签 + 地方植保站防控意见",
    },
    "稻叶蝉": {
        "防治适期": "低龄若虫高峰期（虫口密度达标时）",
        "施药要点": "重点喷稻株中下部；晴天下午施药",
        "抗性管理": "新烟碱类之间易产生交互抗性，务必与其他机制药剂轮换",
        "环境注意": "对蜜蜂高毒——开花植物花期、桑园蚕室附近禁用；稻田养虾蟹区慎用",
        "依据": "农药登记标签 PD20160518 / PD20181108 / PD20181817 + 植保站用药指导",
        "类别说明": "本地库内该类药剂登记对象多为稻飞虱；稻叶蝉与稻飞虱常混合发生，"
                    "新烟碱类药剂可兼治，实际以防标签登记对象为准",
    },
    "白叶枯病": {
        "防治适期": "发病初期（出现中心病株）立即用药；台风暴雨后必须查田补防",
        "施药要点": "重点喷施病株周围；配合排水露田、控氮增钾等农艺措施",
        "抗性管理": "细菌性病害药剂种类少，更要注意轮换与合理混用",
        "环境注意": "噻菌铜等铜制剂不要与碱性农药混用",
        "依据": "农药登记标签 + 地方植保站技术意见（该病登记药剂较少，部分条目待补）",
    },
    "胡麻斑病": {
        "防治适期": "发病初期；该病与缺钾、土壤瘠薄高度相关，先补肥再谈用药",
        "施药要点": "叶面均匀喷雾，连喷 2 次、间隔 7—10 天",
        "抗性管理": "三唑类与甲氧基丙烯酸酯类轮换使用",
        "环境注意": "三唑类药剂对水生生物敏感，注意不要污染水体",
        "依据": "农药登记标签 PD20110564 / PD20140932 / PD20121848（多为纹枯病/稻曲病登记，兼治）",
        "类别说明": "本地库内部分三唑类药剂登记对象为纹枯病/稻曲病，对胡麻斑病为兼治，"
                    "实际以防标签为准",
    },
    "稻瘟病": {
        "防治适期": "叶瘟发病初期；穗颈瘟关键期为破口前 5—7 天、破口期与齐穗期",
        "施药要点": "穗颈瘟只能防不能治，破口期必须打；兑水 40—50 升均匀喷雾",
        "抗性管理": "三环唑、稻瘟灵、甲氧基丙烯酸酯类与生物药剂轮换，每季限用次数",
        "环境注意": "鱼虾蟹套养稻田禁用，田水不得直排；部分药剂对家蚕高毒",
        "依据": "农药登记标签 PD20201042 / PD20081484 / PD20211015 / PD20097891",
    },
}

# 已有药剂补充字段（登记证号 / 施药方法 / 每季次数 / 适期 / 注意）
TREATMENT_ENRICH = {
    "氯虫苯甲酰胺": {
        "reg_no": "PD20220103（甲维·氯虫苯复配登记）", "method": "喷雾", "max_per_season": 1,
        "timing": "移栽后卵孵化盛期至 1—2 龄幼虫期施药 1 次",
        "caution": "鱼虾蟹套养稻田禁用，田水不得直排；开花植物花期、蚕室桑园附近禁用",
        "target": "稻纵卷叶螟", "type": "化学", "source": "农药登记标签 PD20220103",
    },
    "甲维盐（甲氨基阿维菌素苯甲酸盐）": {
        "reg_no": "PD20110861", "method": "喷雾", "max_per_season": 2,
        "timing": "卵孵化高峰期施药，注意喷雾均匀周到",
        "caution": "对鱼类、水蚤、蜜蜂、家蚕、赤眼蜂毒性高；鱼虾蟹套养稻田禁用",
        "target": "稻纵卷叶螟", "type": "化学", "source": "农药登记标签 PD20110861",
    },
    "阿维菌素": {
        "reg_no": "PD20130532", "method": "喷雾", "max_per_season": 2,
        "timing": "卵孵化盛期至低龄幼虫期使用，最好傍晚施药，现配现用",
        "caution": "赤眼蜂等天敌放飞区禁用，鸟类保护区禁用",
        "target": "稻纵卷叶螟", "type": "化学", "source": "农药登记标签 PD20130532",
    },
    "噻虫嗪（阿克泰）": {
        "reg_no": "PD20160518", "method": "喷雾", "max_per_season": 2,
        "timing": "低龄若虫高峰期，晴天下午近傍晚施药",
        "caution": "对蜜蜂高毒，花期与桑园蚕室附近禁用；远离水产养殖区与天敌放飞区",
        "target": "稻飞虱", "type": "化学", "source": "农药登记标签 PD20160518 / PD20132474",
    },
    "吡虫啉": {
        "reg_no": "（政府用药指导，未逐一核对登记证）", "method": "喷雾", "max_per_season": 2,
        "timing": "低龄若虫高峰期；药效与温度正相关，低温效果差",
        "caution": "不可与碱性农药混用；不可污染养蜂养蚕场所与水源",
        "target": "稻飞虱（含白背飞虱）", "type": "化学",
        "source": "地方政府植保站《稻飞虱主要防治药剂用药指导》",
    },
    "烯啶虫胺": {
        "reg_no": "（政府用药指导，未逐一核对登记证）", "method": "喷雾", "max_per_season": 1,
        "timing": "低龄若虫高峰期；速效性好，持效期约 14 天",
        "caution": "对蜜蜂、鱼类、水生物、家蚕有毒；勿与碱性物质混用",
        "target": "稻飞虱（白背飞虱）", "type": "化学",
        "source": "地方政府植保站《稻飞虱主要防治药剂用药指导》",
    },
    "春雷霉素": {
        "reg_no": "PD20081484", "method": "喷雾", "max_per_season": 3,
        "timing": "发病初期开始施药，间隔 7—10 天，连喷 2—3 次",
        "caution": "生物源抗生素类药剂，注意轮换以延缓抗性",
        "target": "稻瘟病（登记）", "type": "生物", "source": "农药登记标签 PD20081484",
    },
    "稻瘟灵（富士一号）": {
        "reg_no": "PD20211015（氟环·稻瘟灵复配）", "method": "喷雾", "max_per_season": 2,
        "timing": "叶瘟发病初期；穗颈瘟在破口前 5—7 天",
        "caution": "注意与三环唑、甲氧基丙烯酸酯类轮换",
        "target": "稻瘟病/稻曲病/纹枯病", "type": "化学", "source": "农药登记标签 PD20211015",
    },
    "肟菌·戊唑醇": {
        "reg_no": "PD20201042", "method": "喷雾", "max_per_season": 2,
        "timing": "稻瘟病在破口前 5—7 天；稻曲病在破口前 5—7 天并间隔 7 天再施 1 次",
        "caution": "鱼虾蟹套养稻田禁用；对家蚕有毒，蚕室桑园附近禁用",
        "target": "稻瘟病/稻曲病", "type": "化学", "source": "农药登记标签 PD20201042",
    },
    "戊唑醇": {
        "reg_no": "PD20121848", "method": "喷雾", "max_per_season": 3,
        "timing": "发病初期开始，间隔 7—10 天",
        "caution": "三唑类，注意与甲氧基丙烯酸酯类轮换",
        "target": "稻曲病", "type": "化学", "source": "农药登记标签 PD20121848",
    },
    "苯醚甲环唑": {
        "reg_no": "PD20110564 / PD20140932（苯甲·丙环唑复配）", "method": "喷雾",
        "max_per_season": 3,
        "timing": "发病前或发病初期，露水干后施药，间隔 7 天左右",
        "caution": "鱼虾蟹套养稻田禁用；赤眼蜂放飞区禁用",
        "target": "纹枯病（对胡麻斑病为兼治）", "type": "化学",
        "source": "农药登记标签 PD20110564 / PD20140932",
    },
}

# 新增药剂（全部来自登记标签，字段齐全）
TREATMENT_ADD = {
    "稻纵卷叶螟": [
        {"name": "甲维·氯虫苯", "spec": "悬浮剂", "dose": "10-14ml/亩", "safe_days": 21,
         "reg_no": "PD20220103", "method": "喷雾", "max_per_season": 1, "type": "化学",
         "timing": "移栽后稻纵卷叶螟卵孵化盛期至 1—2 龄幼虫期，施药 1 次",
         "caution": "鱼或虾蟹套养稻田禁用，田水不得直排；蚕室桑园附近禁用",
         "target": "稻纵卷叶螟", "source": "农药登记标签 PD20220103",
         "safe_source": "农药登记标签 PD20220103（水稻每季 1 次，间隔期 21 天）"},
        {"name": "乙基多杀菌素", "spec": "悬浮剂", "dose": "20-30ml/亩",
         "reg_no": "PD20120240", "method": "喷雾", "type": "生物",
         "timing": "低龄幼虫期施药，注意均匀喷雾",
         "caution": "生物源杀虫剂，对天敌相对友好，仍须避开蚕区与花期",
         "target": "稻纵卷叶螟", "source": "农药登记标签 PD20120240",
         "safe_source": "农药登记标签 PD20120240（间隔期数据多值，待核对标签）"},
    ],
    "稻叶蝉": [
        {"name": "呋虫胺", "spec": "水分散粒剂", "dose": "7.5-10g/亩", "safe_days": 21,
         "reg_no": "PD20181108", "method": "喷雾", "max_per_season": 1, "type": "化学",
         "timing": "水稻飞虱低龄若虫高峰期施药",
         "caution": "开花植物花期禁用；桑园蚕室附近、赤眼蜂放飞区禁用",
         "target": "稻飞虱", "source": "农药登记标签 PD20181108",
         "safe_source": "农药登记标签 PD20181108（水稻间隔期 21 天，每季 1 次）",
         "note": "登记对象为稻飞虱；稻叶蝉常与稻飞虱混合发生，可兼治，以标签为准"},
        {"name": "噻虫胺", "spec": "水分散粒剂", "dose": "5-10g/亩", "safe_days": 14,
         "reg_no": "PD20181817", "method": "喷雾", "max_per_season": 2, "type": "化学",
         "timing": "低龄若虫高峰期",
         "caution": "同新烟碱类，注意对蜜蜂与水生生物的影响",
         "target": "稻飞虱", "source": "农药登记标签 PD20181817",
         "safe_source": "农药登记标签 PD20181817（水稻间隔期 14 天）",
         "note": "登记对象为稻飞虱，与稻叶蝉常混合发生"},
    ],
    "稻瘟病": [
        {"name": "氨基寡糖素", "spec": "水剂", "dose": "200-250ml/亩",
         "reg_no": "PD20097891", "method": "喷雾", "type": "生物",
         "timing": "病害发病前叶面喷雾；水稻稻瘟病连续施药 2 次，间隔 10—14 天",
         "caution": "植物免疫诱抗剂，重在预防，发病后单用效果有限",
         "target": "稻瘟病", "source": "农药登记标签 PD20097891",
         "safe_source": "农药登记标签 PD20097891（未标注安全间隔期，待补）"},
        {"name": "氟环·稻瘟灵", "spec": "乳油", "dose": "80-100ml/亩", "safe_days": 30,
         "reg_no": "PD20211015", "method": "喷雾", "max_per_season": 2, "type": "化学",
         "timing": "叶瘟发病初期；穗颈瘟在破口前 5—7 天",
         "caution": "注意与三环唑等不同机制药剂轮换",
         "target": "稻瘟病/稻曲病/纹枯病", "source": "农药登记标签 PD20211015",
         "safe_source": "农药登记标签 PD20211015（水稻每季 2 次，间隔期 30 天）"},
        {"name": "春雷霉素", "spec": "可湿性粉剂", "dose": "25-40g/亩", "safe_days": 21,
         "reg_no": "PD20081484", "method": "喷雾", "max_per_season": 3, "type": "生物",
         "timing": "发病初期开始，间隔 7—10 天，连喷 2—3 次",
         "caution": "生物源药剂，注意轮换", "target": "稻瘟病",
         "source": "农药登记标签 PD20081484",
         "safe_source": "农药登记标签 PD20081484（水稻每季 3 次，间隔期 21 天）"},
    ],
}


# ══════════════════════════════════════════════════════════════
# 防治指标量化 + 补齐白叶枯病/胡麻斑病（来源逐条标注）
# --------------------------------------------------------------
# 量化指标来源：《2025 年粮食作物重大病虫害防控技术方案》（全国农技推广中心系统发布，
#   farmchina.org.cn），以及陕西省汉中市植保站《当前水稻"两迁害虫"防治技术要点》
#   （其中稻飞虱指标与全国方案一致，互为印证）。
# 药剂来源：农业农村部农药登记标签汇编（第二批 moa2.txt，196 张标签）。
# ══════════════════════════════════════════════════════════════
THRESHOLD_SOURCE_NATIONAL = "《2025 年粮食作物重大病虫害防控技术方案》"
TREATMENT_THRESHOLDS = {
    "稻纵卷叶螟": {
        "量化指标": "分蘖期百丛水稻束叶尖 150 个；孕穗后百丛束叶尖 60 个",
        "施药适期": "卵孵化始盛期至低龄幼虫高峰期施药",
        "指标来源": THRESHOLD_SOURCE_NATIONAL,
        "备注": "分蘖期水稻有补偿能力，可从宽掌握、减少用药",
    },
    "稻叶蝉": {
        "量化指标": "稻飞虱：分蘖至孕穗期百丛虫量 1000 头、穗期 1500 头（本地库该类药剂登记对象即稻飞虱）",
        "施药适期": "低龄若虫高峰期",
        "指标来源": THRESHOLD_SOURCE_NATIONAL + "（汉中市植保站指标一致，互为印证）",
        "备注": "稻叶蝉本身没有国家级统一指标；与稻飞虱混发时参照稻飞虱指标执行",
    },
    "稻水象甲": {
        "量化指标": "暂无国家级统一指标",
        "施药适期": "成虫盛发期与幼虫孵化初期",
        "指标来源": "待补（建议以当地植保站病虫情报为准）",
        "备注": "该虫为区域性检疫性害虫，指标多由地方发布，库内先不写推测值",
    },
    "白叶枯病": {
        "量化指标": "田间出现发病中心（中心病株）时立即施药；重发区在台风、暴雨前后施药预防",
        "施药适期": "发病初期；风雨后必须查田补防",
        "指标来源": THRESHOLD_SOURCE_NATIONAL,
        "备注": "细菌性病害以预防与栽培管理为主，药剂为辅",
    },
    "胡麻斑病": {
        "量化指标": "暂无国家级统一指标（该病未列入全国防控技术方案）",
        "施药适期": "发病初期；先补钾改土，再考虑用药",
        "指标来源": "待补（建议按当地植保站预报与品种抗性掌握）",
        "备注": "该病与缺钾、土壤瘠薄高度相关，农艺措施优先于用药",
    },
    "稻瘟病": {
        "量化指标": "叶瘟：田间初见病斑时施药；穗瘟：破口前 3—5 天预防施药，齐穗期第二次",
        "施药适期": "叶瘟初见病斑；穗颈瘟破口前 3—5 天（关键）",
        "指标来源": THRESHOLD_SOURCE_NATIONAL,
        "备注": "保护性药剂可选三环唑、丙硫唑；治疗性可选吡唑醚菌酯",
    },
}

# 补齐短板：白叶枯病与稻瘟病的登记药剂（全部来自登记标签）
TREATMENT_ADD_MORE = {
    "白叶枯病": [
        {"name": "噻唑锌", "spec": "20% 悬浮剂", "dose": "100-150ml/亩",
         "reg_no": "PD20096932", "method": "喷雾", "type": "化学",
         "timing": "发病初期（出现发病中心）施药；重发区台风暴雨前后预防",
         "caution": "登记对象为细菌性条斑病，属同类细菌性病害用药；"
                    "官方技术方案亦推荐噻唑锌用于水稻细菌性病害",
         "target": "细菌性条斑病（同类细菌性病害）", "source": "农药登记标签 PD20096932",
         "safe_source": "农药登记标签 PD20096932（水稻未标注间隔期，待补）"},
        {"name": "噻霉酮", "spec": "微乳剂", "dose": "60-100ml/亩",
         "reg_no": "PD20160650", "method": "喷雾", "type": "化学",
         "timing": "发病初期施药",
         "caution": "登记对象为细菌性条斑病；官方方案推荐噻霉酮用于细菌性病害",
         "target": "细菌性条斑病（同类细菌性病害）", "source": "农药登记标签 PD20160650",
         "safe_source": "农药登记标签 PD20160650（水稻未标注间隔期，待补）"},
        {"name": "枯草芽孢杆菌", "spec": "可湿性粉剂", "dose": "60g/亩",
         "reg_no": "PD20140340", "method": "喷雾", "type": "生物",
         "timing": "发病前或发病初期施药，重在预防",
         "caution": "生物制剂，避免与杀菌性强的化学药剂同时混用",
         "target": "白叶枯病（直接登记）", "source": "农药登记标签 PD20140340",
         "safe_source": "农药登记标签 PD20140340（生物制剂，水稻未标注间隔期）"},
    ],
    "稻瘟病": [
        {"name": "枯草芽孢杆菌", "spec": "可分散油悬浮剂", "dose": "50-60ml/亩",
         "reg_no": "PD20181893", "method": "喷雾", "type": "生物",
         "timing": "发病前或初期施药，重在预防",
         "caution": "生物制剂，建议与化学药剂轮换，减缓抗性",
         "target": "稻瘟病（直接登记）", "source": "农药登记标签 PD20181893",
         "safe_source": "农药登记标签 PD20181893（生物制剂，未标注间隔期）"},
        {"name": "补骨脂种子提取物", "spec": "微乳剂", "dose": "45-60ml/亩",
         "reg_no": "PD20190020", "method": "喷雾", "type": "生物",
         "timing": "发病初期施药",
         "caution": "植物源免疫诱抗类药剂，重在预防与早期干预",
         "target": "稻瘟病（直接登记）", "source": "农药登记标签 PD20190020",
         "safe_source": "农药登记标签 PD20190020（植物源，水稻未标注间隔期）"},
    ],
}


def _apply_treatment_enrichment():
    """把增强字段合并进本地防治库，并把新药剂追加到对应类别（按名称去重）"""
    # 库内名字常带品牌后缀（如「氯虫苯甲酰胺（康宽）」），所以先精确匹配、再按子串匹配，
    # 并且用「关键字长的优先」避免「戊唑醇」误配到「肟菌·戊唑醇」。
    keys = sorted(TREATMENT_ENRICH, key=len, reverse=True)
    for tr in TREATMENT_DB.values():
        for d in tr.get("drugs", []):
            name = d.get("name", "") or ""
            extra = TREATMENT_ENRICH.get(name)
            if extra is None:
                for k in keys:
                    if k in name:
                        extra = TREATMENT_ENRICH[k]
                        break
            if extra:
                for k, v in extra.items():
                    d.setdefault(k, v)
    # 注意顺序：类别要点与量化指标都往同一份 meta 里合并，
    # 不能用 setdefault("meta", 常量) —— 那会在先建了空 meta 后把要点整块丢掉（踩过）。
    for cls, meta in TREATMENT_CLASS_META.items():
        if cls in TREATMENT_DB:
            TREATMENT_DB[cls].setdefault("meta", {}).update(meta)
    for cls, th in TREATMENT_THRESHOLDS.items():
        if cls in TREATMENT_DB:
            TREATMENT_DB[cls].setdefault("meta", {}).update(th)
    added = 0
    merged = dict(TREATMENT_ADD)
    for cls, drugs in TREATMENT_ADD_MORE.items():
        merged[cls] = list(merged.get(cls, [])) + list(drugs)
    for cls, drugs in merged.items():
        tr = TREATMENT_DB.get(cls)
        if not tr:
            continue
        have = {d.get("name") for d in tr.setdefault("drugs", [])}
        for d in drugs:
            if d.get("name") not in have:
                tr["drugs"].append(dict(d))
                have.add(d.get("name"))
                added += 1
    log.info("本地防治库增强：新增药剂 %d 条", added)
    return added


def _apply_drug_safe_days():
    """把上面那张表填进本地防治库（已有值不覆盖，例如库内自带的 PD20110596）"""
    for key, days, src in sorted(DRUG_SAFE_DAYS, key=lambda x: -len(x[0])):
        for tr in TREATMENT_DB.values():
            for d in tr.get("drugs", []):
                if key in (d.get("name") or "") and not d.get("safe_days"):
                    d["safe_days"] = days
                    d["safe_source"] = src


def drug_safety_gaps():
    """哪些药剂还缺安全间隔期（缺口要看得见，而不是悄悄按 0 处理）"""
    gaps = []
    for cls, tr in TREATMENT_DB.items():
        for d in tr.get("drugs", []):
            if not d.get("safe_days"):
                gaps.append({"类别": cls, "药剂": d.get("name", ""),
                             "规格": d.get("spec", "")})
    return gaps

TREATMENT_DB = {
    "稻纵卷叶螟": {
        "symptom": "幼虫吐丝将稻叶纵卷成苞，在苞内取食叶肉，形成白叶。严重时田间一片枯白，远观如「火烧」状。",
        "pest_img": "resources/treatment/稻纵卷叶螟/pest.jpg",
        "agri": [
            "合理施肥，避免偏施氮肥造成叶片过嫩",
            "适时晒田，降低田间湿度",
            "冬季清除田边杂草，减少越冬虫源",
        ],
        "bio": [
            "释放赤眼蜂（Trichogramma）寄生卵块",
            "使用Bt制剂（苏云金芽孢杆菌）喷洒",
        ],
        "threshold": "分蘖期百丛有虫苞30-50个，穗期百丛有虫苞20-30个",
        "drugs": [
            {"name": "氯虫苯甲酰胺（康宽）", "spec": "20% SC", "dose": "10-15ml/亩", "img": "resources/treatment/稻纵卷叶螟/drug_1.jpg"},
            {"name": "甲维盐（甲氨基阿维菌素苯甲酸盐）", "spec": "5% WG", "dose": "10-15g/亩", "img": "resources/treatment/稻纵卷叶螟/drug_2.jpg"},
            {"name": "阿维菌素", "spec": "1.8% EC", "dose": "50-60ml/亩", "img": "resources/treatment/稻纵卷叶螟/drug_3.jpg"},
        ],
    },
    "稻水象甲": {
        "symptom": "成虫取食稻叶形成白色纵条斑，幼虫蛀食稻根，导致植株黄弱、分蘖减少，严重时整株枯死。",
        "pest_img": "resources/treatment/稻水象甲/pest.jpg",
        "agri": [
            "严格检疫，防止随秧苗、稻草传播",
            "冬季翻耕灭茬，减少越冬虫源",
            "推迟移栽期，避开成虫迁飞高峰",
            "加强水层管理，适时晒田",
        ],
        "bio": [
            "稻田养鸭，利用天敌如步行虫、蜘蛛",
        ],
        "threshold": "秧田每平方米有成虫5头以上，本田百丛有成虫30头以上",
        "drugs": [
            # 2026-10-13 更正：原「氟虫腈（锐劲特）」属国家停止使用品种（三部门公告第1157号，
            # 2009-10-01 起除卫生用/玉米等旱田种子包衣外停止销售和使用），已替换为登记在册的替代药。
            {"name": "氯虫·噻虫嗪（40% 水分散粒剂）", "spec": "40% WG", "dose": "6-8克/亩",
             "img": "resources/treatment/稻水象甲/drug_1.jpg",
             "source": "农药登记证 PD20110596", "safe_days": 21, "max_per_season": 2},
            {"name": "氯虫苯甲酰胺（康宽）", "spec": "20% SC", "dose": "10ml/亩", "img": "resources/treatment/稻水象甲/drug_2.jpg"},
            {"name": "高效氯氟氰菊酯", "spec": "2.5% EC", "dose": "30-40ml/亩", "img": "resources/treatment/稻水象甲/drug_3.jpg"},
        ],
    },
    "稻叶蝉": {
        "symptom": "成、若虫刺吸稻株汁液，叶片出现白色斑点，严重时整叶枯黄。可传播水稻矮缩病、黄矮病等病毒病。",
        "pest_img": "resources/treatment/稻叶蝉/pest.jpg",
        "agri": [
            "选用抗性品种，清除田边杂草",
            "合理布局，避免混栽「桥梁田」",
            "灯光诱杀（黑光灯/频振式杀虫灯），每30-50亩一盏",
        ],
        "bio": [
            "保护蜘蛛、瓢虫等自然天敌",
        ],
        "threshold": "秧田每平方米有虫50头以上，本田百丛有虫300-500头",
        "drugs": [
            {"name": "吡虫啉", "spec": "10% WP", "dose": "20-30g/亩", "img": "resources/treatment/稻叶蝉/drug_1.jpg"},
            {"name": "噻虫嗪（阿克泰）", "spec": "25% WG", "dose": "6-8g/亩", "img": "resources/treatment/稻叶蝉/drug_2.jpg"},
            {"name": "烯啶虫胺", "spec": "10% AS", "dose": "30-50ml/亩", "img": "resources/treatment/稻叶蝉/drug_3.jpg"},
        ],
    },
    "白叶枯病": {
        "symptom": "叶片边缘出现水渍状黄色条纹，后沿叶缘扩展为灰白色枯死。湿度大时有黄白色菌脓，干燥后形成鱼籽状颗粒。",
        "pest_img": "resources/treatment/白叶枯病/pest.jpg",
        "agri": [
            "选用抗病品种，种子用强氯精500倍液浸种12小时消毒",
            "合理施肥，增施硅钾肥，避免偏施氮肥",
            "浅水灌溉，避免深水淹灌",
            "台风暴雨后立即排水，降低田间湿度",
        ],
        "bio": [],
        "threshold": "台风暴雨前需提前预防，发病初期立即施药",
        "drugs": [
            {"name": "噻菌铜（龙克菌）", "spec": "20% SC", "dose": "100-125ml/亩", "img": "resources/treatment/白叶枯病/drug_1.jpg"},
            {"name": "中生菌素", "spec": "3% WP", "dose": "40-60g/亩", "img": "resources/treatment/白叶枯病/drug_2.jpg"},
            {"name": "春雷霉素", "spec": "2% AS", "dose": "80-100ml/亩", "img": "resources/treatment/白叶枯病/drug_3.jpg"},
        ],
    },
    "胡麻斑病": {
        "symptom": "叶片出现椭圆形褐色小斑点，形状如芝麻粒（胡麻粒），周围有黄色晕圈。缺钾田、砂质田、酸性田发生较重。",
        "pest_img": "resources/treatment/胡麻斑病/pest.jpg",
        "agri": [
            "核心措施：补施钾肥——该病本质上是缺素症的表现",
            "增施腐熟有机肥，改良酸性土壤（施石灰）",
            "合理灌溉，避免干旱胁迫",
        ],
        "bio": [],
        "threshold": "分蘖期病叶率10%以上，抽穗期剑叶病叶率5%以上",
        "drugs": [
            {"name": "苯醚甲环唑", "spec": "10% WG", "dose": "40-60g/亩", "img": "resources/treatment/胡麻斑病/drug_1.jpg"},
            {"name": "戊唑醇", "spec": "430g/L SC", "dose": "15-20ml/亩", "img": "resources/treatment/胡麻斑病/drug_2.jpg"},
            {"name": "嘧菌酯", "spec": "250g/L SC", "dose": "20-30ml/亩", "img": "resources/treatment/胡麻斑病/drug_3.jpg"},
        ],
    },
    "稻瘟病": {
        "symptom": "分为苗瘟、叶瘟、节瘟、穗颈瘟。叶瘟呈梭形暗绿色病斑，穗颈瘟造成白穗、绝收。穗颈瘟危害最大，破口期和齐穗期各防一次至关重要。",
        "pest_img": "resources/treatment/稻瘟病/pest.jpg",
        "agri": [
            "选用抗病品种（最经济有效的措施）",
            "合理施肥：氮磷钾配合，切忌偏施氮肥",
            "适时晒田，湿润灌溉",
            "处理带病稻草，减少菌源",
        ],
        "bio": [],
        "threshold": "破口初期（抽穗5%时）和齐穗期各施药一次，施药后4小时内遇雨需补喷",
        "drugs": [
            {"name": "三环唑", "spec": "75% WP", "dose": "20-30g/亩", "img": "resources/treatment/稻瘟病/drug_1.jpg"},
            {"name": "吡唑醚菌酯", "spec": "25% EC", "dose": "30-40ml/亩", "img": "resources/treatment/稻瘟病/drug_2.jpg"},
            {"name": "稻瘟灵（富士一号）", "spec": "40% EC", "dose": "80-100ml/亩", "img": "resources/treatment/稻瘟病/drug_3.jpg"},
            {"name": "肟菌·戊唑醇", "spec": "75% WG", "dose": "15-20g/亩", "img": "resources/treatment/稻瘟病/drug_4.jpg"},
        ],
    },
}


# ══════════════════════════════════════════════════════════════
# 用户自维护层：不动代码，把自己的农药标签补进本地库
# --------------------------------------------------------------
# 设计取舍：
# · 用户数据存 APP_DIR/treatment/user_db.json，独立于代码——升级程序不会覆盖；
# · **内置条目只读**（不可改不可删），用户条目可增删改；同名时用户数据优先，
#   因为农技员最清楚自己手上那份标签；
# · 导出/导入走同一个 JSON，方便农技站之间交换（一份当地用药清单）。
# ══════════════════════════════════════════════════════════════
USER_DB_PATH = APP_DIR / "treatment" / "user_db.json"
_BUILTIN_DRUGS = {}           # 内置药剂快照（首次合并前深拷贝，之后每次重建都用它）
USER_DB_VERSION = 1
USER_DRUG_FIELDS = ["name", "spec", "dose", "safe_days", "reg_no", "method",
                    "max_per_season", "timing", "caution", "target", "type",
                    "source", "note"]


def load_user_db():
    """读取用户补充数据；文件不存在或损坏时返回空库（不影响内置库）"""
    try:
        if USER_DB_PATH.exists():
            data = json.loads(USER_DB_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("drugs", {})
                data.setdefault("thresholds", {})
                return data
    except Exception as e:
        log.warning("读取用户防治库失败（已忽略）：%s", e)
    return {"version": USER_DB_VERSION, "drugs": {}, "thresholds": {}}


def save_user_db(data):
    try:
        USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = dict(data or {})
        data["version"] = USER_DB_VERSION
        data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        USER_DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        return True
    except OSError as e:
        log.warning("保存用户防治库失败：%s", e)
        return False


def _clean_drug(d):
    """只保留白名单字段，避免用户文件里塞进奇怪的键"""
    out = {}
    for k in USER_DRUG_FIELDS:
        v = d.get(k)
        if v in (None, ""):
            continue
        if k == "safe_days":
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        if k == "max_per_season":
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        out[k] = v
    return out


def user_add_drug(cls_name, drug):
    """新增/覆盖一条用户药剂（同类别同名视为修改）"""
    drug = _clean_drug(drug or {})
    if not drug.get("name") or cls_name not in TREATMENT_DB:
        return None, "名称不能为空，且类别必须是六类之一"
    data = load_user_db()
    items = data["drugs"].setdefault(cls_name, [])
    for i, d in enumerate(items):
        if d.get("name") == drug["name"]:
            items[i] = drug
            save_user_db(data)
            apply_user_db()
            log_event("user_db_drug", action="update", cls=cls_name, name=drug["name"])
            return drug, "已更新"
    items.append(drug)
    save_user_db(data)
    apply_user_db()
    log_event("user_db_drug", action="add", cls=cls_name, name=drug["name"])
    return drug, "已新增"


def user_delete_drug(cls_name, name):
    """删除一条用户药剂；内置条目不允许删（返回 False 并说明原因）"""
    data = load_user_db()
    items = data["drugs"].get(cls_name) or []
    keep = [d for d in items if d.get("name") != name]
    if len(keep) == len(items):
        builtin = any(d.get("name") == name for d in TREATMENT_DB.get(cls_name, {}).get("drugs", []))
        return False, ("「%s」是内置条目，不能删除（可在同名用户条目里覆盖它的字段）"
                       % name) if builtin else "没找到这条用户药剂"
    data["drugs"][cls_name] = keep
    save_user_db(data)
    apply_user_db()
    log_event("user_db_drug", action="delete", cls=cls_name, name=name)
    return True, "已删除"


def user_set_threshold(cls_name, fields):
    """修改某个类别的量化指标（覆盖内置）"""
    if cls_name not in TREATMENT_DB:
        return False, "类别不存在"
    data = load_user_db()
    cur = data["thresholds"].setdefault(cls_name, {})
    for k in ("量化指标", "施药适期", "指标来源", "备注"):
        v = (fields or {}).get(k)
        if v not in (None, ""):
            cur[k] = v
    save_user_db(data)
    apply_user_db()
    log_event("user_db_threshold", cls=cls_name)
    return True, "已保存"


def apply_user_db():
    """把用户数据合并进内置库（幂等：新增 / 覆盖 / 撤销都能正确反映）

    踩过的坑：只做「新增+覆盖」的话，用户在界面上删掉一条后，内存里的库还留着它
    （自检当场报出 32 条而不是 31 条）。所以这里先把上一次合并的结果还原，再重新合并。
    """
    data = load_user_db()
    wanted = {}
    for cls, drugs in (data.get("drugs") or {}).items():
        wanted[cls] = {d.get("name") for d in drugs if d.get("name")}
    added = updated = 0
    # 做法：每次都用「首次合并前的内置快照」重建整个药剂列表，再把用户数据叠上去。
    # 这样增/改/删天然幂等，不依赖 origin 之类的记账（之前正是记账出错导致删不掉）。
    if not _BUILTIN_DRUGS:
        for _cls, _tr in TREATMENT_DB.items():
            _BUILTIN_DRUGS[_cls] = [dict(x) for x in _tr.get("drugs", [])]
    for cls, tr in TREATMENT_DB.items():
        base = [dict(x) for x in _BUILTIN_DRUGS.get(cls, [])]
        index = {d.get("name"): i for i, d in enumerate(base)}
        for d in (data.get("drugs", {}).get(cls) or []):
            d = _clean_drug(d)
            if not d.get("name"):
                continue
            if d["name"] in index:
                base[index[d["name"]]].update(d)
                base[index[d["name"]]]["origin"] = "user"
                updated += 1
            else:
                fresh = dict(d)
                fresh["origin"] = "user"
                base.append(fresh)
                index[d["name"]] = len(base) - 1
                added += 1
        tr["drugs"] = base
    for cls, th in (data.get("thresholds") or {}).items():
        tr = TREATMENT_DB.get(cls)
        if tr and th:
            tr.setdefault("meta", {}).update(th)
    return added, updated


def user_db_stats():
    data = load_user_db()
    drugs = sum(len(v or []) for v in (data.get("drugs") or {}).values())
    ths = len([k for k, v in (data.get("thresholds") or {}).items() if v])
    return {"用户药剂": drugs, "用户指标": ths,
            "文件": str(USER_DB_PATH), "更新时间": data.get("updated", "")}


def export_user_db(path):
    """导出成一份可分享的 JSON（农技站之间交换当地用药清单）"""
    try:
        Path(path).write_text(json.dumps(load_user_db(), ensure_ascii=False, indent=1),
                              encoding="utf-8")
        return True, str(path)
    except OSError as e:
        return False, str(e)


def import_user_db(path, merge=True):
    """导入用户库；merge=True 时按类别合并（同名覆盖），否则整体替换"""
    try:
        incoming = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(incoming, dict):
            return False, "文件格式不对"
    except Exception as e:
        return False, "读取失败：%s" % e
    if merge:
        data = load_user_db()
        for cls, drugs in (incoming.get("drugs") or {}).items():
            bucket = data["drugs"].setdefault(cls, [])
            for d in drugs:
                hit = next((x for x in bucket if x.get("name") == d.get("name")), None)
                if hit is None:
                    bucket.append(_clean_drug(d))
                else:
                    hit.update(_clean_drug(d))
        for cls, th in (incoming.get("thresholds") or {}).items():
            data["thresholds"].setdefault(cls, {}).update(th or {})
    else:
        data = {"version": USER_DB_VERSION, "drugs": incoming.get("drugs") or {},
                "thresholds": incoming.get("thresholds") or {}}
    save_user_db(data)
    added, updated = apply_user_db()
    log_event("user_db_import", path=str(path), added=added, updated=updated)
    return True, "已导入：新增 %d 条、更新 %d 条" % (added, updated)


# 本地库增强与安全间隔期填充（都必须在 TREATMENT_DB 定义之后执行）
_apply_treatment_enrichment()
_apply_drug_safe_days()
apply_user_db()          # 用户补充数据最后合并（同名优先于内置）

# ── 农药合规名单（离线 + 带来源，供本地校验与提示词约束） ──────────────
# 三个用途：① 启动时自检本地防治库；② 约束大模型不许推荐禁用农药；③ 扫描 AI 回答中的禁用农药名。
# 数据全部来自官方公告原文，改之前先核对来源（项目铁律：数字要能追到源头）。
#   banned  —— 国家禁用农药品种（全境、所有用途）
#   stopped —— 已停止销售/使用的品种（限特定用途除外）
PESTICIDE_CONTROL = {
    "banned": {
        "source": "农业农村部公告第1039号《国家禁用农药品种清单》（2026-07-13 发布并生效）",
        "url": "https://www.moa.gov.cn:10443/govpublic/ZZYGLS/202607/t20260715_6485813.htm",
        # 类别名（汞制剂 / 砷类 / 铅类）保留在名单里备查，但不参与文本匹配，避免误报
        "names": [
            "六六六", "滴滴涕", "毒杀芬", "二溴氯丙烷", "杀虫脒", "二溴乙烷", "除草醚",
            "艾氏剂", "狄氏剂", "敌枯双", "氟乙酰胺", "甘氟", "毒鼠强", "氟乙酸钠", "毒鼠硅",
            "内吸磷", "硫环磷", "氯唑磷", "甲胺磷", "对硫磷", "甲基对硫磷", "久效磷", "磷胺",
            "氯丹", "灭蚁灵", "六氯苯", "苯线磷", "地虫硫磷", "甲基硫环磷", "磷化钙",
            "磷化镁", "磷化锌", "硫线磷", "蝇毒磷", "治螟磷", "特丁硫磷", "氯磺隆",
            "胺苯磺隆", "甲磺隆", "福美胂", "福美甲胂", "杀扑磷", "2,4-滴丁酯", "百草枯",
            "三氯杀螨醇", "硫丹", "林丹", "氟虫胺", "五氯酚钠", "甲拌磷", "甲基异柳磷",
            "水胺硫磷", "灭线磷", "氧乐果", "克百威", "灭多威", "涕灭威",
        ],
    },
    "stopped": [
        {
            "name": "氟虫腈",
            "alias": ["锐劲特", "fipronil"],
            "scope": "除卫生用、玉米等部分旱田种子包衣剂和专供出口产品外，在我国境内停止销售和使用",
            "source": "农业部、工业和信息化部、环境保护部公告第1157号（2009-10-01 起）",
            "url": "https://www.moa.gov.cn/gk/tzgg_1/gg/200902/t20090227_1226994.htm",
        },
    ],
    # 含禁用名单子串、但本身是另一种农药的名称：匹配前先屏蔽，避免误报
    "exceptions": ["丁硫克百威"],
}


def _control_override_path():
    """可选的外部名单（便于日后更新，不必改代码）：APP_DIR/resources/knowledge/pesticide_control.json"""
    return APP_DIR / "resources" / "knowledge" / "pesticide_control.json"


def _control_lists():
    """返回 (禁用名列表, 停用记录列表)；外部 JSON 存在时与之合并"""
    banned = list(PESTICIDE_CONTROL["banned"]["names"])
    stopped = [dict(s) for s in PESTICIDE_CONTROL["stopped"]]
    p = _control_override_path()
    if p.exists():
        try:
            ext = json.loads(p.read_text(encoding="utf-8"))
            banned = list(dict.fromkeys(banned + list(ext.get("banned", []))))
            have = {s.get("name") for s in stopped}
            stopped += [s for s in ext.get("stopped", []) if s.get("name") not in have]
            log.info("已合并外部农药名单 %s（禁用 %d 项）", p, len(banned))
        except Exception as e:
            log.warning("外部农药名单读取失败，改用内置名单: %s", e)
    return banned, stopped


def _control_terms():
    """参与文本匹配的词 → 命中后展示的规范名"""
    banned, stopped = _control_lists()
    terms = {n: n for n in banned}
    for s in stopped:
        for t in [s.get("name", "")] + list(s.get("alias", [])):
            if t:
                terms[t] = s.get("name", t)
    return terms


def pesticide_hits(text):
    """扫描文本中出现的禁用/停用农药名，返回规范名列表（去重、保持出现顺序）"""
    t = text or ""
    for exc in PESTICIDE_CONTROL["exceptions"]:
        t = t.replace(exc, "＿")          # 屏蔽“李鬼”名称，防止子串误判
    hits = []
    for term, canon in _control_terms().items():
        if term in t and canon not in hits:
            hits.append(canon)
    return hits


def pesticide_status(name):
    """单个农药名的合规状态：('banned'|'stopped'|None, 记录)"""
    for s in _control_lists()[1]:
        if name == s.get("name") or name in (s.get("alias") or []):
            return "stopped", s
    if name in _control_lists()[0]:
        return "banned", {"name": name}
    for h in pesticide_hits(name):
        return "banned", {"name": h}
    return None, None


def audit_treatment_db():
    """启动自检：本地防治库里不许出现禁用/停用农药。返回 [(类别, 药剂名, 命中名单)]"""
    bad = []
    for cls_name, tr in TREATMENT_DB.items():
        for drug in tr.get("drugs", []):
            hits = pesticide_hits("%s %s" % (drug.get("name", ""), drug.get("spec", "")))
            if hits:
                bad.append((cls_name, drug.get("name", ""), hits))
    return bad


def compliance_prompt_clause():
    """注入系统提示词的合规约束：明确禁止推荐的品种，并要求只用本地库里的药剂"""
    banned, stopped = _control_lists()
    stop_txt = "；".join("%s（%s）" % (s.get("name"), s.get("scope", "")) for s in stopped)
    return (
        "【农药合规红线】严禁推荐下列国家禁用农药：%s。"
        "严禁推荐已停止使用的品种：%s。"
        "推荐药剂时只能使用「本地防治库」中列出的药剂，或说明“请按当地植保站指导选用登记药剂”；"
        "不得自行编造药剂名称、含量、亩用量和安全间隔期。"
        % ("、".join(banned), stop_txt)
    )


def banned_warning_html(names):
    """命中禁用农药时给用户看的红字提示"""
    return ("<div style='color:#C0392B; font-size:13px; margin:6px 0;'>"
            "⚠ 以下药剂为国家禁用/停止使用品种，请勿使用：%s</div>" % "、".join(names))


# ══════════════════════════════════════════════════════════════
# Agent 工具集 · 结构化输出校验 · 审计留痕
# --------------------------------------------------------------
# 设计原则（对应评审的「技术规范性 / 人机协同」）：
#   1. 只把「只读」工具交给模型：查本地防治库、查合规名单、查报告历史、
#      查反馈统计、按新阈值重检、查天气。写操作（导报告/落反馈）不开放给模型，
#      必须由用户在界面上点——这是责任边界，也是可追溯的前提。
#   2. 每次工具调用都写审计日志（logs/llm_log.jsonl）。
#   3. 大模型给的「诊断卡」逐项对本地库校验后才展示，未收录/禁用都标出来。
AGENT_MAX_TOOL_ROUNDS = 3

# ══════════════════════════════════════════════════════════════
# 地块地图：让 Agent 会看地图、会排优先级
# --------------------------------------------------------------
# 地图数据来自 rice_geo 的本地档案库（geo/records.db，含 WGS-84 与 GCJ-02 两套坐标）。
# 这里只做「读 + 算」：落点、导入、导出这类写操作仍在界面上由人确认，
# 不交给模型——和现有 11 个工具保持同一条边界。
# ══════════════════════════════════════════════════════════════
_GEO_STATE = {"tried": False, "store": None}
MAP_HALF_LIFE_DAYS = 14        # 风险权重的时间衰减半衰期（天）
MAP_CELL_M = 500               # 热点分箱边长（米）


def geo_store():
    """惰性接入地块地图模块；模块缺失时不拖累主程序，工具会给出明确提示"""
    if not _GEO_STATE["tried"]:
        _GEO_STATE["tried"] = True
        try:
            import rice_geo
            _GEO_STATE["store"] = rice_geo.GeoStore()
            log.info("地块地图已接入：%s", _GEO_STATE["store"].root)
        except Exception as e:
            log.warning("地块地图模块不可用：%s", e)
            _GEO_STATE["store"] = None
    return _GEO_STATE["store"]


def haversine_m(lon1, lat1, lon2, lat2):
    """两点球面距离（米）。用 WGS-84 原始坐标算，不要拿 GCJ-02 算距离"""
    try:
        r = 6371000.0
        p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
        dp = p2 - p1
        dl = math.radians(float(lon2) - float(lon1))
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(min(1.0, a)))
    except (TypeError, ValueError):
        return 0.0


def _geo_records(days=None):
    """已落点的记录（可按天数过滤）；地图模块不可用时返回 None"""
    st = geo_store()
    if st is None:
        return None
    recs = [r for r in st.all_records()
            if r.get("lon") is not None and r.get("lat") is not None]
    if days:
        cut = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%d")
        recs = [r for r in recs if (r.get("shot_at") or "")[:10] >= cut]
    return recs


def _parse_shot(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime((s or "")[:len(fmt) + 2].strip(), fmt)
        except ValueError:
            continue
    return None


def map_record_weight(rec, now=None, half_life=MAP_HALF_LIFE_DAYS):
    """单条记录的风险权重 = 时间衰减 × 置信度。

    时间衰减用半衰期：14 天前的记录权重减半。写成这样是为了可解释——
    「为什么这块地排在前面」能一条条算给用户看，而不是一个黑箱分数。
    """
    now = now or datetime.now()
    t = _parse_shot(rec.get("shot_at"))
    if t is None:
        return 0.0
    age = max(0.0, (now - t).total_seconds() / 86400.0)
    decay = 0.5 ** (age / float(half_life or MAP_HALF_LIFE_DAYS))
    conf = float(rec.get("conf") or 0.0)
    return decay * (0.5 + min(max(conf, 0.0), 1.0) * 0.5)


def plot_risk_rows(days=None, now=None, half_life=MAP_HALF_LIFE_DAYS):
    """按地块聚合风险，降序返回。没有地块名的记录归入「未命名地块」。"""
    recs = _geo_records(days)
    if recs is None:
        return None
    groups = {}
    for r in recs:
        key = (r.get("plot") or "").strip() or "未命名地块"
        groups.setdefault(key, []).append(r)
    rows = []
    for plot, rs in groups.items():
        weight = sum(map_record_weight(r, now, half_life) for r in rs)
        kinds = Counter(r.get("cls") or "未检出" for r in rs)
        repeat = max(kinds.values()) if kinds else 1
        # 同一个问题反复出现 → 说明上一轮没控制住，给一点重复加权
        score = weight * (1 + 0.3 * max(0, repeat - 1))
        last_rec = max(rs, key=lambda r: r.get("shot_at") or "")
        rows.append({"plot": plot, "score": score, "点数": len(rs),
                     "主要问题": kinds.most_common(1)[0][0] if kinds else "",
                     "类别分布": dict(kinds),
                     "最近时间": (last_rec.get("shot_at") or "")[:16],
                     "lon": last_rec.get("lon"), "lat": last_rec.get("lat"),
                     "_recs": sorted(rs, key=lambda r: r.get("shot_at") or "", reverse=True)})
    rows.sort(key=lambda x: -x["score"])
    top = rows[0]["score"] if rows else 0.0
    for r in rows:
        r["风险分"] = round(r["score"] / top * 100, 1) if top > 0 else 0.0
    return rows


def map_overview_data():
    st = geo_store()
    if st is None:
        return {"错误": "地块地图模块不可用（缺少 rice_geo.py 或依赖）"}
    all_recs = st.all_records()
    placed = [r for r in all_recs if r.get("lon") is not None]
    kinds = Counter(r.get("cls") or "未检出" for r in placed)
    plots = sorted({(r.get("plot") or "").strip() for r in placed if (r.get("plot") or "").strip()})
    times = sorted((r.get("shot_at") or "") for r in placed if r.get("shot_at"))
    return {"记录总数": len(all_recs), "已落点": len(placed),
            "待落点": len(all_recs) - len(placed), "地块数": len(plots),
            "涉及地块": plots[:20], "类别分布": dict(kinds),
            "时间范围": [times[0][:10], times[-1][:10]] if times else [],
            "数据目录": str(st.root)}


def map_hotspots_data(cell_m=MAP_CELL_M, days=60, top=5):
    """把点位按边长约 cell_m 米的网格分箱，找出检出最密集的几片区域"""
    recs = _geo_records(days)
    if recs is None:
        return {"错误": "地块地图模块不可用"}
    cells = {}
    for r in recs:
        lat, lon = float(r["lat"]), float(r["lon"])
        dlat = cell_m / 111320.0
        dlon = cell_m / max(1.0, 111320.0 * math.cos(math.radians(lat)))
        key = (round(lat / dlat), round(lon / dlon))
        c = cells.setdefault(key, {"点数": 0, "lat": 0.0, "lon": 0.0, "类别": Counter(),
                                   "地块": Counter(), "最近": ""})
        c["点数"] += 1
        c["lat"] += lat
        c["lon"] += lon
        c["类别"][r.get("cls") or "未检出"] += 1
        if (r.get("plot") or "").strip():
            c["地块"][r["plot"].strip()] += 1
        c["最近"] = max(c["最近"], (r.get("shot_at") or "")[:16])
    out = []
    for c in cells.values():
        out.append({"中心": [round(c["lon"] / c["点数"], 6), round(c["lat"] / c["点数"], 6)],
                    "点数": c["点数"], "主要类别": c["类别"].most_common(1)[0][0],
                    "类别分布": dict(c["类别"]),
                    "涉及地块": [p for p, _ in c["地块"].most_common(3)],
                    "最近时间": c["最近"]})
    out.sort(key=lambda x: -x["点数"])
    return {"分箱边长_米": cell_m, "统计天数": days, "热点数": len(out), "热点": out[:top]}


def patrol_plan_data(max_stops=5, days=None):
    """巡田顺序建议：按风险排序选点，再用最近邻贪心排路线（不追求最优，够用且可解释）"""
    rows = plot_risk_rows(days)
    if rows is None:
        return {"错误": "地块地图模块不可用"}
    stops = [r for r in rows if r.get("lon") is not None][:max(1, int(max_stops))]
    if not stops:
        return {"顺序": [], "总里程_km": 0, "说明": "还没有已落点的记录"}
    order = [stops[0]]
    rest = stops[1:]
    while rest:
        cur = order[-1]
        nxt = min(rest, key=lambda r: haversine_m(cur["lon"], cur["lat"], r["lon"], r["lat"]))
        order.append(nxt)
        rest.remove(nxt)
    total = sum(haversine_m(order[i]["lon"], order[i]["lat"],
                            order[i + 1]["lon"], order[i + 1]["lat"])
                for i in range(len(order) - 1))
    return {"坐标系": "WGS-84（地图显示时会自动转 GCJ-02，偏移 300~600 米是坐标系差异不是数据错误）",
            "顺序": [{"第几站": i + 1, "地块": r["plot"], "风险分": r["风险分"],
                      "主要问题": r["主要问题"], "点数": r["点数"],
                      "坐标": [round(r["lon"], 6), round(r["lat"], 6)]}
                     for i, r in enumerate(order)],
            "总里程_km": round(total / 1000.0, 2),
            "说明": "按风险分从高到低选点，再用最近邻贪心排顺序；里程是直线距离估算"}


def map_digest_text(days=30, top=3):
    """给模型看的区域简报（纯文本，省 token）"""
    ov = map_overview_data()
    if "错误" in ov:
        return "地块地图暂不可用：" + ov["错误"]
    lines = ["地块地图简报（统计最近 %d 天）" % days,
             "记录 %d 条（已落点 %d、待落点 %d），地块 %d 个"
             % (ov["记录总数"], ov["已落点"], ov["待落点"], ov["地块数"]),
             "类别分布：%s" % json.dumps(ov["类别分布"], ensure_ascii=False)]
    rows = plot_risk_rows(days) or []
    if rows:
        lines.append("风险排序（风险分 0-100，含时间衰减）：")
        for r in rows[:top]:
            lines.append("  · %s：风险分 %.1f，%d 个点位，主要为%s，最近 %s"
                         % (r["plot"], r["风险分"], r["点数"], r["主要问题"], r["最近时间"]))
    hs = map_hotspots_data(days=days, top=2)
    if hs.get("热点"):
        lines.append("密集区域：")
        for h in hs["热点"]:
            lines.append("  · 坐标 %s 附近 %d 个点位，主要为%s"
                         % (h["中心"], h["点数"], h["主要类别"]))
    return "\n".join(lines)


def nearby_records_data(lon, lat, radius_m=500, days=None, limit=10):
    """某点附近的历史记录（判断「这块地以前出过什么」）"""
    recs = _geo_records(days)
    if recs is None:
        return {"错误": "地块地图模块不可用"}
    hit = []
    for r in recs:
        d = haversine_m(lon, lat, r.get("lon"), r.get("lat"))
        if d <= float(radius_m):
            hit.append({"距离_米": round(d), "时间": (r.get("shot_at") or "")[:16],
                        "类别": r.get("cls") or "未检出", "置信度": round(float(r.get("conf") or 0), 2),
                        "地块": r.get("plot") or "", "备注": r.get("note") or "",
                        "坐标": [round(float(r["lon"]), 6), round(float(r["lat"]), 6)]})
    hit.sort(key=lambda x: x["距离_米"])
    kinds = Counter(h["类别"] for h in hit)
    return {"查询半径_米": radius_m, "命中": len(hit), "类别分布": dict(kinds),
            "记录": hit[:limit]}


AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "get_treatment",
        "description": "查询本地防治库中某个病虫害的完整防治方案（农业防治/生物防治/防治指标/推荐药剂）",
        "parameters": {"type": "object", "properties": {
            "cls_name": {"type": "string", "enum": CLASS_NAMES,
                         "description": "病虫害名称，必须是这六类之一"}},
            "required": ["cls_name"]}}},
    {"type": "function", "function": {
        "name": "check_pesticide",
        "description": "校验一段文字里是否出现国家禁用或停止使用的农药（如氟虫腈、百草枯等）",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "要校验的文字，通常是准备推荐给农户的药剂清单"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "list_reports",
        "description": "查询最近导出过的识别报告记录（时间、类别、置信度）",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "最近多少天，默认 7"}}}}},
    {"type": "function", "function": {
        "name": "feedback_stats",
        "description": "统计用户标记的误检/漏检反馈数量和分布",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "re_detect",
        "description": "用新的置信度阈值重新检测当前这张照片。怀疑漏检时（尤其病害）可以调低阈值再跑一次",
        "parameters": {"type": "object", "properties": {
            "threshold": {"type": "number", "description": "置信度阈值 0.05~0.95，默认 0.4"}}}}},
    {"type": "function", "function": {
        "name": "verify_with_vision",
        "description": "把检测框区域裁出来交给视觉大模型复核类别。低置信度、病害三类、或症状与判断不符时用它二次确认",
        "parameters": {"type": "object", "properties": {
            "box_index": {"type": "integer", "description": "第几个检测框（从 0 开始，默认 0）"},
            "question": {"type": "string", "description": "想问视觉模型的问题，可留空"}}}}},
    {"type": "function", "function": {
        "name": "batch_scan",
        "description": "把某个文件夹里的图片批量跑一遍检测（含自动复核），返回每张的结论与汇总。用于「把这一批图处理掉」这类任务",
        "parameters": {"type": "object", "properties": {
            "folder": {"type": "string", "description": "图片文件夹路径，留空则用默认收件箱"},
            "limit": {"type": "integer", "description": "最多处理多少张，默认 20"}}}}},
    {"type": "function", "function": {
        "name": "daily_digest",
        "description": "生成本地运营数据摘要：最近若干天的检测次数、类别分布、复核一致率、反馈与实训数据",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "统计最近多少天，默认 7"}}}}},
    {"type": "function", "function": {
        "name": "list_sessions",
        "description": "查询最近几次识别会话（时间、病虫害类别、严重度、轮次）。问「上次/前几天那块地怎么样」时先查它",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "最近多少天，默认 14"}}}}},
    {"type": "function", "function": {
        "name": "get_session",
        "description": "读取某次会话的详情：诊断结论、结构化诊断卡、药剂清单与问答摘要",
        "parameters": {"type": "object", "properties": {
            "session_id": {"type": "string", "description": "会话编号；留空表示最近一次"}}}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查天气并判断病害流行风险。支持城市名、地块名或经纬度；"
                       "地块名会自动从地块地图取坐标（田块头顶的天气比城市预报准）",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，例如 南京"},
            "plot": {"type": "string", "description": "地块名（如 东片1号田），与 city 二选一"},
            "lon": {"type": "number", "description": "经度（WGS-84），有坐标时优先"},
            "lat": {"type": "number", "description": "纬度（WGS-84）"},
            "days": {"type": "integer", "description": "看未来几天，默认 3，最多 7"}}}}},
    {"type": "function", "function": {
        "name": "spray_window",
        "description": "算未来施药窗口：逐小时按温度/风速/降雨概率/高温时段评分，"
                       "给出可以下田打药的具体时段与受限原因。用户问「什么时候能打药」时用它",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名"},
            "plot": {"type": "string", "description": "地块名，会从地块地图取坐标"},
            "lon": {"type": "number", "description": "经度"},
            "lat": {"type": "number", "description": "纬度"},
            "hours": {"type": "integer", "description": "看未来多少小时，默认 48"}}}}},
    {"type": "function", "function": {
        "name": "spray_log",
        "description": "查施药记录，并判断药后有没有被雨冲掉（要不要补施）。"
                       "用户问「我昨天打了药、今天下雨要不要补」时用它",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "查最近多少天的记录，默认 14"},
            "plot": {"type": "string", "description": "只看某个地块，可留空"}}}}},
    # ── 地块地图（空间分析）──
    {"type": "function", "function": {
        "name": "map_overview",
        "description": "查看地块地图的数据总览：记录数、已落点/待落点、地块数、类别分布、时间范围",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "plot_risk",
        "description": "按地块聚合风险并排序，回答「哪块地最该先处理」。"
                       "风险分 0-100，含时间衰减（14 天半衰期）与重复出现加权",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "只统计最近多少天，留空表示全部"},
            "top": {"type": "integer", "description": "返回前几名，默认 5"}}}}},
    {"type": "function", "function": {
        "name": "map_hotspots",
        "description": "找出病虫害最密集的区域（按约 500 米网格分箱），用于判断扩散中心",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "统计最近多少天，默认 60"},
            "cell_m": {"type": "integer", "description": "网格边长（米），默认 500"},
            "top": {"type": "integer", "description": "返回前几个热点，默认 5"}}}}},
    {"type": "function", "function": {
        "name": "nearby_records",
        "description": "查询某个坐标附近的历史记录，用于判断「这块地以前出过什么」",
        "parameters": {"type": "object", "properties": {
            "lon": {"type": "number", "description": "经度（WGS-84）"},
            "lat": {"type": "number", "description": "纬度（WGS-84）"},
            "radius_m": {"type": "number", "description": "半径（米），默认 500"},
            "days": {"type": "integer", "description": "只查最近多少天，留空表示全部"}},
            "required": ["lon", "lat"]}}},
    {"type": "function", "function": {
        "name": "patrol_plan",
        "description": "给出巡田顺序建议：按风险选点并排好路线，含总里程估算",
        "parameters": {"type": "object", "properties": {
            "max_stops": {"type": "integer", "description": "最多几个地块，默认 5"},
            "days": {"type": "integer", "description": "只统计最近多少天"}}}}},
    {"type": "function", "function": {
        "name": "map_digest",
        "description": "生成地块地图简报：记录概况、风险排序前几名、密集区域——"
                       "需要综合谈区域情况时用它",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "统计最近多少天，默认 30"}}}}},
]
TOOL_LABELS = {
    "get_treatment": "查询本地防治库",
    "verify_with_vision": "裁图交给视觉模型复核",
    "batch_scan": "批量检测一个文件夹",
    "daily_digest": "汇总最近运行数据",
    "list_sessions": "查询历史会话",
    "get_session": "读取会话详情",
    "check_pesticide": "校验农药合规名单",
    "list_reports": "查询报告历史",
    "feedback_stats": "统计误检反馈",
    "re_detect": "按新阈值重新检测",
    "get_weather": "查询天气与病害风险",
    "spray_window": "计算施药窗口",
    "spray_log": "查施药记录与补施判断",
    "map_overview": "查看地块地图总览",
    "plot_risk": "按地块排风险优先级",
    "map_hotspots": "找病虫害密集区域",
    "nearby_records": "查附近历史记录",
    "patrol_plan": "排巡田路线",
    "map_digest": "生成地块地图简报",
}

AUDIT_LOG = APP_DIR / "logs" / "llm_log.jsonl"
EVENT_LOG = APP_DIR / "logs" / "events.jsonl"
LAST_IMAGE_SOURCE = ""      # 最近一次加载的图片路径（记录到地图时用来读 EXIF 时间与坐标）


def _append_jsonl(path, record):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("写日志失败 %s: %s", path, e)


def log_llm_call(**fields):
    """审计：每次模型调用都留痕（模型、耗时、token、工具、是否命中缓存）"""
    fields.setdefault("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fields.setdefault("app_version", APP_VERSION)
    _append_jsonl(AUDIT_LOG, fields)


def log_event(kind, **fields):
    """运行台账：检测/反馈/实训等事件，用于赛后统计与支撑材料"""
    fields["kind"] = kind
    fields.setdefault("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    _append_jsonl(EVENT_LOG, fields)


def read_jsonl(path, limit=None):
    if not path.exists():
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out[-limit:] if limit else out


# ── 成本记账（按官方价目估算；元 / 百万 tokens） ──────────────────
# 来源：DeepSeek 模型与价格页（2026-10 现行）。高峰时段 = 北京时间周一至周五 9:00-12:00、14:00-18:00
# （法定节假日无法在本地判断，按工作日近似；估算值仅供成本感知，不用于结算）
DEEPSEEK_PRICES = {
    "deepseek-flash":   {"in_miss": (2.0, 1.0), "in_hit": (0.04, 0.02), "out": (8.0, 4.0)},
    "deepseek-v4-pro":  {"in_miss": (9.0, 4.5), "in_hit": (0.30, 0.15), "out": (27.0, 13.5)},
}


def is_peak_now(when=None):
    when = when or datetime.now()
    return when.weekday() < 5 and (9 <= when.hour < 12 or 14 <= when.hour < 18)


def estimate_cost(model, usage, when=None):
    """按 token 用量估算费用（元）；未知模型/无用量返回 None"""
    price = DEEPSEEK_PRICES.get((model or "").strip())
    if not price or not isinstance(usage, dict):
        return None
    peak = 0 if is_peak_now(when) else 1
    miss = int(usage.get("prompt_cache_miss_tokens")
               or usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    out = int(usage.get("completion_tokens") or 0)
    return round((miss * price["in_miss"][peak] + hit * price["in_hit"][peak]
                  + out * price["out"][peak]) / 1_000_000, 6)


def usage_cost_summary():
    """从审计日志汇总 token 与估算费用（设置页/台账展示用）"""
    rows = [r for r in read_jsonl(AUDIT_LOG) if r.get("event") == "answer"]
    tokens_in = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows)
    tokens_out = sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows)
    costs = [c for c in (r.get("cost") for r in rows) if isinstance(c, (int, float))]
    return {"调用次数": len(rows), "输入tokens": tokens_in, "输出tokens": tokens_out,
            "估算费用元": round(sum(costs), 4) if costs else None,
            "有记账的调用": len(costs)}


# ── 回答缓存：同一张图 + 同一类别 + 同一提示词版本直接回放，不重复付费 ──
# 只缓存「首次分析」的文本结果（追问上下文不同，不缓存；工具结果更不缓存，会过期）。
LLM_CACHE_DIR = APP_DIR / ".cache" / "llm"
PROMPT_VERSION = "v3"          # 系统提示词/约束一变就改这里，避免回放旧口径的答案


def _cache_key(cls_name, prompt, image, model):
    h = hashlib.sha1()
    h.update(("%s|%s|%s|%s" % (PROMPT_VERSION, model, cls_name, prompt)).encode("utf-8"))
    if image is not None and getattr(image, "shape", None) is not None:
        try:
            small = cv2.resize(image, (64, 64))
            h.update(small.tobytes())
        except Exception:
            pass
    return h.hexdigest()[:24]


def cache_get(key):
    try:
        p = LLM_CACHE_DIR / ("%s.json" % key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("读缓存失败: %s", e)
    return None


def cache_put(key, text, model):
    try:
        LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (LLM_CACHE_DIR / ("%s.json" % key)).write_text(json.dumps(
            {"text": text, "model": model, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
             "prompt_version": PROMPT_VERSION}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.warning("写缓存失败: %s", e)


# ── 会话记忆（Agent 的长期记忆） ──────────────────────────────
# 为什么要有：没有记忆的 agent 每次都是从零开始，「上周那块地后来怎么样」这类问题
# 根本无从回答。这里把每次识别的结论、结构化诊断卡与问答历史落盘，
# 并做成两个只读工具暴露给模型。图片只存 768px 压缩副本（本地保存，不上传）。
SESSION_DIR = APP_DIR / "sessions"
SESSION_INDEX = SESSION_DIR / "index.json"


def _session_image_path(sid):
    return SESSION_DIR / ("%s.jpg" % sid)


def save_session(cls_name, conf=0.0, severity=None, mode="image", chat=None,
                 card=None, image=None, session_id=None):
    """保存/更新一次识别会话，返回 session_id"""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    turns = []
    for entry in (chat or []):
        role, content = entry[0], entry[1]
        if role in ("user", "ai", "ai_initial", "system"):
            turns.append([role, content])
    record = {"id": sid, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "cls": cls_name, "conf": round(float(conf or 0.0), 4),
              "severity": (severity or {}).get("level"),
              "severity_text": (severity or {}).get("text"),
              "mode": mode, "model": get_model_name(), "turns": turns[-40:],
              "card": card or None}
    try:
        (SESSION_DIR / ("%s.json" % sid)).write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        if image is not None and getattr(image, "shape", None) is not None:
            h, w = image.shape[:2]
            if max(h, w) > 768:
                s = 768.0 / max(h, w)
                image = cv2.resize(image, (int(w * s), int(h * s)))
            cv2.imwrite(str(_session_image_path(sid)), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 72])
        items = [x for x in load_session_index() if x.get("id") != sid]
        items.insert(0, {"id": sid, "time": record["time"], "cls": cls_name,
                         "conf": record["conf"], "severity": record["severity"],
                         "turns": len(turns), "mode": mode,
                         "card_chemicals": [c.get("name") for c in
                                            ((card or {}).get("chemicals") or [])]})
        SESSION_INDEX.write_text(json.dumps(items[:200], ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    except Exception as e:
        log.warning("保存会话失败: %s", e)
    return sid


def load_session_index():
    """会话索引（按时间倒序）"""
    try:
        if SESSION_INDEX.exists():
            data = json.loads(SESSION_INDEX.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("读取会话索引失败: %s", e)
    return []


def load_session(sid):
    """读取一次会话的完整记录"""
    try:
        p = SESSION_DIR / ("%s.json" % sid)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("读取会话失败: %s", e)
    return None


def _tool_list_sessions(args):
    days = int(args.get("days") or 14)
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for it in load_session_index():
        try:
            if datetime.strptime(it.get("time", "")[:16], "%Y-%m-%d %H:%M") < cutoff:
                continue
        except ValueError:
            pass
        out.append({"会话编号": it.get("id"), "时间": it.get("time"), "类别": it.get("cls"),
                    "严重度": it.get("severity"), "问答轮次": it.get("turns")})
    return json.dumps({"天数": days, "条数": len(out), "会话": out[:15]}, ensure_ascii=False)


def _tool_get_session(args):
    sid = str(args.get("session_id") or "").strip()
    if not sid:
        idx = load_session_index()
        if not idx:
            return json.dumps({"结论": "还没有历史会话"}, ensure_ascii=False)
        sid = idx[0].get("id", "")
    rec = load_session(sid)
    if not rec:
        return json.dumps({"结论": "没有找到会话 %s" % sid}, ensure_ascii=False)
    qa = [[t[0], str(t[1])[:200]] for t in (rec.get("turns") or [])
          if t[0] in ("user", "ai", "ai_initial")][-6:]
    return json.dumps({
        "会话编号": rec.get("id"), "时间": rec.get("time"), "类别": rec.get("cls"),
        "置信度": rec.get("conf"), "严重度": rec.get("severity"),
        "诊断卡": (rec.get("card") or {}).get("diagnosis"),
        "诊断卡药剂": [c.get("name") for c in ((rec.get("card") or {}).get("chemicals") or [])],
        "最近问答": qa,
        "说明": "会话保存在本机 sessions 目录，未上传",
    }, ensure_ascii=False)


# ── 工具实现（全部只读） ──────────────────────────────────────
def _tool_get_treatment(args):
    cls = str(args.get("cls_name", "")).strip()
    tr = TREATMENT_DB.get(cls)
    if not tr:
        return "未找到该类别。可选：%s" % "、".join(TREATMENT_DB.keys())
    payload = {
        "类别": cls, "症状": tr.get("symptom", ""), "农业防治": tr.get("agri", []),
        "生物防治": tr.get("bio", []), "防治指标": tr.get("threshold", ""),
        "推荐药剂": [{"名称": d.get("name"), "规格": d.get("spec"),
                      "亩用量": d.get("dose"), "安全间隔期": d.get("safe_days"),
                      "登记证号": d.get("reg_no") or "待补",
                      "施药方法": d.get("method") or "",
                      "每季最多使用": d.get("max_per_season"),
                      "防治适期": d.get("timing") or "",
                      "注意事项": d.get("caution") or "",
                      "登记对象": d.get("target") or "",
                      "药剂类型": d.get("type") or "",
                      "备注": d.get("note") or ""}
                     for d in tr.get("drugs", [])],
        "数据来源": "本地防治库（药剂剂量与间隔期以农药标签为准）",
    }
    if tr.get("meta"):
        payload["防治要点"] = tr["meta"]
    # 闭环的另一半：这个类别最近打过没有、打的是什么——避免重复施药与抗性
    try:
        cut = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        recent = [r for r in load_spray_records()
                  if (r.get("time") or "")[:10] >= cut
                  and (cls in (r.get("cls") or "") or (r.get("cls") or "") in cls)]
        if recent:
            payload["近期施药"] = {
                "次数": len(recent),
                "记录": [{"时间": r.get("time"), "地块": r.get("plot") or "未填写",
                          "药剂": r.get("drug"), "亩用量": r.get("dose") or "",
                          "来源": r.get("source") or "manual"} for r in recent[-3:]],
                "提示": "近 14 天已针对该对象施药，注意轮换药剂、遵守安全间隔期，避免重复施药",
            }
    except Exception as e:
        log.debug("查询近期施药失败：%s", e)
    return json.dumps(payload, ensure_ascii=False)


def _tool_check_pesticide(args):
    text = str(args.get("text", ""))
    hits = pesticide_hits(text)
    detail = []
    for name in hits:
        status, rec = pesticide_status(name)
        if status == "stopped":
            detail.append({"名称": name, "状态": "已停止使用", "依据": rec.get("source", "")})
        else:
            detail.append({"名称": name, "状态": "国家禁用",
                           "依据": PESTICIDE_CONTROL["banned"]["source"]})
    return json.dumps({"命中": detail, "结论": "存在问题，禁止使用" if hits else "未发现禁用/停用农药"},
                      ensure_ascii=False)


def _tool_list_reports(args):
    days = int(args.get("days") or 7)
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for it in load_report_index():
        t = str(it.get("time", ""))
        if t:
            try:
                if datetime.strptime(t[:16], "%Y-%m-%d %H:%M") < cutoff:
                    continue
            except ValueError:
                pass
        out.append({"时间": t, "类别": it.get("cls_name"), "置信度": it.get("conf")})
    return json.dumps({"天数": days, "条数": len(out), "记录": out[:20]}, ensure_ascii=False)


def _tool_feedback_stats(args=None):
    rows = load_feedback_rows()
    pairs = Counter((r.get("predicted", ""), r.get("user_choice", "")) for r in rows)
    misses = [r for r in rows if r.get("mode") == "miss" or r.get("predicted") == "(无检出)"]
    return json.dumps({
        "反馈总数": len(rows), "其中漏检": len(misses),
        "模型判断→用户判断 前5": [{"模型": p, "用户": u, "次数": n}
                                  for (p, u), n in pairs.most_common(5)],
    }, ensure_ascii=False)


def _tool_re_detect(args, image=None):
    if image is None:
        return "当前没有可复检的图片"
    try:
        conf = float(args.get("threshold") or CONF_THRESHOLD)
    except (TypeError, ValueError):
        conf = CONF_THRESHOLD
    conf = max(0.05, min(0.95, conf))
    try:
        model, _ = get_model()
        res = model(image, conf=conf, imgsz=640, verbose=False)
        dets = [{"类别": CLASS_NAMES[int(b.cls.item())], "置信度": round(float(b.conf.item()), 3)}
                for b in res[0].boxes]
        return json.dumps({"阈值": conf, "检出数": len(dets), "检出": dets,
                           "说明": "阈值越低越不容易漏检，但误检会变多"}, ensure_ascii=False)
    except Exception as e:
        return "重检失败：%s" % e


# ── 视觉复核：裁图 → 让视觉大模型独立判断 ──────────────────────
# 动机：六类里病害三类召回偏低（白叶枯病尤甚），而低置信度恰好说明模型不确定。
# 与其直接抛给农户，不如让第二个模型独立看一遍，再让人来裁决分歧。
# 两条路径共用这个函数：① agent 主动调用（工具）；② 低置信度自动触发（级联）。
PEST_CLASSES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉"]      # 虫害三类
DISEASE_CLASSES = ["白叶枯病", "胡麻斑病", "稻瘟病"]     # 病害三类：召回短板，默认必复核


def same_category(cls_a, cls_b):
    """两个类别是否同属虫害或同属病害。

    为什么要分：虫害与病害的处置完全不同（杀虫剂 vs 杀菌剂）。
    · 跨大类分歧（虫 ↔ 病）→ 可能用错药，必须提醒 + 人工裁决；
    · 同类分歧（病 ↔ 病）→ 处置方向一致，记台账即可，不打扰用户。
    """
    if not cls_a or not cls_b:
        return False
    if cls_a in PEST_CLASSES and cls_b in PEST_CLASSES:
        return True
    if cls_a in DISEASE_CLASSES and cls_b in DISEASE_CLASSES:
        return True
    return False


def parse_class_from_text(text):
    """从自由文本里抓出类别名（视觉模型不一定严格只回一个词）"""
    t = text or ""
    for name in CLASS_NAMES:
        if name in t:
            return name
    return ""


def vision_recheck(image, box=None, question=None, box_index=0, source="tool"):
    """裁出指定框（默认整图）交给视觉模型复核。

    返回 dict：{ok, verdict, raw, model, box_index} 或 {ok: False, error}
    """
    if image is None or getattr(image, "shape", None) is None:
        return {"ok": False, "error": "当前没有可复核的图片"}
    if not model_supports_vision():
        return {"ok": False, "error": "当前模型不支持看图（%s）" % get_model_name()}
    crop = image
    if box:
        x1, y1, x2, y2 = [int(v) for v in box]
        sub = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if sub.size:
            crop = sub
    h, w = crop.shape[:2]
    if max(h, w) < 320:                     # 小图放大，视觉模型对小图更吃力
        s = 320.0 / max(1, max(h, w))
        crop = cv2.resize(crop, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return {"ok": False, "error": "裁图编码失败"}
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    q = (question or
         "这是水稻病虫害图片。请判断属于以下哪一类：%s。只回答类别名，并给一句依据。"
         % "、".join(CLASS_NAMES))
    payload = {"model": get_model_name(), "max_tokens": 200, "stream": False,
               "temperature": 0, "messages": [{"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                   {"type": "text", "text": q}]}]}
    if CONFIG.get("llm_provider", "deepseek") == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    try:
        resp = HTTP_SESSION.post(api_endpoint(),
                                 headers={"Authorization": "Bearer " + get_api_key(),
                                          "Content-Type": "application/json"},
                                 json=payload, timeout=(10, 90))
        if resp.status_code != 200:
            return {"ok": False, "error": "视觉复核失败：HTTP %s" % resp.status_code}
        data = resp.json()
        text = ((data["choices"][0]["message"].get("content")) or "").strip()
        usage = data.get("usage") or {}
        log_llm_call(event="vision_recheck", model=get_model_name(), box_index=box_index,
                     source=source, usage=usage,
                     cost=estimate_cost(get_model_name(), usage))
        return {"ok": True, "verdict": parse_class_from_text(text), "raw": text[:300],
                "model": get_model_name(), "box_index": box_index}
    except Exception as e:
        return {"ok": False, "error": "视觉复核异常：%s" % e}


def should_auto_verify(cls_name, conf):
    """自动复核的触发规则：置信度低，或属于病害三类（召回短板）。

    返回 (是否复核, 原因)。规则写在配置里，方便赛前按实测调。
    """
    if not CONFIG.get("cascade_auto", True):
        return False, "已关闭自动复核"
    if not CONFIG.get("enable_llm", False) or not get_api_key():
        return False, "未启用云端 AI"
    if not model_supports_vision():
        return False, "当前模型不支持看图"
    thr = float(CONFIG.get("cascade_conf", 0.6))
    if conf and conf < thr:
        return True, "置信度 %.0f%% 低于 %.0f%%" % (conf * 100, thr * 100)
    if cls_name in DISEASE_CLASSES:
        return True, "%s 属病害三类（召回短板）" % cls_name
    return False, "置信度与类别都不需要复核"


def _tool_verify_with_vision(args, image=None, boxes=None):
    """agent 主动调用：把检测框裁出来交给视觉大模型复核（二次确认）"""
    box = None
    idx = 0
    if boxes:
        try:
            idx = int(args.get("box_index") or 0)
        except (TypeError, ValueError):
            idx = 0
        idx = max(0, min(idx, len(boxes) - 1))
        box = boxes[idx].get("box")
    res = vision_recheck(image, box, args.get("question"), idx, source="tool")
    if not res.get("ok"):
        return res.get("error", "复核失败")
    return json.dumps({"复核模型": res["model"], "检测框序号": idx,
                       "视觉模型判断": res["raw"] or "（空）",
                       "归类": res["verdict"] or "未识别出类别",
                       "说明": "这是第二个模型的独立判断，与主判断不一致时以现场症状为准"},
                      ensure_ascii=False)


def _tool_batch_scan(args):
    """批处理：把文件夹里的图跑一遍（检测 + 按需复核），返回逐张结论与汇总"""
    folder = str(args.get("folder") or "").strip() or None
    try:
        limit = int(args.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 50))
    if folder and not Path(folder).is_dir():
        # 模型容易臆造目录名：这里必须明确报错并给出可用目录，否则"成功返回空结果"会被误当成干完了
        usable = [str(p) for p in (WATCH_DIR, Path(res_path("手机端可跑版/samples")),
                                   Path(res_path("dataset_fieldlike/images/val")))
                  if p.is_dir()]
        log_event("task_bad_arg", tool="batch_scan", folder=str(folder))
        return json.dumps({"错误": "目录不存在：%s" % folder,
                           "可用目录": usable,
                           "建议": "省略 folder 参数即处理默认收件箱；或让用户给出真实路径"},
                          ensure_ascii=False)
    files = list_watch_candidates(folder, limit)
    if not files:
        return json.dumps({"结论": "没有待处理的新图片",
                           "目录": str(folder or WATCH_DIR)}, ensure_ascii=False)
    rows = [process_image_file(p) for p in files]
    by_cls = Counter(r["cls"] or "未检出" for r in rows)
    agree = [r for r in rows if r.get("verify_agree") != ""]
    return json.dumps({
        "处理张数": len(rows), "目录": str(folder or WATCH_DIR),
        "类别分布": dict(by_cls),
        "复核次数": len(agree),
        "复核一致": sum(1 for r in agree if r.get("verify_agree") is True),
        "需要人工看的": [{"文件": r["file"], "原因": alert_level(r)}
                     for r in rows if alert_level(r)],
        "逐张结果": [{"文件": r["file"], "类别": r["cls"] or "未检出",
                  "置信度": r["conf"], "强度": r["severity"], "复核": r["verify"]}
                 for r in rows[:20]],
        "台账": str(WATCH_RESULTS),
    }, ensure_ascii=False)


def _tool_daily_digest(args):
    """主动简报：不等人问「最近怎么样」，自己把本地数据汇总出来"""
    try:
        days = int(args.get("days") or 7)
    except (TypeError, ValueError):
        days = 7
    cutoff = datetime.now() - timedelta(days=days)
    # 界面检测（detection）与无人值守巡检（watch_process）都算"检测"
    events = [e for e in read_jsonl(EVENT_LOG)
              if e.get("kind") in ("detection", "watch_process")]
    recent = []
    for e in events:
        try:
            if datetime.strptime(e.get("time", "")[:19], "%Y-%m-%d %H:%M:%S") >= cutoff:
                recent.append(e)
        except ValueError:
            continue
    s = usage_summary()
    s["统计天数"] = days
    s["期间检测次数"] = len(recent)
    s["期间类别分布"] = dict(Counter(e.get("cls") or "未检出" for e in recent))
    # 地块地图：把区域风险一并带出来，简报就不只是"跑了多少次"
    try:
        ov = map_overview_data()
        if "错误" not in ov and ov.get("已落点"):
            rows = plot_risk_rows(days) or []
            s["地块地图"] = {
                "已落点": ov["已落点"], "地块数": ov["地块数"], "待落点": ov["待落点"],
                "风险前三": [{"地块": r["plot"], "风险分": r["风险分"],
                              "主要问题": r["主要问题"], "点数": r["点数"]}
                             for r in rows[:3]],
            }
    except Exception as e:
        log.debug("简报附带地图信息失败：%s", e)
    return json.dumps(s, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# 天气：按地块查、算施药窗口、判病害流行风险
# --------------------------------------------------------------
# 数据源仍是 Open-Meteo（免 Key）。相比"查城市 3 天预报"，这里做了三件更实用的事：
# 1) 支持经纬度/地块名 → 直接查田块头顶的天气（地图库里有地块坐标）；
# 2) 逐小时评分出「施药窗口」——温度、风速、降雨、高温时段四条硬规则；
# 3) 把未来天气换算成稻瘟病/白叶枯病/胡麻斑病的流行风险，并能与地块检出记录联动。
# 结果缓存 30 分钟：Agent 可能连着调好几次，没必要重复打接口。
# ══════════════════════════════════════════════════════════════
WEATHER_CACHE = APP_DIR / "cache" / "weather.json"
WEATHER_TTL_MIN = 30
# 缓存结构版本：字段有增减时把它 +1，旧缓存自动失效。
# 踩过的坑：加了「过去逐小时」之后，30 分钟内的旧缓存仍被命中，
# 于是"药后有没有下雨"一直拿不到数据，看起来像功能没生效。
WEATHER_SCHEMA = 3
SPRAY_TEMP_OK = (15.0, 30.0)      # 适宜施药温度（℃）
SPRAY_WIND_OK = 4.0               # 适宜风速上限（m/s，约 3 级）
SPRAY_RAIN_PROB_OK = 30           # 降雨概率上限（%）
SPRAY_PEAK_HOURS = (11, 15)       # 正午高温时段，避开（评分里扣分）
SPRAY_DAY_HOURS = (6, 18)         # 可作业时段：窗口只在这个区间里连
SPRAY_MIN_SCORE = 60              # 参考分（窗口由硬约束决定，评分只用于排序）


def _weather_cache_load():
    try:
        if WEATHER_CACHE.exists():
            data = json.loads(WEATHER_CACHE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as e:
        log.debug("读取天气缓存失败：%s", e)
    return {}


def _weather_cache_put(key, payload):
    try:
        data = _weather_cache_load()
        data[key] = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": payload}
        # 只留最近 40 条，避免无限增长
        if len(data) > 40:
            for k in sorted(data, key=lambda x: data[x].get("ts", ""))[:len(data) - 40]:
                data.pop(k, None)
        WEATHER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WEATHER_CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        log.debug("写天气缓存失败：%s", e)


def geocode_city(city):
    """城市名 → 经纬度（Open-Meteo 地理编码，免 Key）"""
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": city, "count": 1, "language": "zh"}, timeout=8).json()
    if not r.get("results"):
        return None
    loc = r["results"][0]
    return {"lon": loc["longitude"], "lat": loc["latitude"],
            "name": loc.get("name") or city,
            "admin": loc.get("admin1") or ""}


def plot_coords(plot):
    """按地块名取坐标（取该地块最近一条记录的位置）"""
    for row in (plot_risk_rows() or []):
        if plot and (plot in row["plot"] or row["plot"] in plot):
            return row.get("lon"), row.get("lat"), row["plot"]
    return None, None, None


def fetch_weather(city=None, lon=None, lat=None, days=7, hours=48, use_cache=True, past_days=7):
    """取天气：优先用坐标，其次地块名，最后城市名。

    返回 dict：地点/坐标/当前/逐日/逐小时/来源（fresh|cache|stale）/生成时间
    """
    place = None
    plot_name = None
    if lon is None or lat is None:
        if city:
            hit = plot_coords(city)          # 先当地块名试一下（农户嘴里说的多是田块名）
            if hit[0] is not None:
                lon, lat, plot_name = hit
                place = {"name": plot_name, "admin": "地块坐标"}
            else:
                loc = geocode_city(city)
                if not loc:
                    return {"错误": "没查到地点：%s" % city}
                lon, lat, place = loc["lon"], loc["lat"], loc
        else:
            place = geocode_city("南京")
            lon, lat = place["lon"], place["lat"]
    if place is None:
        place = {"name": "地块坐标" if plot_name else "指定坐标",
                 "admin": ""}
    key = "%.3f,%.3f,%d,%d" % (float(lon), float(lat), int(days), int(past_days))
    now = datetime.now()
    if use_cache:
        cached = _weather_cache_load().get(key)
        if cached:
            try:
                ts = datetime.strptime(cached["ts"], "%Y-%m-%d %H:%M:%S")
                age_min = (now - ts).total_seconds() / 60.0
                if age_min <= WEATHER_TTL_MIN and \
                        (cached.get("data") or {}).get("_schema") == WEATHER_SCHEMA:
                    out = dict(cached["data"])
                    out["来源"] = "cache"
                    out["缓存时间"] = cached["ts"]
                    return out
            except (KeyError, ValueError):
                pass
    try:
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "hourly": ("temperature_2m,relative_humidity_2m,precipitation_probability,"
                       "precipitation,wind_speed_10m,wind_gusts_10m,cloud_cover"),
            "daily": ("temperature_2m_max,temperature_2m_min,precipitation_sum,"
                      "precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max"),
            # 注意：不要传 forecast_hours——实测它会让返回的 hourly 只剩未来小时，
            # past_days 的数据会被吞掉（历史降水拿到 0 条）。改为多取几天再自己切片。
            "timezone": "Asia/Shanghai", "forecast_days": max(3, min(int(days), 16)),
            # 过去 7 天的逐小时降水：判断「打完药之后到底下没下雨」靠它（实测数据，不是预报）
            "past_days": max(1, min(int(past_days), 14))}, timeout=10).json()
    except Exception as e:
        cached = _weather_cache_load().get(key)
        if cached and (cached.get("data") or {}).get("_schema") == WEATHER_SCHEMA:
            out = dict(cached["data"])
            out["来源"] = "stale"
            out["缓存时间"] = cached.get("ts", "")
            out["提示"] = "取实时天气失败（%s），以下为上次缓存" % e
            return out
        return {"错误": "天气查询失败：%s" % e}
    cur = resp.get("current") or {}
    hr = resp.get("hourly") or {}
    dy = resp.get("daily") or {}
    hourly, past = [], []
    now_key = now.strftime("%Y-%m-%dT%H:00")
    for i, t in enumerate(hr.get("time", [])):
        row = {"时间": t,
               "温度": hr.get("temperature_2m", [None])[i],
               "湿度": hr.get("relative_humidity_2m", [None])[i],
               "降雨概率": hr.get("precipitation_probability", [None])[i],
               "降雨量": hr.get("precipitation", [None])[i],
               "风速": hr.get("wind_speed_10m", [None])[i],
               "阵风": hr.get("wind_gusts_10m", [None])[i],
               "云量": hr.get("cloud_cover", [None])[i]}
        (past if t < now_key else hourly).append(row)
    hourly = hourly[:max(1, min(int(hours), 240))]      # 只要前 N 个未来小时
    daily = []
    for i, t in enumerate(dy.get("time", [])):
        daily.append({"日期": t,
                      "最高温": dy.get("temperature_2m_max", [None])[i],
                      "最低温": dy.get("temperature_2m_min", [None])[i],
                      "降雨量mm": dy.get("precipitation_sum", [None])[i],
                      "降雨概率%": dy.get("precipitation_probability_max", [None])[i],
                      "最大风速": dy.get("wind_speed_10m_max", [None])[i],
                      "最大阵风": dy.get("wind_gusts_10m_max", [None])[i]})
    out = {"地点": place.get("name", ""), "上级": place.get("admin", ""),
           "坐标": [round(float(lon), 4), round(float(lat), 4)],
           "当前": {"温度": cur.get("temperature_2m"), "湿度": cur.get("relative_humidity_2m"),
                    "风速": cur.get("wind_speed_10m"), "降水": cur.get("precipitation")},
           "逐日": daily, "逐小时": hourly, "过去逐小时": past,
           "_schema": WEATHER_SCHEMA,
           "来源": "fresh", "生成时间": now.strftime("%Y-%m-%d %H:%M")}
    _weather_cache_put(key, out)
    log_event("weather_query", place=out["地点"], lon=out["坐标"][0], lat=out["坐标"][1])
    return out


def spray_score(hour):
    """单个小时的施药适宜度（0-100）。规则写在明面上，方便解释给用户听。"""
    t, w = hour.get("温度"), hour.get("风速")
    p, rain = hour.get("降雨概率"), hour.get("降雨量")
    try:
        hh = int(str(hour.get("时间", ""))[11:13])
    except ValueError:
        hh = 12
    score = 100.0
    if t is None:
        return 0.0, ["缺少温度数据"]
    reasons = []
    if t < 10 or t > 35:
        score -= 60
        reasons.append("温度 %.0f℃ 超出安全施药区间" % t)
    elif t < SPRAY_TEMP_OK[0]:
        score -= 25
        reasons.append("温度偏低（%.0f℃）影响药效" % t)
    elif t > 32:
        score -= 35
        reasons.append("高温 %.0f℃ 易药害" % t)
    elif t > SPRAY_TEMP_OK[1]:
        score -= 15
        reasons.append("温度偏高（%.0f℃）" % t)
    if w is not None:
        if w > 2 * SPRAY_WIND_OK:
            score -= 60
            reasons.append("风速 %.1f m/s 过大，飘移风险高" % w)
        elif w > SPRAY_WIND_OK:
            score -= 25
            reasons.append("风速 %.1f m/s 偏大" % w)
    if p is not None:
        if p > 2 * SPRAY_RAIN_PROB_OK:
            score -= 50
            reasons.append("降雨概率 %d%%，药液易被冲刷" % p)
        elif p > SPRAY_RAIN_PROB_OK:
            score -= 20
            reasons.append("降雨概率 %d%% 偏高" % p)
    if rain and rain > 0.2:
        score -= 30
        reasons.append("有降水 %.1f mm" % rain)
    if SPRAY_PEAK_HOURS[0] <= hh < SPRAY_PEAK_HOURS[1]:
        score -= 20
        reasons.append("正午高温时段")
    return max(0.0, min(100.0, score)), reasons


def spray_hour_ok(hour):
    """硬约束：这一小时到底能不能下田打药（评分只是排序，能不能打看这个）

    为什么单独做一层：只靠扣分会出现"风速 7.5 m/s 仍算适宜"这种荒唐结论——
    飘移和冲刷是一票否决项，不该被别的加分抵消掉。
    """
    t, w = hour.get("温度"), hour.get("风速")
    p, rain = hour.get("降雨概率"), hour.get("降雨量")
    try:
        hh = int(str(hour.get("时间", ""))[11:13])
    except ValueError:
        hh = 12
    if t is None or w is None:
        return False, "缺少温度/风速数据"
    if not (SPRAY_TEMP_OK[0] <= t <= SPRAY_TEMP_OK[1]):
        return False, "温度 %.0f℃ 超出 %.0f-%.0f℃" % (t, SPRAY_TEMP_OK[0], SPRAY_TEMP_OK[1])
    if w > SPRAY_WIND_OK:
        return False, "风速 %.1f m/s 超过 %.1f m/s（飘移风险）" % (w, SPRAY_WIND_OK)
    if p is not None and p > SPRAY_RAIN_PROB_OK:
        return False, "降雨概率 %d%% 超过 %d%%" % (p, SPRAY_RAIN_PROB_OK)
    if rain and rain > 0.2:
        return False, "正在降水 %.1f mm" % rain
    if not (SPRAY_DAY_HOURS[0] <= hh < SPRAY_DAY_HOURS[1]):
        return False, "%02d 点不在可作业时段（%02d-%02d 点）" % (hh, SPRAY_DAY_HOURS[0],
                                                                SPRAY_DAY_HOURS[1])
    return True, ""


def _block_category(block):
    """把硬约束的说明归类，便于统计「主要卡在哪一条」"""
    if block.startswith("风速"):
        return "风速过大"
    if block.startswith("温度"):
        return "温度不适宜"
    if block.startswith("降雨概率"):
        return "降雨概率高"
    if block.startswith("正在降水"):
        return "正在下雨"
    if block.startswith("缺少"):
        return "数据缺失"
    if "不在可作业时段" in block:
        return "非作业时段（夜间）"
    return block or "其他"


def spray_windows(wx, min_hours=2, top=2):
    """把「硬约束都满足」的连续小时连成窗口"""
    hours = wx.get("逐小时") or []
    if not hours:
        return []
    scored = []
    for h in hours:
        s, why = spray_score(h)
        ok, block = spray_hour_ok(h)
        scored.append({"时间": h["时间"], "温度": h["温度"], "风速": h["风速"],
                       "降雨概率": h["降雨概率"], "评分": round(s), "可用": ok,
                       "不宜原因": ([block] if block else []) + why})
    windows, cur = [], []
    for row in scored:
        if row["可用"]:
            cur.append(row)
        else:
            if len(cur) >= min_hours:
                windows.append(cur)
            cur = []
    if len(cur) >= min_hours:
        windows.append(cur)
    out = []
    for w in windows:
        avg = sum(x["评分"] for x in w) / float(len(w))
        out.append({"开始": w[0]["时间"], "结束": w[-1]["时间"], "时长小时": len(w),
                    "平均评分": round(avg, 1),
                    "温度区间": [min(x["温度"] for x in w), max(x["温度"] for x in w)],
                    "最大风速": max(x["风速"] for x in w),
                    "最大降雨概率": max(x["降雨概率"] for x in w),
                    "建议": _spray_advice(w[0]["时间"])})
    out.sort(key=lambda x: -x["平均评分"])
    return out[:top]


def _spray_advice(start):
    try:
        hh = int(str(start)[11:13])
    except ValueError:
        return "择机施药"
    if hh < 9:
        return "清晨施药：露水干后、气温未升，药效好且安全"
    if hh < 16:
        return "上午/午后施药：注意避开正午高温"
    return "傍晚施药：避开高温，药液停留时间长"

# ── 地块级施药窗口（地图徽章用） ──────────────────────────────
# 状态四色：绿=今天有窗口，黄=窗口很窄，红=今天不宜，灰=天气不可用。
# 颜色只表达「今天能不能打药」这一件事，别的信息都在点开的详情里。
SPRAY_STATUS_META = {
    "ok":      {"label": "可施药",   "color": "#2e7d32"},
    "narrow":  {"label": "窗口很窄", "color": "#f9a825"},
    "no":      {"label": "不宜施药", "color": "#c62828"},
    "unknown": {"label": "天气未知", "color": "#607d8b"},
}
SPRAY_NARROW_HOURS = 2          # 短于这个时长的窗口算「窄」，得掐着点去


def _today_spray(wx, today=None):
    """从一份天气里挑出「今天」的施药窗口与不宜原因。

    只看今天：逐小时里混着明天后天，不切出来会把明天的窗口当成今天的。
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    hours = [h for h in (wx.get("逐小时") or [])
             if str(h.get("时间") or "").startswith(today)]
    wins = spray_windows({"逐小时": hours}, min_hours=1, top=2)
    reason = ""
    if not wins:
        # 没窗口时告诉用户「主要卡在哪一条」，比只说「不宜」有用得多
        reasons = Counter()
        for h in hours:
            ok, block = spray_hour_ok(h)
            if not ok and block:
                reasons[_block_category(block)] += 1
        reason = reasons.most_common(1)[0][0] if reasons else "今天已无可用时段"
    return hours, wins, reason


def plot_spray_status(plot, lon, lat, today=None):
    """单个地块：今天能不能打药（天气走 30 分钟缓存，重复看不会重复请求）"""
    if lon is None or lat is None:
        return {"plot": plot, "status": "unknown", "label": "无坐标", "reason": "地块还没有落点"}
    try:
        wx = fetch_weather(lon=lon, lat=lat)
    except Exception as e:
        log.warning("地块 %s 查天气失败：%s", plot, e)
        return {"plot": plot, "status": "unknown", "label": "查询失败", "reason": str(e)[:40]}
    if wx.get("错误"):
        return {"plot": plot, "status": "unknown", "label": "天气不可用",
                "reason": str(wx["错误"])[:40]}
    hours, wins, reason = _today_spray(wx, today)
    cur = wx.get("当前") or {}
    base = {"plot": plot, "lon": float(lon), "lat": float(lat),
            "source": wx.get("来源") or "",
            "now": {"温度": cur.get("温度"), "风速": cur.get("风速"), "降水": cur.get("降水")}}
    if not hours:
        return dict(base, status="unknown", label="无逐小时数据", reason="接口没返回今天的逐小时")
    if wins:
        w = wins[0]
        narrow = w["时长小时"] < SPRAY_NARROW_HOURS
        return dict(base, status=("narrow" if narrow else "ok"),
                    label="%s–%s" % (str(w["开始"])[11:16], str(w["结束"])[11:16]),
                    window=w, reason="")
    return dict(base, status="no", label="今天不宜施药", reason=reason)


class PlotWeatherThread(QThread):
    """逐个地块查天气算窗口。

    网络请求慢（每块地一次），必须放后台，否则地块一多界面就卡死。
    """
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(list)

    def __init__(self, plots, parent=None):
        super().__init__(parent)
        self.plots = list(plots or [])

    def run(self):
        out = []
        total = len(self.plots)
        for i, item in enumerate(self.plots):
            if self.isInterruptionRequested():
                break
            name, lon, lat = item
            self.progress.emit(i + 1, total, name)
            try:
                out.append(plot_spray_status(name, lon, lat))
            except Exception as e:
                log.warning("地块 %s 查天气异常：%s", name, e)
                out.append({"plot": name, "status": "unknown", "label": "查询异常",
                            "reason": str(e)[:40]})
        self.done.emit(out)


def disease_weather_risk(wx):
    """把未来天气换算成病害流行风险（依据写在明细里，不做黑箱结论）"""
    days = (wx.get("逐日") or [])[:3]
    hours = (wx.get("逐小时") or [])[:72]
    if not days:
        return {}
    rainy_days = sum(1 for d in days
                     if (d.get("降雨概率%") or 0) > 60 or (d.get("降雨量mm") or 0) > 5)
    storm = any(((d.get("降雨量mm") or 0) > 25) or ((d.get("降雨概率%") or 0) > 80)
                for d in days)
    temps = [d for d in days if d.get("最高温") is not None]
    mean_high = (sum(d["最高温"] for d in temps) / len(temps)) if temps else None
    wet_hours = sum(1 for h in hours if (h.get("湿度") or 0) >= 90
                    or (h.get("降雨量") or 0) > 0.2)
    out = {}
    # 稻瘟病：主判据是「连阴雨 + 适温」——适温高湿是流行的有利条件，
    # 但连阴雨（≥2 个雨天）本身就已经构成高湿环境，不再额外要求逐小时高湿时长，
    # 逐小时高湿只作为「没有连阴雨但持续高湿」时的补充判据。
    if mean_high is not None:
        if 18 <= mean_high <= 30 and rainy_days >= 2:
            out["稻瘟病"] = {"等级": "高",
                             "依据": "未来 3 天有 %d 天降雨、日均高温 %.0f℃，"
                                     "高湿时段 %d 小时——典型流行条件" % (rainy_days, mean_high, wet_hours)}
        elif 18 <= mean_high <= 30 and (rainy_days >= 1 or wet_hours >= 24):
            out["稻瘟病"] = {"等级": "中",
                             "依据": "有降雨或持续高湿（%d 小时）且温度适宜（日均高温 %.0f℃）"
                                     % (wet_hours, mean_high)}
        else:
            out["稻瘟病"] = {"等级": "低",
                             "依据": "未来 3 天降雨偏少或温度不适宜（日均高温 %s）"
                                     % ("%.0f℃" % mean_high if mean_high else "未知")}
    # 白叶枯病：台风暴雨 + 高温
    if storm and (mean_high or 0) >= 25:
        out["白叶枯病"] = {"等级": "高",
                           "依据": "未来 3 天有暴雨/强降雨过程且高温（%.0f℃）——风雨后伤口易侵染" % mean_high}
    elif storm:
        out["白叶枯病"] = {"等级": "中", "依据": "有强降雨过程，注意风雨后查田"}
    else:
        out["白叶枯病"] = {"等级": "低", "依据": "未来 3 天无强降雨过程"}
    # 胡麻斑病：高温高湿
    if (mean_high or 0) >= 28 and wet_hours >= 18:
        out["胡麻斑病"] = {"等级": "中", "依据": "高温高湿（%.0f℃、高湿 %d 小时），缺钾田块易发" % (mean_high, wet_hours)}
    else:
        out["胡麻斑病"] = {"等级": "低", "依据": "温湿度条件一般"}
    return out


def weather_plot_alert(wx, days=14):
    """天气风险 × 地块检出的联动判断：有检出 + 未来有利 = 真正要盯的地块"""
    rows = plot_risk_rows(days) or []
    risk = disease_weather_risk(wx)
    hot = [k for k, v in risk.items() if v.get("等级") == "高"]
    if not rows or not hot:
        return {}
    hits = []
    for r in rows:
        named = [k for k in hot if k in (r.get("类别分布") or {})]
        if named:
            hits.append({"地块": r["plot"], "风险分": r["风险分"],
                         "检出过的病害": named, "最近": r["最近时间"],
                         "建议": "该地块近 %d 天检出过%s，未来天气有利于其流行，建议优先查田并安排保护性施药"
                                 % (days, "、".join(named))})
    return {"未来高风险病害": hot, "需要关注的地块": hits} if hits else {"未来高风险病害": hot}


# ── 极端天气预警（台风大风 / 暴雨 / 寒露风 / 高温热害）──
EXTREME_GUST_WARN = 17.2       # 阵风 ≥8 级（m/s）
EXTREME_GUST_SEVERE = 24.5     # 阵风 ≥10 级
EXTREME_RAIN_WARN = 50.0       # 24 小时降水量 ≥50mm（暴雨）
EXTREME_RAIN_SEVERE = 100.0    # ≥100mm（大暴雨）
COLD_DEW_TEMP = 22.0           # 寒露风：日均温 ≤22℃（粳稻；籼稻约 23℃）
COLD_DEW_DAYS = 3
HEAT_WAVE_TEMP = 35.0          # 高温热害：日最高温 ≥35℃
HEAT_WAVE_DAYS = 3
MICROCLIMATE_NOTE = ("坐标天气是网格预报（几公里分辨率），田块微气候（地形、林带、水层、"
                     "施药方式）会造成偏差；农事与用药决策请以田间实测和当地植保站指导为准。")


def extreme_weather_alerts(wx):
    """未来 3 天的极端天气预警。阈值都是常量，看得见也改得动，不做黑箱判断。"""
    daily = (wx.get("逐日") or [])[:3]
    out = []
    if not daily:
        return out
    gusts = [d.get("最大阵风") for d in daily if d.get("最大阵风") is not None]
    if gusts:
        g = max(gusts)
        if g >= EXTREME_GUST_SEVERE:
            out.append({"类型": "大风/台风", "等级": "严重", "数值": "%.1f m/s" % g,
                        "依据": "未来 3 天阵风最大 %.1f m/s（10 级以上）" % g,
                        "应对": "暂停田间作业与施药；加固育秧棚与防虫网；提前清沟排水；过后扶正倒伏稻株并防细菌性病害"})
        elif g >= EXTREME_GUST_WARN:
            out.append({"类型": "大风", "等级": "警告", "数值": "%.1f m/s" % g,
                        "依据": "未来 3 天阵风最大 %.1f m/s（8 级以上）" % g,
                        "应对": "不要施药（飘移与药害风险）；等风停后再打"})
    rains = [d.get("降雨量mm") or 0 for d in daily]
    if rains:
        r = max(rains)
        if r >= EXTREME_RAIN_SEVERE:
            out.append({"类型": "暴雨", "等级": "严重", "数值": "%.0f mm" % r,
                        "依据": "未来 3 天单日降水量最大 %.0f mm（大暴雨）" % r,
                        "应对": "提前疏通排水沟防淹苗；雨后 1-2 天查白叶枯病与细菌性条斑病，发现中心病株立即用药；已施药地块注意药液被冲刷"})
        elif r >= EXTREME_RAIN_WARN:
            out.append({"类型": "暴雨", "等级": "警告", "数值": "%.0f mm" % r,
                        "依据": "未来 3 天单日降水量最大 %.0f mm（暴雨）" % r,
                        "应对": "做好排水；避开降雨前后施药"})
    means, months = [], set()
    for d in daily:
        h, l = d.get("最高温"), d.get("最低温")
        if h is not None and l is not None:
            means.append((h + l) / 2.0)
        months.add(str(d.get("日期", ""))[5:7])
    if (len(means) >= COLD_DEW_DAYS and all(m <= COLD_DEW_TEMP for m in means[:COLD_DEW_DAYS])
            and (months & {"09", "10"})):
        out.append({"类型": "寒露风", "等级": "警告",
                    "数值": "日均温 %.1f℃" % (sum(means[:COLD_DEW_DAYS]) / COLD_DEW_DAYS),
                    "依据": "连续 %d 天日均温 ≤%.0f℃ 且正值 9-10 月抽穗扬花期" % (COLD_DEW_DAYS, COLD_DEW_TEMP),
                    "应对": "灌 5-10cm 深水以水调温；叶面喷磷钾肥提高抗逆；抽穗期遇低温可喷保温剂，降低空秕率"})
    highs = [d.get("最高温") for d in daily if d.get("最高温") is not None]
    if len(highs) >= HEAT_WAVE_DAYS and all(h >= HEAT_WAVE_TEMP for h in highs[:HEAT_WAVE_DAYS]):
        out.append({"类型": "高温热害", "等级": "警告",
                    "数值": "最高温 %.0f℃" % max(highs[:HEAT_WAVE_DAYS]),
                    "依据": "连续 %d 天日最高温 ≥%.0f℃" % (HEAT_WAVE_DAYS, HEAT_WAVE_TEMP),
                    "应对": "灌深水降温、以水调温；避开中午施药；关注花粉活力与结实率"})
    return out


# ── 施药记录与「雨后要不要补施」──
SPRAY_LOG_PATH = APP_DIR / "spray" / "spray_log.json"
# 内吸性/耐冲刷药剂：同样雨量下比触杀性药剂更耐受冲刷
SYSTEMIC_DRUGS = ("吡虫啉", "噻虫嗪", "氯虫苯甲酰胺", "三环唑", "稻瘟灵", "吡唑醚菌酯",
                  "烯啶虫胺", "噻呋酰胺", "苯醚甲环唑", "咪鲜胺", "春雷霉素", "嘧菌酯")
RAIN_WASH_RULES = {"内吸性": (10.0, 40.0), "触杀性": (5.0, 20.0)}   # (药后6h, 药后24h) mm


def load_spray_records():
    try:
        if SPRAY_LOG_PATH.exists():
            data = json.loads(SPRAY_LOG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("读取施药记录失败：%s", e)
    return []


def add_spray_record(plot, drug, cls="", dose="", when=None, note="", by="",
                     source="manual", safe_days=None, detection=None, photo="",
                     plan_from=""):
    """记录一次施药（写操作，由用户在界面上确认后调用）

    比"手工台账"多存三样东西，闭环才成立：
    · detection：这次的识别依据（类别/置信度/定位数）——事后能回答"当时凭什么打的"
    · plan_from/safe_days：方案出处与安全间隔期——能回答"按哪个方案、什么时候能收"
    · photo：施药当时的田况图副本——原图删了也能回溯
    """
    when = when or datetime.now()
    rec = {"time": when.strftime("%Y-%m-%d %H:%M:%S"), "plot": str(plot or "").strip(),
           "drug": str(drug or "").strip(), "cls": str(cls or "").strip(),
           "dose": str(dose or "").strip(), "note": str(note or "").strip(),
           "by": str(by or "").strip(), "source": str(source or "manual"),
           "safe_days": safe_days, "detection": detection or {},
           "photo": str(photo or ""), "plan_from": str(plan_from or ""),
           "recorded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    recs = load_spray_records()
    recs.append(rec)
    try:
        SPRAY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPRAY_LOG_PATH.write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    except OSError as e:
        log.warning("写入施药记录失败：%s", e)
        return None
    log_event("spray_record", plot=rec["plot"], drug=rec["drug"])
    return rec


def treatments_audit():
    """本地防治库体检：逐条列出缺什么。

    为什么要这个：库的价值取决于「每条都能追到源头」。
    缺登记证号、缺间隔期、缺防治适期的条目必须看得见，否则用着用着就变成了猜测。
    """
    out = {"类别数": len(TREATMENT_DB), "药剂条目": 0,
           "有登记证号": 0, "有安全间隔期": 0, "有防治适期": 0, "生物农药": 0,
           "缺登记证号": [], "缺安全间隔期": [], "缺防治适期": [],
           "类别缺防治要点": [], "类别缺依据": []}
    for cls, tr in TREATMENT_DB.items():
        for d in tr.get("drugs", []):
            out["药剂条目"] += 1
            name = "%s / %s" % (cls, d.get("name"))
            reg = str(d.get("reg_no") or "")
            if reg and "待补" not in reg and "未逐一核对" not in reg:
                out["有登记证号"] += 1
            else:
                out["缺登记证号"].append(name)
            if d.get("safe_days"):
                out["有安全间隔期"] += 1
            else:
                out["缺安全间隔期"].append(name)
            if d.get("timing"):
                out["有防治适期"] += 1
            else:
                out["缺防治适期"].append(name)
            if d.get("type") == "生物":
                out["生物农药"] += 1
        meta = tr.get("meta") or {}
        if not meta:
            out["类别缺防治要点"].append(cls)
        elif not meta.get("依据"):
            out["类别缺依据"].append(cls)
    return out


def treatment_plan_rows(cls_name):
    """本地防治库给这个类别的防治方案（药剂带规格/亩用量/安全间隔期/登记依据）"""
    tr = TREATMENT_DB.get(cls_name) or {}
    rows = []
    for d in tr.get("drugs", []):
        rows.append({"名称": d.get("name", ""), "规格": d.get("spec", ""),
                     "亩用量": d.get("dose", ""), "安全间隔期": d.get("safe_days"),
                     "每季最多": d.get("max_per_season"),
                     "登记依据": d.get("source", "")})
    return {"类别": cls_name, "防治指标": tr.get("threshold", ""),
            "农业防治": tr.get("agri", []), "生物防治": tr.get("bio", []),
            "推荐药剂": rows, "来源": "本地防治库 TREATMENT_DB"}


def save_spray_photo(img_bgr, prefix="spray"):
    """存一份施药当时的田况图（768px JPEG），原图删了也能回溯"""
    try:
        if img_bgr is None or not hasattr(img_bgr, "shape"):
            return ""
        h, w = img_bgr.shape[:2]
        if max(h, w) > 768:
            s = 768.0 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)))
        d = SPRAY_LOG_PATH.parent / "photos"
        d.mkdir(parents=True, exist_ok=True)
        p = d / ("%s_%s.jpg" % (prefix, datetime.now().strftime("%Y%m%d_%H%M%S")))
        cv2.imwrite(str(p), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return str(p)
    except Exception as e:
        log.warning("保存施药照片失败：%s", e)
        return ""


def spray_precheck(cls_name="", plot="", drug="", when=None, wx=None):
    """施药前四条校验：农药合规 / 天气窗口 / 安全间隔期 / 重复施药

    这就是"闭环"里最值钱的一步：登记之前先把能拦住的问题拦住。
    任何一条不过都不阻止登记（现实里农户可能已经打了），但会明确写清风险。
    """
    when = when or datetime.now()
    out = {"时间": when.strftime("%Y-%m-%d %H:%M"), "地块": plot or "未填写",
           "防治对象": cls_name or "", "药剂": drug or ""}
    # ① 农药合规
    hits = pesticide_hits(drug) if drug else []
    out["农药合规"] = {"通过": not hits, "命中": hits,
                     "说明": ("命中禁用/停用名单：%s，禁止使用" % "、".join(hits)) if hits
                             else "未命中国家禁用/停用名单"}
    # ② 天气窗口（取未来第一个小时作为"现在"）
    try:
        w = wx or fetch_weather(city=plot or None, days=3)
        hours = w.get("逐小时") or []
        if hours:
            ok, block = spray_hour_ok(hours[0])
            wins = spray_windows(w, min_hours=2, top=1)
            out["天气窗口"] = {"当前适宜": bool(ok), "不宜原因": block,
                             "当前": {"时间": hours[0].get("时间"), "温度": hours[0].get("温度"),
                                      "风速": hours[0].get("风速"),
                                      "降雨概率": hours[0].get("降雨概率")},
                             "最近可用窗口": wins[0] if wins else None,
                             "极端天气": extreme_weather_alerts(w)}
        else:
            out["天气窗口"] = {"当前适宜": None, "说明": "没有取到逐小时数据"}
    except Exception as e:
        out["天气窗口"] = {"当前适宜": None, "说明": "天气查询失败：%s" % e}
    # ③ 安全间隔期
    safe = drug_safe_days(drug)
    out["安全间隔期"] = {"天数": safe,
                       "说明": ("距收获需 ≥%s 天，请注意采收时间" % safe) if safe
                               else "本地库没有该药剂的间隔期数据，请以标签为准"}
    # ④ 重复施药 / 抗性
    recent = []
    for r in load_spray_records():
        if plot and plot not in (r.get("plot") or ""):
            continue
        if drug and drug[:4] and drug[:4] not in (r.get("drug") or ""):
            continue
        try:
            days_ago = (when - datetime.strptime(r["time"][:19], "%Y-%m-%d %H:%M:%S")).days
        except (ValueError, KeyError):
            continue
        if 0 <= days_ago <= 10:
            recent.append({"时间": r.get("time"), "距今天数": days_ago,
                           "药剂": r.get("drug"), "地块": r.get("plot") or "未填写"})
    out["重复施药"] = {"近期同药剂次数": len(recent), "记录": recent,
                     "说明": ("同地块同药剂 %d 天内已用过 %d 次，注意轮换药剂、避免抗性"
                              % (10, len(recent))) if recent else "近 10 天没有同药剂记录"}
    blocked = (not out["农药合规"]["通过"]) or (out["天气窗口"].get("当前适宜") is False)
    out["是否建议现在施药"] = not blocked
    if not out["农药合规"]["通过"]:
        out["结论"] = "⛔ 不建议现在施药：" + out["农药合规"]["说明"]
    elif out["天气窗口"].get("当前适宜") is False:
        out["结论"] = "⚠ 当前天气不宜施药：" + str(out["天气窗口"].get("不宜原因"))
    else:
        out["结论"] = "✅ 四条校验通过，可以按方案施药"
    return out

def drug_is_systemic(drug):
    d = str(drug or "")
    return any(k in d for k in SYSTEMIC_DRUGS)


def drug_safe_days(drug):
    """从本地防治库查该药剂的安全间隔期（查不到返回 None）"""
    d = str(drug or "")
    for tr in TREATMENT_DB.values():
        for item in tr.get("drugs", []):
            name = item.get("name", "")
            if d and name and (d in name or name in d or d[:3] in name):
                return item.get("safe_days")
    return None


def rain_between(wx, start_str, hours):
    """从 start_str 起的 hours 小时内累计降水量（用过去逐小时实测数据，不是预报）"""
    past = wx.get("过去逐小时") or []
    if not past:
        return None
    start = None
    s = str(start_str or "")
    for fmt, n in (("%Y-%m-%dT%H", 13), ("%Y-%m-%d %H:%M", 16), ("%Y-%m-%d", 10)):
        try:
            start = datetime.strptime(s[:n], fmt)
            break
        except ValueError:
            continue
    if start is None:
        return None
    # 降水是逐小时桶：把施药时刻向下取整到整点，并把施药所在的那个小时也计入。
    # 取舍理由：宁可略微高估雨量（提示补施）也不要漏报——漏报的代价是白打一遍药。
    start = start.replace(minute=0, second=0, microsecond=0)
    total, hit = 0.0, False
    for h in past:
        try:
            t = datetime.strptime(str(h.get("时间"))[:13], "%Y-%m-%dT%H")
        except ValueError:
            continue
        delta = (t - start).total_seconds() / 3600.0
        if 0 <= delta < hours:
            total += float(h.get("降雨量") or 0)
            hit = True
    return total if hit else None


# ── 收获期与「最早可收日期」──
HARVEST_PATH = APP_DIR / "spray" / "harvest.json"


def load_harvest_dates():
    """{地块: {date, note, updated}} —— 计划什么时候收"""
    try:
        if HARVEST_PATH.exists():
            data = json.loads(HARVEST_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("读取收获期失败：%s", e)
    return {}


def set_harvest_date(plot, date_str, note=""):
    """登记某个地块的计划收获期（写操作，人点触发）"""
    plot = (plot or "").strip() or "未命名地块"
    try:
        datetime.strptime(str(date_str).strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None
    data = load_harvest_dates()
    data[plot] = {"date": str(date_str).strip()[:10], "note": str(note or "").strip(),
                  "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        HARVEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        HARVEST_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        log.warning("保存收获期失败：%s", e)
        return None
    log_event("harvest_set", plot=plot, date=data[plot]["date"])
    return data[plot]


def earliest_harvest(plot=None, days=90):
    """按「末次施药 + 该药剂安全间隔期」推算最早可收日期，并与登记收获期比对

    这是闭环的最后一环：打完药之后，农户真正关心的下一个问题就是"什么时候能收"。
    缺安全间隔期数据时不推算、不猜——直接说明缺什么。
    """
    cut = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    per_plot = {}
    for r in load_spray_records():
        if (r.get("time") or "")[:10] < cut:
            continue
        p = (r.get("plot") or "").strip() or "未命名地块"
        per_plot.setdefault(p, []).append(r)
    harvest = load_harvest_dates()
    targets = [plot] if plot else sorted(set(per_plot) | set(harvest))
    out = []
    for p in targets:
        rs = sorted(per_plot.get(p, []), key=lambda x: x.get("time") or "")
        last = rs[-1] if rs else None
        hd = (harvest.get(p) or {}).get("date")
        item = {"地块": p, "登记收获期": hd, "施药次数": len(rs)}
        if last:
            drug = last.get("drug") or ""
            safe = last.get("safe_days") or drug_safe_days(drug)
            item["末次施药"] = (last.get("time") or "")[:10]
            item["末次药剂"] = drug
            item["安全间隔期"] = safe
            if safe:
                try:
                    base = datetime.strptime((last.get("time") or "")[:10], "%Y-%m-%d")
                    eh = base + timedelta(days=int(safe))
                    item["最早可收日期"] = eh.strftime("%Y-%m-%d")
                    item["距今天数"] = (datetime.now() - eh).days
                except ValueError:
                    item["最早可收日期"] = None
            else:
                item["最早可收日期"] = None
                item["缺什么"] = "该药剂没有安全间隔期数据（库里 %d 种药剂待补），无法推算最早可收日期" % len(drug_safety_gaps())
        else:
            item["最早可收日期"] = None
        eh_s = item.get("最早可收日期")
        if hd and eh_s:
            if hd < eh_s:
                try:
                    gap = (datetime.strptime(eh_s, "%Y-%m-%d") - datetime.strptime(hd, "%Y-%m-%d")).days
                except ValueError:
                    gap = 0
                item["结论"] = "⚠ 登记收获期 %s 早于最早可收日期 %s，存在农残风险，建议推迟 %d 天" % (hd, eh_s, gap)
                item["是否安全"] = False
            else:
                item["结论"] = "✅ 登记收获期 %s 在安全间隔期之后，可按计划收" % hd
                item["是否安全"] = True
        elif hd:
            item["结论"] = "已登记收获期 %s；%s" % (hd, item.get("缺什么") or "缺少施药记录，无法推算" )
            item["是否安全"] = None
        elif eh_s:
            item["结论"] = "最早可收日期 %s（%s）" % (eh_s, "已过" if item["距今天数"] >= 0 else "还差 %d 天" % -item["距今天数"])
            item["是否安全"] = None
        else:
            item["结论"] = "尚未登记收获期，也没有可推算的依据"
            item["是否安全"] = None
        out.append(item)
    return out

def spray_rain_check(days=7, wx=None):
    """逐条施药记录判断「药后有没有被雨冲掉」，给出要不要补施的建议"""
    cut = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recs = [r for r in load_spray_records() if (r.get("time") or "")[:10] >= cut]
    if not recs:
        return []
    out = []
    cache = {}
    for r in recs:
        plot = r.get("plot") or ""
        w = wx
        if w is None and plot and plot_coords(plot)[0] is not None:
            if plot not in cache:
                cache[plot] = fetch_weather(city=plot, days=3)
            w = cache[plot]
        r6 = rain_between(w, r.get("time"), 6) if w else None
        r24 = rain_between(w, r.get("time"), 24) if w else None
        kind = "内吸性" if drug_is_systemic(r.get("drug")) else "触杀性"
        lim6, lim24 = RAIN_WASH_RULES[kind]
        # 把"没数据"的三种情况分开说：刚打完还没结算 / 超出历史范围 / 确实缺数据。
        # 混成一句"无法判断"会让人以为功能坏了（真实踩过）。
        hours_since = None
        try:
            hours_since = (datetime.now() - datetime.strptime(
                r["time"][:19], "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0
        except (ValueError, KeyError):
            pass
        if r6 is None and r24 is None:
            if hours_since is not None and hours_since < 6:
                conclusion = "等待结算"
                need_h = max(1, int(6 - hours_since) + 1)
                advice = ("刚施药 %.1f 小时，降水数据还没结算（逐小时数据有延迟）；"
                          "约 %d 小时后再查就能给出确切判断" % (hours_since, need_h))
            elif hours_since is not None and hours_since > 7 * 24:
                conclusion, advice = "超出历史范围", "施药已超过 7 天，接口只提供最近 7 天的逐小时降水"
            else:
                conclusion, advice = "无法判断", "缺少该时段的历史降水数据（可能是缓存过期），可稍后重试"
        elif (r6 is not None and r6 >= lim6) or (r24 is not None and r24 >= lim24):
            conclusion = "建议补施"
            advice = ("药后 6 小时降雨 %.1f mm、24 小时 %.1f mm，超过%s药剂的耐受阈值（%.0f/%.0f mm）；"
                      "保护性药剂被冲刷后基本失效，建议天晴后按标签剂量补施一次" % (r6 or 0, r24 or 0, kind, lim6, lim24))
        elif (r6 or 0) > 0 or (r24 or 0) > 0:
            conclusion = "影响不大"
            advice = ("药后 6 小时降雨 %.1f mm、24 小时 %.1f mm，未超过%s药剂阈值（%.0f/%.0f mm），一般不用补施；"
                      "若为触杀性药剂且雨势集中，可隔天查一次防效" % (r6 or 0, r24 or 0, kind, lim6, lim24))
        else:
            conclusion, advice = "药后无雨", "药后 24 小时内没有降水，药效未受影响"
        try:
            days_ago = (datetime.now() - datetime.strptime(r["time"][:19], "%Y-%m-%d %H:%M:%S")).days
        except (ValueError, KeyError):
            days_ago = None
        out.append({"地块": plot or "未填写", "药剂": r.get("drug") or "",
                    "防治对象": r.get("cls") or "", "施药时间": r.get("time"),
                    "距今天数": days_ago, "药剂类型": kind,
                    "药后6小时雨量mm": None if r6 is None else round(r6, 1),
                    "药后24小时雨量mm": None if r24 is None else round(r24, 1),
                    "结论": conclusion, "建议": advice,
                    "安全间隔期": r.get("safe_days") or drug_safe_days(r.get("drug")),
                    "来源": r.get("source") or "manual",
                    "识别依据": (r.get("detection") or {}).get("cls") or r.get("cls") or "",
                    "当时置信度": (r.get("detection") or {}).get("conf"),
                    "有照片存档": bool(r.get("photo")),
                    "备注": r.get("note") or ""})
    return out

def weather_summary_text(wx, windows=None):
    """一段可直接给人/给模型看的天气结论"""
    if "错误" in wx:
        return wx["错误"]
    unit = {"温度": "℃", "湿度": "%", "风速": "m/s", "降水": "mm"}
    cur = wx.get("当前") or {}
    lines = ["%s（%s）天气：%s" % (wx.get("地点") or "地块", wx.get("坐标"),
                                  "、".join("%s %s%s" % (k, v, unit.get(k, ""))
                                            for k, v in cur.items() if v is not None))]
    for d in (wx.get("逐日") or [])[:3]:
        lines.append("  · %s：%s~%s℃，降雨 %s mm / 概率 %s%%"
                     % (d["日期"], d.get("最低温"), d.get("最高温"),
                        d.get("降雨量mm"), d.get("降雨概率%")))
    alerts = extreme_weather_alerts(wx)
    for a in alerts:
        lines.append("⚠ 极端天气预警（%s）：%s——%s" % (a["等级"], a["依据"], a["应对"]))
    risk = disease_weather_risk(wx)
    if risk:
        lines.append("病害流行风险：" + "；".join(
            "%s %s（%s）" % (k, v["等级"], v["依据"]) for k, v in risk.items()))
    windows = windows if windows is not None else spray_windows(wx)
    if windows:
        w = windows[0]
        lines.append("最佳施药窗口：%s ~ %s（%d 小时，平均评分 %.0f）——%s"
                     % (w["开始"][5:16].replace("T", " "), w["结束"][11:16],
                        w["时长小时"], w["平均评分"], w["建议"]))
    else:
        lines.append("未来 48 小时没有达到及格线的施药窗口（雨/风/高温不满足），建议等天气转好")
    return "\n".join(lines)


def _weather_args(args):
    """工具的公共参数解析：城市名 / 地块名 / 经纬度"""
    try:
        lon = float(args["lon"]) if args.get("lon") is not None else None
        lat = float(args["lat"]) if args.get("lat") is not None else None
    except (TypeError, ValueError):
        lon = lat = None
    place = str(args.get("city") or args.get("plot") or "").strip()
    return place or None, lon, lat


def _tool_get_weather(args):
    """天气：支持城市名 / 地块名 / 经纬度，附病害流行风险与最佳施药窗口"""
    place, lon, lat = _weather_args(args)
    try:
        days = max(3, min(int(args.get("days") or 3), 7))
    except (TypeError, ValueError):
        days = 3
    wx = fetch_weather(city=place, lon=lon, lat=lat, days=days)
    if "错误" in wx:
        return wx["错误"]
    out = {"地点": wx.get("地点"), "坐标": wx.get("坐标"), "当前": wx.get("当前"),
           "未来数日": (wx.get("逐日") or [])[:days],
           "病害流行风险": disease_weather_risk(wx),
           "最佳施药窗口": spray_windows(wx)[:1],
           "来源": wx.get("来源"), "生成时间": wx.get("生成时间")}
    if wx.get("提示"):
        out["提示"] = wx["提示"]
    alerts = extreme_weather_alerts(wx)
    if alerts:
        out["极端天气预警"] = alerts
    out["口径说明"] = MICROCLIMATE_NOTE
    alert = weather_plot_alert(wx)
    if alert:
        out["地块联动"] = alert
    log_event("tool_get_weather", place=out.get("地点"), risk=list(out["病害流行风险"]))
    return json.dumps(out, ensure_ascii=False)


def _tool_spray_window(args):
    """施药窗口：逐小时评分后给出可执行的时段建议（含受限原因）"""
    place, lon, lat = _weather_args(args)
    try:
        hours = max(12, min(int(args.get("hours") or 48), 168))
    except (TypeError, ValueError):
        hours = 48
    wx = fetch_weather(city=place, lon=lon, lat=lat, days=3, hours=hours)
    if "错误" in wx:
        return wx["错误"]
    windows = spray_windows(wx, min_hours=2, top=3)
    bad = Counter()
    for h in (wx.get("逐小时") or []):
        ok, block = spray_hour_ok(h)
        if not ok:
            bad[_block_category(block)] += 1
    payload = {"地点": wx.get("地点"), "坐标": wx.get("坐标"),
               "判定规则": "硬约束（一票否决）：温度 %.0f-%.0f℃、风速≤%.0f m/s、"
                           "降雨概率≤%d%%、无降水、且只算 %02d-%02d 点可作业时段；"
                           "连续 ≥2 小时满足才成一个窗口，评分只用于排序"
                           % (SPRAY_TEMP_OK[0], SPRAY_TEMP_OK[1], SPRAY_WIND_OK,
                              SPRAY_RAIN_PROB_OK, SPRAY_DAY_HOURS[0], SPRAY_DAY_HOURS[1]),
               "窗口": windows,
               "受限原因Top": [{"原因": k, "小时数": v} for k, v in bad.most_common(3)],
               "来源": wx.get("来源"), "生成时间": wx.get("生成时间")}
    alerts = extreme_weather_alerts(wx)
    if alerts:
        payload["极端天气预警"] = alerts
        if any(a.get("类型") in ("大风/台风", "暴雨") for a in alerts):
            payload["提醒"] = "极端天气期间不要施药：药液会被吹散或冲刷，等天气转好再安排"
    payload["口径说明"] = MICROCLIMATE_NOTE
    if not windows:
        payload["结论"] = "未来 %d 小时没有达到及格线的施药窗口，建议等天气转好" % hours
    log_event("tool_spray_window", place=payload.get("地点"), windows=len(windows))
    return json.dumps(payload, ensure_ascii=False)


def _tool_map_overview(args):
    data = map_overview_data()
    if "错误" in data:
        return "地块地图不可用：%s（请确认 rice_geo.py 与 geo/ 目录存在）" % data["错误"]
    if not data["记录总数"]:
        return ("地块地图还是空的：还没有任何记录。"
                "可以在「地块地图」页导入带 GPS 的田间照片，或用「记录到地图」把识别结果存档。")
    return json.dumps(data, ensure_ascii=False)


def _tool_plot_risk(args):
    days = args.get("days")
    top = int(args.get("top") or 5)
    rows = plot_risk_rows(days)
    if rows is None:
        return "地块地图不可用（缺少 rice_geo.py 或依赖）"
    if not rows:
        return "最近没有已落点的记录，无法排序" if days else "还没有已落点的记录"
    out = []
    for r in rows[:top]:
        out.append({"地块": r["plot"], "风险分": r["风险分"], "点位数": r["点数"],
                    "主要问题": r["主要问题"], "类别分布": r["类别分布"],
                    "最近时间": r["最近时间"],
                    "证据": [{"时间": x.get("shot_at", "")[:16],
                              "类别": x.get("cls") or "未检出",
                              "置信度": round(float(x.get("conf") or 0), 2)}
                             for x in r["_recs"][:3]]})
    return json.dumps({"统计范围": ("最近 %s 天" % days) if days else "全部记录",
                       "口径": "风险分 = Σ(时间衰减 0.5^(天/14) × 置信度系数) × 重复出现加权，归一到 0-100",
                       "排序": out}, ensure_ascii=False)


def _tool_map_hotspots(args):
    data = map_hotspots_data(cell_m=int(args.get("cell_m") or MAP_CELL_M),
                             days=int(args.get("days") or 60),
                             top=int(args.get("top") or 5))
    if "错误" in data:
        return "地块地图不可用：%s" % data["错误"]
    if not data["热点"]:
        return "统计范围内没有已落点的记录"
    return json.dumps(data, ensure_ascii=False)


def _tool_nearby_records(args):
    try:
        lon = float(args["lon"])
        lat = float(args["lat"])
    except (KeyError, TypeError, ValueError):
        return json.dumps({"错误": "需要提供有效的 lon / lat"}, ensure_ascii=False)
    data = nearby_records_data(lon, lat, radius_m=float(args.get("radius_m") or 500),
                               days=args.get("days"))
    if "错误" in data:
        return "地块地图不可用：%s" % data["错误"]
    if not data["命中"]:
        return json.dumps({"查询半径_米": data["查询半径_米"], "命中": 0,
                           "结论": "这个点附近没有历史记录"}, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def _tool_patrol_plan(args):
    data = patrol_plan_data(max_stops=int(args.get("max_stops") or 5), days=args.get("days"))
    if "错误" in data:
        return "地块地图不可用：%s" % data["错误"]
    if not data["顺序"]:
        return "还没有已落点的记录，排不出巡田路线"
    return json.dumps(data, ensure_ascii=False)


def _tool_map_digest(args):
    return map_digest_text(days=int(args.get("days") or 30))


def _tool_spray_log(args):
    """施药记录 + 药后被雨冲刷的判断（回答「要不要补施」）"""
    try:
        days = max(1, min(int(args.get("days") or 14), 60))
    except (TypeError, ValueError):
        days = 14
    plot = str(args.get("plot") or "").strip()
    cut = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recs = [r for r in load_spray_records() if (r.get("time") or "")[:10] >= cut]
    if plot:
        recs = [r for r in recs if plot in (r.get("plot") or "")]
    if not recs:
        return json.dumps({"查询范围": "最近 %d 天" % days, "记录数": 0,
                           "结论": "这期间没有施药记录。可以在首页「天气与施药窗口」卡片里点「记录施药」登记一次，"
                                   "登记后我就能判断药后有没有被雨冲掉。"}, ensure_ascii=False)
    checks = spray_rain_check(days=days)
    if plot:
        checks = [c for c in checks if plot in (c.get("地块") or "")]
    need = [c for c in checks if c.get("结论") == "建议补施"]
    payload = {"查询范围": "最近 %d 天" % days, "记录数": len(recs),
               "记录": [{"地块": r.get("plot") or "未填写", "药剂": r.get("drug") or "",
                         "防治对象": r.get("cls") or "", "时间": r.get("time"),
                         "备注": r.get("note") or ""} for r in recs],
               "雨后冲刷判断": checks,
               "结论": ("有 %d 次施药可能被雨冲刷，建议补施" % len(need)) if need
                       else "药后降水均未超过耐受阈值，暂不需要补施",
               "收获期与最早可收": earliest_harvest(plot or None),
               "口径说明": MICROCLIMATE_NOTE}
    log_event("tool_spray_log", days=days, records=len(recs), need_respray=len(need))
    return json.dumps(payload, ensure_ascii=False)

def run_agent_tool(name, args, image=None, boxes=None):
    """执行一个工具调用，返回给模型的文本结果；异常也返回文本，不抛给线程"""
    try:
        if name == "verify_with_vision":
            return _tool_verify_with_vision(args, image, boxes)
        if name == "get_treatment":
            return _tool_get_treatment(args)
        if name == "check_pesticide":
            return _tool_check_pesticide(args)
        if name == "list_reports":
            return _tool_list_reports(args)
        if name == "feedback_stats":
            return _tool_feedback_stats(args)
        if name == "re_detect":
            return _tool_re_detect(args, image)
        if name == "get_weather":
            return _tool_get_weather(args)
        if name == "spray_window":
            return _tool_spray_window(args)
        if name == "spray_log":
            return _tool_spray_log(args)
        if name == "batch_scan":
            return _tool_batch_scan(args)
        if name == "daily_digest":
            return _tool_daily_digest(args)
        if name == "list_sessions":
            return _tool_list_sessions(args)
        if name == "get_session":
            return _tool_get_session(args)
        if name == "map_overview":
            return _tool_map_overview(args)
        if name == "plot_risk":
            return _tool_plot_risk(args)
        if name == "map_hotspots":
            return _tool_map_hotspots(args)
        if name == "nearby_records":
            return _tool_nearby_records(args)
        if name == "patrol_plan":
            return _tool_patrol_plan(args)
        if name == "map_digest":
            return _tool_map_digest(args)
        return "未知工具：%s" % name
    except Exception as e:
        log.exception("工具执行失败: %s", name)
        return "工具 %s 执行失败：%s" % (name, e)


# ── 结构化输出校验（诊断卡） ──────────────────────────────────
def _local_drug_names():
    names = []
    for tr in TREATMENT_DB.values():
        for d in tr.get("drugs", []):
            if d.get("name"):
                names.append(d["name"])
    return names


def validate_diagnosis(card, severity=None):
    """把大模型给的诊断卡逐项对本地库校验，返回 (校验后的 card, 警告列表)。

    校验规则（技术规范性：AI 输出必须可核验）：
      · 药剂名命中本地防治库   → source=local
      · 命中禁用/停用名单       → source=banned（标红，禁止使用）
      · 都不命中               → source=unknown（提示人工核实登记与用量）
      · 自报严重度与图像检测不符 → 记警告，以现场为准
    """
    card = dict(card or {})
    warnings = []
    local = _local_drug_names()
    chemicals = []
    for c in (card.get("chemicals") or []):
        if isinstance(c, str):
            c = {"name": c}
        name = str(c.get("name", "")).strip()
        hits = pesticide_hits(name)
        if hits:
            src, note = "banned", "命中禁用/停用名单：%s" % "、".join(hits)
            warnings.append("诊断卡中的「%s」属禁用或停止使用农药，禁止使用" % name)
        elif any(name and (name in ln or ln in name) for ln in local):
            src, note = "local", "命中本地防治库"
        else:
            src, note = "unknown", "本地防治库未收录，需人工核实登记与用量"
        chemicals.append({"name": name, "dose": c.get("dose", ""),
                          "reason": c.get("reason", ""), "source": src, "note": note})
    card["chemicals"] = chemicals
    lv = (severity or {}).get("level")
    claimed = str(card.get("severity_level", "")).strip()
    if lv and claimed and claimed != lv and lv != "无":
        warnings.append("大模型判断严重度为「%s」，图像检测为「%s」，以现场实际情况为准"
                        % (claimed, lv))
    card["_warnings"] = warnings
    return card, warnings


# ── 运行台账：把「用了多少、效果如何」变成可导出的数字 ──────────────
# ══════════════════════════════════════════════════════════════
# 无人值守流水线（自动巡检 / 批处理）
# --------------------------------------------------------------
# 这是"更像 agent"的关键一块：不等人提问。
#   监听到新照片 → 检测 → （低置信度/病害类）自动视觉复核 → 会话落库 → 异常才提醒
# 复用界面上同一套模型、合规名单与会话存储，结果全部可追溯。
WATCH_DIR = APP_DIR / "inbox"
WATCH_RESULTS = APP_DIR / "watch" / "results.csv"
WATCH_COLUMNS = ["time", "file", "cls", "conf", "boxes", "severity",
                 "verify", "verify_agree", "phone_cls", "phone_agree", "note"]
# phone_cls / phone_agree：小程序或手机端自己的识别结论，以及它与电脑端是否一致。
# 台账里必须看得见——否则"手机端说是稻瘟病、电脑端说是白叶枯病"这种事只躺在审计日志里。


# ── 小程序/手机端结果同步（中转服务 → 收件箱） ──
RELAY_STATE_PATH = APP_DIR / "watch" / "relay_state.json"


def phone_sidecar_path(image_path):
    """手机端结论的伴随文件：与图片同名 + .phone.json"""
    return Path(str(image_path) + ".phone.json")


def read_phone_sidecar(image_path):
    """读手机端随图上传的识别结论；没有就返回 None（普通照片走原有流程）"""
    p = phone_sidecar_path(image_path)
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception as e:
        log.debug("读手机端结论失败：%s", e)
    return None


def _relay_state():
    try:
        if RELAY_STATE_PATH.exists():
            return json.loads(RELAY_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"since": "", "base": "", "synced": 0}


def _save_relay_state(state):
    try:
        RELAY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RELAY_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    except OSError as e:
        log.warning("保存同步状态失败：%s", e)


def _write_phone_sidecar(image_path, rec, rid=""):
    """把手机端结论写成图片的伴随文件（图片已拉过、结论后补传时也会刷新）"""
    # 来源要听调用方的：手机直传写「手机直传」，中转拉下来的写「小程序」。
    # 原来这里写死成「小程序」，手机传的图也被标成小程序，来源就分不出来了。
    side = {"id": rid or rec.get("id") or "",
            "来源": rec.get("来源") or "小程序",
            "label": rec.get("label") or "", "score": rec.get("score"),
            "boxes": rec.get("boxes"), "model": rec.get("model") or "",
            "lat": rec.get("lat"), "lng": rec.get("lng"),
            "address": rec.get("address") or "", "note": rec.get("note") or "",
            "device": rec.get("device") or "",
            "when": rec.get("created_at_str") or ""}
    try:
        phone_sidecar_path(image_path).write_text(
            json.dumps(side, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        log.warning("写手机端伴随文件失败：%s", e)
    return side


# 来源代码（写进地块地图库，地图上据此给点上色）
SOURCE_CODES = {"手机直传": "phone", "小程序": "mini", "电脑本地": "pc"}

_PHONE_UPLOAD_CACHE = {"mtime": None, "names": set()}


def _phone_upload_names():
    """手机直传过的文件名。

    打来源标记的代码是后加的，早先传的图没有伴随文件 —— 但每次手机上传都会
    记一条 phone_upload 事件，靠它把这些图也认出来。
    按事件台账的修改时间来失效：程序常驻时，手机刚传的图必须立刻认得出来，
    不能只在启动时读一次（那会把新图全判成「电脑本地」）。
    """
    try:
        mt = EVENT_LOG.stat().st_mtime if EVENT_LOG.exists() else 0.0
    except OSError:
        mt = 0.0
    if _PHONE_UPLOAD_CACHE["mtime"] != mt:
        try:
            _PHONE_UPLOAD_CACHE["names"] = {
                e.get("file") for e in read_jsonl(EVENT_LOG)
                if e.get("kind") == "phone_upload" and e.get("file")}
        except Exception as e:
            log.debug("读手机上传事件失败：%s", e)
        _PHONE_UPLOAD_CACHE["mtime"] = mt
    return _PHONE_UPLOAD_CACHE["names"]


def watch_source(image_path):
    """这条记录从哪来：小程序（公网中转）/ 手机直传（局域网）/ 电脑本地。

    台账 CSV 里没存这个字段，所以按文件名回查伴随文件；
    没有伴随文件的老图再回查事件台账。
    """
    side = read_phone_sidecar(image_path)
    if side:
        src = str(side.get("来源") or "")
        if "小程序" in src or "中转" in src:
            return "小程序"
        if "手机" in src:
            return "手机直传"
        return "手机直传"          # 有伴随文件但来源为空：只可能是手机那条链路写的
    try:
        if Path(image_path).name in _phone_upload_names():
            return "手机直传"
    except Exception:
        pass
    return "电脑本地"


def relay_pull(base_url="", token="", limit=100, folder=None, reset=False):
    """从中转服务拉新记录：图片落进收件箱 + 手机端结论写成伴随文件。

    这样小程序识别过的照片也会走「自动巡检」：检测、复核、提醒、落库、地图落点，
    并额外给出「手机端 vs 电脑端」是否一致的对比——而不是另开一条无人看的 CSV。
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return {"ok": False, "error": "没有填写中转服务地址（形如 http://127.0.0.1:9000）"}
    folder = Path(folder or WATCH_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    state = _relay_state()
    # 中转的 since 语义是「时间戳」（created_at 秒），不是记录 id——这点务必对齐，
    # 否则第二次同步会直接 400。
    try:
        since = 0.0 if reset else float(state.get("since") or 0)
    except (TypeError, ValueError):
        since = 0.0
    # 注意头名：中转服务认 X-Auth-Token（写错会 401，而且只有设了口令才暴露——
    # 之前端到端测试没设口令，这个错一直没被发现，是部署自检抓出来的）
    headers = {"X-Auth-Token": token} if token else {}
    try:
        r = requests.get(base + "/api/records",
                         params={"since": since, "limit": max(1, min(int(limit), 500))},
                         headers=headers, timeout=15)
        if r.status_code != 200:
            return {"ok": False, "error": "中转服务返回 HTTP %s" % r.status_code}
        payload = r.json()
    except Exception as e:
        return {"ok": False, "error": "连不上中转服务：%s" % e}
    records = payload.get("records") if isinstance(payload, dict) else payload
    records = records or []
    saved, skipped, last_id = [], 0, since
    max_created = since
    for rec in records:
        rid = str(rec.get("id") or "")
        if not rid:
            continue
        ext = str(rec.get("ext") or ".jpg")
        dst = folder / ("phone_%s%s" % (rid, ext))
        try:
            max_created = max(max_created, float(rec.get("created_at") or 0))
        except (TypeError, ValueError):
            pass
        if dst.exists():
            # 图已经拉过了，但结论可能这次才补传上来——把伴随文件刷新一遍
            if rec.get("label"):
                _write_phone_sidecar(dst, rec, rid)
            skipped += 1
            last_id = rid
            continue
        try:
            ir = requests.get(base + "/api/image/" + rid, headers=headers, timeout=30)
            if ir.status_code != 200 or not ir.content:
                log.warning("下载 %s 失败：HTTP %s", rid, ir.status_code)
                continue
            dst.write_bytes(ir.content)          # 原字节落盘
        except Exception as e:
            log.warning("下载 %s 异常：%s", rid, e)
            continue
        _write_phone_sidecar(dst, rec, rid)
        saved.append(dst.name)
        last_id = rid
    state.update({"since": max_created or since, "last_id": last_id, "base": base,
                  "synced": int(state.get("synced", 0)) + len(saved),
                  "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    _save_relay_state(state)
    log_event("relay_pull", base=base, new=len(saved), skipped=skipped)
    return {"ok": True, "新增": len(saved), "已存在跳过": skipped,
            "文件": saved[:20], "状态": state}


def relay_sync_report():
    st = _relay_state()
    pending = [p for p in list_watch_candidates(WATCH_DIR, 500) if read_phone_sidecar(p)]
    return {"中转地址": st.get("base") or "（未配置）",
            "上次同步": st.get("last_sync") or "（从未）",
            "累计同步": st.get("synced", 0),
            "收件箱里来自手机端的图": len(pending)}


def process_image_file(path, conf=None, do_cascade=True):
    """处理一张图：检测 → 可选级联复核 → 落会话 + 事件。返回记录 dict（不抛异常）。"""
    rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "file": os.path.basename(str(path)), "cls": "", "conf": 0.0, "boxes": 0,
           "severity": "", "verify": "", "verify_agree": "", "note": ""}
    # 来源：手机直传 / 小程序 / 电脑本地。以前 rec 里没有这个字段，
    # 自动落点写库时取到的永远是 None → 所有点在地图上都成了「电脑本地」。
    try:
        rec["source"] = SOURCE_CODES.get(watch_source(path), "pc")
    except Exception:
        rec["source"] = "pc"
    try:
        img = cv2.imread(str(path))
        if img is None:
            rec["note"] = "读不了图"
            return rec
        # 手机端/小程序已经用**另一套模型**识别过：不再用本地模型重复识别，
        # 直接采用手机端结论（2026-10-14 按用户要求；两套模型结论不同，重复识别只会打架）
        _side = read_phone_sidecar(path) or {}
        _plabel = (_side.get("label") or "").strip()
        if _plabel:
            _pconf = round(float(_side.get("score") or 0.0), 3)
            dets = [{"name": _plabel, "conf": _pconf, "box": []}]
            sev = {"level": "未评估（手机端）"}
            rec.update({"boxes": int(_side.get("boxes") or 0),
                        "severity": "未评估（手机端）",
                        "cls": _plabel, "conf": _pconf,
                        "source": "phone", "phone_cls": _plabel,
                        "phone_conf": _side.get("score"),
                        "note": "采用手机端识别结果（本地未重复识别）"})
        else:
            model, _ = get_model()
            res = model(img, conf=conf or CONF_THRESHOLD, imgsz=640, verbose=False)
            dets = [{"name": CLASS_NAMES[int(b.cls.item())], "conf": float(b.conf.item()),
                     "box": b.xyxy[0].tolist()} for b in res[0].boxes]
            sev = assess_severity(img.shape, dets)
            rec["boxes"] = len(dets)
            rec["severity"] = sev["level"]
        if dets:
            best = max(dets, key=lambda x: x["conf"])
            rec["cls"], rec["conf"] = best["name"], round(best["conf"], 3)
            need, why = should_auto_verify(best["name"], best["conf"])
            if _plabel:
                need = False          # 手机端结论直接采信，本地不再请第二个模型复核
            if do_cascade and need:
                v = vision_recheck(img, best["box"], None, 0, source="watch")
                if v.get("ok"):
                    rec["verify"] = v.get("verdict") or ""
                    rec["verify_agree"] = bool(rec["verify"]) and rec["verify"] == best["name"]
                else:
                    rec["note"] = v.get("error", "")[:80]
        else:
            rec["cls"] = ""
            rec["note"] = "未检出目标"
        # 来源标记：带伴随文件的 = 手机传上来的（手机直传/小程序），否则是电脑本地处理
        rec["source"] = "phone" if read_phone_sidecar(path) else "pc"
        # 手机端（小程序）识别过的图会带伴随文件：把它的结论一并记录并对比
        side = read_phone_sidecar(path)
        if side:
            rec["phone_cls"] = side.get("label") or ""
            rec["phone_conf"] = side.get("score")
            if rec["phone_cls"]:
                # _plabel 为真 = 本地直接采用了手机端结论，两边必然一致（不再做对比）；
                # 否则才是真正的"端云对比"：本地模型 vs 手机端模型
                rec["phone_agree"] = True if _plabel else (rec["cls"] == rec["phone_cls"])
            rec["phone_from"] = side.get("来源") or "手机端"
        sid = save_session(rec["cls"] or "未检出", rec["conf"], sev, mode="watch",
                           chat=[("system", "自动巡检：%s" % rec["file"])], image=img)
        rec["session"] = sid
        log_event("watch_process", **{k: v for k, v in rec.items() if k != "time"})
        _append_watch_row(rec)
        # 写一份识别结果伴随文件：手机传图页面靠它把结论回显给现场的人
        try:
            Path(str(path) + ".result.json").write_text(json.dumps({
                "cls": rec.get("cls") or "未检出", "conf": rec.get("conf") or 0.0,
                "boxes": rec.get("boxes") or 0, "severity": rec.get("severity") or "",
                "verify": rec.get("verify") or "", "phone_cls": rec.get("phone_cls") or "",
                "phone_agree": rec.get("phone_agree"), "time": rec.get("time"),
            }, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.debug("写识别结果伴随文件失败：%s", e)
        # 带有效 GPS 的照片顺手落进地块档案库——手机传图那条链路也就真能上地图了。
        # （界面上写过"有 GPS 会自动落点"，之前其实没接，这里补上承诺。）
        try:
            import rice_geo
            meta = rice_geo.read_photo_meta(str(path))
            _lon = meta.get("lon") if meta.get("has_gps") else None
            _lat = meta.get("lat") if meta.get("has_gps") else None
            _csrc = "exif"
            if _lon is None:
                # 图里没有 EXIF 定位时（安卓系统相册选择器会剥离位置），
                # 用传图时现取的当前位置兜底——坐标来自浏览器定位，属 WGS-84
                _sd = read_phone_sidecar(path) or {}
                if _sd.get("lat") is not None and _sd.get("lng") is not None:
                    # 注意：这是**上传时**的定位，不是拍摄时的——只在上传者明确声明
                    # "我此刻就在现场"时才有意义。所以来源单独标 phone-onsite，
                    # 界面上与 EXIF 定位区分开，便于复核；绝不当成确证位置。
                    _lon, _lat, _csrc = float(_sd["lng"]), float(_sd["lat"]), "phone-onsite"
                    rec["note"] = (rec.get("note") or "") + "｜落点=上传时现场定位（未经核实）"
            if _lon is not None and _lat is not None:
                st = geo_store()
                sha = rice_geo.sha1_of_file(str(path))
                if st is not None and not st.has_sha1(sha):
                    st.add_record(
                        shot_at=(meta.get("taken_at") or datetime.now()).strftime(
                            "%Y-%m-%d %H:%M:%S"),
                        lon=_lon, lat=_lat, coord_src=_csrc,
                        cls=rec.get("cls") or "", conf=float(rec.get("conf") or 0),
                        boxes=rec.get("boxes") or 0,
                        note="自动巡检落库（含 GPS·%s）" % (
                            "手机" if rec.get("source") == "phone" else "电脑"),
                        source_path=str(path), device=meta.get("device") or "",
                        source=rec.get("source") or "pc")
                    log_event("watch_geo_add", file=os.path.basename(str(path)),
                              lon=meta["lon"], lat=meta["lat"])
        except Exception as e:
            log.debug("巡检自动落库失败：%s", e)
    except Exception as e:
        log.exception("巡检处理失败: %s", path)
        rec["note"] = "异常：%s" % e
    return rec


def _append_watch_row(rec):
    try:
        WATCH_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        # 台账表头变过：老文件先按新表头迁移一次，否则新行比老行多列，Excel 打开会错位
        if WATCH_RESULTS.exists():
            try:
                with open(WATCH_RESULTS, "r", newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.reader(f))
            except OSError:
                rows = []
            if rows and rows[0] != WATCH_COLUMNS:
                # 注意：新列要插在 note **之前**，不能直接追加到末尾——
                # 追加会让老行的 note 内容跑到 phone_cls 列下面（我自己踩过这个坑）。
                old_cols = rows[0]
                tail = [c for c in WATCH_COLUMNS if c not in old_cols]      # 新增的列
                fixed = [WATCH_COLUMNS]
                for r in rows[1:]:
                    row = dict(zip(old_cols, r))
                    fixed.append([str(row.get(c, "") or "") for c in WATCH_COLUMNS])
                log.info("台账迁移：老表头 %d 列 → 新表头 %d 列（新增 %s）",
                         len(old_cols), len(WATCH_COLUMNS), ",".join(tail))
                with open(WATCH_RESULTS, "w", newline="", encoding="utf-8-sig") as f:
                    csv.writer(f).writerows(fixed)
                log.info("巡检台账已按新表头迁移（%d 行）", len(fixed) - 1)
        new = not WATCH_RESULTS.exists()
        with open(WATCH_RESULTS, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(WATCH_COLUMNS)
            w.writerow([rec.get(c, "") for c in WATCH_COLUMNS])
    except OSError as e:
        log.warning("写巡检台账失败: %s", e)


def list_watch_candidates(folder=None, limit=50):
    """待处理图片：按修改时间从旧到新，跳过已处理过的文件名"""
    folder = Path(folder or WATCH_DIR)
    if not folder.is_dir():
        return []
    done = {r.get("file") for r in read_jsonl(EVENT_LOG)[-500:]
            if r.get("kind") == "watch_process"}
    files = [p for p in sorted(folder.glob("*"), key=lambda p: p.stat().st_mtime)
             if p.suffix.lower() in KNOWN_EXTENSIONS and p.name not in done]
    return files[:limit]


def alert_level(rec):
    """什么情况需要主动提醒人。

    复核分歧只在「跨大类」时提醒（虫↔病）：同类分歧记台账就够了，
    否则视觉模型的一点噪声就会天天弹窗，提醒很快没人看。
    """
    if not rec:
        return ""
    if (rec.get("verify_agree") is False
            and not same_category(rec.get("cls"), rec.get("verify"))):
        return "两个模型判断冲突（虫害/病害方向不同），建议人工确认"
    if rec.get("severity") == "重":
        return "检出强度为「重」，建议尽快处理"
    return ""


def usage_summary():
    """汇总运行数据：检测、AI 调用、工具调用、反馈、实训（支撑材料用）"""
    events = read_jsonl(EVENT_LOG)
    det = [e for e in events if e.get("kind") in ("detection", "watch_process")]
    watched = [e for e in events if e.get("kind") == "watch_process"]
    ans = [e for e in events if e.get("kind") == "analysis_done"]
    tools = [e for e in events if e.get("kind") == "tool_call"]
    train = [e for e in events if e.get("kind") == "training_answer"]
    cards = [e for e in events if e.get("kind") == "diagnosis_card"]
    casc = [e for e in events if e.get("kind") == "cascade_result"]
    casc_done = [e for e in casc if "agrees" in e]
    fb = load_feedback_rows()
    llm = read_jsonl(AUDIT_LOG)
    return {
        "检测次数": len(det),
        "检出类别分布": dict(Counter(e.get("cls") for e in det if e.get("cls"))),
        "未检出次数": sum(1 for e in det if not e.get("cls")),
        "其中无人值守处理": len(watched),
        "AI 分析次数": len(ans),
        "AI 回答被检出禁用农药": sum(1 for e in ans if e.get("banned")),
        "工具调用次数": len(tools),
        "工具使用分布": dict(Counter(e.get("tool") for e in tools if e.get("tool"))),
        "诊断卡生成次数": len(cards),
        "诊断卡告警次数": sum(1 for e in cards if e.get("warnings")),
        "视觉复核次数": len(casc_done),
        "视觉复核一致率%": (round(sum(1 for e in casc_done if e.get("agrees"))
                            / len(casc_done) * 100, 1) if casc_done else None),
        "人工裁决次数": sum(1 for e in events if e.get("kind") == "cascade_resolve"),
        "模型调用（审计条数）": len(llm),
        "AI 调用成本": usage_cost_summary(),
        "反馈条数": len(fb),
        "其中漏检反馈": sum(1 for r in fb if r.get("mode") == "miss"),
        "实训答题数": len(train),
        "实训正确率%": (round(sum(1 for r in train if r.get("correct")) / len(train) * 100, 1)
                    if train else None),
        "模型在实训中的正确率%": (round(sum(1 for r in train if r.get("model_correct")) / len(train) * 100, 1)
                          if train else None),
    }


def export_usage_ledger(path=None):
    """把台账写成 CSV（两列：指标 / 数值），供参赛支撑材料直接引用"""
    s = usage_summary()
    path = Path(path or (APP_DIR / "运行台账_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["指标", "数值"])
        for k, v in s.items():
            w.writerow([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
    return path, s


def allowed_pesticide_clause():
    """给模型的白名单提示：只允许推荐本地库里的药剂"""
    return "可推荐药剂（本地防治库）：%s。库外药剂必须注明「需人工核实登记与用量」。" % "、".join(_local_drug_names())


# ── 治理方案资源目录初始化 ───────────────────────────────────
def init_treatment_dirs():
    """在可写目录下创建治理方案图片目录，方便用户自行补充图片"""
    for cls_name in CLASS_NAMES:
        d = APP_DIR / "resources" / "treatment" / cls_name
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            log.warning("创建治理方案目录失败 %s: %s", d, e)


KNOWN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
                    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}


def _candidates(path):
    """生成所有待尝试的文件路径（双资源根目录 + 替换常见扩展名），避免重复生成已有扩展名"""
    if not path:
        return
    p = Path(path)
    if p.is_absolute():
        bases = [str(p)]
    else:
        bases = [str(root / p) for root in _resource_roots()]

    seen = set()
    for base in bases:
        stem, orig_ext = os.path.splitext(base)
        for cand in (base,) + tuple(stem + ext for ext in KNOWN_EXTENSIONS if ext != orig_ext):
            if cand not in seen:
                seen.add(cand)
                yield cand


def load_treatment_image(path, target_w=180, target_h=140):
    """加载治理方案图片，自动匹配 jpg/png/bmp 格式，失败返回 None"""
    if not path:
        return None
    for candidate in _candidates(path):
        if os.path.exists(candidate):
            pix = QPixmap(candidate)
            if not pix.isNull():
                return pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return None

def _wrapped_text_height(text, font, width):
    """按给定宽度估算自动换行后 QLabel 需要的像素高度。

    QLabel 开了 wordWrap 之后，布局往往只按"一行"给高度，文字一换行就被裁掉
    （药剂卡显示不全就是这个原因）。这里用 QFontMetrics 把高度先算死。
    """
    fm = QFontMetrics(font)
    rect = fm.boundingRect(0, 0, max(20, int(width)), 10000,
                           Qt.TextWordWrap | Qt.AlignCenter, text)
    return rect.height() + 4


# 药剂行里文字栏的最小宽度。高度就是按这个宽度算的，两者必须一致，
# 否则窗口再窄一点，长名字还是会多折一行被裁掉。
DRUG_TEXT_MIN_W = 150


def make_drug_row(drug):
    """一行药剂：左侧药品图片，右侧名称 + 规格/用量。

    原来是"两列图片卡 + 下面再列一遍纯文本"：卡片只有 140px 宽，
    名称/规格/用量三四行文字挤在里面必然被裁，而且同样的信息还显示了两遍。
    改成一行一药后文字有整栏宽度，也不会重复。
    """
    row = QWidget()
    row.setStyleSheet(f"background-color: {NAU['card']}; border-radius: {RADIUS['inner']}px; "
                      f"border: 1px solid {NAU['border']};")
    lay = QHBoxLayout(row)
    lay.setContentsMargins(10, 10, 14, 10)
    lay.setSpacing(12)

    size = 76
    thumb = QLabel()
    thumb.setFixedSize(size, size)
    thumb.setAlignment(Qt.AlignCenter)
    pix = load_treatment_image(drug.get("img", ""), size - 8, size - 8)
    if pix:
        thumb.setPixmap(pix)
        thumb.setStyleSheet(f"background: {NAU['bg']}; border-radius: 10px; border: none;")
    else:
        thumb.setText("待补图")
        thumb.setFont(ui_font(9))
        thumb.setStyleSheet(f"background: {NAU['bg']}; border-radius: 10px; border: none; "
                            f"color: {NAU['subtext']};")
    lay.addWidget(thumb)

    text_box = QWidget()
    text_box.setMinimumWidth(DRUG_TEXT_MIN_W)      # 与下面的高度预算保持一致
    text_box.setStyleSheet("background: transparent; border: none;")
    col = QVBoxLayout(text_box)
    col.setContentsMargins(0, 2, 0, 2)
    col.setSpacing(4)

    name_text = drug.get("name", "")
    name_font = ui_font(13, True)
    name = QLabel(name_text)
    name.setFont(name_font)
    name.setWordWrap(True)
    name.setMinimumHeight(_wrapped_text_height(name_text, name_font, DRUG_TEXT_MIN_W))
    name.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
    col.addWidget(name)

    detail_text = " · ".join(p for p in (drug.get("spec", ""), drug.get("dose", "")) if p)
    detail_font = ui_font(11)
    detail = QLabel(detail_text)
    detail.setFont(detail_font)
    detail.setWordWrap(True)
    detail.setMinimumHeight(_wrapped_text_height(detail_text, detail_font, DRUG_TEXT_MIN_W))
    detail.setStyleSheet(f"color: {NAU['primary']}; background: transparent; border: none;")
    col.addWidget(detail)
    col.addStretch()

    lay.addWidget(text_box, 1)
    return row


def make_image_card(image_path, label_text, w=180, h=140):
    """创建一个图片卡片组件（图片+标题）"""
    card = QWidget()
    card.setStyleSheet(f"background-color: {NAU['card']}; border-radius: {RADIUS['inner']}px; border: 1px solid {NAU['border']};")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)

    pix = load_treatment_image(image_path, w, h)
    if pix:
        img_label = QLabel()
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setFixedSize(w, h)
        img_label.setStyleSheet("border-radius: 14px; background: transparent; border: none;")
        layout.addWidget(img_label, alignment=Qt.AlignCenter)
    else:
        placeholder = QLabel(f"{icon_img('image', 30, 'gray', gap=False)}<br>待添加图片")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setFixedSize(w, h)
        placeholder.setFont(ui_font(11))
        placeholder.setStyleSheet(f"""
            background-color: {NAU['primary_light']};
            border-radius: 14px;
            color: {NAU['subtext']};
            border: 2px dashed {NAU['border']};
        """)
        layout.addWidget(placeholder, alignment=Qt.AlignCenter)

    name_label = QLabel(label_text)
    name_label.setFont(ui_font(11))
    name_label.setAlignment(Qt.AlignCenter)
    name_label.setWordWrap(True)
    # 宽度固定死（而不是只设 maximumWidth），再按同一宽度算高度：
    # 只设最大宽度时布局仍按一行分配高度，文字换行后就被裁掉
    name_label.setFixedWidth(w + 20)
    name_label.setMinimumHeight(_wrapped_text_height(label_text, name_label.font(), w + 20))
    name_label.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
    layout.addWidget(name_label, alignment=Qt.AlignCenter)

    return card

# ── 南农大官方配色（从校徽 & 辅助图形中提取） ──────────────────
NAU = {
    # ──「稻视界」视觉识别（取自项目设计稿）──
    # 南农绿为主色、金为点缀、米白为底；金只用在深绿底上或做装饰线，
    # 因为 #F3B31B 在白底上对比度只有 1.86:1，当正文色不达标（WCAG AA）。
    "primary": "#007A3F",        # 南农绿（设计稿原色）
    "primary_hover": "#00602F",
    "primary_deep": "#00502A",   # 深绿底：品牌 hero
    "primary_light": "#E8F2EC",
    "gold": "#F3B31B",           # 金：装饰线、强调、深绿底上的文字
    "gold_light": "#FFDE59",
    "accent_gold": "#F3B31B",    # 兼容旧名
    "accent_blue": "#2C6E8F",
    "bg": "#F7F6F1",             # 米白底（设计稿底色）
    "card": "#FFFFFF",
    "dark": "#12301F",           # 墨绿黑
    "text": "#2F3A33",
    # 次要文字：中文笔画细，颜色一浅就"没墨"，所以比常规灰阶再深一档
    "subtext": "#4A544F",        # 对白底 7.77:1
    "muted": "#6E7873",          # 对白底 4.63:1（原来 3.13:1 不达标）
    "border": "#E4E7E0",
    "divider": "#EFF1EA",
    "red": "#C0392B",
    "warning": "#D9822B",
    "shadow": QColor(18, 48, 31, 26),
}

# 三种来源的配色（都过 WCAG AA 对白底）：小程序=蓝、手机直传=绿、电脑本地=灰。
# 判据是伴随文件：中转和手机直传都会写 .phone.json，电脑本地的图没有。
SOURCE_META = {
    "小程序":   {"color": NAU["accent_blue"], "dot": "●"},
    "手机直传": {"color": NAU["primary"],     "dot": "●"},
    "电脑本地": {"color": NAU["muted"],       "dot": "●"},
}

BRAND_DIR = Path(res_path("resources/brand"))


def brand_pixmap(name, height=None, width=None):
    """取品牌素材（主楼剪影 / 稻视界 logo / 校徽）；缺失返回 None"""
    p = BRAND_DIR / name
    if not p.exists():
        return None
    pix = QPixmap(str(p))
    if pix.isNull():
        return None
    if height:
        return pix.scaledToHeight(height, Qt.SmoothTransformation)
    if width:
        return pix.scaledToWidth(width, Qt.SmoothTransformation)
    return pix


def _sized(lbl, key, bold=None, weight=None):
    """把 Fluent 标签的字号对齐到 TYPE 阶梯。

    只改 QFont，不加样式表 —— 给 Fluent 标签套自定义样式表会打乱它的字体度量。
    weight 用来做字重层级：Bold 最重、Medium 中间、Normal 最轻，
    有了中间档就不用"要么不加粗要么死粗"，层级会细腻很多。
    """
    old = lbl.font()
    if weight is not None:
        w = weight
    elif bold is not None:
        w = QFont.Bold if bold else QFont.Normal
    else:
        w = old.weight()
    lbl.setFont(ui_font(TYPE[key], weight=w))     # 走字体栈，西文用 Inter
    return lbl


def t_block(text, parent):
    """区块标题（Subtitle 级）—— 用 Medium，和正文拉开但不至于死粗"""
    return _sized(StrongBodyLabel(text, parent), "subtitle", weight=QFont.Medium)


def t_body(text, parent, muted=False, color=None):
    """正文（Body 级）。muted 的说明文字同样加半档字重"""
    lbl = _sized(BodyLabel(text, parent), "body",
                 weight=QFont.Medium if muted else None)
    if color:
        lbl.setTextColor(color, color)
    elif muted:
        lbl.setTextColor(NAU["subtext"], NAU["subtext"])
    return lbl


def t_caption(text, parent, color=None):
    """说明 / 元信息（Caption 级）。

    小字用 Medium 而不是 Regular：中文笔画多，Regular 在小字号下细笔画会糊成灰的，
    看起来像"没墨"。加半档字重立刻实起来。
    """
    lbl = _sized(CaptionLabel(text, parent), "caption", weight=QFont.Medium)
    if color:
        lbl.setTextColor(color, color)
    return lbl


def page_title(text, parent):
    """页面主标题：左侧金色竖条 + 标题。

    竖条用独立控件画，而不是给 TitleLabel 加 border-left —— 给 Fluent 标签
    套自定义样式表会打乱它的字体度量，标题会被裁掉（实测踩过）。
    """
    row = QWidget(parent)
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)
    bar = QFrame(row)
    bar.setFixedSize(4, 30)
    bar.setStyleSheet(f"background-color: {NAU['gold']}; border-radius: 2px;")
    lay.addWidget(bar, 0, Qt.AlignVCenter)
    lbl = TitleLabel(text, row)
    # 页面标题用衬线（display_font），和品牌调性一致
    tf = display_font(TYPE["title"])
    tf.setLetterSpacing(QFont.PercentageSpacing, 103)   # 中文大标题微开字距更"立得住"
    lbl.setFont(tf)
    lay.addWidget(lbl, 0, Qt.AlignVCenter)
    lay.addStretch()
    # 不加这句的话这个行容器会纵向撑满，把副标题挤到页面中间去
    row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    row.setFixedHeight(46)        # 统一标题行高，页面纵向节奏才稳定
    return row


# ── 排版阶梯（点号）──────────────────────────────────────────
# 层级靠"字号 + 字重 + 颜色"三者共同拉开，而不是只靠字号。
# 相邻层级至少差 4pt，否则看起来会"糊在一起"。
TYPE = {
    "display": 30,     # 首页主标题：最大最显眼
    "title": 22,       # 页面标题
    "subtitle": 16,    # 区块标题 / 品牌副标题
    "body": 11,        # 正文
    "caption": 9,      # 说明、元信息
}
# 文字颜色也分三级：主 / 次 / 弱
TEXT_STRONG = "#FFFFFF"      # 深色底上的主文字
TEXT_MUTED_LIGHT = "#9FC4AE"  # 深色底上的弱文字

REPORT_W = 1240          # A4 竖版 @150dpi


def _wrap_text(probe, text, font, max_w):
    """按像素宽度折行（中英混排都能断）"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if probe.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def export_report(path, result_list, orig_img, cls_name, conf, ai_text, device):
    """生成识别报告（.png 或 .pdf）：品牌头 + 标注图 + 识别结果 + 治理方案 + AI 建议"""
    from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont

    G, GOLD, DARK, SUB = NAU["primary"], NAU["gold"], NAU["dark"], NAU["subtext"]
    W, PAD = REPORT_W, 72
    inner = W - PAD * 2

    def font(size, bold=False):
        # 报告也用同一套字体，观感和界面统一
        for name in (("HarmonyOS_Sans_SC_Bold.ttf", "NotoSansSC-Bold.otf") if bold
                     else ("HarmonyOS_Sans_SC_Regular.ttf", "NotoSansSC-Regular.otf")):
            p = FONT_DIR / name
            if p.exists():
                break
        try:
            return PILFont.truetype(str(p), size)
        except Exception:
            return PILFont.load_default()

    f_h1, f_h2 = font(30, True), font(22, True)
    f_big, f_body, f_small = font(24, True), font(18), font(15)
    probe = PILDraw.Draw(PILImage.new("RGB", (8, 8)))
    LH = 30

    tr = TREATMENT_DB.get(cls_name or "", {})
    ops = []          # (高度, 绘制函数(draw, y))

    def add(h, fn):
        ops.append((h, fn))

    # 识别结果
    add(40, lambda d, y: d.text((PAD, y), "识别结果", font=f_h2, fill=GOLD))
    name_txt = f"病虫害：{cls_name}" if cls_name else "未检测到病虫害"
    add(38, lambda d, y: d.text((PAD, y), name_txt, font=f_big, fill=DARK))
    add(30, lambda d, y: d.text((PAD, y), f"置信度：{conf * 100:.2f}%", font=f_body, fill=SUB))
    sev = assess_severity(getattr(orig_img, "shape", None), result_list)
    add(28, lambda d, y: d.text((PAD, y), "检出强度：%s ｜ %s" % (sev["level"], sev["text"]),
                                font=f_small, fill=SUB))
    add(24, lambda d, y: d.text((PAD, y), "（%s）" % sev["basis"], font=f_small, fill=SUB))
    for cname in CLASS_NAMES:
        c = next((r["conf"] for r in result_list if r["name"] == cname), 0.0)
        def bar(d, y, cname=cname, c=c):
            d.text((PAD, y + 2), cname, font=f_small, fill=DARK)
            bx, bw = PAD + 130, inner - 220
            d.rounded_rectangle([bx, y + 5, bx + bw, y + 15], 5, fill="#EDEFEA")
            if c > 0:
                d.rounded_rectangle([bx, y + 5, bx + max(8, int(bw * c)), y + 15], 5, fill=G)
            d.text((bx + bw + 16, y), f"{c * 100:.0f}%", font=f_small, fill=SUB)
        add(26, bar)

    # 治理方案
    if tr:
        add(46, lambda d, y: d.text((PAD, y + 12), "治理方案", font=f_h2, fill=GOLD))

        def section(title, items, color=DARK):
            add(32, lambda d, y, tt=title: d.text((PAD, y), tt, font=f_body, fill=G))
            for it in items:
                for ln in _wrap_text(probe, "· " + it, f_body, inner - 10):
                    add(LH, lambda d, y, s=ln: d.text((PAD + 8, y), s, font=f_body, fill=color))
        section("农业防治", tr.get("agri", []))
        if tr.get("bio"):
            section("生物防治", tr["bio"])
        add(32, lambda d, y: d.text((PAD, y), "防治指标", font=f_body, fill=G))
        for ln in _wrap_text(probe, "· " + tr.get("threshold", ""), f_body, inner - 10):
            add(LH, lambda d, y, s=ln: d.text((PAD + 8, y), s, font=f_body, fill=NAU["red"]))
        add(32, lambda d, y: d.text((PAD, y), "推荐药剂", font=f_body, fill=G))
        for drug in tr.get("drugs", []):
            txt = "· %s ｜ %s ｜ %s" % (drug.get("name", ""), drug.get("spec", ""), drug.get("dose", ""))
            for ln in _wrap_text(probe, txt, f_body, inner - 10):
                add(LH, lambda d, y, s=ln: d.text((PAD + 8, y), s, font=f_body, fill=DARK))

    # 合规红线：AI 建议里出现禁用/停用农药时，报告上直接标红（不静默删改模型原话）
    _hits = pesticide_hits(ai_text)
    if _hits:
        add(30, lambda d, y, hs=_hits: d.text(
            (PAD, y), "⚠ AI 建议中出现禁用/停用农药：%s，请勿使用" % "、".join(hs),
            font=f_small, fill=NAU["red"]))

    # AI 建议（截断，避免报告过长）
    if ai_text.strip():
        add(46, lambda d, y: d.text((PAD, y + 12), "AI 分析建议", font=f_h2, fill=GOLD))
        shown = 0
        for para in ai_text.split("\n"):
            para = para.strip().lstrip("#").strip()
            if not para:
                continue
            for ln in _wrap_text(probe, para, f_body, inner):
                add(LH, lambda d, y, s=ln: d.text((PAD, y), s, font=f_body, fill=DARK))
                shown += 1
            if shown > 26:
                add(LH, lambda d, y: d.text((PAD, y), "……（完整内容见程序内 AI 面板）",
                                            font=f_body, fill=SUB))
                break

    # 标注图高度
    # 先放大再画框：annotate_image 的字号是固定的 30px，300x300 的小图上会显得巨大
    src = orig_img
    if src is not None and src.shape[1] < 900:
        s = 900 / src.shape[1]
        src = cv2.resize(src, (900, int(src.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    ann = annotate_image(src, result_list) if result_list else src
    pil_img = PILImage.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    iw = inner
    ih = int(pil_img.height * iw / pil_img.width)
    if ih > 520:
        ih = 520
        iw = int(pil_img.width * ih / pil_img.height)
    pil_img = pil_img.resize((iw, ih), PILImage.LANCZOS)

    HEAD, FOOT = 150, 76
    body_h = sum(h for h, _ in ops) + 40
    H = HEAD + 40 + ih + 34 + body_h + FOOT + 40

    canvas = PILImage.new("RGB", (W, H), "#FFFFFF")
    d = PILDraw.Draw(canvas)

    # ── 品牌头 ──
    d.rectangle([0, 0, W, HEAD], fill=NAU["primary_deep"])
    lx = PAD
    logo = BRAND_DIR / "稻视界logo_金_透明.png"
    if logo.exists():
        lg = PILImage.open(logo).convert("RGBA")
        lg = lg.resize((int(lg.width * 52 / lg.height), 52), PILImage.LANCZOS)
        canvas.paste(lg, (lx, 34), lg)
        lx += lg.width + 14
    d.text((lx, 34), "稻视界", font=font(38, True), fill="#FFFFFF")
    d.text((lx + 2, 82), "水稻病虫害智能诊断报告", font=font(19), fill=GOLD)
    sil = BRAND_DIR / "主楼剪影_金_透明.png"
    if sil.exists():
        si = PILImage.open(sil).convert("RGBA")
        si = si.resize((int(si.width * 116 / si.height), 116), PILImage.LANCZOS)
        canvas.paste(si, (W - PAD - si.width, HEAD - si.height), si)

    y = HEAD + 26
    d.text((PAD, y), "检测时间：%s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           font=f_small, fill=SUB)
    d.text((PAD + 330, y), "模型：YOLOv8n（%s）" % MODEL_INFO["训练数据集"], font=f_small, fill=SUB)
    d.text((PAD + 760, y), "运行设备：%s" % ("GPU (CUDA)" if device != "cpu" else "CPU"),
           font=f_small, fill=SUB)
    y += 34

    # 标注图
    canvas.paste(pil_img, ((W - iw) // 2, y))
    d.rectangle([(W - iw) // 2, y, (W - iw) // 2 + iw, y + ih], outline="#E4E7E0", width=1)
    y += ih + 34

    # 主体
    for h, fn in ops:
        fn(d, y)
        y += h

    # ── 品牌页脚 ──
    d.line([PAD, H - FOOT, W - PAD, H - FOOT], fill=GOLD, width=1)
    d.text((PAD, H - FOOT + 22), "稻视界 · 南京农业大学 · 诚朴勤仁", font=f_small, fill=SUB)
    d.text((W - PAD - 300, H - FOOT + 22), "本报告由「稻视界」自动生成，仅供参考",
           font=f_tiny if False else f_small, fill=SUB)

    if str(path).lower().endswith(".pdf"):
        canvas.save(path, "PDF", resolution=150.0)
    else:
        canvas.save(path)
    return path


REPORT_DIR = APP_DIR / "reports"
REPORT_INDEX = REPORT_DIR / "index.json"
REPORT_THUMB_DIR = REPORT_DIR / "thumbs"


def load_report_index():
    """读取导出记录（按时间倒序）"""
    try:
        if REPORT_INDEX.exists():
            items = json.loads(REPORT_INDEX.read_text(encoding="utf-8"))
            return sorted(items, key=lambda x: x.get("time", ""), reverse=True)
    except Exception as e:
        log.warning("读取报告索引失败: %s", e)
    return []


def save_report_index(items):
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_INDEX.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        log.error("写入报告索引失败: %s", e)


def _make_report_thumb(path, key):
    """给报告生成缩略图（PDF 跳过）"""
    if not str(path).lower().endswith(".png"):
        return ""
    try:
        from PIL import Image as PILImage
        im = PILImage.open(path)
        im.thumbnail((320, 460), PILImage.LANCZOS)
        REPORT_THUMB_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_THUMB_DIR / (key + ".jpg")
        im.convert("RGB").save(out, quality=88)
        return str(out)
    except Exception as e:
        log.warning("生成报告缩略图失败: %s", e)
        return ""


def add_report_record(path, cls_name, conf, device):
    """登记一条导出记录"""
    now = datetime.now()
    key = now.strftime("%Y%m%d_%H%M%S")
    items = load_report_index()
    items = [x for x in items if x.get("path") != str(path)]      # 同路径覆盖
    items.append({
        "path": str(path),
        "thumb": _make_report_thumb(path, key),
        "cls_name": cls_name or "未识别",
        "conf": float(conf or 0.0),
        "device": device,
        "time": now.strftime("%Y-%m-%d %H:%M"),
    })
    save_report_index(items)
    return items


# ── 误检反馈队列（E1：把「识别不对」变成可训练的数据） ──────────────
# 设计要点：图片必须存副本。农户拍的原图随时会被下一张覆盖或被删，
# 只记一行 CSV 等于什么都没留下；有了副本才能进标注流程。
FEEDBACK_DIR = APP_DIR / "feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "feedback.csv"
FEEDBACK_COLUMNS = ["time", "image", "predicted", "conf", "boxes", "severity",
                    "user_choice", "note", "mode", "model", "app_version"]


def append_feedback_record(image=None, predicted="", conf=0.0, boxes=0,
                           severity="", user_choice="", note="", mode="image",
                           image_path=None):
    """记录一次「识别不对」：保存图片副本 + 追加一行 CSV，返回图片副本路径"""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(predicted or "unknown")).strip("_")[:20]
    img_name = "%s_%s.jpg" % (stamp, safe or "unknown")
    saved = ""
    if image is not None and getattr(image, "shape", None) is not None:
        try:
            cv2.imwrite(str(FEEDBACK_DIR / img_name), image)
            saved = img_name
        except Exception as e:
            log.warning("反馈图片写入失败: %s", e)
    elif image_path and os.path.exists(image_path):
        try:
            shutil.copy2(image_path, FEEDBACK_DIR / img_name)
            saved = img_name
        except Exception as e:
            log.warning("反馈图片复制失败: %s", e)
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), saved, predicted,
           "%.4f" % float(conf or 0.0), int(boxes or 0), severity or "",
           user_choice or "", note or "", mode or "",
           MODEL_INFO.get("模型版本", ""), APP_VERSION]
    new_file = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FEEDBACK_COLUMNS)
        w.writerow(row)
    log.info("已记录误检反馈：%s → %s", predicted, user_choice or "(未填)")
    log_event("feedback", predicted=predicted, user_choice=user_choice, mode=mode,
              conf=round(float(conf or 0.0), 3))
    return str(FEEDBACK_DIR / img_name) if saved else ""


def load_feedback_rows():
    """读取反馈队列（供 checks/feedback_review.py 统计，也用来自检）"""
    if not FEEDBACK_CSV.exists():
        return []
    with open(FEEDBACK_CSV, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def brand_title(text, parent, size=20):
    """带金色竖线的区块标题（稻视界的视觉语言）"""
    lbl = SubtitleLabel(text, parent)
    lbl.setStyleSheet(f"border-left: 3px solid {NAU['gold']}; padding-left: 10px;")
    f = lbl.font()
    f.setPointSize(TYPE["subtitle"])
    f.setWeight(QFont.Medium)
    lbl.setFont(f)
    return lbl


# ── 设计令牌：圆角 / 间距 / 阴影统一在这里改，别在控件里随手写数字 ──
RADIUS = {"window": 18, "card": 16, "inner": 12, "chip": 8, "button": 28, "input": 10}
SPACE = {"xs": 4, "sm": 8, "md": 14, "lg": 20, "xl": 28}
SHADOW = {"sm": (6, 2), "md": (12, 3), "lg": (20, 6)}

# ── 界面图标（Lucide，ISC 许可，见 resources/icons/NOTICE.txt）──────────
# make_icons.py 把 resources/icons/_src/*.svg 渲染成三种色调的 PNG：
#   green —— 浅色背景（AI 面板、状态栏）  white —— 深色按钮  gray —— 标题栏/次要按钮
ICON_DIR = Path(res_path("resources/icons"))


def icon_img(name, size=20, tone="green", gap=True):
    """图标名 → 内联 <img>；文件缺失时返回空串（宁可不显示，也不退回 emoji）。

    QTextEdit / QLabel 的富文本支持 file:// 本地图片，不用往 document 注册资源。
    """
    path = ICON_DIR / tone / (name + ".png")
    if not path.exists():
        return ""
    size = int(size)
    return ("<img src='%s' width='%d' height='%d' style='vertical-align:middle;'>%s"
            % (path.as_uri(), size, size, "&nbsp;" if gap else ""))


def icon_qicon(name, tone="gray", size=18, active_tone=None):
    """图标名 → QIcon；active_tone 用来在鼠标悬停时换一套颜色（例如关闭按钮）。"""
    normal = ICON_DIR / tone / (name + ".png")
    if not name or not normal.exists():
        return None
    icon = QIcon(str(normal))
    if active_tone and active_tone != tone:
        active = ICON_DIR / active_tone / (name + ".png")
        if active.exists():
            icon.addPixmap(QPixmap(str(active)), QIcon.Active)
    return icon


def icon_button(name, tone="gray", size=18, active_tone=None):
    """只有图标的 QPushButton；找不到图标文件时就是个空按钮（不会退回 emoji）。"""
    btn = QPushButton()
    icon = icon_qicon(name, tone, size, active_tone)
    if icon is not None:
        btn.setIcon(icon)
        btn.setIconSize(QSize(size, size))
    return btn


# AI 回答的小标题关键词 → 图标名（原来这里塞的是 emoji，一眼就是 AI 生成的）
AI_HEADING_ICONS = [
    (("防治", "治理", "方案", "措施", "建议", "方法"), "clipboard-list"),
    (("药剂", "药物", "用药", "农药", "推荐药"), "pill"),
    (("诊断", "症状", "判断", "识别", "表现", "特征"), "stethoscope"),
    (("农业", "农艺", "栽培", "耕作", "种植", "田间"), "sprout"),
    (("注意", "警告", "严重", "风险", "危害"), "triangle-alert"),
    (("生物", "天敌", "寄生"), "bug"),
    (("综合", "总结", "汇总", "概述"), "clipboard-check"),
    (("气候", "天气", "环境", "湿度", "温度"), "cloud-sun"),
]

# ── LLM API 配置 ──────────────────────────────────────────
# 配置文件放在可写目录（打包后为 exe 同级），避免写进只读的临时解压目录
CONFIG_PATH = str(APP_DIR / "config.json")
# 桌面程序标配：记住窗口大小/位置；版本号用于"关于"对话框
APP_VERSION = "3.8.0"
SETTINGS = QSettings(str(APP_DIR / "settings.ini"), QSettings.IniFormat)
DEFAULT_CONFIG = {
    # 云边联动：小程序跨网络时用得上——小程序传公网中转，电脑端主动拉（不需要公网 IP）
    "relay_base": "",
    "relay_token": "",
    "llm_provider": "deepseek",
    "deepseek_api_key": "",
    # 2026-10-13 更新：deepseek-chat / deepseek-reasoner 官方公告已于 2026-07-24 停止服务。
    # 现行模型：deepseek-flash（V4.1 Flash，原生多模态、1M 上下文、支持 Tool Calls）/ deepseek-v4-pro（不支持看图）
    "deepseek_model": "deepseek-flash",
    "dashscope_api_key": "",
    "qwen_model": "qwen-vl-max",
    # 自定义服务：任何 OpenAI 兼容端点都能接（硅基流动 / 智谱 / OpenRouter / 本地 Ollama、vLLM）
    "custom_base_url": "",
    "custom_api_key": "",
    "custom_model": "",
    # 看图能力：auto 按「服务商 + 模型名」自动判断，on/off 手动强制（自建网关、私有模型用得上）
    "vision_override": "auto",
    # 深度思考：DeepSeek 的思考模式默认开启，思维链会吃掉 token 预算、首字延迟也长。
    # 农户问答用不上长思维链，默认关；需要时在设置页打开。
    "llm_thinking": False,
    # 结构化「诊断卡」：回答出完后，再把结论抽成 JSON 并逐项核验（技术规范性）
    "enable_diagnosis_card": True,
    # 回答缓存：同一张图重复分析直接回放（省钱；只缓存首次分析）
    "enable_llm_cache": True,
    # 视觉复核级联：低置信度或病害三类时，自动请第二个模型独立看一遍
    "cascade_auto": True,
    "cascade_conf": 0.6,
    "enable_llm": False,
    "llm_max_tokens": 2000,
}

# 已停止服务的模型名 → 现行模型名。继续用旧名字只会拿到 404，这里自动升级并写日志。
RETIRED_MODELS = {
    "deepseek-chat": "deepseek-flash",
    "deepseek-reasoner": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
}


def _migrate_config(cfg):
    """把配置里已下线的模型名换成现行名字（不静默：写日志，用户能在日志里看到）"""
    old = cfg.get("deepseek_model", "")
    if old in RETIRED_MODELS:
        cfg["deepseek_model"] = RETIRED_MODELS[old]
        log.warning("模型 %s 已停止服务，自动改用 %s（见 DeepSeek 更新日志 2026-07-24）",
                    old, cfg["deepseek_model"])
    return cfg

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(saved)
            cfg = _migrate_config(cfg)
            log.info("已加载配置 %s（服务商: %s / 模型: %s / 开关: %s）",
                     CONFIG_PATH, cfg.get("llm_provider"), cfg.get("deepseek_model"),
                     cfg.get("enable_llm"))
            return cfg
        except Exception as e:
            log.warning("读取配置失败，改用默认配置: %s", e)
    else:
        log.info("未发现 config.json，使用默认配置")
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    """原子写入配置文件：先写临时文件再替换，避免中途崩溃留下半个 JSON。

    注意：config.json 里保存的是明文 API Key，请勿随程序一起打包分发。
    """
    tmp_path = CONFIG_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CONFIG_PATH)
        try:
            os.chmod(CONFIG_PATH, 0o600)      # POSIX 下收紧权限；Windows 由 ACL 决定
        except OSError:
            pass
        log.info("配置已保存到 %s", CONFIG_PATH)
    except OSError as e:
        log.error("保存配置失败: %s", e)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

CONFIG = load_config()

PROVIDER_KEYS = {"deepseek": "deepseek_api_key", "qwen": "dashscope_api_key",
                "custom": "custom_api_key"}
PROVIDER_MODEL_KEYS = {"deepseek": "deepseek_model", "qwen": "qwen_model",
                       "custom": "custom_model"}
PROVIDER_LABELS = {"deepseek": "DeepSeek", "qwen": "千问（阿里云）", "custom": "自定义服务"}
# 环境变量优先（便于 CI / 临时测试时不落盘 Key）
PROVIDER_ENV = {"deepseek": "DEEPSEEK_API_KEY", "qwen": "DASHSCOPE_API_KEY",
                "custom": "RICE_LLM_API_KEY"}

# ── 看图能力：按「服务商 + 模型名」判断，不再写死某一家 ──────────────
# 依据：DeepSeek 官方文档「图像理解」——deepseek-flash（V4.1 Flash）原生支持图片输入，
#       deepseek-v4-pro 不支持；千问按模型名里的 vl 判断；自定义服务各家命名不一，走关键词 + 手动开关。
VISION_MODELS = {
    "deepseek": {"deepseek-flash": True, "deepseek-v4-pro": False},
    "qwen": {},
}
VISION_HINTS = ("vl", "vision", "multimodal", "omni", "gpt-4o", "gpt-4.1", "gpt-5",
                "gemini", "claude-3", "claude-4", "glm-4v", "internvl", "minicpm-v",
                "llava", "step-1v", "yi-vision", "moonshot-v1-vision")


def get_api_key(provider=None):
    p = provider or CONFIG.get("llm_provider", "deepseek")
    return os.environ.get(PROVIDER_ENV.get(p, ""), "") or CONFIG.get(PROVIDER_KEYS.get(p, ""), "")

def get_model_name(provider=None):
    p = provider or CONFIG.get("llm_provider", "deepseek")
    return CONFIG.get(PROVIDER_MODEL_KEYS.get(p, ""), "") or ""

def provider_label():
    """给界面用的一行字：服务商 · 模型 ·（能否看图）"""
    p = CONFIG.get("llm_provider", "deepseek")
    model = get_model_name() or "未填模型"
    return "%s · %s%s" % (PROVIDER_LABELS.get(p, p), model,
                          " · 可看图" if model_supports_vision(p, model) else " · 纯文本")

def model_supports_vision(provider=None, model=None, override=None):
    """这个「服务商 + 模型」能不能读图。

    判断顺序：手动强制 > 显式表格 > 模型名关键词。
    拿不准时设置页可以强制开/关，代码里不猜死了。
    """
    provider = provider or CONFIG.get("llm_provider", "deepseek")
    model = (model if model is not None else get_model_name()) or ""
    model = model.lower()
    override = override if override is not None else CONFIG.get("vision_override", "auto")
    if override == "on":
        return True
    if override == "off":
        return False
    table = VISION_MODELS.get(provider) or {}
    if model in table:
        return table[model]
    if provider == "deepseek":
        # 现行只有 Flash 系能看图；v4-pro 与更早的文本模型不行
        return model.startswith("deepseek-flash") or "vision" in model
    return any(h in model for h in VISION_HINTS)


# ── HTTP 会话：复用连接 + 有限重试，避免每次请求都重新做 TLS 握手 ──
def _build_http_session():
    session = requests.Session()
    if Retry is None:                      # 极端情况下 urllib3 不可用，退化为无重试
        return session
    kwargs = dict(total=2, connect=2, read=0, status=2, other=0, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["POST"]), raise_on_status=False)
    try:
        retry = Retry(**kwargs)
    except TypeError:                      # urllib3 < 1.26 用的是 method_whitelist
        kwargs.pop("allowed_methods")
        retry = Retry(method_whitelist=frozenset(["POST"]), **kwargs)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


HTTP_SESSION = _build_http_session()
API_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}


def api_endpoint(provider=None):
    """指定/当前服务商的 chat/completions 地址；自定义服务按用户填的 base_url 拼"""
    p = provider or CONFIG.get("llm_provider", "deepseek")
    if p == "custom":
        base = (CONFIG.get("custom_base_url") or "").strip().rstrip("/")
        if not base:
            return ""
        return base if base.endswith("/chat/completions") else base + "/chat/completions"
    return API_ENDPOINTS.get(p, API_ENDPOINTS["deepseek"])


def provider_chain():
    """故障降级顺序：当前服务商打头，其余「配了 Key + 有模型 + 有地址」的依次兜底"""
    cur = CONFIG.get("llm_provider", "deepseek")
    order = [cur] + [p for p in ("deepseek", "qwen", "custom") if p != cur]
    return [p for p in order if get_api_key(p) and get_model_name(p) and api_endpoint(p)]


class LLMAnalysisThread(QThread):
    """调用大模型做分析/追问。

    流式优先：chunk 信号持续回传「已收到的全文」，界面边收边显示；
    服务端不支持流式（首个分片为空）时自动退回一次性请求。
    """
    chunk = pyqtSignal(str)
    reasoning = pyqtSignal(str)      # 思维链（思考模式下先于正文到达）
    tool_used = pyqtSignal(str, str)  # (工具名, 中文标签) —— 界面显示「正在查本地防治库…」
    fallback = pyqtSignal(str)        # 主服务商失败、自动切换备用时的通知
    result_ready = pyqtSignal(str)
    failed = pyqtSignal(str)

    SYSTEM_PROMPT = ("你是水稻病虫害防治专家，请用通俗中文回答用户问题。"
                     "回答分点、先给结论再给做法，不编造农药名称和用量。")

    def __init__(self, cls_name, image=None, history=None, prompt=None,
                 severity=None, boxes=None, parent=None):
        super().__init__(parent)
        self.cls_name = cls_name
        self.boxes = boxes or []   # 检测框，供 verify_with_vision 裁图复核
        self.image = image        # OpenCV BGR numpy array, 可选
        self.history = history    # 对话历史 [{"role":"user"/"assistant","content":...}], 可选
        self.prompt = prompt      # 追问时直接传入问题，不再用猴子补丁覆盖 _build_prompt
        self.severity = severity  # assess_severity() 的结果，可选
        self._stop = False
        self._resp = None
        self._reasoning = ""        # 思维链累计（只用来显示「在思考」，不展示内容）
        self.use_tools = True       # 是否允许模型调用只读工具（自检可关掉做对照）
        self.usage = None           # 服务端返回的 token 用量（用于成本记账）
        self._last_error = ""       # 最近一次失败原因（供降级链汇总后统一上报）

    def _build_prompt(self):
        if self.prompt is not None:
            return self.prompt
        sev = self.severity or {}
        evidence = ""
        if sev.get("boxes"):
            # 把可算的量交给模型，避免它凭空判断「严重程度」
            evidence = ("检测数据（供你判断严重程度，不要臆造数字）：共 %d 个检测框，"
                        "检出区域约占画面 %.1f%%，最大单框占 %.1f%%，平均置信度 %.0f%%。"
                        "注意：%s\n"
                        % (sev["boxes"], sev["area_ratio"] * 100,
                           sev["max_ratio"] * 100, sev["mean_conf"] * 100,
                           sev.get("basis", "")))
        return (
            f"你是水稻病虫害防治专家。检测到【{self.cls_name}】。\n"
            f"{evidence}"
            f"1. 确认诊断：该病虫害的典型症状\n"
            f"2. 严重程度判断（结合上面的检测数据说，并说明这只是图像尺度的估计）\n"
            f"3. 3条最有效的防治建议\n"
            f"4. 推荐1-2种药剂及用量（只能从本地防治库中选，不要编造）\n"
            f"请用通俗中文详细回答。"
        )

    def _system_prompt(self):
        return (self.SYSTEM_PROMPT + compliance_prompt_clause() + allowed_pesticide_clause()
                + "你可以调用只读工具查本地数据：防治库、合规名单、报告历史、反馈统计、天气，"
                  "也可以调低阈值重新检测当前照片。凡涉及具体数字（用量、历史、天气）先调工具，不要凭空回答。"
                  "写操作（导出报告、提交反馈）由用户在界面上确认，你不能代办。")

    def _image_to_base64(self, img_bgr):
        """OpenCV BGR 图像 → 压缩后 base64 JPEG（最长边不超过 768px）

        顺带一个隐私好处：图像走 OpenCV 解码再重新编码，EXIF（含 GPS）不会带上去。
        这条是「碰巧安全」，改动这里时要留意别把原图字节直接发出去。
        """
        try:
            if img_bgr is None or not hasattr(img_bgr, 'shape'):
                return ""
            h, w = img_bgr.shape[:2]
            max_side = 768
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as e:
            log.warning("图像编码失败，本次请求将只发送文本: %s", e)
            return ""

    def _build_messages(self):
        system_msg = {"role": "system", "content": self._system_prompt()}
        provider = CONFIG["llm_provider"]
        msgs = [system_msg]

        # 追加对话历史
        if self.history:
            msgs.extend(self.history)

        # 构建当前用户消息：带不带图由「模型能力」决定，不再写死服务商
        if self.image is not None and not model_supports_vision(provider):
            log.info("当前模型 %s（%s）不支持看图，本次只发文本；如需看图请在设置里换模型",
                     get_model_name(), provider)
        if self.image is not None and model_supports_vision(provider):
            b64 = self._image_to_base64(self.image)
            if b64:
                user_msg = {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": self._build_prompt()}
                ]}
            else:
                user_msg = {"role": "user", "content": self._build_prompt()}
        else:
            user_msg = {"role": "user", "content": self._build_prompt()}

        msgs.append(user_msg)
        return msgs

    def _payload_for(self, provider, messages, thinking_on):
        """按服务商组装请求体（thinking 参数只有 DeepSeek 系认；思考模式下 temperature 不生效）"""
        payload = {"model": get_model_name(provider), "messages": messages,
                   "max_tokens": CONFIG["llm_max_tokens"], "temperature": 0.3}
        if provider == "deepseek":
            # 让服务端在末尾分片里回传 token 用量，用于成本记账（OpenAI 兼容字段）
            payload["stream_options"] = {"include_usage": True}
        if provider == "deepseek" or (provider == "custom" and thinking_on):
            payload["thinking"] = {"type": "enabled" if thinking_on else "disabled"}
            if thinking_on:
                payload["max_tokens"] = max(int(CONFIG["llm_max_tokens"]), 4000)
                payload.pop("temperature", None)
        return payload

    def run(self):
        if not CONFIG["enable_llm"]:
            self.failed.emit("AI 分析未启用，请在设置中配置 API Key")
            return
        chain = provider_chain()
        if not chain:
            self.failed.emit("没有可用的服务商：请检查 API Key、模型名与接口地址")
            return
        import time as _time
        t0 = _time.perf_counter()
        thinking_on = bool(CONFIG.get("llm_thinking", False))
        messages = self._build_messages()       # 消息在多次尝试之间复用（工具结果不能丢）
        # 首次分析（无追问、无历史、带图）才查缓存
        ckey = None
        if (CONFIG.get("enable_llm_cache", True) and self.prompt is None
                and not self.history and self.image is not None):
            ckey = _cache_key(self.cls_name, self._build_prompt(), self.image,
                              get_model_name(chain[0]))
            hit = cache_get(ckey)
            if hit and hit.get("text"):
                log_llm_call(event="answer", provider=chain[0], model=hit.get("model"),
                             cache_hit=True, chars=len(hit["text"]), seconds=0.0)
                log_event("llm_cache_hit", cls=self.cls_name)
                self.chunk.emit(hit["text"])
                self.result_ready.emit(hit["text"])
                return
        tools_used = []
        errors = []
        # ── 服务商降级链：主服务商失败就换下一个，全失败才报错 ──
        for pi, provider in enumerate(chain):
            if self._stop:
                return
            if pi:
                self.fallback.emit("已切换到备用服务商：%s"
                                   % PROVIDER_LABELS.get(provider, provider))
            endpoint = api_endpoint(provider)
            headers = {"Authorization": f"Bearer {get_api_key(provider)}",
                       "Content-Type": "application/json"}
            failed_here = False
            try:
                # ── Agent 循环：模型可以先调工具，再基于工具结果作答 ──
                for round_i in range(AGENT_MAX_TOOL_ROUNDS + 1):
                    body = self._payload_for(provider, messages, thinking_on)
                    if self.use_tools and round_i < AGENT_MAX_TOOL_ROUNDS:
                        body["tools"] = AGENT_TOOLS
                        body["tool_choice"] = "auto"
                    content, calls = self._request(endpoint, headers, body, stream=True)
                    if content is None and not self._stop:   # 不支持流式 → 退回一次性
                        content, calls = self._request(endpoint, headers, body, stream=False)
                    if content is None:
                        if self._stop:
                            return
                        failed_here = True                   # 换下一个服务商
                        break
                    if calls and not self._stop:
                        messages.append(self._assistant_tool_message(content or "", calls))
                        for c in calls:
                            label = TOOL_LABELS.get(c["name"], c["name"])
                            self.tool_used.emit(c["name"], label)   # 界面显示「正在查…」
                            try:
                                targs = json.loads(c.get("arguments") or "{}")
                            except ValueError:
                                targs = {}
                            result = run_agent_tool(c["name"], targs, self.image, self.boxes)
                            tools_used.append(c["name"])
                            log_llm_call(event="tool", provider=provider, model=get_model_name(provider),
                                         tool=c["name"], args=targs)
                            messages.append({"role": "tool",
                                             "tool_call_id": c.get("id") or ("call_%d" % len(tools_used)),
                                             "content": result})
                        continue
                    if ckey and not tools_used:      # 用了工具的回答不缓存（含实时数据）
                        cache_put(ckey, content or "", get_model_name(provider))
                    self.result_ready.emit(content or "")
                    log_llm_call(event="answer", provider=provider, model=get_model_name(provider),
                                 rounds=round_i + 1, tools=tools_used, chars=len(content or ""),
                                 seconds=round(_time.perf_counter() - t0, 2),
                                 thinking=thinking_on, usage=self.usage,
                                 cost=estimate_cost(get_model_name(provider), self.usage))
                    return
                if not failed_here:
                    self.failed.emit("工具调用轮次已达上限（%d 轮），请换个问法"
                                     % AGENT_MAX_TOOL_ROUNDS)
                    return
                errors.append("%s：%s" % (provider, self._last_error or "无返回"))
            except requests.exceptions.Timeout:
                errors.append("%s：请求超时" % provider)
            except requests.exceptions.RequestException as e:
                errors.append("%s：网络异常 %s" % (provider, e))
            except Exception as e:
                log.exception("AI 请求异常（%s）", provider)
                errors.append("%s：%s" % (provider, e))
        if not self._stop:
            self.failed.emit("所有服务商均调用失败 → " + "；".join(errors))

    def _assistant_tool_message(self, content, calls):
        """把「模型提出工具调用」的那一轮 assistant 消息补回上下文（否则模型会重复调同一个工具）"""
        msg = {"role": "assistant", "content": content or None,
               "tool_calls": [{"id": c.get("id") or ("call_%d" % i), "type": "function",
                               "function": {"name": c.get("name", ""),
                                            "arguments": c.get("arguments") or "{}"}}
                              for i, c in enumerate(calls)]}
        if self._reasoning:
            # DeepSeek 官方：请求带 tools 时，历史轮次的思维链需要回传
            msg["reasoning_content"] = self._reasoning
        return msg

    def _request(self, endpoint, headers, payload, stream):
        """发一次请求。返回 (正文, 工具调用列表)；出错返回 (None, None)"""
        body = dict(payload, stream=True) if stream else payload
        resp = HTTP_SESSION.post(endpoint, headers=headers, json=body,
                                 timeout=(10, 300 if stream else 120), stream=stream)
        self._resp = resp
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:300]
            self._last_error = "API 错误 [%s]: %s" % (resp.status_code, str(detail)[:200])
            log.warning("模型接口返回错误：%s", self._last_error)
            try:
                resp.close()
            except Exception:
                pass
            return None, None
        if not stream:
            data = resp.json()
            try:
                msg = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                self._last_error = "返回格式异常: %s" % str(data)[:200]
                return None, None
            if isinstance(data.get("usage"), dict):
                self.usage = data["usage"]
            rc = msg.get("reasoning_content") or ""
            if rc:
                self._reasoning = rc
                self.reasoning.emit(rc)
            calls = [{"id": tc.get("id"),
                      "name": (tc.get("function") or {}).get("name", ""),
                      "arguments": (tc.get("function") or {}).get("arguments") or "{}"}
                     for tc in (msg.get("tool_calls") or [])]
            return msg.get("content") or "", calls
        got, calls = self._consume_sse(resp)
        if not got and not calls and not self._stop:
            log.info("流式返回为空，改用一次性请求重试")
            return None, None
        return got, calls

    def _consume_sse(self, resp):
        """解析 OpenAI 兼容的 SSE 流。

        · 正文：chunk 每次回传「累计全文」，界面不必自己拼接
        · 工具调用：tool_calls 是分片下发的，按 index 累加 name/arguments
        返回 (正文, 工具调用列表)
        """
        parts = []
        slots = {}
        try:
            for raw in resp.iter_lines(decode_unicode=False):
                if self._stop:
                    break
                if not raw:
                    continue
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    if isinstance(obj.get("usage"), dict):   # 末尾分片带用量
                        self.usage = obj["usage"]
                    delta = obj["choices"][0].get("delta") or {}
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if think:
                    # 思维链不展示内容，只让界面知道「它在思考」
                    self._reasoning += think
                    self.reasoning.emit(self._reasoning)
                for tc in (delta.get("tool_calls") or []):
                    idx = int(tc.get("index") or 0)
                    slot = slots.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if piece:
                    parts.append(piece)
                    self.chunk.emit("".join(parts))
        except Exception as e:
            if not self._stop:                 # 用户主动停止不算异常
                log.warning("流式读取中断: %s", e)
        finally:
            try:
                resp.close()
            except Exception:
                pass
        calls = [slots[k] for k in sorted(slots) if slots[k].get("name")]
        return "".join(parts), calls

    def stop(self):
        """请求停止生成：关掉连接，已经收到的内容照常显示"""
        self._stop = True
        resp = self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════
# 任务规划与执行（plan → 人工确认 → 执行 → 汇报）
# --------------------------------------------------------------
# 与「聊天」的区别：聊天是问一句答一句；这里是接一个目标，
# 先由模型把它拆成可审阅的步骤（只用只读工具），用户点了「执行」才动手，
# 每步都有进度、失败会重试一次、最后给结论。这是 agent 的任务式形态。
TASK_MAX_STEPS = 6
TASK_ALLOWED_TOOLS = [t["function"]["name"] for t in AGENT_TOOLS]


def _tools_manifest():
    return "\n".join("- %s：%s" % (t["function"]["name"], t["function"]["description"])
                     for t in AGENT_TOOLS)


def build_task_plan_prompt(goal):
    return (
        "你是任务规划器。用户目标：%s\n\n"
        "可以使用的只读工具：\n%s\n\n"
        "要求：\n"
        "1. 只输出 JSON，格式：{\"goal\":\"...\",\"steps\":[{\"tool\":\"工具名\","
        "\"args\":{...},\"why\":\"为什么需要这一步\"}],\"note\":\"给用户的说明\"}\n"
        "2. 最多 %d 步；每步只能用上面列出的工具；args 用工具要求的字段。\n"
        "3. 用户没有明确给出路径时，args 里不要写路径（省略该字段），由工具使用默认值；"
        "严禁臆造目录名或文件名。\n"
        "4. 现有工具做不到的事，就在 note 里说明差距，不要编造工具或数据。\n"
        % (goal, _tools_manifest(), TASK_MAX_STEPS))


def validate_task_plan(plan):
    """校验计划：未知工具剔除、步数上限、args 必须是对象。返回 (plan, warnings)"""
    warnings = []
    plan = dict(plan or {})
    steps = []
    for s in (plan.get("steps") or []):
        if not isinstance(s, dict):
            continue
        name = str(s.get("tool", "")).strip()
        if name not in TASK_ALLOWED_TOOLS:
            warnings.append("剔除了不存在的工具：%s" % name)
            continue
        args = s.get("args") or {}
        if not isinstance(args, dict):
            warnings.append("%s 的参数格式不对，已置空" % name)
            args = {}
        steps.append({"tool": name, "args": args, "why": str(s.get("why", ""))[:120]})
        if len(steps) >= TASK_MAX_STEPS:
            break
    plan["steps"] = steps
    plan["_warnings"] = warnings
    return plan, warnings


def call_llm_json(prompt, max_tokens=900, timeout=(10, 120)):
    """一次性 JSON 调用（规划/抽取共用）。返回 (dict, error)"""
    endpoint = api_endpoint()
    if not endpoint:
        return None, "未配置接口地址"
    payload = {"model": get_model_name(),
               "messages": [{"role": "system", "content": "你只输出合法 JSON。"},
                            {"role": "user", "content": prompt}],
               "response_format": {"type": "json_object"},
               "max_tokens": max_tokens, "temperature": 0}
    if CONFIG.get("llm_provider", "deepseek") == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    try:
        resp = HTTP_SESSION.post(endpoint, json=payload, timeout=timeout,
                                 headers={"Authorization": "Bearer " + get_api_key(),
                                          "Content-Type": "application/json"})
        if resp.status_code != 200:
            return None, "HTTP %s" % resp.status_code
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = text.strip(fence).strip()
            if text[:4].lower() == "json":
                text = text[4:].strip()
        usage = data.get("usage") or {}
        log_llm_call(event="json_call", model=get_model_name(), usage=usage,
                     cost=estimate_cost(get_model_name(), usage))
        return json.loads(text), ""
    except Exception as e:
        return None, str(e)


def run_task_step(step, image=None, boxes=None):
    """执行一步（含一次重试）。返回 (ok, 文本结果)"""
    name = step.get("tool", "")
    args = step.get("args") or {}
    if name not in TASK_ALLOWED_TOOLS:      # 计划校验之外再挡一次（防御性）
        return False, "不允许的工具：%s" % name
    for attempt in (1, 2):
        try:
            out = run_agent_tool(name, args, image, boxes)
        except Exception as e:
            out = "异常：%s" % e
        text = str(out)
        bad = ((not out) or text.startswith("工具 %s 执行失败" % name)
               or "异常：" in text[:20]
               # 工具自己报的错（例如目录不存在）也算失败，别让汇总把它当成功
               or text.lstrip().startswith('{"错误"') or '"错误":' in text[:40])
        if not bad:
            return True, str(out)
        if attempt == 2:
            return False, str(out)
        log.info("任务步骤 %s 第 1 次失败，重试", name)
    return False, ""


def summarize_task(plan, results):
    """把执行结果拼成给模型的汇报材料"""
    lines = ["任务目标：%s" % plan.get("goal", "")]
    for i, (step, ok, out) in enumerate(results, 1):
        lines.append("%d. %s（%s）→ %s" % (i, step.get("tool"), "成功" if ok else "失败",
                                        str(out)[:600]))
    return "\n".join(lines)


class TaskPlanThread(QThread):
    """把一句话目标拆成可审阅的步骤（只规划，不动手）"""
    planned = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, goal, parent=None):
        super().__init__(parent)
        self.goal = goal

    def run(self):
        if not CONFIG.get("enable_llm", False) or not get_api_key():
            self.failed.emit("未启用云端 AI，无法规划")
            return
        data, err = call_llm_json(build_task_plan_prompt(self.goal))
        if data is None:
            self.failed.emit("规划失败：%s" % err)
            return
        plan, warns = validate_task_plan(data)
        plan["goal"] = plan.get("goal") or self.goal
        log_event("task_planned", goal=plan["goal"], steps=len(plan["steps"]), warnings=warns)
        self.planned.emit(plan)


TASK_MAX_REPLANS = 2       # 失败后最多自动重规划几次（防死循环）


def build_replan_prompt(goal, done_text, fail_text):
    return (
        "你是任务规划器。原目标：%s\n\n"
        "已经执行过的步骤与结果：\n%s\n\n"
        "其中这些步骤失败了：\n%s\n\n"
        "请给出修正后的**剩余步骤**：\n"
        "1. 只输出还需要新执行的步骤，不要重复已经成功的步骤。\n"
        "2. 最多 %d 步；只能用这些工具：\n%s\n"
        "3. 失败若是参数问题（例如目录不存在），就修正参数重试；"
        "若确实做不到，就返回空的 steps 并在 note 里说明原因。\n"
        "4. 只输出 JSON：{\"steps\":[{\"tool\":\"...\",\"args\":{...},\"why\":\"...\"}],"
        "\"note\":\"...\"}\n"
        % (goal, done_text, fail_text, TASK_MAX_STEPS, _tools_manifest()))


def plan_signature(step):
    """步骤指纹：用于去重（同工具同参数 = 同一步）"""
    return "%s|%s" % (step.get("tool", ""), json.dumps(step.get("args") or {},
                                                     ensure_ascii=False, sort_keys=True))


class TaskRunThread(QThread):
    """按计划逐步执行；失败会自动重规划（plan → act → observe → replan）"""
    step_started = pyqtSignal(int, dict)
    step_done = pyqtSignal(int, dict, bool, str)
    replanning = pyqtSignal(int, str)          # (第几轮重规划, 原因说明)
    task_done = pyqtSignal(dict, list)
    failed = pyqtSignal(str)

    def __init__(self, plan, image=None, boxes=None, max_replans=TASK_MAX_REPLANS, parent=None):
        super().__init__(parent)
        self.plan = plan or {}
        self.image = image
        self.boxes = boxes or []
        self.max_replans = max_replans
        self._stop = False
        self.rounds = 0

    def _execute(self, plan):
        results = []
        for i, step in enumerate(plan.get("steps") or []):
            if self._stop:
                break
            self.step_started.emit(i, step)
            ok, out = run_task_step(step, self.image, self.boxes)
            results.append((step, ok, out))
            self.step_done.emit(i, step, ok, out)
            log_event("task_step", tool=step.get("tool"), ok=ok)
        return results

    def _replan(self, goal, all_results, fails):
        """把「已经做了什么、什么失败了」交给规划器，换一份修正计划"""
        if not CONFIG.get("enable_llm", False) or not get_api_key():
            log.info("未启用云端 AI，跳过重规划")
            return None
        done = "\n".join("%d. %s（%s）→ %s" % (i, s.get("tool"), "成功" if ok else "失败",
                                             str(o)[:200])
                         for i, (s, ok, o) in enumerate(all_results, 1))
        fail_txt = "\n".join("- %s：%s" % (s.get("tool"), str(o)[:200]) for s, _, o in fails)
        data, err = call_llm_json(build_replan_prompt(goal, done, fail_txt))
        if data is None:
            log.warning("重规划失败：%s", err)
            return None
        plan, warns = validate_task_plan(data)
        goal_sig = {plan_signature(s) for s, ok, _ in all_results if ok}
        steps = [s for s in plan.get("steps", []) if plan_signature(s) not in goal_sig]
        plan["steps"] = steps
        plan["_warnings"] = (warns or []) + (
            ["已剔除 %d 个已成功执行过的重复步骤"
             % (len(validate_task_plan(data)[0]["steps"]) - len(steps))] if steps else [])
        log_event("task_replan", steps=len(steps), warnings=plan["_warnings"])
        return plan

    def run(self):
        try:
            plan = self.plan
            all_results = []
            goal = plan.get("goal", "")
            while True:
                if self._stop:
                    break
                results = self._execute(plan)
                all_results += results
                fails = [(s, ok, o) for s, ok, o in results if not ok]
                if (not fails) or self.rounds >= self.max_replans or self._stop:
                    break
                self.rounds += 1
                why = "第 %d 步失败：%s" % (len(all_results), str(fails[0][2])[:80])
                self.replanning.emit(self.rounds, why)
                new_plan = self._replan(goal, all_results, fails)
                if not new_plan or not new_plan.get("steps"):
                    note = (new_plan or {}).get("note") or "规划器认为现有工具无法完成剩余部分"
                    log.info("不再重规划：%s", note)
                    if new_plan is not None:
                        all_results.append(({"tool": "(重规划)", "args": {}}, False, note))
                    break
                plan = new_plan
            self.task_done.emit(plan, all_results)
        except Exception as e:
            log.exception("任务执行异常")
            self.failed.emit(str(e))

    def stop(self):
        self._stop = True


class TaskQueue(QObject):
    """任务队列：一次只跑一个任务，后来的排队（互斥，避免多任务抢同一套对话状态）"""
    started = pyqtSignal(dict)
    step_started = pyqtSignal(int, dict)
    step_done = pyqtSignal(int, dict, bool, str)
    replanning = pyqtSignal(int, str)
    finished = pyqtSignal(dict, list)
    failed = pyqtSignal(str)
    changed = pyqtSignal(list)          # 队列快照（界面显示用）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.thread = None
        self.current = None

    def snapshot(self):
        out = []
        if self.current:
            out.append({"goal": self.current.get("plan", {}).get("goal", ""), "state": "执行中"})
        for it in self.items:
            out.append({"goal": it.get("plan", {}).get("goal", ""), "state": "排队中"})
        return out

    def is_busy(self):
        return self.thread is not None and self.thread.isRunning()

    def submit(self, plan, image=None, boxes=None):
        """入队；返回排队位置（1 = 立即开始）"""
        self.items.append({"plan": plan or {}, "image": image, "boxes": boxes or []})
        # 位置 = 正在跑的 1 个 + 前面排队的（items 里不含正在跑的那个）
        # 注意：不能用 is_busy()——线程 start() 之后 isRunning() 还可能短暂为 False，
        # 会把"排队第 2 个"错报成"立即执行"。current 是 _pump 里同步设置的，更可靠。
        running = self.current is not None or self.is_busy()
        pos = (len(self.items) + 1) if running else 1
        log_event("task_enqueue", goal=(plan or {}).get("goal"),
                  steps=len((plan or {}).get("steps") or []), position=pos)
        self.changed.emit(self.snapshot())
        self._pump()
        return pos

    def _pump(self):
        if self.is_busy() or not self.items:
            return
        item = self.items.pop(0)
        self.current = item
        t = _track_thread(TaskRunThread(item["plan"], item["image"], item["boxes"]))
        self.thread = t
        t.step_started.connect(self.step_started)
        t.step_done.connect(self.step_done)
        t.replanning.connect(self.replanning)
        t.task_done.connect(self._on_done)
        t.failed.connect(self._on_failed)
        self.started.emit(item["plan"])
        self.changed.emit(self.snapshot())
        t.start()

    def _on_done(self, plan, results):
        self.finished.emit(plan, results)
        self.current = None
        self.thread = None
        self.changed.emit(self.snapshot())
        self._pump()                     # 接着跑下一个

    def _on_failed(self, msg):
        self.failed.emit(msg)
        self.current = None
        self.thread = None
        self.changed.emit(self.snapshot())
        self._pump()

    def cancel_current(self):
        """停止当前任务：队列继续往下走"""
        if self.is_busy():
            self.thread.stop()
            log_event("task_cancel", goal=(self.current or {}).get("plan", {}).get("goal"))

    def clear_pending(self):
        self.items = []
        self.changed.emit(self.snapshot())


class VisionCascadeThread(QThread):
    """低置信度 / 病害三类自动触发的视觉复核：让第二个模型独立判断一次。

    与工具版共用 vision_recheck()，区别是这里由规则触发、结果要交给用户裁决。
    """
    verdict = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cls_name, conf, image, boxes=None, reason="", parent=None):
        super().__init__(parent)
        self.cls_name = cls_name
        self.conf = conf
        self.image = image
        self.boxes = boxes or []
        self.reason = reason

    def run(self):
        box = None
        if self.boxes:
            try:
                best = max(self.boxes, key=lambda x: x.get("conf", 0.0))
                box = best.get("box")
            except Exception:
                box = None
        res = vision_recheck(self.image, box, None, 0, source="cascade")
        if not res.get("ok"):
            self.failed.emit(res.get("error", "复核失败"))
            return
        out = dict(res)
        out["cls_name"] = self.cls_name
        out["conf"] = self.conf
        out["reason"] = self.reason
        out["agrees"] = bool(res.get("verdict")) and res["verdict"] == self.cls_name
        self.verdict.emit(out)


class DiagnosisCardThread(QThread):
    """把 AI 的长篇回答整理成结构化「诊断卡」，并逐项对本地库校验。

    为什么要多这一步（对应评审的「技术规范性 / 人机协同」）：
      · 自由文本无法核验，结构化字段才能逐项校验（药剂在不在库、是否禁用）；
      · 校验结论直接展示给用户，AI 输出不是黑箱；
      · 报告与赛后统计可以直接吃结构化字段。
    """
    card_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cls_name, answer, severity=None, parent=None):
        super().__init__(parent)
        self.cls_name = cls_name
        self.answer = answer or ""
        self.severity = severity or {}

    def _prompt(self):
        return (
            "把下面这段水稻病虫害诊断建议，整理成 JSON（只输出 JSON，不要解释）。\n"
            "字段：diagnosis(诊断结论,字符串)、severity_level(轻/中/重 之一)、"
            "actions(防治措施,字符串数组,最多4条)、"
            "chemicals(药剂数组,每项 {name, dose, reason})、"
            "safety(安全间隔期与注意事项,字符串)、uncertain(不确定或需要人工确认的点,字符串数组)。\n"
            "硬要求：药剂名只能来自原文；原文没写用量就留空字符串，不要编造。\n"
            "检测类别：%s；图像检出强度：%s。\n\n原文：\n%s"
            % (self.cls_name or "未检出", self.severity.get("level", "未计算"), self.answer[:4000]))

    def run(self):
        if not CONFIG["enable_llm"] or not get_api_key():
            self.failed.emit("未启用 AI")
            return
        endpoint = api_endpoint()
        if not endpoint:
            self.failed.emit("未配置接口地址")
            return
        payload = {
            "model": get_model_name(),
            "messages": [{"role": "system", "content": "你是信息抽取助手，只输出合法 JSON。"},
                         {"role": "user", "content": self._prompt()}],
            "response_format": {"type": "json_object"},
            "max_tokens": 900,
            "temperature": 0,
        }
        if CONFIG.get("llm_provider", "deepseek") == "deepseek":
            payload["thinking"] = {"type": "disabled"}     # 抽取任务不需要思维链
        headers = {"Authorization": f"Bearer {get_api_key()}",
                   "Content-Type": "application/json"}
        t0 = datetime.now()
        try:
            resp = HTTP_SESSION.post(endpoint, headers=headers, json=payload,
                                     timeout=(10, 120), stream=False)
            if resp.status_code != 200:
                self.failed.emit("诊断卡请求失败 [%s]" % resp.status_code)
                return
            msg = resp.json()["choices"][0]["message"]
            raw = (msg.get("content") or "").strip()
            fence = chr(96) * 3                            # 去掉代码块围栏（不写反引号，避免转义麻烦）
            if raw.startswith(fence):
                raw = raw.strip(fence).strip()
                if raw[:4].lower() == "json":
                    raw = raw[4:].strip()
            try:
                data = json.loads(raw)
            except ValueError:
                self.failed.emit("模型没有返回合法 JSON")
                return
            card, warnings = validate_diagnosis(data, self.severity)
            card["_model"] = get_model_name()
            log_llm_call(event="diagnosis_card", model=get_model_name(),
                         chemicals=[c.get("name") for c in card.get("chemicals", [])],
                         warnings=warnings,
                         seconds=round((datetime.now() - t0).total_seconds(), 2))
            self.card_ready.emit(card)
        except Exception as e:
            log.warning("诊断卡生成异常: %s", e)
            self.failed.emit(str(e))


def render_diagnosis_card(card, severity=None):
    """把校验后的诊断卡渲染成对话区里的一块 HTML（来源标注 + 警告分色显示）"""
    def esc(s):
        return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    rows = []
    lv = (severity or {}).get("level") or card.get("severity_level") or "—"
    rows.append("<div style='font-size:13px; margin:2px 0;'><b>诊断：</b>%s　"
                "<b>检出强度：</b>%s</div>" % (esc(card.get("diagnosis") or "—"), esc(lv)))
    actions = card.get("actions") or []
    if actions:
        rows.append("<div style='font-size:13px; margin:6px 0 2px 0;'><b>措施：</b></div>")
        for a in actions[:4]:
            rows.append("<div style='font-size:13px; margin-left:10px;'>· %s</div>" % esc(a))
    chems = card.get("chemicals") or []
    if chems:
        rows.append("<div style='font-size:13px; margin:6px 0 2px 0;'><b>药剂（逐条核验）：</b></div>")
        for c in chems:
            src = c.get("source")
            if src == "banned":
                color, mark = "#C0392B", "⛔ 禁用/停用"
            elif src == "local":
                color, mark = "#0F7B4A", "✅ 本地防治库"
            else:
                color, mark = "#B7791F", "⚠ 库外，需人工核实"
            rows.append("<div style='font-size:13px; margin-left:10px; color:%s;'>· %s %s"
                        "　<span style='color:#5F6B66;'>%s</span></div>"
                        % (color, esc(c.get("name")), esc(c.get("dose") or ""), mark))
    if card.get("safety"):
        rows.append("<div style='font-size:13px; margin:6px 0 2px 0;'><b>安全间隔：</b>%s</div>"
                    % esc(card["safety"]))
    for u in (card.get("uncertain") or [])[:3]:
        rows.append("<div style='font-size:12px; color:#5F6B66; margin-left:10px;'>？%s</div>" % esc(u))
    for w in (card.get("_warnings") or []):
        rows.append("<div style='font-size:12.5px; color:#C0392B; margin-top:4px;'>⚠ %s</div>" % esc(w))
    rows.append("<div style='font-size:11.5px; color:#5F6B66; margin-top:6px;'>"
                "结构化结果由 %s 生成，药剂名已对本地防治库与禁用名单逐条核验；"
                "如与实际情况不符，请点右上角「识别不对」反馈。</div>"
                % esc(card.get("_model") or "AI"))
    return ("<div style='background:#F2F7F4; border-left:3px solid #0F7B4A; border-radius:6px;"
            " padding:10px 12px; margin:8px 0;'>"
            "<div style='font-size:13px; color:#0F7B4A; font-weight:bold; margin-bottom:4px;'>"
            "诊断卡 · AI 生成 · 已按本地库校验</div>" + "".join(rows) + "</div>")


# ── 全局模型单例（避免重复加载，解决返回首页卡顿） ──────────
_global_model = None
_global_device = None


def get_model():
    global _global_model, _global_device
    if _global_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"未找到模型权重文件：{MODEL_PATH}")
        _global_device = 0 if torch.cuda.is_available() else "cpu"
        log.info("加载模型 %s（设备: %s）", MODEL_PATH, _global_device)
        _global_model = YOLO(MODEL_PATH)
        _global_model.to(_global_device)
        _global_model(torch.zeros((1, 3, 640, 640)).to(_global_device), verbose=False)
        log.info("模型加载完成")
    return _global_model, _global_device


# ── 后台线程登记表（退出前统一等待，避免 QThread 未结束就被析构） ──
_ACTIVE_THREADS = []


def _track_thread(thread):
    """登记运行中的 QThread，结束后自动移出列表。"""
    _ACTIVE_THREADS.append(thread)
    thread.finished.connect(
        lambda: _ACTIVE_THREADS.remove(thread) if thread in _ACTIVE_THREADS else None)
    return thread


def _wait_active_threads(timeout_ms=2000):
    """退出前等待后台线程收尾（带超时，避免卡住退出流程）。"""
    for t in list(_ACTIVE_THREADS):
        try:
            if t.isRunning():
                t.wait(timeout_ms)
        except RuntimeError:
            pass


_BGR888 = getattr(QImage, "Format_BGR888", None)   # PyQt5 >= 5.14 才有；有它就不用先 cvtColor


def bgr_to_qpixmap(img_bgr, size=None):
    """OpenCV BGR ndarray → QPixmap（可选按尺寸等比缩放）。

    QImage 不复制外部缓冲区，这里紧接着 fromImage 转成 QPixmap（深拷贝），
    不会出现 numpy 数组被回收后的野指针问题。
    """
    h, w = img_bgr.shape[:2]
    if _BGR888 is not None:
        qimg = QImage(img_bgr.data, w, h, img_bgr.strides[0], _BGR888)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if size is not None:
        pix = pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


_FONT_CANDIDATES = [
    str(FONT_DIR / "HarmonyOS_Sans_SC_Bold.ttf"),   # 随程序分发的字体优先
    str(FONT_DIR / "NotoSansSC-Bold.otf"),
    str(FONT_DIR / "NotoSansSC-Medium.otf"),
    FONT_PATH,
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


@lru_cache(maxsize=32)
def _load_font(size):
    """按候选顺序加载中文字体，全部失败时退回 PIL 默认字体。

    结果做 LRU 缓存：摄像机模式每帧都会调用，避免反复解析字体文件。
    """
    for fp in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    log.warning("未找到可用的中文字体，标注文字可能显示为方框")
    return ImageFont.load_default()


def annotate_image(img_bgr, boxes):
    """
    在 OpenCV BGR 图像上绘制检测框和中文标签。
    使用 PIL 绘制中文文本，避免 cv2.putText 不支持中文的问题。
    返回标注后的 BGR 图像。

    视觉规范（与品牌色板 NAU 对齐）：
    · 虫害框用品牌金、病害框用品牌红，延续全应用「红＝需要警惕」的语义；
    · 标签为圆角药丸、默认贴在框上方，贴到画面边缘时移回框内，不会超出画面；
    · 框线带半透明白色打底，深色叶丛 / 高光叶面上都看得清；
    · 字号与线宽随图片尺寸缩放，摄像头小帧不压画面，手机大图不显小。
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    oh, ow = rgb.shape[:2]
    short = min(oh, ow)
    fs = int(max(24, min(48, short * 0.055)))
    lw = max(2, min(8, round(short * 0.004)))
    font = _load_font(fs)
    pad_x = max(8, fs // 3)
    pad_y = max(4, fs // 5)

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    for item in boxes:
        x1, y1, x2, y2 = map(int, item["box"])
        text = "%s %s%%" % (item["name"], ("%.1f" % (float(item["conf"]) * 100)))
        if item["name"] in PEST_CLASSES:
            fill, ink = (243, 179, 27, 255), (18, 48, 31, 255)     # 品牌金 + 墨绿字
        else:
            fill, ink = (192, 57, 43, 255), (255, 255, 255, 255)   # 品牌红 + 白字

        # 检测框：白色打底描边 + 类色圆角框（小图上线宽细，描边要更淡才不抢主线）
        rad = max(0, min(max(6, lw * 3), (x2 - x1) // 4, (y2 - y1) // 4))
        d.rounded_rectangle([x1, y1, x2, y2], radius=rad,
                            outline=(255, 255, 255, 100 if lw <= 2 else 150),
                            width=lw + 2)
        d.rounded_rectangle([x1, y1, x2, y2], radius=rad, outline=fill, width=lw)

        # 标签药丸：优先贴框上方；上方放不下就移进框内顶部
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pw, ph = tw + pad_x * 2, th + pad_y * 2
        gap = lw + 2
        px = min(max(0, x1), max(0, ow - pw - 1))
        py = y1 - ph - gap
        if py < 0:
            py = max(0, min(y1 + gap, oh - ph - 1))
        d.rounded_rectangle([px, py, px + pw, py + ph], radius=ph // 2, fill=fill)
        d.text((px + pad_x - tb[0], py + (ph - th) // 2 - tb[1]), text,
               font=font, fill=ink)

    out = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


# ── 检出强度量化（图像尺度估计，不是田间病情指数） ────────────────
# 说明：检测框面积占比回答的是「这张照片上检出区域占多大」，不等于田间病叶率/病情指数。
# 用「检出区域」而不是「病斑」：虫害三类（卷叶螟/水象甲/叶蝉）检出的不是病斑。
# 分级阈值是图像尺度的经验值，只用来给农户一个量级；文本里始终带 basis 说明。
SEVERITY_BANDS = ((0.01, "轻"), (0.05, "中"))     # <1% 轻；1%~5% 中；≥5% 重
SEVERITY_BASIS = "按检测框占画面面积估计，不是田间病情指数，仅供参考"


def assess_severity(img_shape, result_list):
    """把检测框换算成可解释的检出强度：框数、面积占比、最大框占比、平均置信度"""
    h, w = (int(img_shape[0]), int(img_shape[1])) if img_shape else (0, 0)
    area = float(max(1, h * w))
    ratios, confs = [], []
    for item in result_list or []:
        x1, y1, x2, y2 = item.get("box", (0, 0, 0, 0))
        ratios.append(max(0.0, (x2 - x1) * (y2 - y1)) / area)
        confs.append(float(item.get("conf", 0.0)))
    total = min(1.0, sum(ratios))
    level = "无"
    if ratios:
        level = "重"
        for edge, name in SEVERITY_BANDS:
            if total < edge:
                level = name
                break
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    text = ("未检出目标" if not ratios else
            "检出 %d 处 · 检出区域约占画面 %.1f%% · 平均置信度 %.0f%%"
            % (len(ratios), total * 100, mean_conf * 100))
    return {"level": level, "boxes": len(ratios), "area_ratio": total,
            "max_ratio": max(ratios) if ratios else 0.0, "mean_conf": mean_conf,
            "text": text, "basis": SEVERITY_BASIS}


class InferenceThread(QThread):
    """后台推理线程。

    信号不叫 finished：那是 QThread 的内置信号，同名覆盖容易出现难以排查的行为差异。
    """
    result_ready = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, model, image, parent=None):
        super().__init__(parent)
        self.model = model
        self.image = image

    def run(self):
        try:
            results = self.model(self.image, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
            result_list = []
            for box in results[0].boxes:
                result_list.append({
                    "name": CLASS_NAMES[int(box.cls.item())],
                    "conf": float(box.conf.item()),
                    "box": box.xyxy[0].tolist()
                })
            self.result_ready.emit(result_list)
        except Exception as e:
            # 必须回传失败，否则主线程永远停在识别中，按钮也不会恢复
            log.exception("推理失败")
            self.failed.emit(f"推理失败: {e}")


def watch_thumb_pixmap(image_path, size=(72, 54)):
    """列表缩略图。

    走 PIL 的 draft()：JPEG 只解 1/8 分辨率，比 cv2.imread 全解快一个量级 ——
    收件箱里动辄 4~5MB 的手机原图，全解的话十几条就能把页面卡住。
    """
    try:
        with Image.open(image_path) as im:
            im.draft("RGB", size)
            im.thumbnail(size)
            rgb = im.convert("RGB")
            data = rgb.tobytes("raw", "RGB")
            qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
    except Exception as e:
        log.debug("缩略图生成失败（%s）：%s", image_path, e)
        return None


def md_to_html(text):
    """把 AI 返回的 Markdown 转成 HTML

    output_format="html5"：Python-Markdown 默认用 XHTML 序列化，自闭合标签会输出成
    <hr /> <br />，导致 _style_md 里针对 <hr> 的样式替换永远不生效。
    """
    try:
        return markdown.markdown(text, extensions=['extra', 'nl2br'], output_format='html5')
    except Exception as e:
        log.warning("Markdown 渲染失败，退化为纯文本: %s", e)
        return text.replace("\n", "<br>")


# ═══════════════════════════════════════════════════════════════════
#  Fluent 界面层（PyQt-Fluent-Widgets，GPL-3.0，见 resources/icons/NOTICE.txt）
#  一个主窗口 + 左侧导航，替代原来 5 个各自手绘无边框窗口的写法
# ═══════════════════════════════════════════════════════════════════
from qfluentwidgets import (FluentWindow, NavigationItemPosition, FluentIcon as FIF,
                            setTheme, Theme, setThemeColor, TitleLabel, SubtitleLabel,
                            StrongBodyLabel, BodyLabel, CaptionLabel, LargeTitleLabel,
                            PrimaryPushButton, PushButton, TransparentPushButton,
                            TransparentToolButton,
                            HyperlinkButton, CardWidget, SimpleCardWidget, ElevatedCardWidget,
                            HeaderCardWidget, ImageLabel, ProgressBar, IndeterminateProgressBar,
                            ScrollArea, TextEdit, LineEdit, PasswordLineEdit, ComboBox,
                            SwitchButton, InfoBar, InfoBarPosition, MessageBox,
                            MessageBoxBase, EditableComboBox, SegmentedWidget, CheckBox,
                            isDarkTheme)


def _page(title, subtitle=""):
    """所有页面的公共外壳：标题 + 副标题 + 内容容器"""
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(36, 28, 36, 28)
    lay.setSpacing(6)
    lay.addWidget(TitleLabel(title, page))
    if subtitle:
        cap = CaptionLabel(subtitle, page)
        lay.addWidget(cap)
    lay.addSpacing(12)
    return page, lay


def _scroll_body(parent):
    """可滚动的内容区（内容多时自动出滚动条）"""
    scroll = ScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.enableTransparentBackground()
    body = QWidget()
    body.setStyleSheet("background: transparent;")
    box = QVBoxLayout(body)
    box.setContentsMargins(0, 0, 12, 0)
    box.setSpacing(12)
    scroll.setWidget(body)
    return scroll, box


def _drug_card(drug, parent):
    """一行药剂：缩略图 + 名称 + 规格/用量"""
    card = SimpleCardWidget(parent)
    lay = QHBoxLayout(card)
    lay.setContentsMargins(12, 10, 14, 10)
    lay.setSpacing(12)
    img = ImageLabel(card)
    img.setFixedSize(64, 64)
    img.setAlignment(Qt.AlignCenter)
    pix = load_treatment_image(drug.get("img", ""), 60, 60)
    if pix:
        img.setPixmap(pix)
    else:
        img.setText("无图")
    lay.addWidget(img)
    col = QVBoxLayout()
    col.setSpacing(3)
    col.addWidget(_sized(StrongBodyLabel(drug.get("name", ""), card), "body", weight=QFont.Medium))
    detail = " · ".join(p for p in (drug.get("spec", ""), drug.get("dose", "")) if p)
    if detail:
        col.addWidget(t_caption(detail, card, color=NAU["subtext"]))
    # 专业信息一行：登记证号 / 安全间隔期 / 每季次数（缺的显示「待补」，不藏缺口）
    bits = []
    if drug.get("reg_no"):
        bits.append(str(drug["reg_no"]).split("（")[0])
    bits.append("间隔期 %s 天" % drug["safe_days"] if drug.get("safe_days") else "间隔期待补")
    if drug.get("max_per_season"):
        bits.append("每季≤%s 次" % drug["max_per_season"])
    if drug.get("type"):
        bits.append(drug["type"])
    col.addWidget(t_caption(" · ".join(bits), card, color=NAU["subtext"]))
    tip = "；".join(p for p in (drug.get("timing") and "适期：" + drug["timing"],
                                drug.get("caution") and "注意：" + drug["caution"],
                                drug.get("note")) if p)
    if tip:
        card.setToolTip(tip)
    col.addStretch()
    lay.addLayout(col, 1)
    return card


# ── 首页 ──────────────────────────────────────────────────────
class BrandHero(QWidget):
    """首页品牌区。

    排版要点（对齐 Harness 欢迎页的处理）：
      1. 主标题居中、字号最大、字重最重 —— 一眼就知道这是什么程序
      2. 副标题次一级，用品牌金，和主标题拉开对比
      3. 元信息最小最弱，不抢视线
      4. 主楼剪影退到底部当底纹，只做氛围，不参与信息层级
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("brandHero")
        # QWidget 子类必须开这个属性，样式表里的 background-color 才会被绘制，
        # 否则整个品牌区是透明的（白底白字，等于看不见）
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(248)
        self.setStyleSheet(
            f"#brandHero {{ background-color: {NAU['primary_deep']};"
            f" border-radius: {RADIUS['card']}px; }}")

        self.sil = None
        sil_pix = brand_pixmap("主楼剪影_金_透明.png", height=132)
        if sil_pix is not None:
            self.sil = ImageLabel(self)
            self.sil.setPixmap(sil_pix)
            self.sil.setStyleSheet("background: transparent;")
            self.sil.lower()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.setSpacing(0)
        lay.addStretch(1)

        logo_pix = brand_pixmap("稻视界logo_金_透明.png", height=52)
        if logo_pix is not None:
            logo = ImageLabel(self)
            logo.setPixmap(logo_pix)
            logo.setAlignment(Qt.AlignCenter)
            lay.addWidget(logo, 0, Qt.AlignHCenter)
            lay.addSpacing(12)

        # 主标题：字距拉开一点，中文大标题这样更"立得住"
        title = QLabel('<span style="letter-spacing:5px;">稻视界</span>', self)
        title.setFont(display_font(TYPE["display"]))   # 衬线，品牌主标题
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_STRONG}; background: transparent;")
        lay.addWidget(title)
        lay.addSpacing(10)

        sub = QLabel('<span style="letter-spacing:2px;">水稻病虫害智能诊断系统</span>', self)
        sf = sub.font()
        sf.setPointSize(TYPE["subtitle"])
        sub.setFont(sf)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {NAU['gold']}; background: transparent;")
        lay.addWidget(sub)
        lay.addSpacing(10)

        # 小字号 + 自定义样式色在深色底上会出现贯穿横线的渲染瑕疵，
        # 所以元信息用 Fluent 自己的 CaptionLabel（走框架绘制路径）
        meta = CaptionLabel("南京农业大学　·　YOLOv8n　·　6 类病虫害　·　本地推理", self)
        meta.setTextColor(TEXT_MUTED_LIGHT, TEXT_MUTED_LIGHT)
        meta.setAlignment(Qt.AlignCenter)
        lay.addWidget(meta)
        lay.addStretch(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.sil is not None:      # 剪影铺在右下角当底纹
            self.sil.move(self.width() - self.sil.width() - 16,
                          self.height() - self.sil.height() - 8)


class WeatherThread(QThread):
    """后台取天气：点一下不该卡住界面（接口最慢几秒）"""
    done = pyqtSignal(dict, str, list)
    failed = pyqtSignal(str)

    def __init__(self, place=None, lon=None, lat=None, parent=None):
        super().__init__(parent)
        self.place, self.lon, self.lat = place, lon, lat

    def run(self):
        try:
            wx = fetch_weather(city=self.place, lon=self.lon, lat=self.lat, days=3)
            if "错误" in wx:
                self.failed.emit(wx["错误"])
                return
            windows = spray_windows(wx)
            self.done.emit(wx, weather_summary_text(wx, windows), windows)
        except Exception as e:
            log.exception("天气查询线程异常")
            self.failed.emit(str(e))

class HomeInterface(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("homeInterface")
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 24)
        root.setSpacing(16)

        root.addWidget(BrandHero(self))

        row = QHBoxLayout()
        row.setSpacing(14)
        # 主次分明：拍照上传是主流程（实心主按钮），实时识别是并列功能（描边次按钮）
        btn_cam = PrimaryPushButton(FIF.PHOTO, "拍照上传识别", self)
        btn_cam.setMinimumHeight(46)
        btn_live = PushButton(FIF.CAMERA, "打开摄像头实时识别", self)
        btn_live.setMinimumHeight(46)
        btn_cam.clicked.connect(lambda: self.main.go(self.main.imageInterface))
        btn_live.clicked.connect(lambda: self.main.go(self.main.cameraInterface))
        row.addWidget(btn_cam)
        row.addWidget(btn_live)
        root.addLayout(row)

        dev = "GPU (CUDA)" if self.main.device != "cpu" else "CPU"
        info = HeaderCardWidget("模型信息", self)
        rows = [("模型版本", MODEL_INFO["模型版本"]), ("参数量", MODEL_INFO["Params"]),
                ("计算量", MODEL_INFO["FLOPs"]), ("输入尺寸", MODEL_INFO["Input Size"]),
                ("训练数据集", MODEL_INFO["训练数据集"]),
                ("识别类别", "、".join(CLASS_NAMES)), ("运行设备", dev)]
        # HeaderCardWidget.viewLayout 是横向的，套一层竖直布局才能一行一项
        host = QWidget(info)
        grid = QVBoxLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(7)
        for k, v in rows:
            line = QHBoxLayout()
            k_lbl = BodyLabel(k, host)
            k_lbl.setFixedWidth(96)
            k_lbl.setTextColor(NAU["muted"], NAU["muted"])   # 标签弱化，数值读起来是主角
            line.addWidget(k_lbl)
            line.addWidget(BodyLabel(str(v), host), 1)
            grid.addLayout(line)
        info.viewLayout.addWidget(host)
        root.addWidget(info)

        # 天气与施药窗口：农户最关心的两个问题——会不会发病、什么时候能打药
        self.weather_card = HeaderCardWidget("天气与施药窗口", self)
        whost = QWidget(self.weather_card)
        wgrid = QVBoxLayout(whost)
        wgrid.setContentsMargins(0, 0, 0, 0)
        wgrid.setSpacing(6)
        self.weather_body = BodyLabel(
            "点「查询天气」获取天气、病害流行风险、极端天气预警与最佳施药窗口；"
            "填地块名会用地块地图里的坐标查田块头顶的天气。", whost)
        self.weather_body.setWordWrap(True)
        wgrid.addWidget(self.weather_body)
        self.spray_body = BodyLabel("", whost)
        self.spray_body.setWordWrap(True)
        wgrid.addWidget(self.spray_body)
        wrow = QHBoxLayout()
        self.weather_input = LineEdit(whost)
        self.weather_input.setPlaceholderText("城市名或地块名（留空 = 南京）")
        self.weather_input.returnPressed.connect(self.refresh_weather)
        wrow.addWidget(self.weather_input, 1)
        self.btn_weather = PushButton(FIF.SYNC, "查询天气", whost)
        self.btn_weather.clicked.connect(self.refresh_weather)
        wrow.addWidget(self.btn_weather)
        self.btn_spray = PushButton(FIF.EDIT, "记录施药", whost)
        self.btn_spray.setToolTip("登记一次施药，之后会自动判断药后有没有被雨冲掉")
        self.btn_spray.clicked.connect(self.record_spray)
        wrow.addWidget(self.btn_spray)
        self.btn_respray = PushButton(FIF.UPDATE, "查补施", whost)
        self.btn_respray.setToolTip("按实测降水判断最近几次施药要不要补施")
        self.btn_respray.clicked.connect(self.refresh_spray_check)
        wrow.addWidget(self.btn_respray)
        self.btn_harvest = PushButton(FIF.DATE_TIME, "收获期", whost)
        self.btn_harvest.setToolTip("登记计划收获期，自动推算最早可收日期并检查农残风险")
        self.btn_harvest.clicked.connect(self.set_harvest)
        wrow.addWidget(self.btn_harvest)
        wgrid.addLayout(wrow)
        self.weather_card.viewLayout.addWidget(whost)
        root.addWidget(self.weather_card)
        self._weather_busy = False
        QTimer.singleShot(1200, self.refresh_weather)      # 进首页顺手取一次
        QTimer.singleShot(1600, self.refresh_spray_check)  # 顺带看看有没有要补施的

        root.addStretch()
        dev = "GPU (CUDA)" if self.main.device != "cpu" else "CPU"
        self.status = CaptionLabel(
            f"● 模型已就绪　|　运行设备 {dev}　|　识别类别 {len(CLASS_NAMES)} 类　|　"
            f"v{APP_VERSION}　|　F1 查看关于与快捷键", self)
        self.status.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status)


    # ── 天气卡片 ──
    def refresh_weather(self):
        if getattr(self, "_weather_busy", False):
            return
        self._weather_busy = True
        self.btn_weather.setEnabled(False)
        self.weather_body.setText("正在查询天气…")
        place = self.weather_input.text().strip() or None
        self.weather_thread = _track_thread(WeatherThread(place))
        self.weather_thread.done.connect(self._on_weather)
        self.weather_thread.failed.connect(self._on_weather_fail)
        self.weather_thread.start()

    def _on_weather(self, wx, text, windows):
        self._weather_busy = False
        self.btn_weather.setEnabled(True)
        if windows:
            w = windows[0]
            tail = ("\n\n▶ 最佳施药窗口：%s ~ %s（%d 小时，平均评分 %.0f）"
                    % (w["开始"][5:16].replace("T", " "), w["结束"][11:16],
                       w["时长小时"], w["平均评分"]))
        else:
            tail = "\n\n▶ 未来 48 小时没有及格的施药窗口，建议等天气转好"
        self.weather_body.setText(text + tail)
        log_event("weather_card", place=wx.get("地点"), source=wx.get("来源"),
                  windows=len(windows))

    # ── 施药记录与补施判断 ──
    def record_spray(self):
        """记录一次施药（写操作，人点触发）"""
        try:
            plots = [r["plot"] for r in (plot_risk_rows() or [])]
        except Exception:
            plots = []
        dlg = SprayDialog(plots=plots, parent=self.window())
        if not dlg.exec():
            return
        v = dlg.values()
        if not v["drug"]:
            InfoBar.warning("药剂必填", "至少要填药剂名称", duration=3000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        rec = add_spray_record(plot=v["plot"], drug=v["drug"], cls=v["cls"],
                               when=v["when"], note=v["note"])
        if rec is None:
            InfoBar.error("保存失败", "无法写入施药记录文件", duration=4000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已记录施药", "%s · %s · %s"
                        % (rec["plot"] or "未填地块", rec["drug"], rec["time"][5:16]),
                        duration=4000, position=InfoBarPosition.TOP, parent=self)
        self.refresh_spray_check()

    def refresh_spray_check(self):
        """按实测降水判断最近的施药要不要补施"""
        try:
            checks = spray_rain_check(days=14)
        except Exception as e:
            log.warning("补施判断失败：%s", e)
            checks = []
        if not checks:
            self.spray_body.setText("还没有施药记录：点「记录施药」登记一次，"
                                    "之后会自动用实测降水判断药后有没有被雨冲掉。")
            return
        need = [c for c in checks if c.get("结论") == "建议补施"]
        lines = [("⚠ 有 %d 次施药可能被雨冲刷，建议补施：" % len(need)) if need
                 else "施药记录 %d 条，药后降水均未超过耐受阈值。" % len(checks)]
        for c in checks[-2:]:
            lines.append("　· %s %s（%s，药后6h %.1fmm / 24h %.1fmm）：%s"
                         % (c["地块"], c["药剂"], (c["施药时间"] or "")[5:16],
                            c["药后6小时雨量mm"] or 0, c["药后24小时雨量mm"] or 0,
                            c["结论"]))
            if c.get("安全间隔期"):
                lines.append("　　安全间隔期 %s 天，注意采收时间" % c["安全间隔期"])
        for h in earliest_harvest():
            if h.get("结论") and ("可收" in h["结论"] or "风险" in h["结论"]):
                lines.append("　· %s：%s" % (h["地块"], h["结论"]))
        self.spray_body.setText("\n".join(lines))
        log_event("spray_check", records=len(checks), need=len(need))

    def set_harvest(self):
        """登记计划收获期（写操作，人点触发）"""
        try:
            plots = [r["plot"] for r in (plot_risk_rows() or [])]
        except Exception:
            plots = []
        plots += [x["地块"] for x in earliest_harvest()]
        dlg = HarvestDialog(plots=sorted({p for p in plots if p}), parent=self.window())
        if not dlg.exec():
            return
        v = dlg.values()
        rec = set_harvest_date(v["plot"], v["date"], v["note"])
        if rec is None:
            InfoBar.warning("日期格式不对", "请按 YYYY-MM-DD 填写，例如 2026-11-05", duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已登记收获期", "%s · %s" % (v["plot"] or "未命名地块", rec["date"]),
                        duration=4000, position=InfoBarPosition.TOP, parent=self)
        self.refresh_spray_check()

    def _on_weather_fail(self, msg):
        self._weather_busy = False
        self.btn_weather.setEnabled(True)
        self.weather_body.setText("天气查询失败：%s（不影响其它功能，可稍后重试）" % msg)

# ── 拖拽区域 ──────────────────────────────────────────────────
class DropArea(SimpleCardWidget):
    """支持拖入图片的区域"""
    dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(320)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.addStretch()
        self.art = None
        art = brand_pixmap("主楼剪影_淡金_透明.png", height=120)
        if art is not None:
            self.art = ImageLabel(self)
            self.art.setPixmap(art)
            self.art.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.art)
        self.image = ImageLabel(self)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setText("把图片拖到这里\n或点击下方按钮选择")
        self.image.setWordWrap(True)
        lay.addWidget(self.image)
        lay.addStretch()
        self._hover = False

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True
            self.update()

    def dragLeaveEvent(self, e):
        self._hover = False
        self.update()

    def dropEvent(self, e):
        self._hover = False
        self.update()
        for url in e.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                self.dropped.emit(fp)
                break


# ── 图片识别页 ────────────────────────────────────────────────
class ImageInterface(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("imageInterface")
        self.current_image = None
        self._busy = False

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(12)
        root.addWidget(page_title("图片识别", self))
        root.addWidget(CaptionLabel("支持拖拽图片，或从电脑中选择；支持 jpg / png / bmp", self))
        root.addSpacing(8)

        self.drop = DropArea(self)
        self.drop.dropped.connect(self.load_image)
        root.addWidget(self.drop, 1)

        self.bar = IndeterminateProgressBar(self, start=False)
        self.bar.setVisible(False)
        root.addWidget(self.bar)

        self.status = CaptionLabel("", self)
        root.addWidget(self.status)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.btn_pick = PrimaryPushButton(FIF.FOLDER, "选择图片", self)
        self.btn_pick.setMinimumHeight(42)
        self.btn_run = PushButton(FIF.PLAY, "开始识别", self)
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setEnabled(False)
        self.btn_miss = PushButton(getattr(FIF, "FEEDBACK", FIF.EDIT), "报告漏检", self)
        self.btn_miss.setMinimumHeight(42)
        self.btn_miss.setEnabled(False)
        self.btn_miss.setToolTip("田里确实有病、模型没检出：记下来进待标注队列（漏检是最危险的错误）")
        row.addStretch()
        row.addWidget(self.btn_pick)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_miss)
        row.addStretch()
        root.addLayout(row)

        self.btn_pick.clicked.connect(self.pick)
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_miss.clicked.connect(self._report_miss)

    def pick(self):
        if self._busy:
            return
        fp, _ = QFileDialog.getOpenFileName(self, "选择要识别的图片", "",
                                            "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if fp:
            self.load_image(fp)

    def load_image(self, file_path):
        if self._busy:
            return
        global LAST_IMAGE_SOURCE
        LAST_IMAGE_SOURCE = str(file_path)      # 结果页「记录到地图」要靠它读拍摄时间与 GPS
        img = cv2.imread(file_path)
        if img is None:
            InfoBar.error("读取失败", "无法读取该图片文件", duration=3000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        self.current_image = img
        if self.drop.art is not None:
            self.drop.art.setVisible(False)      # 有图了就收起剪影
        self.drop.image.setPixmap(bgr_to_qpixmap(img, self.drop.size() - QSize(40, 40)))
        self.btn_run.setEnabled(True)
        self.btn_miss.setEnabled(True)
        self.status.setText(f"已载入：{os.path.basename(file_path)}　{img.shape[1]}×{img.shape[0]}")

    def run_inference(self):
        if self._busy or self.current_image is None:
            return
        self._busy = True
        self.btn_pick.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.start()
        self.status.setText("正在识别…")
        self.thread = _track_thread(InferenceThread(self.main.model, self.current_image))
        self.thread.result_ready.connect(self._done)
        self.thread.failed.connect(self._failed)
        self.thread.start()

    def _reset(self):
        self._busy = False
        self.bar.stop()
        self.bar.setVisible(False)
        self.btn_pick.setEnabled(True)
        self.btn_run.setEnabled(self.current_image is not None)
        self.btn_miss.setEnabled(self.current_image is not None)

    def _failed(self, msg):
        self._reset()
        self.status.setText("识别失败：" + msg)
        InfoBar.error("识别失败", msg, duration=5000, position=InfoBarPosition.TOP, parent=self)

    def _report_miss(self):
        """漏检反馈：模型没检出、但用户知道田里有病 —— 这类样本最有价值"""
        if self.current_image is None:
            return
        dlg = FeedbackDialog("未检出目标（漏检）", self)
        if not dlg.exec():
            return
        choice, note = dlg.values()
        try:
            saved = append_feedback_record(
                image=self.current_image, predicted="(无检出)", conf=0.0, boxes=0,
                severity="无", user_choice=choice, note=note, mode="miss")
        except Exception as e:
            log.exception("写入漏检反馈失败")
            InfoBar.error("反馈保存失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已记录", "漏检样本已存入 feedback 队列（%s）"
                        % (os.path.basename(saved) if saved else "CSV 已写入"),
                        duration=4000, position=InfoBarPosition.TOP, parent=self)

    def _done(self, result_list):
        self._reset()
        if not result_list:
            self.status.setText("未检测到符合阈值的病虫害，可换一张更清晰的图片")
            InfoBar.warning("未检测到目标", "没有超过置信度阈值的检测框", duration=3000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        self.status.setText(f"识别完成，共 {len(result_list)} 个检测框")
        self.main.show_result(result_list, self.current_image, "image")


# ── 摄像头页 ──────────────────────────────────────────────────
class CameraInterface(QWidget):
    INFERENCE_INTERVAL = 30

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("cameraInterface")
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.frame_count = 0
        self._busy = False
        self._alive = True
        self._last_results = []

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(12)
        root.addWidget(page_title("实时识别", self))
        root.addWidget(CaptionLabel("摄像头画面 + 实时检测框；检测到目标后自动跳到结果页", self))
        root.addSpacing(8)

        self.video = ImageLabel(self)
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumHeight(380)
        self._show_idle_art()
        root.addWidget(self.video, 1)

        self.status = CaptionLabel("", self)
        root.addWidget(self.status)

        row = QHBoxLayout()
        row.addStretch()
        self.btn_toggle = PrimaryPushButton(FIF.CAMERA, "开启摄像头", self)
        self.btn_toggle.setMinimumHeight(42)
        self.btn_toggle.clicked.connect(self.toggle)
        row.addWidget(self.btn_toggle)
        row.addStretch()
        root.addLayout(row)

    def _show_idle_art(self):
        """未开摄像头时用淡化主楼剪影占位，品牌元素贯穿全应用"""
        art = brand_pixmap("主楼剪影_淡金_透明.png", height=150)
        if art is not None:
            self.video.setPixmap(art)
        else:
            self.video.setText("摄像头未开启")

    def toggle(self):
        if self.cap is None:
            self.open_camera()
        else:
            self.close_camera()
            self.status.setText("摄像头已关闭")

    def open_camera(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for idx in range(3):
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    self.cap = cap
                    break
                cap.release()
            if self.cap is not None:
                break
        if self.cap is None:
            InfoBar.error("摄像头打不开", "没有找到可用摄像头设备", duration=4000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._alive = True
        self.frame_count = 0
        self.timer.start(33)
        self.btn_toggle.setText("关闭摄像头")
        self.status.setText("摄像头已开启 · 正在实时识别")

    def close_camera(self):
        self._alive = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self._show_idle_art()
        self.btn_toggle.setText("开启摄像头")

    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_count += 1
        if self._last_results:
            display = annotate_image(frame, self._last_results)
        else:
            display = frame.copy()
            cv2.putText(display, "Scanning...", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (60, 200, 120), 2)
        self.video.setPixmap(bgr_to_qpixmap(display, self.video.size()))
        if self.frame_count % self.INFERENCE_INTERVAL == 0 and not self._busy:
            self._busy = True
            self.current_frame = frame.copy()
            self.thread = _track_thread(InferenceThread(self.main.model, self.current_frame))
            self.thread.result_ready.connect(self._on_inference)
            self.thread.failed.connect(self._on_error)
            self.thread.start()

    def _on_error(self, msg):
        self._busy = False
        if self._alive:
            self.status.setText("识别失败：" + msg)

    def _on_inference(self, result_list):
        self._busy = False
        if not self._alive:
            return
        if result_list:
            self._last_results = result_list
            frame = self.current_frame
            self.close_camera()
            self.main.show_result(result_list, frame, "camera")
        else:
            self._last_results = []

    def leave(self):
        """离开本页时释放摄像头"""
        if self.cap is not None:
            self.close_camera()


# ── 结果页 ────────────────────────────────────────────────────
def speakable(text):
    """把 Markdown 变成适合朗读的纯文本：去掉标记符号，把换行变成停顿"""
    t = text or ""
    fence = chr(96) * 3                                    # 三个反引号
    t = re.sub(fence + r".*?" + fence, "", t, flags=re.S)  # 代码块整段丢掉
    t = t.replace(chr(96), "")
    t = re.sub(r"[#*>_~|]", "", t)
    t = re.sub(r"^\s*[-·•]\s*", "，", t, flags=re.M)        # 列表项变停顿
    t = re.sub(r"^\s*\d+[.、]\s*", "，", t, flags=re.M)
    t = re.sub(r"\n{2,}", "。", t)
    t = re.sub(r"\n", "，", t)
    t = re.sub(r"[，。]{2,}", "。", t)
    return t.strip()


ASR_DIR = Path(res_path("resources/asr"))
ASR_MODEL_NAME = "vosk-model-small-cn-0.22"


def ascii_path(path):
    """把可能含中文的路径换成一个纯 ASCII 路径。

    Vosk 底层的 Kaldi 在 Windows 上用 ANSI 窄字符读文件，路径里有中文就报
    "Failed to create a model"。这里优先取 8.3 短路径（不复制文件），
    拿不到就把模型复制到一个纯英文目录再用。
    """
    p = str(path)
    if all(ord(c) < 128 for c in p):
        return p
    # 1) 8.3 短路径（瞬间，无复制）
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(p, buf, 1024):
            short = buf.value
            if short and all(ord(c) < 128 for c in short) and os.path.isdir(short):
                log.info("语音模型改用短路径: %s", short)
                return short
    except Exception as e:
        log.debug("取短路径失败: %s", e)
    # 2) 复制到纯英文目录（按顺序试，第一个成功的就用）
    name = os.path.basename(p)
    for base in ("C:/Users/Public/rice_dp1/asr",
                 "C:/ProgramData/rice_dp1/asr",
                 "D:/rice_dp1/asr"):
        dest = os.path.join(base, name)
        if os.path.isdir(dest):
            return dest
        try:
            os.makedirs(base, exist_ok=True)
            shutil.copytree(p, dest)
            log.info("语音模型已复制到: %s", dest)
            return dest
        except Exception as e:
            log.warning("复制语音模型到 %s 失败: %s", base, e)
    raise RuntimeError(
        "语音模型路径含中文，且没有可写的英文目录。\n"
        "请手动把 resources\\asr 整个文件夹复制到 C:\\Users\\Public\\ 下再试。")


class ListenThread(QThread):
    """麦克风录音 + Vosk 离线识别。

    用流式识别（边说边出字），年长用户能立刻看到识别结果，说错了马上重说；
    整段录完再识别会让人干等 1-3 秒，体验差很多。
    """

    partial = pyqtSignal(str)      # 中间结果
    final = pyqtSignal(str)        # 最终结果
    failed = pyqtSignal(str)

    def __init__(self, model_dir, parent=None):
        super().__init__(parent)
        self.model_dir = model_dir
        self._stop = False

    def run(self):
        try:
            import json as _json
            import queue as _queue
            import sounddevice as _sd
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            rec = KaldiRecognizer(Model(self.model_dir), 16000)
            audio = _queue.Queue()

            def _cb(indata, frames, t_info, status):
                audio.put(bytes(indata))

            with _sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                                    channels=1, callback=_cb):
                while not self._stop:
                    if rec.AcceptWaveform(audio.get()):
                        txt = _json.loads(rec.Result()).get("text", "").replace(" ", "")
                        if txt:
                            self.final.emit(txt)
                            return
                    else:
                        p = _json.loads(rec.PartialResult()).get("partial", "").replace(" ", "")
                        if p:
                            self.partial.emit(p)
            txt = _json.loads(rec.FinalResult()).get("text", "").replace(" ", "")
            if txt:
                self.final.emit(txt)
        except ImportError:
            self.failed.emit("缺少语音识别组件，请先安装：pip install vosk sounddevice")
        except Exception as e:
            log.exception("语音识别失败")
            self.failed.emit(str(e))

    def stop(self):
        self._stop = True


# Edge 神经网络语音：音质接近真人，免费无需 key，但要联网
VOICE_OPTIONS = [
    ("晓晓 · 女声（推荐）", "zh-CN-XiaoxiaoNeural"),
    ("晓伊 · 女声活泼", "zh-CN-XiaoyiNeural"),
    ("云希 · 男声", "zh-CN-YunxiNeural"),
    ("云扬 · 男声播音", "zh-CN-YunyangNeural"),
    ("云健 · 男声沉稳", "zh-CN-YunjianNeural"),
]


class ChatTTSSpeakThread(QThread):
    """本地 ChatTTS。

    模型用类级缓存常驻内存 —— 第二次开始点朗读几乎立刻出声，
    这正是换掉 Edge 的原因（Edge 每次都要跨境往返 2 秒以上）。
    """

    ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    _engine = None

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text

    def run(self):
        try:
            import tempfile
            import time as _time
            import ChatTTS
            import torch
            import torchaudio
            if ChatTTSSpeakThread._engine is None:
                log.info("首次加载 ChatTTS 模型，可能要等一会儿…")
                eng = ChatTTS.Chat()
                eng.load(source="modelscope")
                ChatTTSSpeakThread._engine = eng
            torch.manual_seed(42)      # 固定种子，否则每次音色都不一样
            wavs = ChatTTSSpeakThread._engine.infer([self.text])
            out = os.path.join(tempfile.gettempdir(),
                               "rice_chattts_%d.wav" % int(_time.time() * 1000))
            torchaudio.save(out, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                self.ok.emit(out)
            else:
                self.failed.emit("合成结果为空")
        except Exception as e:
            self.failed.emit(str(e))


PIPER_DIR = Path(res_path("resources/tts"))
PIPER_MODEL = PIPER_DIR / "zh_CN-huayan-medium.onnx"


def piper_synth(voice, text, wav_path):
    """兼容 piper-tts 各版本，并确保真的写入了音频。

    坑：1.3.x 的 synthesize(text, syn_config) 第二个参数是配置而不是文件，
    传 wave 对象进去不报错，直接静默产出空文件。所以按 API 存在性判断，
    并在写完后检查帧数，空的话换下一种方式重试。
    """
    import inspect
    import wave
    rate = getattr(getattr(voice, 'config', None), 'sample_rate', 22050)

    def _write(how):
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(rate))
            if how == 'wav':
                voice.synthesize_wav(text, wf)          # 1.3.x 专用方法
            elif how == 'two':
                voice.synthesize(text, wf)              # 1.2.x
            else:
                for chunk in voice.synthesize(text):    # 1.3.x 生成器
                    wf.writeframes(chunk.audio_int16_bytes)
        with wave.open(wav_path, 'rb') as r:
            return r.getnframes()

    order = []
    if hasattr(voice, 'synthesize_wav'):
        order.append('wav')
    try:
        n = len(inspect.signature(voice.synthesize).parameters)
    except (TypeError, ValueError):
        n = 2
    order.append('two' if n >= 2 else 'gen')
    order.append('gen')

    last = 0
    for how in order:
        try:
            frames = _write(how)
        except Exception as e:
            print('      方式 %s 报错: %s' % (how, str(e)[:60]))
            continue
        if frames > 0:
            return frames, rate, how
        last = frames
    return last, rate, None

class PiperSpeakThread(QThread):
    """本地神经语音（Piper + ONNX）。

    完全离线，农户在田里也能用。模型只有 60MB，CPU 就能跑。
    piper-tts 各版本 API 略有差异，这里两种调用方式都试。
    """

    ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, text, model_path, parent=None):
        super().__init__(parent)
        self.text = text
        self.model_path = model_path

    def run(self):
        try:
            import tempfile
            import time as _time
            import wave
            from piper import PiperVoice
            out = os.path.join(tempfile.gettempdir(),
                               "rice_piper_%d.wav" % int(_time.time() * 1000))
            voice = PiperVoice.load(str(self.model_path))
            frames, rate, how = piper_synth(voice, self.text, out)
            log.info("Piper 合成: %d 帧 @ %sHz (方式 %s)", frames, rate, how)
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                self.ok.emit(out)
            else:
                self.failed.emit("合成结果为空")
        except Exception as e:
            self.failed.emit(str(e))


class EdgeSpeakThread(QThread):
    """先用 Edge 神经语音合成 mp3，再交给主线程播放。

    Windows 自带的 SAPI 是 Vista 时代的老引擎，读起来很机械；
    Edge 这套是神经网络语音，免费且不需要 key。没网时回退 SAPI。
    """

    ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, text, voice, parent=None, rate="-8%"):
        super().__init__(parent)
        self.text = text
        self.voice = voice
        self.rate = rate

    def run(self):
        try:
            import tempfile
            import time as _time
            # 用自己的最小客户端，不用 edge-tts 包：
            # 那个包在 websockets 上用 headers= 传参，会顶掉库的默认头，握手必 403
            from edge_tts_lite import synthesize
            out = os.path.join(tempfile.gettempdir(),
                               "rice_tts_%d.mp3" % int(_time.time() * 1000))
            n = synthesize(self.text, self.voice, out, rate=self.rate)
            if n and n > 1000:
                self.ok.emit(out)
            else:
                self.failed.emit("合成结果为空")
        except Exception as e:
            self.failed.emit(str(e))


class SpeakThread(QThread):
    """后台朗读。

    SAPI 的 Speak 是阻塞调用，直接在主线程调会把界面冻住；
    而且 COM 对象必须在使用它的线程里初始化（CoInitialize）。
    这里用异步朗读 + 轮询，才能随时"停止"。
    """

    done = pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self._voice = None
        self._stop = False

    def run(self):
        pythoncom = None
        try:
            import pythoncom as _pc
            import win32com.client
            pythoncom = _pc
            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            for v in voice.GetVoices():          # 优先中文发音人
                if "Chinese" in v.GetDescription():
                    voice.Voice = v
                    break
            voice.Rate = -1                       # 稍慢一点，年长用户听得清
            voice.Volume = 100
            self._voice = voice
            voice.Speak(self.text, 1)             # 1 = 异步
            while voice.Status.RunningState != 1:  # 1 = 读完
                if self._stop:
                    voice.Speak("", 3)            # 清空队列并停止
                    break
                self.msleep(80)
        except Exception as e:
            log.warning("语音朗读失败: %s", e)
        finally:
            self._voice = None
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
            self.done.emit()

    def stop(self):
        self._stop = True


class FeedbackDialog(MessageBoxBase):
    """「识别不对」对话框：让用户选实际情况 + 填备注。

    只收「实际是什么」，不收框位置——框的位置仍需人工在标注工具里画，
    所以这里只负责把可疑样本挑出来，不假装能自动变成训练标签。
    """

    CHOICES = list(CLASS_NAMES) + ["不是病虫害（背景/其他作物）", "说不清，但确实不对"]

    def __init__(self, predicted, parent=None):
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel("反馈识别结果", self))
        tip = BodyLabel("本次模型判断：%s\n请选择实际情况（会保存图片副本到 feedback 目录）："
                        % (predicted or "未检出目标"), self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.combo = ComboBox(self)
        self.combo.addItems(self.CHOICES)
        self.viewLayout.addWidget(self.combo)
        self.note = LineEdit(self)
        self.note.setPlaceholderText("补充说明（可选）：病斑很小 / 光照太暗 / 拍的是穗部…")
        self.viewLayout.addWidget(self.note)
        self.yesButton.setText("提交反馈")
        self.cancelButton.setText("取消")
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(440)

    def values(self):
        return self.combo.currentText(), self.note.text().strip()


class HarvestDialog(MessageBoxBase):
    """登记计划收获期：登记后自动推算「最早可收日期」并检查冲突"""

    def __init__(self, plots=None, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.viewLayout.addWidget(SubtitleLabel("登记计划收获期", self))
        tip = BodyLabel("按「末次施药 + 该药剂安全间隔期」推算最早可收日期；"
                        "若计划收获期早于它，会提示农残风险与需要推迟的天数。", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.plot = LineEdit(self)
        hint = "、".join((plots or [])[:4])
        self.plot.setPlaceholderText("地块名（可留空）" + ("，例如：" + hint if hint else ""))
        self.viewLayout.addWidget(self.plot)
        self.date = LineEdit(self)
        self.date.setText((datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
        self.date.setPlaceholderText("计划收获日期 YYYY-MM-DD")
        self.viewLayout.addWidget(self.date)
        self.note = LineEdit(self)
        self.note.setPlaceholderText("备注（可选）")
        self.viewLayout.addWidget(self.note)
        self.yesButton.setText("保存")
        self.cancelButton.setText("取消")
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(460)

    def values(self):
        return {"plot": self.plot.text().strip(), "date": self.date.text().strip(),
                "note": self.note.text().strip()}

class PrecheckThread(QThread):
    """施药前校验放后台：要联网取天气，不能卡住对话框"""
    done = pyqtSignal(dict)

    def __init__(self, cls_name, plot, drug, when=None, parent=None):
        super().__init__(parent)
        self.cls_name, self.plot, self.drug, self.when = cls_name, plot, drug, when

    def run(self):
        try:
            self.done.emit(spray_precheck(cls_name=self.cls_name, plot=self.plot,
                                          drug=self.drug, when=self.when))
        except Exception as e:
            log.exception("施药前校验失败")
            self.done.emit({"结论": "校验失败：%s" % e, "是否建议现在施药": None})


class SprayPlanDialog(MessageBoxBase):
    """识别 → 按方案施药并登记：方案来自本地防治库，登记前先做四条校验"""

    def __init__(self, cls_name, conf, boxes, plots=None, parent=None, autorun=True):
        # MessageBoxBase 会调用 parent.width()，所以不能传 None
        super().__init__(parent or QApplication.activeWindow())
        self.cls_name = cls_name or ""
        self.plan = treatment_plan_rows(self.cls_name)
        self.viewLayout.addWidget(SubtitleLabel("按方案施药并登记", self))
        tip = BodyLabel("识别结论：%s（置信度 %.0f%%，%d 个定位）\n方案来源：%s"
                        % (self.cls_name or "未识别", (conf or 0) * 100, boxes,
                           self.plan.get("来源", "")), self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.viewLayout.addWidget(StrongBodyLabel("推荐药剂（来自本地防治库）", self))
        self.drug = ComboBox(self)
        self._drug_rows = self.plan.get("推荐药剂") or []
        if not self._drug_rows:
            self.drug.addItems(["（本地库暂无该类别药剂）"])
        for d in self._drug_rows:
            extra = []
            if d.get("规格"):
                extra.append(d["规格"])
            if d.get("亩用量"):
                extra.append(d["亩用量"])
            if d.get("安全间隔期"):
                extra.append("间隔期 %s 天" % d["安全间隔期"])
            self.drug.addItem("%s%s" % (d.get("名称", ""),
                                        ("（" + " · ".join(extra) + "）") if extra else ""),
                             userData=d)
        self.viewLayout.addWidget(self.drug)
        row_plot = QHBoxLayout()
        self.plot = LineEdit(self)
        self.plot.setPlaceholderText("地块名（可留空）" + ("，例如：" + "、".join((plots or [])[:3]) if plots else ""))
        row_plot.addWidget(self.plot, 1)
        self.viewLayout.addLayout(row_plot)
        self.dose = LineEdit(self)
        self.dose.setPlaceholderText("亩用量（自动带出本地库剂量，可改）")
        self.viewLayout.addWidget(self.dose)
        self.when = LineEdit(self)
        self.when.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.viewLayout.addWidget(self.when)
        self.note = LineEdit(self)
        self.note.setPlaceholderText("备注（可选）：天气、机具、面积…")
        self.viewLayout.addWidget(self.note)
        self.check_label = BodyLabel("正在校验（农药合规 / 天气窗口 / 安全间隔期 / 重复施药）…", self)
        self.check_label.setWordWrap(True)
        self.viewLayout.addWidget(self.check_label)
        self.btn_recheck = PushButton(FIF.SYNC, "重新校验", self)
        self.btn_recheck.clicked.connect(self.run_precheck)
        self.viewLayout.addWidget(self.btn_recheck)
        self.yesButton.setText("登记这次施药")
        self.cancelButton.setText("取消")
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(520)
        self.drug.currentIndexChanged.connect(self._fill_dose)
        self._fill_dose()
        self._thread = None
        if autorun:            # 自检里关掉：校验要联网取天气，不适合放进离屏用例
            QTimer.singleShot(80, self.run_precheck)

    def _current_drug_row(self):
        data = self.drug.currentData()
        return data if isinstance(data, dict) else {}

    def _fill_dose(self):
        row = self._current_drug_row()
        if row.get("亩用量"):
            self.dose.setText(row["亩用量"])

    def run_precheck(self):
        if self._thread is not None and self._thread.isRunning():
            return
        self.check_label.setText("正在校验（农药合规 / 天气窗口 / 安全间隔期 / 重复施药）…")
        when = None
        try:
            when = datetime.strptime(self.when.text().strip()[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            pass
        row = self._current_drug_row()
        self._thread = _track_thread(PrecheckThread(self.cls_name, self.plot.text().strip(),
                                                     row.get("名称", ""), when))
        self._thread.done.connect(self._show_precheck)
        self._thread.start()

    def _show_precheck(self, res):
        parts = []
        if res.get("结论"):
            parts.append(str(res["结论"]))
        wx_ = res.get("天气窗口") or {}
        if wx_.get("当前适宜") is False:
            parts.append("· 天气：%s" % wx_.get("不宜原因"))
        elif wx_.get("最近可用窗口"):
            w = wx_["最近可用窗口"]
            parts.append("· 最近可用窗口：%s ~ %s（%d 小时）"
                         % (w["开始"][5:16].replace("T", " "), w["结束"][11:16], w["时长小时"]))
        for a in (wx_.get("极端天气") or []):
            parts.append("· ⚠ %s（%s）：%s" % (a["类型"], a["等级"], a["应对"][:40]))
        rep = res.get("重复施药") or {}
        if rep.get("近期同药剂次数"):
            parts.append("· %s" % rep.get("说明"))
        safe = res.get("安全间隔期") or {}
        if safe.get("天数"):
            parts.append("· 安全间隔期 %s 天" % safe["天数"])
        self.check_label.setText("\n".join(parts) or "校验完成")

    def values(self):
        row = self._current_drug_row()
        when = datetime.now()
        try:
            when = datetime.strptime(self.when.text().strip()[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            pass
        return {"plot": self.plot.text().strip(), "drug": row.get("名称", "").split("（")[0].strip(),
                "drug_full": row.get("名称", ""), "dose": self.dose.text().strip(),
                "when": when, "note": self.note.text().strip(),
                "safe_days": row.get("安全间隔期"), "plate": row}

class SprayDialog(MessageBoxBase):
    """记录一次施药（写操作，必须人点）——记录后才能判断药后被雨冲刷要不要补施"""

    def __init__(self, plots=None, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.viewLayout.addWidget(SubtitleLabel("记录一次施药", self))
        tip = BodyLabel("登记后，「天气与施药窗口」会结合实测降水判断药后有没有被雨冲掉。", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.plot = LineEdit(self)
        hint = "、".join((plots or [])[:4])
        self.plot.setPlaceholderText("地块名（可留空）" + ("，例如：" + hint if hint else ""))
        self.viewLayout.addWidget(self.plot)
        self.drug = ComboBox(self)
        drugs = sorted({d.get("name", "") for tr in TREATMENT_DB.values()
                        for d in tr.get("drugs", []) if d.get("name")})
        self.drug.addItems(drugs or ["（本地库暂无药剂）"])
        self.viewLayout.addWidget(self.drug)
        self.cls = ComboBox(self)
        self.cls.addItems([""] + list(CLASS_NAMES))
        self.viewLayout.addWidget(self.cls)
        self.when = LineEdit(self)
        self.when.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.when.setPlaceholderText("施药时间 YYYY-MM-DD HH:MM")
        self.viewLayout.addWidget(self.when)
        self.note = LineEdit(self)
        self.note.setPlaceholderText("备注（可选）：用量、天气、机具…")
        self.viewLayout.addWidget(self.note)
        self.yesButton.setText("保存记录")
        self.cancelButton.setText("取消")
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(460)

    def values(self):
        when = None
        try:
            when = datetime.strptime(self.when.text().strip()[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            when = datetime.now()
        return {"plot": self.plot.text().strip(), "drug": self.drug.currentText().strip(),
                "cls": self.cls.currentText().strip(), "when": when,
                "note": self.note.text().strip()}

class ResultInterface(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("resultInterface")
        self.cls_name = None
        self.conf = 0.0
        self.result_list = []
        self.orig_img = None
        self.mode = "image"
        self._chat_history = []
        self._api_history = []
        self._img_tag_cache = {}
        self._ai_busy = False
        self._pending_user_msg = None
        self._followup_with_image = False   # 追问是否带原图（每轮会多花图片 token，默认关）
        self._ai_font_size = TYPE["body"]   # 正文字号（pt）
        self.severity = None                # assess_severity() 的结果
        self._stream_role = None            # 正在流式输出的角色
        self._stream_index = None           # 当前 AI 条目在对话区里的下标（工具提示会插在它前面）
        self._session_id = None             # 本次会话编号（落盘后由 save_session 返回）
        self._diag_card = None              # 结构化诊断卡（随会话一起保存）
        self.cascade_state = None           # 视觉复核结果（人工裁决用）
        self._revised = False               # 本轮是否已自我修正过（避免反复返工）
        self._last_answer = ""              # 最近一版回答（返工时作为上下文）
        self._pending_plan = None           # 待用户确认的执行计划
        # 任务队列：一次只跑一个任务（互斥），后来的排队
        self.task_queue = TaskQueue(self)
        self.task_queue.started.connect(self._on_task_started)
        self.task_queue.step_started.connect(self._on_task_step_started)
        self.task_queue.step_done.connect(self._on_task_step_done)
        self.task_queue.replanning.connect(self._on_task_replanning)
        self.task_queue.finished.connect(self._on_task_done)
        self.task_queue.failed.connect(self._on_task_failed)
        self.task_queue.changed.connect(self._on_queue_changed)
        self._stream_timer = QTimer(self)   # 流式刷新节流：每 120ms 重建一次，别每个分片都重建
        self._stream_timer.setInterval(120)
        self._stream_timer.timeout.connect(lambda: self._rebuild_chat(focus="bottom"))

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 12)
        root.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(page_title("识别结果与治理方案", self))
        head.addStretch()
        self.btn_stop = TransparentPushButton("停止生成", self)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self._stop_generate)
        head.addWidget(self.btn_stop)
        self.btn_feedback = PushButton(getattr(FIF, "FEEDBACK", FIF.EDIT), "识别不对", self)
        self.btn_feedback.setToolTip("记录这次结果和你的判断，进入 feedback 待标注队列，用于改进模型")
        self.btn_feedback.clicked.connect(self._feedback)
        head.addWidget(self.btn_feedback)
        self.btn_spray = PushButton(FIF.EDIT, "按方案施药并登记", self)
        self.btn_spray.setToolTip("用本地防治库的推荐方案登记这次施药；登记前会校验农药合规、"
                                  "天气窗口、安全间隔期与重复施药")
        self.btn_spray.clicked.connect(self.spray_from_plan)
        head.addWidget(self.btn_spray)
        self.btn_map = PushButton(FIF.GLOBE, "记录到地图", self)
        self.btn_map.setToolTip("把这次识别结果连同照片的拍摄时间与 GPS 存进地块地图档案库")
        self.btn_map.clicked.connect(self.record_to_map)
        head.addWidget(self.btn_map)
        self.btn_export = PushButton(FIF.SAVE, "导出报告", self)
        self.btn_export.clicked.connect(self._export)
        head.addWidget(self.btn_export)
        self.btn_again = PushButton(FIF.SYNC, "再识别一张", self)
        self.btn_again.clicked.connect(self._again)
        head.addWidget(self.btn_again)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left(), 26)
        body.addWidget(self._build_mid(), 34)
        body.addWidget(self._build_right(), 40)
        root.addLayout(body, 1)

        foot = QWidget(self)
        fv = QVBoxLayout(foot)
        fv.setContentsMargins(0, 6, 0, 0)
        fv.setSpacing(6)
        rule = QFrame(foot)
        rule.setFixedHeight(2)
        rule.setStyleSheet(f"background-color: {NAU['gold']}; border: none;")
        fv.addWidget(rule)
        fr = QHBoxLayout()
        fr.setContentsMargins(2, 0, 2, 0)
        mark = CaptionLabel("稻视界 · 南京农业大学 · 诚朴勤仁", foot)
        mark.setTextColor(NAU["subtext"], NAU["subtext"])
        fr.addWidget(mark)
        fr.addStretch()
        mini = brand_pixmap("主楼剪影_淡金_透明.png", height=24)
        if mini is not None:
            pic = ImageLabel(foot)
            pic.setPixmap(mini)
            fr.addWidget(pic)
        fv.addLayout(fr)
        self.footer = foot
        root.addSpacing(56)            # 给浮在底部的页脚留位置

        self._show_empty_state()   # 必须放在三栏创建之后


    # ── 左栏：识别结果 ──
    def _build_left(self):
        card = SimpleCardWidget(self)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(8)
        lay.addWidget(brand_title("识别结果", card))
        self.lbl_cls = _sized(StrongBodyLabel("—", card), "subtitle", weight=QFont.Bold)
        self.lbl_cls.setTextColor(NAU["primary"], NAU["primary"])
        lay.addWidget(self.lbl_cls)
        self.lbl_conf = t_caption("", card)
        lay.addWidget(self.lbl_conf)
        self.bar_conf = ProgressBar(card)
        self.bar_conf.setRange(0, 100)
        lay.addWidget(self.bar_conf)
        self.lbl_sev = t_caption("", card)      # 检出强度（图像尺度估计）
        self.lbl_sev.setWordWrap(True)
        lay.addWidget(self.lbl_sev)
        # 视觉复核级联的展示区：一致就报平安，不一致就在这里让人裁决
        self.cascade_box = QWidget(card)
        self.cascade_lay = QVBoxLayout(self.cascade_box)
        self.cascade_lay.setContentsMargins(0, 2, 0, 2)
        self.cascade_lay.setSpacing(6)
        self.cascade_box.setVisible(False)
        lay.addWidget(self.cascade_box)
        self.img_view = ImageLabel(card)
        self.img_view.setAlignment(Qt.AlignCenter)
        self.img_view.setMinimumHeight(180)
        lay.addWidget(self.img_view)
        lay.addWidget(t_block("各类别置信度", card))
        self.cls_bars = {}
        for name in CLASS_NAMES:
            row = QHBoxLayout()
            lbl = t_caption(name, card, color=NAU["muted"])   # 未检出时的弱化色
            lbl.setFixedWidth(100)                            # 放得下「白叶枯病 100%」
            bar = ProgressBar(card)
            bar.setRange(0, 100)
            bar.setFixedHeight(6)
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            lay.addLayout(row)
            self.cls_bars[name] = (lbl, bar)
        lay.addStretch()
        return card

    # ── 中栏：治理方案 ──
    def _build_mid(self):
        card = SimpleCardWidget(self)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 12, 12, 12)   # 4px 网格
        scroll, box = _scroll_body(card)
        box.addWidget(brand_title("治理方案", scroll))
        self.mid_box = box
        lay.addWidget(scroll)
        return card

    # ── 右栏：AI 分析 ──
    def _build_right(self):
        card = SimpleCardWidget(self)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(brand_title("AI 智能分析", card))
        head.addStretch()
        self.badge = CaptionLabel("", card)
        head.addWidget(self.badge)
        ic = getattr(FIF, "VOLUME", None) or getattr(FIF, "SPEAKERS", None)
        self.btn_speak = (PushButton(ic, "朗读", card) if ic is not None
                          else PushButton("朗读", card))
        self.btn_speak.setToolTip("朗读 AI 的回答（再点一次停止）")
        self.btn_speak.clicked.connect(self._toggle_speak)
        head.addWidget(self.btn_speak)
        self.cmb_voice = ComboBox(card)
        for label, _v in VOICE_OPTIONS:
            self.cmb_voice.addItem(label)
        self.cmb_voice.setFixedWidth(150)
        self.cmb_voice.setToolTip("朗读音色（Edge 神经网络语音）")
        head.addWidget(self.cmb_voice)
        btn_minus = TransparentPushButton("A-", card)
        btn_plus = TransparentPushButton("A+", card)
        btn_minus.setFixedWidth(38)
        btn_plus.setFixedWidth(38)
        btn_minus.clicked.connect(lambda: self._change_font(-1))
        btn_plus.clicked.connect(lambda: self._change_font(1))
        head.addWidget(btn_minus)
        head.addWidget(btn_plus)
        lay.addLayout(head)

        self.ai_view = TextEdit(card)
        self.ai_view.setReadOnly(True)
        self.ai_view.setPlaceholderText("识别完成后自动调用大模型分析…")
        lay.addWidget(self.ai_view, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.ai_input = LineEdit(card)
        self.ai_input.setPlaceholderText("输入问题追问 AI…")
        self.ai_input.returnPressed.connect(self._send)
        self.btn_send = PrimaryPushButton("发送", card)
        self.btn_send.clicked.connect(self._send)
        mic_ic = getattr(FIF, "MICROPHONE", None)
        self.btn_mic = (PushButton(mic_ic, "语音", card) if mic_ic is not None
                        else PushButton("语音", card))
        self.btn_mic.setToolTip("点击开始说话，说完再点一次结束")
        self.btn_mic.clicked.connect(self._toggle_listen)
        row.addWidget(self.btn_mic)
        self.btn_withimg = TransparentPushButton("带图 · 关", card)
        self.btn_withimg.setCheckable(True)
        self.btn_withimg.setToolTip("追问时把原图一起发给模型")
        self.btn_withimg.toggled.connect(self._toggle_withimg)
        row.addWidget(self.btn_withimg)
        self.btn_plan = PushButton(getattr(FIF, "ROBOT", FIF.EDIT), "规划执行", card)
        self.btn_plan.setToolTip("把输入框里的话当成一个任务：先出可审阅的执行计划，你确认后再执行")
        self.btn_plan.clicked.connect(self._plan_task)
        row.addWidget(self.btn_plan)
        row.addWidget(self.ai_input, 1)
        row.addWidget(self.btn_send)
        lay.addLayout(row)

        # 计划确认条：模型给了计划但不自动执行，等人点「执行」
        self.plan_bar = QWidget(card)
        pb = QHBoxLayout(self.plan_bar)
        pb.setContentsMargins(0, 0, 0, 0)
        pb.setSpacing(8)
        self.plan_label = CaptionLabel("", card)
        self.plan_label.setWordWrap(True)
        pb.addWidget(self.plan_label, 1)
        self.btn_run_plan = PrimaryPushButton("执行计划", card)
        self.btn_run_plan.clicked.connect(self._run_plan)
        pb.addWidget(self.btn_run_plan)
        self.btn_cancel_plan = TransparentPushButton("取消", card)
        self.btn_cancel_plan.clicked.connect(self._cancel_plan)
        pb.addWidget(self.btn_cancel_plan)
        self.btn_stop_task = PushButton("停止任务", card)
        self.btn_stop_task.setToolTip("停掉正在执行的任务（排队的会继续）")
        self.btn_stop_task.clicked.connect(lambda: self.task_queue.cancel_current())
        self.btn_stop_task.setVisible(False)
        pb.addWidget(self.btn_stop_task)
        self.btn_clear_queue = TransparentPushButton("清空排队", card)
        self.btn_clear_queue.clicked.connect(lambda: self.task_queue.clear_pending())
        self.btn_clear_queue.setVisible(False)
        pb.addWidget(self.btn_clear_queue)
        self.plan_bar.setVisible(False)
        lay.addWidget(self.plan_bar)
        return card

    # ── 数据填充 ──
    def resizeEvent(self, event):
        """页脚用绝对定位浮在底部：三栏内容多时，放进布局会被挤成 0 高"""
        super().resizeEvent(event)
        foot = getattr(self, "footer", None)
        if foot is not None:
            foot.setGeometry(28, self.height() - 66, self.width() - 56, 40)
            foot.raise_()

    def _show_empty_state(self):
        """空状态：先说清楚"为什么空"和"下一步做什么"（Nielsen #10 帮助与文档）"""
        self.lbl_cls.setText("还没有识别结果")
        self.lbl_conf.setText("请先在「图片识别」或「实时识别」页完成一次检测")
        self.img_view.setText("识别完成后这里会显示带检测框的图片")
        art = brand_pixmap("主楼剪影_淡金_透明.png", height=130)
        if art is not None:
            pic = ImageLabel(self)
            pic.setPixmap(art)
            pic.setAlignment(Qt.AlignCenter)
            self.mid_box.addSpacing(20)
            self.mid_box.addWidget(pic)
        tip = t_body("识别完成后，这里会给出农业防治、生物防治、防治指标和推荐药剂。", self, muted=True)
        tip.setAlignment(Qt.AlignCenter)
        self.mid_box.addWidget(tip)

    def show_result(self, result_list, image, mode):
        self.result_list = result_list

        # 识别完就自动存档到地块档案库（没有 GPS 就落成"待落点"），不必再手点按钮
        try:
            self.record_to_map(auto=True)
        except Exception as e:
            log.debug("识别后自动落图失败：%s", e)
        self.orig_img = image
        self.mode = mode
        best = max(result_list, key=lambda x: x["conf"]) if result_list else None
        self.cls_name = best["name"] if best else None
        self.conf = best["conf"] if best else 0.0
        self.lbl_cls.setText(f"病虫害：{self.cls_name}" if self.cls_name else "未检测到病虫害")
        self.lbl_conf.setText(f"置信度：{self.conf * 100:.2f}%" if best else "")
        self.bar_conf.setValue(int(self.conf * 100))
        # 检出强度：先把可算的量算出来（框数/面积占比/平均置信度），再交给模型和大模型解释
        self.severity = assess_severity(getattr(image, "shape", None), result_list)
        self.lbl_sev.setText(
            "检出强度：%s · %s（%s）" % (self.severity["level"], self.severity["text"],
                                        self.severity["basis"])
            if result_list else "检出强度：未检出目标")
        log_event("detection", mode=mode, cls=self.cls_name, conf=round(self.conf, 3),
                  boxes=len(result_list), severity=(self.severity or {}).get("level"))
        annotated = annotate_image(image, result_list) if result_list else image
        self.img_view.setPixmap(bgr_to_qpixmap(annotated).scaledToWidth(
            320, Qt.SmoothTransformation))
        # 三级颜色层级：主诊断=品牌绿，其余检出=正文色，未检出=弱化灰
        for name, (lbl, bar) in self.cls_bars.items():
            conf = next((r["conf"] for r in result_list if r["name"] == name), 0.0)
            bar.setValue(int(conf * 100))
            if not conf:
                lbl.setText(name)
                lbl.setTextColor(NAU["muted"], NAU["muted"])
                continue
            lbl.setText(f"{name} {conf * 100:.0f}%")
            c = NAU["primary"] if name == self.cls_name else NAU["text"]
            lbl.setTextColor(c, c)
        self._fill_treatment()
        self._start_cascade()      # 复核与 AI 分析并行：低置信度/病害类才触发
        self._start_ai()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _fill_treatment(self):
        box = self.mid_box
        # 清掉除标题以外的内容
        while box.count() > 1:
            item = box.takeAt(1)
            w = item.widget()
            if w is not None:
                w.setParent(None)      # 必须立刻脱离父级：deleteLater 是延迟的，
                w.deleteLater()        # 否则旧内容会盖在新内容上（会看到重影）
            elif item.layout() is not None:
                self._clear_layout(item.layout())
        tr = TREATMENT_DB.get(self.cls_name)
        if not tr:
            box.addWidget(t_body("未识别到具体病虫害，暂无治理方案", self, muted=True))
            box.addStretch()
            return

        def section(text):
            box.addWidget(t_body(text, self, color=NAU["primary"]))

        section("农业防治")
        for m in tr["agri"]:
            box.addWidget(t_body("· " + m, self))
        if tr.get("bio"):
            section("生物防治")
            for m in tr["bio"]:
                box.addWidget(t_body("· " + m, self))
        section("防治指标")
        box.addWidget(t_body("· " + tr["threshold"], self, color=NAU["red"]))
        # 量化指标（官方技术方案口径）——比"注意防治"这种话可执行得多
        q = (tr.get("meta") or {}).get("量化指标")
        if q:
            box.addWidget(t_body("· " + q, self))
            src = (tr.get("meta") or {}).get("指标来源")
            if src:
                box.addWidget(t_caption("　指标来源：" + src, self, color=NAU["subtext"]))
        section("推荐药剂")
        for drug in tr["drugs"]:
            box.addWidget(_drug_card(drug, self))
        box.addStretch()

    # ── 误检反馈（E1） ──
    def spray_from_plan(self):
        """识别 → 按方案施药并登记（写操作，人点触发）"""
        if not self.result_list:
            InfoBar.info("没有识别结果", "先识别一张图片，再按方案登记施药", duration=3000,
                         position=InfoBarPosition.TOP, parent=self)
            return
        best = max(self.result_list, key=lambda d: d.get("conf", 0))
        try:
            plots = [r["plot"] for r in (plot_risk_rows() or [])]
        except Exception:
            plots = []
        dlg = SprayPlanDialog(best.get("name", ""), best.get("conf", 0.0),
                              len(self.result_list), plots=plots, parent=self.window())
        if not dlg.exec():
            return
        v = dlg.values()
        if not v["drug"]:
            InfoBar.warning("没有可选药剂", "本地防治库暂无该类别的药剂，可到首页「记录施药」手工登记",
                            duration=5000, position=InfoBarPosition.TOP, parent=self)
            return
        photo = save_spray_photo(self.orig_img)
        rec = add_spray_record(
            plot=v["plot"], drug=v["drug"], cls=best.get("name", ""), dose=v["dose"],
            when=v["when"], note=v["note"], source="detection", safe_days=v["safe_days"],
            detection={"cls": best.get("name", ""),
                       "conf": round(float(best.get("conf") or 0), 3),
                       "boxes": len(self.result_list),
                       "image": os.path.basename(LAST_IMAGE_SOURCE or ""),
                       "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            photo=photo, plan_from="本地防治库 TREATMENT_DB")
        if rec is None:
            InfoBar.error("保存失败", "无法写入施药记录文件", duration=4000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已登记施药",
                        "%s · %s · %s\n药后雨量与补施判断会在首页「天气与施药窗口」自动核算"
                        % (rec["plot"] or "未填地块", rec["drug"], rec["time"][5:16]),
                        duration=6000, position=InfoBarPosition.TOP, parent=self)
        log_event("spray_from_detection", cls=rec["cls"], drug=rec["drug"],
                  plot=rec["plot"], photo=bool(photo))

    def record_to_map(self, auto=False):
        """把当前识别结果存档到地块地图。

        auto=True：识别完成后自动调用——不弹文件选择框、不打断用户；
        无原图或没结果就静默跳过（留给用户手动点按钮）。
        """
        st = geo_store()
        if st is None:
            InfoBar.warning("地块地图不可用", "缺少 rice_geo.py 或相关依赖",
                            duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        if not self.result_list:
            if not auto:
                InfoBar.info("没有可记录的结果", "先识别一张图片", duration=3000,
                             position=InfoBarPosition.TOP, parent=self)
            return
        src = LAST_IMAGE_SOURCE
        if not src or not Path(src).exists():
            if auto:
                return          # 自动模式下拿不到原图就跳过，不弹窗打断
            src, _ = QFileDialog.getOpenFileName(
                self, "选择这次识别的原始照片（用来读取拍摄时间与 GPS 坐标）", "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)")
            if not src:
                return
        meta = {}
        try:
            if self.geo is None:
                import rice_geo
                self.geo = rice_geo
            meta = self.geo.read_photo_meta(src)
        except Exception as e:
            log.warning("读取照片元数据失败：%s", e)
        best = max(self.result_list, key=lambda d: d.get("conf", 0))
        try:
            rec_id = st.add_record(
                shot_at=(meta.get("taken_at") or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                lon=meta.get("lon"), lat=meta.get("lat"),
                coord_src="exif" if meta.get("has_gps") else "manual",
                cls=best.get("name", ""), conf=float(best.get("conf") or 0),
                boxes=len(self.result_list),
                note="来自结果页（%d 个检测框）" % len(self.result_list),
                source_path=src, device=meta.get("device", ""))
        except Exception as e:
            log.exception("写入地块档案库失败")
            InfoBar.error("记录失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        log_event("map_record", rec_id=rec_id, has_gps=bool(meta.get("has_gps")),
                  cls=best.get("name", ""))
        if meta.get("has_gps"):
            InfoBar.success("已记录到地图",
                            "拍摄 %s · 坐标 %.5f, %.5f"
                            % ((meta.get("taken_at") or datetime.now()).strftime("%m-%d %H:%M"),
                               meta.get("lon"), meta.get("lat")),
                            duration=5000, position=InfoBarPosition.TOP, parent=self)
        else:
            InfoBar.info("已记录，但照片没有 GPS",
                         "已进「待落点」，请到「地块地图」页手动点一下实际位置",
                         duration=6000, position=InfoBarPosition.TOP, parent=self)

    def _feedback(self):
        """把这次结果标成「不对」，写进待标注队列"""
        if self.orig_img is None and not self.result_list:
            InfoBar.warning("还没有可反馈的内容", "请先完成一次识别", duration=3000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        predicted = self.cls_name or "未检出目标"
        dlg = FeedbackDialog(predicted, self)
        if not dlg.exec():
            return
        choice, note = dlg.values()
        sev = self.severity or {}
        try:
            saved = append_feedback_record(
                image=self.orig_img, predicted=predicted, conf=self.conf,
                boxes=sev.get("boxes", len(self.result_list)),
                severity=sev.get("level", ""), user_choice=choice, note=note,
                mode=self.mode)
        except Exception as e:
            log.exception("写入反馈失败")
            InfoBar.error("反馈保存失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已记录", "图片已存入 feedback 待标注队列（%s）"
                        % (os.path.basename(saved) if saved else "CSV 已写入"),
                        duration=4000, position=InfoBarPosition.TOP, parent=self)

    def _export(self):
        """导出识别报告（PNG / PDF）"""
        if not self.result_list:
            InfoBar.warning("没有可导出的内容", "请先完成一次识别", duration=3000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        ai_text = "".join(e[1] + "\n" for e in self._chat_history
                          if e[0] in ("ai_initial", "ai"))
        default = os.path.join(APP_DIR, "识别报告_%s.png" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        path, _ = QFileDialog.getSaveFileName(
            self, "保存识别报告", default, "PNG 图片 (*.png);;PDF 文档 (*.pdf)")
        if not path:
            return
        try:
            export_report(path, self.result_list, self.orig_img, self.cls_name,
                          self.conf, ai_text, self.main.device)
            add_report_record(path, self.cls_name, self.conf, self.main.device)
            self.main.reportInterface.reload()
        except Exception as e:
            log.exception("导出报告失败")
            InfoBar.error("导出失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已导出", os.path.basename(path), duration=4000,
                        position=InfoBarPosition.TOP, parent=self)

    def _again(self):
        self._stop_speak()
        self._stop_threads()
        if self.mode == "camera":
            self.main.go(self.main.cameraInterface)
        else:
            self.main.go(self.main.imageInterface)

    def _stop_speak(self):
        player = getattr(self, "_player", None)
        if player is not None:
            player.stop()
        for name in ("speak_thread", "piper_thread", "chattts_thread", "edge_thread"):
            th = getattr(self, name, None)
            if th is not None and th.isRunning():
                th.stop() if hasattr(th, "stop") else None
        lt = getattr(self, "listen_thread", None)
        if lt is not None and lt.isRunning():
            lt.stop()
            self._reset_mic()

    def _stop_threads(self):
        for t in (getattr(self, "llm_thread", None), getattr(self, "llm_thread2", None)):
            if t is not None and t.isRunning():
                t.stop()                 # 先断流，再等线程收尾
                t.wait(1500)

    # ── AI 分析 ──
    def _start_ai(self):
        self._chat_history = []
        self._api_history = []
        self._stream_role = None
        self._stream_index = None
        self._session_id = None              # 新的一次识别 = 新会话
        self._diag_card = None
        self._revised = False
        self._last_answer = ""
        self.badge.setText(" %s " % provider_label())
        # 能不能带图追问，取决于当前模型；不能看图就别给用户一个点了没用的按钮
        can_see = model_supports_vision()
        self.btn_withimg.setEnabled(can_see)
        self.btn_withimg.setToolTip("追问时把原图一起发给模型（更准，但每轮多花图片 token）"
                                    if can_see else
                                    "当前模型不支持看图（%s），换 deepseek-flash / *-vl-* 之类的模型后再用"
                                    % get_model_name())
        if not self.cls_name:
            self._chat_history = [("system", "没有识别到具体病虫害，可追问或换一张图片")]
            self._rebuild_chat()
            return
        if not CONFIG["enable_llm"] or not get_api_key():
            self._chat_history = [("system",
                "未启用云端 AI 分析。可在左侧「API 设置」中填入 API Key 并开启后重试。")]
            self._rebuild_chat()
            return
        self._chat_history = [("system", f"正在调用 {provider_label()} 分析 {self.cls_name} …")]
        self._stream_index = 0
        self._rebuild_chat()
        self._stream_stopped = False
        self._set_busy(True)
        log_event("analysis_start", cls=self.cls_name, conf=round(self.conf, 3),
                  severity=(self.severity or {}).get("level"),
                  provider=CONFIG.get("llm_provider"), model=get_model_name())
        self.llm_thread = _track_thread(
            LLMAnalysisThread(self.cls_name, self.orig_img, severity=self.severity,
                              boxes=self.result_list))
        self.llm_thread.fallback.connect(self._on_fallback)
        self.llm_thread.chunk.connect(lambda t: self._on_stream(t, "ai_initial"))
        self.llm_thread.reasoning.connect(lambda t: self._on_reasoning(t, "ai_initial"))
        self.llm_thread.tool_used.connect(self._on_tool)
        self.llm_thread.result_ready.connect(self._on_ai_result)
        self.llm_thread.failed.connect(self._on_ai_error)
        self.llm_thread.start()

    def _set_busy(self, busy):
        self._ai_busy = busy
        self.ai_input.setEnabled(not busy)
        self.btn_send.setEnabled(not busy)
        self.btn_stop.setVisible(busy)          # 生成中才出现「停止生成」
        self.ai_input.setPlaceholderText("AI 正在回复…" if busy else "输入问题追问 AI…")

    # ── 任务规划与执行（plan → 确认 → 执行 → 汇报） ──
    def _render_plan_html(self, plan):
        rows = ["<div style='font-size:13px; color:#0F7B4A; font-weight:bold; margin-bottom:4px;'>"
                "执行计划 · 待你确认</div>",
                "<div style='font-size:13px; margin:2px 0;'><b>目标：</b>%s</div>"
                % (plan.get("goal") or "—")]
        for i, s in enumerate(plan.get("steps") or [], 1):
            rows.append("<div style='font-size:13px; margin:6px 0 0 0;'>%d. <b>%s</b>"
                        "<span style='color:#5F6B66;'>（%s）</span></div>"
                        % (i, TOOL_LABELS.get(s.get("tool"), s.get("tool")),
                           s.get("tool")))
            if s.get("why"):
                rows.append("<div style='font-size:12px; color:#5F6B66; margin-left:16px;'>%s</div>"
                            % s["why"])
            if s.get("args"):
                rows.append("<div style='font-size:11.5px; color:#5F6B66; margin-left:16px;'>"
                            "参数：%s</div>"
                            % json.dumps(s["args"], ensure_ascii=False)[:160])
        if plan.get("note"):
            rows.append("<div style='font-size:12.5px; margin-top:6px;'>%s</div>" % plan["note"])
        for w in (plan.get("_warnings") or []):
            rows.append("<div style='font-size:12px; color:#C0392B;'>⚠ %s</div>" % w)
        rows.append("<div style='font-size:11.5px; color:#5F6B66; margin-top:6px;'>"
                    "只使用只读工具；写操作（导报告、落反馈）仍需你在界面上点。</div>")
        return ("<div style='background:#F2F7F4; border-left:3px solid #0F7B4A; border-radius:6px;"
                " padding:10px 12px; margin:8px 0;'>" + "".join(rows) + "</div>")

    def _plan_task(self):
        """把输入框里的话当作目标，先出计划（不自动执行）"""
        goal = self.ai_input.text().strip()
        if not goal:
            InfoBar.info("先说说要做什么",
                         "例如：把收件箱里的图跑一遍，告诉我哪几张需要人工看",
                         duration=5000, position=InfoBarPosition.TOP, parent=self)
            return
        if not CONFIG.get("enable_llm", False) or not get_api_key():
            InfoBar.warning("未启用云端 AI", "规划需要调用大模型，请先在设置里配置",
                            duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        self.ai_input.clear()
        self._chat_history.append(("user", "【任务】" + goal))
        self._chat_history.append(("system", "正在制定执行计划…"))
        self._stream_index = len(self._chat_history) - 1
        self._rebuild_chat(focus="bottom")
        self._set_busy(True)
        self.plan_thread = _track_thread(TaskPlanThread(goal))
        self.plan_thread.planned.connect(self._on_plan)
        self.plan_thread.failed.connect(self._on_plan_failed)
        self.plan_thread.start()

    def _on_plan(self, plan):
        idx = self._stream_entry_index()
        self._stop_stream_timer()
        self._pending_plan = plan
        html = self._render_plan_html(plan)
        if idx is None:
            self._chat_history.append(("card", html))
        else:
            self._chat_history[idx] = ("card", html)
        self._stream_index = None
        self._set_busy(False)
        n = len(plan.get("steps") or [])
        busy = self.task_queue.is_busy()
        self._on_queue_changed(self.task_queue.snapshot())
        if n:
            self.plan_bar.setVisible(True)
            self.btn_run_plan.setVisible(True)      # 忙时也允许提交（会排队）
            self.btn_cancel_plan.setVisible(True)
            self.plan_label.setText(
                ("计划共 %d 步 · 当前有任务在执行，确认后会排队" if busy
                 else "计划共 %d 步，确认后开始执行（可取消）") % n)
        self._rebuild_chat(focus="bottom")

    def _on_plan_failed(self, msg):
        idx = self._stream_entry_index()
        self._stop_stream_timer()
        self._stream_index = None
        if idx is None:
            self._chat_history.append(("error", "规划失败：%s" % msg))
        else:
            self._chat_history[idx] = ("error", "规划失败：%s" % msg)
        self._set_busy(False)
        self._rebuild_chat(focus="bottom")

    def _cancel_plan(self):
        self._pending_plan = None
        self.plan_bar.setVisible(False)
        self._chat_history.append(("system", "已取消该计划，未执行任何步骤。"))
        self._rebuild_chat(focus="bottom")

    def _run_plan(self):
        """确认执行：入队（忙的话排队，不会和正在跑的任务抢状态）"""
        plan = self._pending_plan
        if not plan:
            return
        self._pending_plan = None
        self._chat_history.append(("system", "已提交执行，等待队列调度…"))
        self._stream_index = None
        self._rebuild_chat(focus="bottom")
        pos = self.task_queue.submit(plan, self.orig_img, self.result_list)
        if pos > 1:
            InfoBar.info("已加入队列", "前面还有 %d 个任务，跑完自动开始" % (pos - 1),
                         duration=4000, position=InfoBarPosition.TOP, parent=self)
        log_event("task_run_start", goal=plan.get("goal"),
                  steps=len(plan.get("steps") or []), queue_position=pos)

    def _on_task_started(self, plan):
        self._chat_history.append(("system", "开始执行：%s" % (plan.get("goal") or "任务")))
        self._set_busy(True)
        self._rebuild_chat(focus="bottom")

    def _on_task_replanning(self, round_i, why):
        """自动重规划：把「为什么重来」讲清楚，别让它像无事发生"""
        self._chat_history.append(("system", "↻ 第 %d 次自动重规划（%s）…" % (round_i, why)))
        log_event("task_replan_ui", round=round_i, why=why)
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _on_queue_changed(self, snapshot):
        """队列状态：执行中/排队中一目了然"""
        busy = self.task_queue.is_busy()
        waiting = max(0, len(snapshot) - (1 if busy else 0))
        self.btn_stop_task.setVisible(busy)
        self.btn_clear_queue.setVisible(waiting > 0)
        if busy or waiting:
            self.plan_bar.setVisible(True)
            self.btn_run_plan.setVisible(False)
            self.btn_cancel_plan.setVisible(False)
            parts = []
            if busy:
                parts.append("执行中：%s" % (snapshot[0].get("goal") or "任务")[:24])
            if waiting:
                parts.append("排队 %d 个" % waiting)
            self.plan_label.setText("任务队列　" + "　·　".join(parts))
        elif self._pending_plan:
            self.plan_bar.setVisible(True)
            self.btn_run_plan.setVisible(True)
            self.btn_cancel_plan.setVisible(True)
        else:
            self.plan_bar.setVisible(False)

    def _on_task_step_started(self, i, step):
        self._chat_history.append(("tool", "第 %d 步：%s…"
                                   % (i + 1, TOOL_LABELS.get(step.get("tool"), step.get("tool")))))
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _on_task_step_done(self, i, step, ok, out):
        self._chat_history.append(("system", "%s %s" % ("✅ 完成" if ok else "⚠ 失败",
                                                        str(out)[:150].replace("\n", " "))))
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _on_task_done(self, plan, results):
        ok_n = sum(1 for _, ok, _ in results if ok)
        pending = len(self.task_queue.items)
        if pending:
            # 队列里还有任务：先不汇总，等整条队列跑完再统一汇报（避免几段汇报互相打断）
            self._chat_history.append(("system", "本任务完成：%d/%d 步成功；队列里还有 %d 个任务…"
                                       % (ok_n, len(results), pending)))
            self._rebuild_chat(focus="bottom")
            return
        self._chat_history.append(("system", "任务执行完毕：%d/%d 步成功，正在汇总…"
                                   % (ok_n, len(results))))
        self._rebuild_chat(focus="bottom")
        log_event("task_run_done", goal=plan.get("goal"), steps=len(results), ok=ok_n)
        # 让模型把结果写成一段结论（复用流式通道）
        prompt = ("下面是一次自动化任务的执行结果。请用中文写一段不超过 200 字的结论："
                  "先说结论，再指出需要人工关注的点；不要编造结果里没有的数据。\n\n%s"
                  % summarize_task(plan, results))
        self._stream_stopped = False
        self._stream_index = None
        self.summary_thread = _track_thread(LLMAnalysisThread(
            self.cls_name or "任务", None, None, prompt=prompt, severity=self.severity))
        self.summary_thread.chunk.connect(lambda t: self._on_stream(t, "ai"))
        self.summary_thread.result_ready.connect(self._on_followup)
        self.summary_thread.failed.connect(self._on_task_failed)
        self.summary_thread.start()

    def _on_task_failed(self, msg):
        self._chat_history.append(("error", "任务执行失败：%s" % msg))
        self._set_busy(False)
        self._rebuild_chat(focus="bottom")

    # ── 自我修正：校验发现问题就自己返工（不是把问题丢给用户） ──
    def _maybe_self_revise(self, answer_text=""):
        if self._revised:
            return
        issues = []
        hits = pesticide_hits(answer_text or "")
        if hits:
            issues.append("出现了禁用/停止使用的农药：%s" % "、".join(hits))
        unknown = [c.get("name") for c in (self._diag_card or {}).get("chemicals", [])
                   if c.get("source") == "unknown"]
        if unknown:
            issues.append("本地防治库未收录的药剂：%s" % "、".join(n for n in unknown if n))
        if not issues:
            return
        if not CONFIG.get("enable_llm", False) or not get_api_key():
            return
        self._revised = True
        prompt = ("你上一版回答里存在必须修正的问题：\n- " + "\n- ".join(issues) +
                  "\n请重写最终建议：只用本地防治库里的药剂（库外药剂必须写明「需人工核实登记与用量」，"
                  "禁用农药一律删除）；用量按本地库原值，其余结论保持不变。"
                  "直接给修正后的完整建议，不要解释你在改什么。")
        self._chat_history.append(("system", "↻ 校验发现问题，正在自动修正上一版建议…"))
        self._rebuild_chat(focus="bottom")
        self._stream_stopped = False
        self._stream_index = None
        self._set_busy(True)
        log_event("self_revision_start", cls=self.cls_name, issues=issues)
        self.llm_thread3 = _track_thread(LLMAnalysisThread(
            self.cls_name, None, list(self._api_history), prompt=prompt,
            severity=self.severity, boxes=self.result_list))
        self.llm_thread3.fallback.connect(self._on_fallback)
        self.llm_thread3.tool_used.connect(self._on_tool)
        self.llm_thread3.chunk.connect(lambda t: self._on_stream(t, "ai"))
        self.llm_thread3.result_ready.connect(self._on_revision)
        self.llm_thread3.failed.connect(self._on_revision_error)
        self.llm_thread3.start()

    def _on_revision(self, text):
        idx = self._stream_entry_index()
        self._stop_stream_timer()
        self._stream_index = None
        if not (text or "").strip():
            self._set_busy(False)
            return
        if idx is None:
            self._chat_history.append(("ai", text))
        else:
            self._chat_history[idx] = ("ai", text)
        self._api_history.append({"role": "assistant", "content": text})
        self._set_busy(False)
        self._rebuild_chat(focus="bottom")
        self._save_session()
        log_event("self_revision_done", cls=self.cls_name, chars=len(text),
                  banned=pesticide_hits(text))
        InfoBar.success("已自动修正建议", "禁用/库外药剂已按本地防治库重写",
                        duration=4000, position=InfoBarPosition.TOP, parent=self)

    def _on_revision_error(self, msg):
        self._stop_stream_timer()
        self._stream_index = None
        self._set_busy(False)
        log.info("自动修正未完成：%s", msg)

    # ── 视觉复核级联（低置信度/病害类自动触发 → 冲突人工裁决） ──
    def _clear_cascade(self):
        while self.cascade_lay.count():
            item = self.cascade_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
        self.cascade_box.setVisible(False)

    def _set_cascade_text(self, text, color=None):
        self._clear_cascade()
        lbl = t_caption(text, self, color=color or NAU["subtext"])
        lbl.setWordWrap(True)
        self.cascade_lay.addWidget(lbl)
        self.cascade_box.setVisible(True)

    def _start_cascade(self):
        """按规则决定要不要复核：置信度低于阈值，或属于病害三类"""
        self._clear_cascade()
        self.cascade_state = None
        need, why = should_auto_verify(self.cls_name, self.conf)
        if not need or self.orig_img is None:
            log.info("本次不做视觉复核：%s", why)
            return
        self._set_cascade_text("正在请第二个模型独立复核…（%s）" % why)
        log_event("cascade_start", cls=self.cls_name, conf=round(self.conf, 3), reason=why)
        self.cascade_thread = _track_thread(VisionCascadeThread(
            self.cls_name, self.conf, self.orig_img, self.result_list, why))
        self.cascade_thread.verdict.connect(self._on_cascade)
        self.cascade_thread.failed.connect(
            lambda msg: self._set_cascade_text("复核未完成：%s（不影响主结果）" % msg))
        self.cascade_thread.start()

    def _on_cascade(self, res):
        """复核回来了：一致就报平安；不一致就把裁决权交给用户"""
        self.cascade_state = res
        log_event("cascade_result", cls=self.cls_name, verdict=res.get("verdict"),
                  agrees=res.get("agrees"), reason=res.get("reason"))
        self._clear_cascade()
        if res.get("agrees"):
            self.cascade_box.setVisible(True)
            self.cascade_lay.addWidget(t_caption(
                "✅ 视觉复核一致：%s（%s）" % (res.get("verdict"), res.get("reason")),
                self, color=NAU["primary"]))
            return
        # 同类分歧（病↔病 / 虫↔虫）：处置方向一致，记台账不打扰用户
        if same_category(self.cls_name, res.get("verdict")):
            self.cascade_box.setVisible(True)
            self.cascade_lay.addWidget(t_caption(
                "ℹ 复核给出相近类别（%s），已记入台账，不打断你"
                % (res.get("verdict") or "未识别"), self, color=NAU["subtext"]))
            log_event("cascade_same_category", cls=self.cls_name, verdict=res.get("verdict"))
            return
        self.cascade_box.setVisible(True)
        self.cascade_lay.addWidget(t_caption(
            "⚠ 虫害/病害判断冲突（处置完全不同），请你确认实际是什么：",
            self, color=NAU["red"]))
        self.cascade_lay.addWidget(t_caption(
            "检测模型：%s（%.0f%%）　｜　视觉模型：%s"
            % (self.cls_name, self.conf * 100, res.get("verdict") or "未识别出类别"),
            self, color=NAU["subtext"]))
        row = QHBoxLayout()
        row.setSpacing(6)
        b1 = PushButton("是 %s" % self.cls_name, self)
        b1.clicked.connect(lambda: self._resolve_cascade(self.cls_name))
        row.addWidget(b1)
        if res.get("verdict"):
            b2 = PushButton("是 %s" % res["verdict"], self)
            b2.clicked.connect(lambda: self._resolve_cascade(res["verdict"]))
            row.addWidget(b2)
        b3 = TransparentPushButton("都不是", self)
        b3.clicked.connect(lambda: self._resolve_cascade("都不对/其他"))
        row.addWidget(b3)
        self.cascade_lay.addLayout(row)

    def _resolve_cascade(self, choice):
        """人工裁决 → 写进反馈队列：这条数据就是下次训练的原料"""
        res = self.cascade_state or {}
        try:
            append_feedback_record(
                image=self.orig_img, predicted=self.cls_name, conf=self.conf,
                boxes=len(self.result_list),
                severity=(self.severity or {}).get("level", ""),
                user_choice=choice,
                note="级联裁决：视觉模型=%s；触发原因=%s" % (res.get("verdict"), res.get("reason")),
                mode="cascade")
        except Exception as e:
            log.exception("写入裁决失败")
            InfoBar.error("记录失败", str(e), duration=4000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        log_event("cascade_resolve", cls=self.cls_name, vision=res.get("verdict"),
                  choice=choice)
        self._save_session()                  # 裁决也进会话记忆
        self._set_cascade_text("已记录：你确认是「%s」。这条已进入 feedback 待标注队列。"
                               % choice, NAU["primary"])
        InfoBar.success("已记录裁决", "将用于下一轮补数据", duration=3000,
                        position=InfoBarPosition.TOP, parent=self)

    # ── 流式输出 & 工具调用可视化 ──
    def _insert_history(self, idx, entry):
        """统一插入入口：插入会平移后面的条目，必须同步 _stream_index。

        踩过的坑：诊断卡插到最前面时，_stream_index 还指着旧位置，
        下一批流式分片就写到「修正说明」那条上，把它覆盖掉了。
        """
        idx = max(0, min(idx, len(self._chat_history)))
        self._chat_history.insert(idx, entry)
        if isinstance(self._stream_index, int) and self._stream_index >= idx:
            self._stream_index += 1

    def _stream_entry_index(self):
        """当前 AI 条目在 _chat_history 里的位置（工具提示会插在它前面，所以位置会变）"""
        i = getattr(self, "_stream_index", None)
        if isinstance(i, int) and 0 <= i < len(self._chat_history):
            return i
        return None

    def _on_stream(self, text, role):
        """收到分片：chunk 回传的是「累计全文」，覆盖写进当前 AI 条目；重绘 120ms 节流"""
        self._stream_role = role
        entry_role = "ai_initial" if role == "ai_initial" else "ai"
        idx = self._stream_entry_index()
        if idx is None:
            self._chat_history.append((entry_role, text))
            self._stream_index = len(self._chat_history) - 1
        else:
            self._chat_history[idx] = (entry_role, text)
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _on_tool(self, name, label):
        """工具调用可见化：用户看得到 Agent 正在查什么（人机协同 / 可解释性）"""
        idx = self._stream_entry_index()
        entry = ("tool", "正在%s…" % label)
        if idx is None:
            self._chat_history.append(entry)
        else:
            self._insert_history(idx, entry)          # 插在 AI 条目之前（下标自动顺延）
        log_event("tool_call", tool=name, label=label)
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _on_fallback(self, msg):
        """主服务商失败、自动切到备用：告诉用户，别让它在暗处发生"""
        log_event("provider_fallback", detail=msg)
        InfoBar.warning("已自动切换服务商", msg, duration=4000,
                        position=InfoBarPosition.TOP, parent=self)

    def _on_reasoning(self, text, role):
        """思考模式下：思维链内容不展示，但要让用户看到「它在动」而不是卡死"""
        hint = "深度思考中…（已思考 %d 字）" % len(text or "")
        idx = self._stream_entry_index()
        if idx is not None and self._chat_history[idx][0] == "system":
            self._chat_history[idx] = ("system", hint)
        if not self._stream_timer.isActive():
            self._stream_timer.start()

    def _stop_stream_timer(self):
        if self._stream_timer.isActive():
            self._stream_timer.stop()
        self._stream_role = None

    def _stop_generate(self):
        """停止生成：连接会被关掉，已经收到的内容保留"""
        th = getattr(self, "llm_thread2", None) if self._pending_user_msg \
            else getattr(self, "llm_thread", None)
        self._stream_stopped = True
        if th is not None and th.isRunning():
            th.stop()
        InfoBar.info("已停止生成", "已收到的内容会保留", duration=2500,
                     position=InfoBarPosition.TOP, parent=self)

    def _send(self):
        if self._ai_busy:
            return
        if self.task_queue.is_busy():
            InfoBar.warning("任务正在执行", "等它跑完再提问（可点「停止任务」中断）",
                            duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        msg = self.ai_input.text().strip()
        if not msg:
            return
        self.ai_input.clear()
        self._chat_history.append(("user", msg, None))
        self._chat_history.append(("system", "AI 思考中…"))
        self._rebuild_chat()
        self._pending_user_msg = msg
        self._stream_index = len(self._chat_history) - 1     # 占位符的位置
        self._set_busy(True)
        self._stream_stopped = False
        self.llm_thread2 = _track_thread(
            LLMAnalysisThread(self.cls_name,
                              self.orig_img if self._followup_with_image else None,
                              list(self._api_history), prompt=msg, severity=self.severity))
        self.llm_thread2.fallback.connect(self._on_fallback)
        self.llm_thread2.chunk.connect(lambda t: self._on_stream(t, "ai"))
        self.llm_thread2.reasoning.connect(lambda t: self._on_reasoning(t, "ai"))
        self.llm_thread2.tool_used.connect(self._on_tool)
        self.llm_thread2.result_ready.connect(self._on_followup)
        self.llm_thread2.failed.connect(self._on_followup_error)
        self.llm_thread2.start()

    def _on_ai_result(self, text):
        idx = self._stream_entry_index()
        self._stop_stream_timer()
        if not (text or "").strip() and self._stream_stopped:
            self._stream_index = None
            self._set_busy(False)            # 一个字都没收到就停了：保留界面上已有内容
            return
        if idx is None:
            self._chat_history.append(("ai_initial", text))
        else:
            self._chat_history[idx] = ("ai_initial", text)
        if self._stream_stopped:
            self._chat_history.append(("system", "（已停止生成，以上为部分内容）"))
        self._api_history = [{"role": "assistant", "content": text}]
        self._stream_index = None
        self._set_busy(False)
        self._rebuild_chat(focus="top")      # 首次分析：从头开始读
        self._presynth(text)                 # 顺手在后台把语音合成好
        hits = pesticide_hits(text or "")
        log_event("analysis_done", cls=self.cls_name, chars=len(text or ""),
                  banned=hits)
        if hits:                             # 合规红线：回答里出现禁用农药，立刻提醒并留痕
            InfoBar.warning("检测到禁用农药", "AI 回答中出现：%s，请勿使用" % "、".join(hits),
                            duration=6000, position=InfoBarPosition.TOP, parent=self)
        self._last_answer = text or ""
        self._maybe_self_revise(text or "")   # 有禁用农药就自己返工改一版
        self._save_session()                 # 首次分析先落一次盘（诊断卡到了再更新）
        if CONFIG.get("enable_diagnosis_card", True) and (text or "").strip():
            self.card_thread = _track_thread(
                DiagnosisCardThread(self.cls_name, text, self.severity))
            self.card_thread.card_ready.connect(self._on_card)
            self.card_thread.failed.connect(lambda m: log.info("诊断卡未生成：%s", m))
            self.card_thread.start()

    def _save_session(self, card=None):
        """把本次识别会话落盘（Agent 的长期记忆）：结论 + 诊断卡 + 问答历史 + 图片副本"""
        if not self.cls_name:
            return
        if card is not None:
            self._diag_card = card
        try:
            self._session_id = save_session(
                self.cls_name, self.conf, self.severity, self.mode,
                self._chat_history, self._diag_card, self.orig_img, self._session_id)
            log_event("session_saved", session=self._session_id, cls=self.cls_name,
                      turns=len(self._chat_history))
        except Exception as e:
            log.warning("会话保存失败: %s", e)

    def restore_session(self, sid):
        """载入历史会话：恢复结论、诊断卡与问答历史，可以接着追问（模型能用到上下文）"""
        rec = load_session(sid)
        if not rec:
            InfoBar.warning("会话不存在", "可能已被清理", duration=3000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        self._session_id = sid
        self._diag_card = rec.get("card")
        self.cls_name = rec.get("cls")
        self.conf = float(rec.get("conf") or 0.0)
        self.severity = {"level": rec.get("severity"), "text": rec.get("severity_text") or "",
                         "boxes": 0, "area_ratio": 0.0, "max_ratio": 0.0, "mean_conf": 0.0,
                         "basis": "历史会话记录"}
        self.result_list = []
        self.mode = rec.get("mode") or "image"
        self.lbl_cls.setText("病虫害：%s" % (self.cls_name or "—"))
        self.lbl_conf.setText("置信度：%.2f%%" % (self.conf * 100))
        self.bar_conf.setValue(int(self.conf * 100))
        self.lbl_sev.setText("历史会话：%s ｜ %s" % (rec.get("time"), rec.get("severity") or "—"))
        img_path = _session_image_path(sid)
        if img_path.exists():
            pix = QPixmap(str(img_path))
            if not pix.isNull():
                self.img_view.setPixmap(pix.scaledToWidth(320, Qt.SmoothTransformation))
                self.orig_img = cv2.imread(str(img_path))
        for name, (lbl, bar) in self.cls_bars.items():
            bar.setValue(0)
            lbl.setText(name)
            lbl.setTextColor(NAU["muted"], NAU["muted"])
        self._fill_treatment()
        turns = [tuple(t) for t in (rec.get("turns") or []) if len(t) >= 2]
        self._api_history = []
        for role, content in turns:
            if role == "user":
                self._api_history.append({"role": "user", "content": content})
            elif role in ("ai", "ai_initial"):
                self._api_history.append({"role": "assistant", "content": content})
        self._chat_history = turns or [("system", "这是一条历史会话记录")]
        if self._diag_card:
            self._chat_history.insert(0, ("card", render_diagnosis_card(self._diag_card,
                                                                       self.severity)))
        self._stream_index = None
        self.badge.setText(" 历史会话 ")
        self._rebuild_chat(focus="bottom")
        InfoBar.success("已载入会话", "%s · %s" % (rec.get("time"), rec.get("cls") or "未识别"),
                        duration=3000, position=InfoBarPosition.TOP, parent=self)

    def _on_card(self, card):
        """结构化诊断卡：插在工具提示之后、正文之前"""
        try:
            html = render_diagnosis_card(card, self.severity)
        except Exception as e:
            log.warning("诊断卡渲染失败: %s", e)
            return
        idx = 0
        for i, entry in enumerate(self._chat_history):
            if entry[0] == "tool":
                idx = i + 1
        self._insert_history(idx, ("card", html))
        log_event("diagnosis_card", cls=self.cls_name,
                  chemicals=[c.get("name") for c in card.get("chemicals", [])],
                  warnings=card.get("_warnings", []))
        self._rebuild_chat(focus="keep")
        self._save_session(card)          # 诊断卡一到手就更新会话（记忆里带上结构化结论）
        self._maybe_self_revise(self._last_answer)   # 库外药剂也要返工

    def _on_ai_error(self, msg):
        self._stop_stream_timer()
        self._stream_index = None
        self._chat_history = [("error", msg)]
        self._set_busy(False)
        self._rebuild_chat(focus="top")
        InfoBar.error("AI 调用失败", msg, duration=5000, position=InfoBarPosition.TOP, parent=self)

    def _on_followup(self, text):
        idx = self._stream_entry_index()
        self._stop_stream_timer()
        if not (text or "").strip() and self._stream_stopped:
            self._stream_index = None
            self._set_busy(False)
            return
        if idx is None:
            self._chat_history.append(("ai", text))
        else:
            self._chat_history[idx] = ("ai", text)
        if self._pending_user_msg:
            self._api_history.append({"role": "user", "content": self._pending_user_msg})
            self._pending_user_msg = None
        self._api_history.append({"role": "assistant", "content": text})
        self._stream_index = None
        self._rebuild_chat(focus="question")   # 停在提问处，答案就在下面
        self._presynth(text)
        self._set_busy(False)
        self._save_session()                   # 追问也进记忆

    def _on_followup_error(self, msg):
        idx = self._stream_entry_index()
        self._stop_stream_timer()
        self._pending_user_msg = None
        if idx is None:
            self._chat_history.append(("error", msg))
        else:
            self._chat_history[idx] = ("error", msg)
        self._stream_index = None
        self._rebuild_chat(focus="question")
        self._set_busy(False)
        InfoBar.error("AI 调用失败", msg, duration=5000, position=InfoBarPosition.TOP, parent=self)

    def _toggle_listen(self):
        """语音输入：再点一次结束"""
        th = getattr(self, "listen_thread", None)
        if th is not None and th.isRunning():
            th.stop()
            return
        model_dir = ASR_DIR / ASR_MODEL_NAME
        if not model_dir.is_dir():
            InfoBar.warning("语音模型未就绪",
                            "请把 Vosk 中文模型解压到 resources/asr/ 下",
                            duration=6000, position=InfoBarPosition.TOP, parent=self)
            return
        self.btn_mic.setText("停止")
        self.ai_input.setPlaceholderText("正在听…请说话")
        self.listen_thread = ListenThread(ascii_path(model_dir), self)
        self.listen_thread.partial.connect(self.ai_input.setText)   # 边说边出字
        self.listen_thread.final.connect(self._on_speech)
        self.listen_thread.failed.connect(self._on_speech_failed)
        self.listen_thread.start()

    def _reset_mic(self):
        self.btn_mic.setText("语音")
        self.ai_input.setPlaceholderText("输入问题追问 AI…")

    def _toggle_withimg(self, on):
        """追问带图开关：模型能看图时才允许打开"""
        on = bool(on) and model_supports_vision()
        self.btn_withimg.setChecked(on)
        self._followup_with_image = on
        self.btn_withimg.setText("带图 · 开" if on else "带图 · 关")

    def _on_speech(self, text):
        self.ai_input.setText(text)
        self._reset_mic()
        self.ai_input.setFocus()

    def _on_speech_failed(self, msg):
        self._reset_mic()
        if "Failed to create a model" in msg:
            msg = "模型加载失败：路径含中文。请把 resources/asr 复制到 C:/Users/Public/ 下"
        InfoBar.error("语音识别失败", msg, duration=8000,
                      position=InfoBarPosition.TOP, parent=self)

    def _presynth(self, text):
        """后台预合成。

        AI 回答一到就开始合成 —— 用户读那段字的几秒钟里音频已经好了，
        点「朗读」时直接播，感觉就是"秒出"，不用等网络往返。
        """
        text = speakable(text)
        if not text or len(text) < 4 or len(text) > 3000:
            return
        if getattr(self, "_pre_text", "") == text:
            return
        self._pre_text = text
        self._pre_file = None
        th = ChatTTSSpeakThread(text, self)
        th.ok.connect(self._pre_done)
        th.failed.connect(lambda msg: self._pre_edge(text))
        th.start()

    def _pre_edge(self, text, why=""):
        self._pre_file = None
        voice = VOICE_OPTIONS[max(0, min(self.cmb_voice.currentIndex(),
                                         len(VOICE_OPTIONS) - 1))][1]
        th = EdgeSpeakThread(text, voice, parent=self)
        th.ok.connect(self._pre_done)
        th.start()

    def _pre_done(self, path):
        self._pre_file = path

    def _toggle_speak(self):
        """朗读 / 停止。

        优先播预合成好的（AI 回答到达时已在后台合成），命中就是秒出；
        没命中才现合成，走 ChatTTS → Edge → Piper → SAPI 逐级降级。
        """
        player = getattr(self, "_player", None)
        th = getattr(self, "speak_thread", None)
        et = getattr(self, "edge_thread", None)
        pt = getattr(self, "piper_thread", None)
        ct = getattr(self, "chattts_thread", None)
        if ((player is not None and player.state() == player.PlayingState)
                or (th is not None and th.isRunning())
                or (et is not None and et.isRunning())
                or (pt is not None and pt.isRunning())
                or (ct is not None and ct.isRunning())):
            self._stop_speak()
            self.btn_speak.setText("朗读")
            return
        text = ""
        for role, content in ((e[0], e[1]) for e in self._chat_history):
            if role in ("ai_initial", "ai"):
                text = content
        text = speakable(text)
        if not text:
            InfoBar.info("没有可朗读的内容", "先让 AI 分析一次", duration=2500,
                         position=InfoBarPosition.TOP, parent=self)
            return
        self.btn_speak.setText("停止")
        pre = getattr(self, "_pre_file", None)
        if pre and os.path.exists(pre) and getattr(self, "_pre_text", "") == text:
            self._pre_file = None          # 用掉就清掉，避免重复播
            self._play_audio(pre)
            return
        self._last_speak_text = text
        idx = max(0, min(self.cmb_voice.currentIndex(), len(VOICE_OPTIONS) - 1))
        voice = VOICE_OPTIONS[idx][1]
        # 四级链路：ChatTTS 本地（快，首选）→ Edge 在线（音质最好）
        #          → Piper 本地 → 系统 SAPI（兜底）
        self.chattts_thread = ChatTTSSpeakThread(text, self)
        self.chattts_thread.ok.connect(self._play_audio)
        self.chattts_thread.failed.connect(lambda msg: self._speak_edge(text, voice, msg))
        self.chattts_thread.start()

    def _speak_edge(self, text, voice, why=""):
        """本地 ChatTTS 不可用 → 换 Edge 在线语音"""
        if why:
            log.info("ChatTTS 不可用（%s），改用 Edge", str(why)[:60])
        # parent 必须用关键字传：第三个位置参数是 rate
        self.edge_thread = EdgeSpeakThread(text, voice, parent=self)
        self.edge_thread.ok.connect(self._play_audio)
        self.edge_thread.failed.connect(lambda msg: self._speak_piper(text, msg))
        self.edge_thread.start()

    def _speak_piper(self, text, why):
        """Edge 用不了（多半是网络/地域限制）→ 换本地语音"""
        log.info("Edge 语音不可用（%s），改用本地 Piper", str(why)[:50])
        if not PIPER_MODEL.exists():
            self._speak_sapi(text)
            return
        self.piper_thread = PiperSpeakThread(text, PIPER_MODEL, self)
        self.piper_thread.ok.connect(self._play_audio)
        self.piper_thread.failed.connect(lambda msg: self._speak_sapi(text, msg))
        self.piper_thread.start()

    def _speak_sapi(self, text, why=""):
        """本地也失败 → 系统自带语音兜底"""
        if why:
            log.info("本地 Piper 不可用（%s），回退系统语音", str(why)[:50])
        self.speak_thread = SpeakThread(text, self)
        self.speak_thread.done.connect(lambda: self.btn_speak.setText("朗读"))
        self.speak_thread.start()

    def _play_audio(self, path):
        """播放合成好的音频（Edge 是 mp3，Piper 是 wav）"""
        try:
            from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
            from PyQt5.QtCore import QUrl
            if getattr(self, "_player", None) is None:
                self._player = QMediaPlayer(self)
                self._player.mediaStatusChanged.connect(self._on_media_status)
            self._cur_mp3 = path
            self._player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self._player.play()
        except Exception as e:
            log.warning("播放失败，回退系统语音: %s", e)
            self._speak_sapi(self._last_speak_text or "", str(e))

    def _on_media_status(self, status):
        from PyQt5.QtMultimedia import QMediaPlayer
        if status in (QMediaPlayer.EndOfMedia, QMediaPlayer.InvalidMedia):
            self.btn_speak.setText("朗读")
            p = getattr(self, "_cur_mp3", "")
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
            self._cur_mp3 = ""

    def _change_font(self, delta):
        self._ai_font_size = max(TYPE["body"] - 2, min(TYPE["body"] + 5, self._ai_font_size + delta))
        self._rebuild_chat(focus="keep")     # 调字号别把人甩到底部

    def _img_tag(self, img_bgr, max_w=200):
        if img_bgr is None:
            return ""
        key = (id(img_bgr), max_w)
        cached = self._img_tag_cache.get(key)
        if cached is not None:
            return cached[1]
        try:
            h, w = img_bgr.shape[:2]
            if max(h, w) > max_w:
                s = max_w / max(h, w)
                img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)))
            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            tag = (f"<img src='data:image/jpeg;base64,{b64}' "
                   f"style='max-width:{max_w}px; border-radius:6px;'>")
        except Exception as e:
            log.warning("聊天图片编码失败: %s", e)
            return ""
        self._img_tag_cache[key] = (img_bgr, tag)
        return tag

    def _style_md(self, html, fs):
        """把 Markdown 渲染出来的裸标签补上字号/颜色，适配 Fluent 的浅色底"""
        d, g = NAU["dark"], NAU["primary"]
        # fs 是正文字号（pt），标题按阶梯递增；行高给足，长文才不挤
        html = html.replace('<h1>', f'<h1 style="font-size:{fs + 5}pt; color:{d}; margin:16px 0 8px 0;">')
        html = html.replace('<h2>', f'<h2 style="font-size:{fs + 3}pt; color:{d}; margin:14px 0 6px 0;">')
        html = html.replace('<h3>', f'<h3 style="font-size:{fs + 1}pt; color:{g}; margin:12px 0 4px 0;">')
        html = html.replace('<p>', f'<p style="font-size:{fs}pt; line-height:1.85; margin:4px 0 9px 0;">')
        html = html.replace('<li>', f'<li style="font-size:{fs}pt; line-height:1.75; margin:3px 0;">')
        html = html.replace('<ul>', '<ul style="margin:4px 0 9px 0; padding-left:18px;">')
        html = html.replace('<ol>', '<ol style="margin:4px 0 9px 0; padding-left:18px;">')
        html = html.replace('<strong>', f'<strong style="color:{g};">')
        return html

    def _add_ai_icons(self, html):
        """给 AI 回答的小标题配 Lucide 图标（绿色 PNG，不是 emoji）"""
        size = max(15, min(26, self._ai_font_size + 4))
        for words, icon in AI_HEADING_ICONS:
            img = icon_img(icon, size)
            if not img:
                continue
            pattern = r'(<h[123][^>]*>)(.*?(?:%s).*?)(</h[123]>)' % "|".join(words)

            def _repl(m, img=img):
                return (m.group(0) if "<img" in m.group(2)
                        else m.group(1) + img + m.group(2) + m.group(3))

            html = re.sub(pattern, _repl, html, flags=re.IGNORECASE)
        return html

    def _rebuild_chat(self, focus="bottom"):
        """重建对话区。

        focus 决定滚到哪里 —— 长回答滚到底的话，用户得自己往回翻：
          bottom   滚到底部（发消息、调字号时用）
          top      回到顶部（首次 AI 分析时用：直接从头读）
          question 停在最后一条提问处（追问回答到达时用：答案就在提问下面）
          keep     保持原来的相对位置（调字号时用：别把人甩到底部）
        """
        fs = self._ai_font_size
        parts = []
        last_question = ""
        sb = self.ai_view.verticalScrollBar()
        ratio = (sb.value() / sb.maximum()) if (focus == "keep" and sb.maximum()) else 0.0
        for entry in self._chat_history:
            role, content = entry[0], entry[1]
            if role in ("ai_initial", "ai"):
                hits = pesticide_hits(content)       # 合规红线：模型偶尔会推荐禁用农药
                if hits:
                    log.warning("AI 回答中出现禁用/停用农药: %s", hits)
                html = self._add_ai_icons(self._style_md(md_to_html(content), fs))
                if hits:
                    html = banned_warning_html(hits) + html
                if role == "ai":
                    parts.append(f"<div style='margin:8px 0;'>{html}</div>")
                else:
                    parts.append(html)
            elif role == "user":
                last_question = content
                parts.append(
                    f"<div style='margin:10px 0; padding:8px 12px; background:#E9F3EE; "
                    f"border-radius:8px;'><b>你：</b>"
                    f"<span style='font-size:{fs}px;'>{content}</span></div>")
            elif role == "card":
                parts.append(content)          # 诊断卡：内容已经是 HTML
            elif role == "tool":
                # Agent 的工具调用过程：让用户看到「它在查什么」，而不是黑箱
                parts.append(f"<div style='color:#2F7D5A; font-size:{max(11, fs - 2)}px;"
                             f" margin:6px 0 2px 0;'>🔧 {content}</div>")
            elif role == "error":
                parts.append(f"<div style='color:#C0392B; font-size:{fs}px;'>⚠ {content}</div>")
            else:
                parts.append(f"<div style='color:#5F6B66; font-size:{fs}px;'>{content}</div>")
        if not parts:
            parts.append(f"<div style='color:#5F6B66; font-size:{fs}px;'>识别完成后自动分析…</div>")
        self.ai_view.setHtml("".join(parts))
        # setHtml 之后文档布局是惰性的：此刻 scrollBar.maximum() 还是 0，
        # 直接 setValue 会被夹成 0（实测踩过）。所以推迟一轮事件循环再滚。
        QTimer.singleShot(0, lambda f=focus, q=last_question, r=ratio: self._apply_scroll(f, q, r))

    def _apply_scroll(self, focus, question, ratio=0.0):
        """按 focus 决定滚动位置"""
        sb = self.ai_view.verticalScrollBar()
        if focus == "top":
            sb.setValue(0)
            return
        if focus == "keep":
            sb.setValue(int(sb.maximum() * ratio))
            return
        if focus == "question" and question:
            needle = " ".join(question.split())[:16]
            cur = self.ai_view.document().find(needle) if needle else None
            if cur is not None and not cur.isNull():
                y = self.ai_view.document().documentLayout().blockBoundingRect(cur.block()).top()
                sb.setValue(max(0, int(y) - 10))
                return
        sb.setValue(sb.maximum())


# ── 设置页 ────────────────────────────────────────────────────
# ── 实训考核：把「识别能力」变成可教学的实训流程（教育场景） ──────────
# 设计说明：老师/学生用同一个检测模型做「人机对照」——先自己判断，再看模型判断，
# 最后看知识库讲解。既练人，也暴露模型的短板，教学数据可导出成台账。
TRAIN_DIR = APP_DIR / "training"
TRAIN_RECORDS = TRAIN_DIR / "records.json"
TRAIN_LABEL_MAP = {          # 文件名前缀 → 类别（与数据集命名一致）
    "rice_leaf_roller": "稻纵卷叶螟",
    "rice_water_weevil": "稻水象甲",
    "rice_leafhopper": "稻叶蝉",
    "bacterial_blight": "白叶枯病",
    "brown_spot": "胡麻斑病",
    "rice_blast": "稻瘟病",
}
TRAIN_POOL_DIRS = ("dataset_fieldlike/images/val", "手机端可跑版/samples",
                   "dataset_splitfix/images/val")


def guess_label_from_name(filename):
    """从文件名前缀判断真实类别（数据集的命名约定：<英文类名>_<序号>.jpg）"""
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    for key, label in TRAIN_LABEL_MAP.items():
        if stem.startswith(key):
            return label
    return None


def collect_question_pool(extra_dirs=None, limit_per_dir=400):
    """收集可用作考题的图片（自带真值标签），返回 [{path,label,source}]"""
    roots = []
    for rel in TRAIN_POOL_DIRS:
        p = Path(res_path(rel))
        if p.is_dir():
            roots.append(p)
    for d in (extra_dirs or []):
        p = Path(d)
        if p.is_dir():
            roots.append(p)
    pool = []
    for root in roots:
        for f in sorted(root.glob("*.jpg"))[:limit_per_dir]:
            label = guess_label_from_name(f.name)
            if label:
                pool.append({"path": str(f), "label": label, "source": root.name})
    return pool


def load_train_records():
    try:
        if TRAIN_RECORDS.exists():
            data = json.loads(TRAIN_RECORDS.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("读取实训记录失败: %s", e)
    return []


def append_train_record(**fields):
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_train_records()
    rows.append(fields)
    try:
        TRAIN_RECORDS.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        log.warning("写入实训记录失败: %s", e)


def training_summary(rows):
    """成绩单：人的正确率、平均用时、错题分布、模型正确率（人机对照）"""
    rows = list(rows or [])
    n = len(rows)
    if not n:
        return {"题数": 0}
    human_ok = sum(1 for r in rows if r.get("correct"))
    # 只统计"有人机对照"的题（题库答题没有模型判断这一维，不能拉低模型正确率）
    model_rows = [r for r in rows if r.get("model_correct") is not None]
    model_ok = sum(1 for r in model_rows if r.get("model_correct"))
    secs = [float(r.get("seconds") or 0) for r in rows]
    wrong = Counter(r.get("label") for r in rows if not r.get("correct"))
    return {
        "题数": n,
        "人的正确率": round(human_ok / n * 100, 1),
        "模型正确率": (round(model_ok / len(model_rows) * 100, 1) if model_rows else None),
        "有人机对照的题数": len(model_rows),
        "平均用时秒": round(sum(secs) / n, 1),
        "错题类别": dict(wrong),
        "人机都错": sum(1 for r in model_rows
                    if not r.get("correct") and not r.get("model_correct")),
    }


# ── 自动巡检（无人值守）：不等人提问的 agent 循环 ────────────────
# ══════════════════════════════════════════════════════════════
# 定时自主任务：不等人在场，到点自己干
# --------------------------------------------------------------
# 跨会话的「自主目标」：程序开着的时候，到点自动执行既定动作，
# 结果落盘并在界面上通知。数据缺口提醒会指出"该补哪一类数据"，
# 直接对接训练侧（病害三类样本少是项目已知短板）。
SCHEDULE_PATH = APP_DIR / "schedule.json"
SCHEDULE_LOG = APP_DIR / "watch" / "schedule_log.txt"

DEFAULT_SCHEDULES = [
    {"id": "weekly_digest", "name": "每周数据简报", "action": "digest",
     "freq": "weekly", "weekday": 1, "time": "09:00", "enabled": True, "last_run": ""},
    {"id": "daily_watch", "name": "每天处理收件箱", "action": "watch_once",
     "freq": "daily", "time": "08:30", "enabled": False, "last_run": ""},
    {"id": "weekly_gap", "name": "每周补数据提醒", "action": "gap_check",
     "freq": "weekly", "weekday": 5, "time": "17:00", "enabled": True, "last_run": ""},
]
ACTION_LABELS = {"digest": "数据简报", "watch_once": "处理收件箱", "gap_check": "补数据提醒"}


def load_schedules():
    try:
        if SCHEDULE_PATH.exists():
            data = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
    except Exception as e:
        log.warning("读取定时任务失败: %s", e)
    return [dict(x) for x in DEFAULT_SCHEDULES]


def save_schedules(items):
    try:
        SCHEDULE_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    except OSError as e:
        log.warning("保存定时任务失败: %s", e)


def _schedule_due_at(item, now=None):
    """算出这条任务「本次应该执行的时刻」（今天或本周的那个点）"""
    now = now or datetime.now()
    try:
        hh, mm = [int(x) for x in str(item.get("time", "09:00")).split(":")]
    except ValueError:
        hh, mm = 9, 0
    if item.get("freq") == "weekly":
        # 配置里的 weekday 用 ISO 口径（周一=1），datetime.weekday() 是周一=0
        wd = (int(item.get("weekday", 1)) - 1) % 7
        delta = (now.weekday() - wd) % 7
        cand = (now - timedelta(days=delta)).replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cand > now:
            cand -= timedelta(days=7)
        return cand
    cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand > now:
        cand -= timedelta(days=1)
    return cand


def due_schedules(now=None):
    """到点且还没跑过的任务"""
    now = now or datetime.now()
    out = []
    for item in load_schedules():
        if not item.get("enabled"):
            continue
        due_at = _schedule_due_at(item, now)
        last = str(item.get("last_run") or "")
        if last:
            try:
                if datetime.strptime(last[:19], "%Y-%m-%d %H:%M:%S") >= due_at:
                    continue
            except ValueError:
                pass
        out.append((item, due_at))
    return out


def mark_schedule_run(item_id, when=None):
    items = load_schedules()
    for it in items:
        if it.get("id") == item_id:
            it["last_run"] = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    save_schedules(items)


def data_gap_report():
    """补数据提醒：从反馈、复核分歧、实训错题里看哪几类最缺数据"""
    events = read_jsonl(EVENT_LOG)
    fb = load_feedback_rows()
    wrong = Counter(r.get("predicted") for r in fb
                    if r.get("user_choice") and r.get("user_choice") != r.get("predicted"))
    casc = Counter(e.get("cls") for e in events
                   if e.get("kind") == "cascade_result" and e.get("agrees") is False)
    # 只看"有人机对照且模型答错"的实训记录；题库答题没有模型这一维，不算数据缺口
    train_bad = Counter(e.get("cls") for e in events
                        if e.get("kind") == "training_answer"
                        and e.get("model_correct") is False)
    score = Counter()
    for src, weight in ((wrong, 2), (casc, 2), (train_bad, 1)):
        for k, v in src.items():
            if k:
                score[k] += v * weight
    lines = ["数据缺口提醒（%s）" % datetime.now().strftime("%Y-%m-%d %H:%M")]
    if not score:
        lines.append("· 暂无明显缺口证据：还没有足够的反馈/分歧/实训记录")
    else:
        lines.append("· 按证据强度排序，建议优先补这几类的标注数据：")
        for k, v in score.most_common(5):
            lines.append("  - %s（证据权重 %d：误检 %d、复核分歧 %d、实训错题 %d）"
                         % (k, v, wrong.get(k, 0), casc.get(k, 0), train_bad.get(k, 0)))
        lines.append("· 方法：把 feedback/ 待标注目录里的图补标后并入训练集，"
                     "注意病害三类的独立簇本来就少")
    lines.append("· 参考：检测短板为白叶枯病（五折 mAP50 0.7701），稻瘟病框不准（mAP50-95 0.3994）")
    return "\n".join(lines)


def run_schedule_action(action):
    """执行一个定时动作，返回可直接展示的文本"""
    if action == "digest":
        s = usage_summary()
        lines = ["数据简报（%s）" % datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "· 检测 %s 次（其中无人值守 %s 次），未检出 %s 次"
                 % (s["检测次数"], s.get("其中无人值守处理", 0), s["未检出次数"]),
                 "· 类别分布：%s" % (s["检出类别分布"] or "无"),
                 "· AI 分析 %s 次；工具调用 %s 次；诊断卡 %s 张（告警 %s）"
                 % (s["AI 分析次数"], s["工具调用次数"], s["诊断卡生成次数"],
                    s["诊断卡告警次数"]),
                 "· 视觉复核 %s 次，一致率 %s%%；人工裁决 %s 次"
                 % (s["视觉复核次数"], s["视觉复核一致率%"], s["人工裁决次数"]),
                 "· 反馈 %s 条（漏检 %s）；实训 %s 题，人正确率 %s%%、模型正确率 %s%%"
                 % (s["反馈条数"], s["其中漏检反馈"], s["实训答题数"],
                    s["实训正确率%"], s["模型在实训中的正确率%"])]
        return "\n".join(lines)
    if action == "watch_once":
        files = list_watch_candidates(WATCH_DIR, 50)
        if not files:
            return "收件箱没有新图片（目录：%s）" % WATCH_DIR
        rows = [process_image_file(p) for p in files]
        alerts = [r for r in rows if alert_level(r)]
        return ("处理收件箱：%d 张，%d 条需要关注\n" % (len(rows), len(alerts)) +
                "\n".join("· %s → %s（%s）%s" % (r["file"], r["cls"] or "未检出",
                                             r["severity"], alert_level(r))
                          for r in rows[:10]))
    if action == "gap_check":
        return data_gap_report()
    return "未知动作：%s" % action


def write_schedule_log(text):
    try:
        SCHEDULE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SCHEDULE_LOG, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 50 + "\n" + text + "\n")
    except OSError as e:
        log.warning("写定时任务日志失败: %s", e)


class ScheduleWorker(QThread):
    """执行一个定时动作（放后台，别卡界面）"""
    done = pyqtSignal(str, str)      # (动作, 结果文本)
    failed = pyqtSignal(str, str)

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action

    def run(self):
        try:
            text = run_schedule_action(self.action)
            write_schedule_log(text)
            log_event("schedule_run", action=self.action, ok=True)
            self.done.emit(self.action, text)
        except Exception as e:
            log.exception("定时任务执行失败: %s", self.action)
            log_event("schedule_run", action=self.action, ok=False, error=str(e))
            self.failed.emit(self.action, str(e))


# ══════════════════════════════════════════════════════════════
# 手机传图：局域网直传原文件（不经微信/QQ，不转码，EXIF 完整保留）
# --------------------------------------------------------------
# 为什么要有它：现场用手机拍的照片，走微信/QQ 传到电脑会被重新编码，
# EXIF（含 GPS）就没了——地图模块的自动落点也就失效。这里走局域网直传：
# 浏览器把 File 对象的原始字节 POST 上来，服务端原样落盘，**一个字节都不改**。
# ══════════════════════════════════════════════════════════════
PHONE_UPLOAD_PORT = 8766          # 与手机端演示服务（8765）错开
PHONE_UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".tif", ".tiff"}
# 注意：这里必须是**原始字符串** r"""：页面里 JS 写的 '\n' 若被 Python 先转义成真换行，
# 会把 JS 字符串截断 → 整段脚本语法错误（"Invalid or unexpected token"）→ 页面按钮全部失灵。
PHONE_UPLOAD_PAGE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>稻视界 · 手机传图</title><style>
:root{--pri:#2e7d5b;--bg:#f4f7f5;--sub:#5f6b66;--err:#c62828;--warn:#ef6c00}
*{box-sizing:border-box}
body{font:16px/1.7 -apple-system,"Microsoft YaHei",sans-serif;margin:0;padding:16px 16px 32px;
     background:var(--bg);color:#22312b;-webkit-text-size-adjust:100%}
h2{margin:0 0 4px;font-size:20px}
.host{color:var(--sub);font-size:13px;margin-bottom:12px}
.box{background:#fff;border-radius:12px;padding:14px;box-shadow:0 2px 10px rgba(0,0,0,.07);
     margin-bottom:12px}
.row{display:flex;gap:10px}
button,.btn{flex:1;padding:15px 10px;font-size:16px;border:0;border-radius:10px;
       background:var(--pri);color:#fff;font-weight:600;text-align:center;
       display:block;cursor:pointer;-webkit-tap-highlight-color:transparent}
button.ghost,.btn.ghost{background:#e8efeb;color:#22312b}
button:disabled{opacity:.5}
/* 文件框放在屏幕外而不是 display:none——后者在部分手机浏览器上会点了没反应 */
.picker{position:absolute;left:-9999px;top:0;width:1px;height:1px;opacity:0}
#bar{height:8px;border-radius:4px;background:#e8efeb;overflow:hidden;margin-top:12px;display:none}
#bar i{display:block;height:100%;width:0;background:var(--pri);transition:width .15s}
#log{font-size:13px;color:var(--sub);margin-top:8px;white-space:pre-wrap;min-height:20px}
ul{list-style:none;padding:0;margin:0}
li{display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #eef2ef}
li:last-child{border-bottom:0}
li img{width:52px;height:52px;object-fit:cover;border-radius:8px;background:#e8efeb;flex:0 0 auto}
.nm{font-size:13px;word-break:break-all}
.tag{font-size:12px;padding:1px 6px;border-radius:6px;background:#e8efeb;color:var(--sub)}
.tag.ok{background:#e3f2e8;color:var(--pri)}
.tag.no{background:#fdeaea;color:var(--err)}
.tag.wait{background:#fff3e0;color:var(--warn)}
.res{font-size:12px;color:var(--pri);margin-top:2px}
.tip{font-size:12px;color:var(--sub);margin-top:10px}
</style></head><body>
<h2>稻视界 · 手机传图</h2>
<div class="host" id="host">原图直传 · 不压缩不转码 · GPS 与拍摄时间完整保留 <b>【v2 版】</b></div>

<div class="box">
  <div class="row">
    <!-- 用 label 原生触发：手机上 JS 调 hidden input 的 click() 经常没反应 -->
    <label class="btn" for="fc">📷 拍照</label>
    <label class="btn ghost" for="f">🖼 从相册选</label>
  </div>
  <div class="row" style="margin-top:8px">
    <button class="ghost" id="diag" type="button" style="flex:0 0 auto;padding:8px 14px;font-size:14px">
      测试连接</button>
    <span id="diagout" style="font-size:13px;color:#5f6b66;align-self:center"></span>
  </div>
  <input id="f" class="picker" type="file" accept="image/*" multiple>
  <input id="fc" class="picker" type="file" accept="image/*" capture="environment">
  <div id="bar"><i></i></div>
  <div id="log">选好照片会自动开始上传，传完这里会显示电脑端的识别结果。</div>
  <div class="tip">iPhone 建议：设置 → 相机 → 格式 → 选「兼容性最佳」（HEIC 格式电脑端可能读不到位置）</div>
</div>

<div class="box">
  <div style="font-weight:600;margin-bottom:6px">本次已传 <span id="n">0</span> 张</div>
  <ul id="list"></ul>
</div>

<script>
var $ = function(id){ return document.getElementById(id); };
var items = [];      // {name, url, gps, result}

function setLog(t){ $('log').textContent = t; }

function renderList(){
  $('n').textContent = items.length;
  $('list').innerHTML = items.map(function(it){
    var tag = it.gps === true ? '<span class="tag ok">含 GPS</span>'
            : it.gps === false ? '<span class="tag no">无 GPS</span>'
            : '<span class="tag">上传中</span>';
    var res = it.result ? '<div class="res">电脑端：' + it.result + '</div>' : '';
    return '<li><img src="' + it.url + '"><div><div class="nm">' + it.name + ' ' + tag + '</div>' + res + '</div></li>';
  }).join('');
}

function upload(fileObj, url, onProgress){
  return new Promise(function(resolve, reject){
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload?name=' + encodeURIComponent(fileObj.name));
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = function(e){
      if (e.lengthComputable) { onProgress(e.loaded / e.total); }
    };
    xhr.onload = function(){
      try { resolve(JSON.parse(xhr.responseText)); }
      catch(e){ reject(new Error('服务端返回异常 HTTP ' + xhr.status)); }
    };
    xhr.onerror = function(){ reject(new Error('网络中断')); };
    xhr.send(fileObj);
  });
}

function pollResult(it, tries){
  if (tries > 20) { return; }               // 约 60 秒还没结果就不再等
  setTimeout(function(){
    fetch('/result?name=' + encodeURIComponent(it.saved))
      .then(function(r){ return r.json(); })
      .then(function(j){
        if (j && j.ok && j.result) {
          it.result = j.result.cls + '（' + Math.round((j.result.conf || 0) * 100) + '%）'
                    + (j.result.phone_cls ? '｜手机端：' + j.result.phone_cls : '');
          renderList();
        } else { pollResult(it, tries + 1); }
      })
      .catch(function(){ pollResult(it, tries + 1); });
  }, 3000);
}

async function handle(files){
  if (!files || !files.length) { return; }
  var total = files.length, done = 0, ok = 0, fail = 0;
  $('bar').style.display = 'block';
  for (var i = 0; i < total; i++) {
    var f = files[i];
    var it = { name: f.name, url: URL.createObjectURL(f), gps: null, saved: '', result: '' };
    items.push(it); renderList();
    setLog('正在上传 ' + (i + 1) + '/' + total + '：' + f.name);
    var attempt = 0, res = null;
    while (attempt < 2 && !res) {                 // 失败自动重试一次
      attempt++;
      try {
        res = await upload(f, null, function(p){
          $('bar').firstElementChild.style.width =
            Math.round(((i + p) / total) * 100) + '%';
        });
      } catch (e) {
        if (attempt >= 2) { setLog('❌ ' + f.name + '：' + e.message); fail++; }
      }
    }
    if (res && res.ok) {
      ok++; done++;
      it.gps = res.gps ? !!res.gps.has_gps : null;
      it.saved = res.saved;
      renderList();
      pollResult(it, 0);
    }
  }
  $('bar').firstElementChild.style.width = '100%';
  setLog('完成：成功 ' + ok + ' 张' + (fail ? '，失败 ' + fail + ' 张（可重选再传）' : '')
         + '\n电脑端识别结果会自动出现在上面列表里。');
  setTimeout(function(){ $('bar').style.display = 'none'; $('bar').firstElementChild.style.width = '0'; }, 1200);
  $('f').value = ''; $('fc').value = '';
}

// 打开选择器交给 <label for>，这里只处理选完之后的上传
$('f').onchange = function(){ handle(this.files); };

// 现场自检：连不上时一眼看出是"页面旧了"还是"网络不通"
$('diag').onclick = function(){
  var o = $('diagout');
  o.textContent = '检测中…';
  var t0 = Date.now();
  fetch('/ping?t=' + Date.now()).then(function(r){ return r.json(); }).then(function(j){
    o.textContent = '✅ 通（' + (Date.now() - t0) + 'ms）收件箱：' + (j.inbox || '');
  }).catch(function(e){
    o.textContent = '❌ 连不上：' + e.message + '（页面可能是缓存的旧版，换地址重开）';
  });
};
$('fc').onchange = function(){ handle(this.files); };

fetch('/whoami').then(function(r){ return r.json(); }).then(function(j){
  if (j && j.host) { $('host').textContent = '连接到：' + j.host + ' · 原图直传不转码'; }
}).catch(function(){});
</script></body></html>"""


class _PhoneUploadHandler(BaseHTTPRequestHandler):
    """只做两件事：给手机一个上传页、把上传的原文件原样落盘"""

    inbox = None          # 由 PhoneUploadServer 注入
    on_saved = None       # 回调（保存成功后通知界面）

    def log_message(self, fmt, *args):
        log.debug("手机传图 %s", fmt % args)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # 手机浏览器缓存很凶：页面和 /result 回执都不能缓存，
        # 否则改完页面手机上还是旧的、轮询结果也永远拿到第一次的响应
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(200, PHONE_UPLOAD_PAGE, "text/html; charset=utf-8")
        elif self.path.startswith("/ping"):
            self._send(200, json.dumps({"ok": True, "inbox": str(self.inbox)}))
        elif self.path.startswith("/whoami"):
            # 手机页显示"连到哪台电脑"，避免连错设备还以为传成功了
            self._send(200, json.dumps({"ok": True, "host": socket.gethostname(),
                                        "inbox": str(self.inbox)}, ensure_ascii=False))
        elif self.path.startswith("/result"):
            # 识别结果回执：手机上传后轮询这里，几秒就能在手机上看到电脑端的结论
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name = os.path.basename((q.get("name") or [""])[0])
            hit = None
            if name:
                rp = Path(self.inbox) / (name + ".result.json")
                if rp.exists():
                    try:
                        hit = json.loads(rp.read_text(encoding="utf-8"))
                    except Exception as e:
                        log.debug("读识别结果失败：%s", e)
            self._send(200, json.dumps({"ok": hit is not None, "result": hit},
                                       ensure_ascii=False))
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        if self.path.split("?")[0] != "/upload":
            self._send(404, json.dumps({"ok": False, "error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n <= 0 or n > 80 * 1024 * 1024:
                self._send(400, json.dumps({"ok": False, "error": "文件大小不合法"},
                                           ensure_ascii=False))
                return
            raw = self.rfile.read(n)
            # 文件名从查询串来；只取 basename，防目录穿越
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name = os.path.basename((q.get("name") or ["photo.jpg"])[0]) or "photo.jpg"
            ext = Path(name).suffix.lower()
            if ext not in PHONE_UPLOAD_EXTS:
                self._send(400, json.dumps({"ok": False, "error": "不支持的格式：%s" % ext},
                                           ensure_ascii=False))
                return
            inbox = Path(self.inbox)
            inbox.mkdir(parents=True, exist_ok=True)
            dst = inbox / ("%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), name))
            dst.write_bytes(raw)                    # ← 原字节落盘，不转码
            # 立刻读一次 EXIF，把「有没有 GPS」回给手机，现场就能确认
            gps = None
            try:
                import rice_geo
                meta = rice_geo.read_photo_meta(str(dst))
                gps = {"has_gps": bool(meta.get("has_gps")), "lon": meta.get("lon"),
                       "lat": meta.get("lat")}
            except Exception as e:
                log.debug("读手机上传图 EXIF 失败：%s", e)
            log_event("phone_upload", file=dst.name, bytes=len(raw),
                      has_gps=bool(gps and gps.get("has_gps")))
            if callable(self.on_saved):
                try:
                    self.on_saved(str(dst), gps or {})
                except Exception:
                    log.exception("手机传图回调失败")
            # 打一个来源标记（复用 .phone.json 伴随文件）：让识别结果能区分
            # "手机传上来的"还是"电脑本地的"，地图与记录管理里都能看出来
            try:
                _write_phone_sidecar(dst, {"来源": "手机直传", "label": "", "score": None})
            except Exception as e:
                log.debug("写手机来源标记失败：%s", e)
            warn = ""
            if ext in (".heic", ".heif"):
                warn = ("HEIC 格式电脑端可能读不到位置信息：iPhone 可在"
                        "「设置 → 相机 → 格式」里改成「兼容性最佳」后重拍")
            self._send(200, json.dumps({"ok": True, "saved": dst.name,
                                        "bytes": len(raw), "gps": gps, "提示": warn},
                                       ensure_ascii=False))
        except Exception as e:
            log.exception("手机传图失败")
            self._send(500, json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))


class PhoneUploadServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class PhoneUploadThread(QThread):
    """在后台跑局域网接收服务；界面关掉就停"""
    saved = pyqtSignal(str, dict)      # 落盘路径, gps 信息

    def __init__(self, inbox, port=PHONE_UPLOAD_PORT, parent=None):
        super().__init__(parent)
        self.inbox, self.port = Path(inbox), int(port)
        self.httpd = None

    def run(self):
        try:
            handler = type("_H", (_PhoneUploadHandler,),
                           {"inbox": self.inbox, "on_saved": self._on_saved})
            self.httpd = PhoneUploadServer(("0.0.0.0", self.port), handler)
            log.info("手机传图服务已启动：http://%s:%d/", lan_ip(), self.port)
            self.httpd.serve_forever(poll_interval=0.3)
        except Exception as e:
            log.exception("手机传图服务启动失败：%s", e)

    def _on_saved(self, path, gps):
        self.saved.emit(path, gps or {})

    def stop(self):
        try:
            if self.httpd is not None:
                self.httpd.shutdown()
                self.httpd.server_close()
        except Exception:
            pass


def qr_matrix(text, border=3):
    """文本 → 二维码矩阵。qrcode 随程序分发（vendor/qrcode，BSD-3），离线可用。"""
    vp = BASE_DIR / "vendor"
    if vp.is_dir() and str(vp) not in sys.path:
        sys.path.insert(0, str(vp))
    try:
        import qrcode
    except ImportError:
        log.info("未随程序分发 qrcode，二维码不可用（可手输网址）")
        return None
    try:
        q = qrcode.QRCode(border=border, error_correction=qrcode.constants.ERROR_CORRECT_M)
        q.add_data(text)
        q.make(fit=True)
        return q.get_matrix()
    except Exception as e:
        log.warning("生成二维码失败：%s", e)
        return None


def qr_pixmap(text, size=200):
    """矩阵 → QPixmap（黑白 + 静区；静区不能省，否则扫码容易失败）"""
    m = qr_matrix(text)
    if not m:
        return None
    n = len(m)
    buf = bytearray()
    for row in m:
        for v in row:
            buf += b"\x00\x00\x00" if v else b"\xff\xff\xff"
    try:
        img = QImage(bytes(buf), n, n, n * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(img.scaled(size, size, Qt.KeepAspectRatio,
                                           Qt.FastTransformation))
    except Exception as e:
        log.warning("二维码转图像失败：%s", e)
        return None


def lan_ip():
    """取本机在局域网里的地址（手机要访问的那个）"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 80))       # 不会真的发包，只为拿到出口网卡地址
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class WatchThread(QThread):
    """后台巡检线程：轮询收件箱 → 新图就走完整流水线（检测 + 按需复核 + 落库）"""
    processed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    finished_once = pyqtSignal(int)

    def __init__(self, folder, interval=5.0, run_once=False, do_cascade=True,
                 limit=20, parent=None):
        super().__init__(parent)
        self.folder = str(folder)
        self.interval = float(interval)
        self.run_once = bool(run_once)
        self.do_cascade = do_cascade
        self.limit = limit
        self._stop = False

    def run(self):
        total = 0
        while not self._stop:
            try:
                files = list_watch_candidates(self.folder, self.limit)
                for f in files:
                    if self._stop:
                        break
                    rec = process_image_file(f, do_cascade=self.do_cascade)
                    total += 1
                    self.processed.emit(rec)
            except Exception as e:
                log.exception("巡检线程异常")
                self.failed.emit(str(e))
            if self.run_once:
                break
            for _ in range(int(max(1.0, self.interval) * 10)):   # 分片睡眠，停止响应快
                if self._stop:
                    break
                self.msleep(100)
        self.finished_once.emit(total)

    def stop(self):
        self._stop = True


class PhoneUploadDialog(MessageBoxBase):
    """手机传图：局域网直传原文件，不转码、GPS 完整保留"""

    def __init__(self, folder, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.folder = Path(folder)
        self.thread = None
        self.count = 0
        self.viewLayout.addWidget(SubtitleLabel("手机传图（原图直传）", self))
        tip = BodyLabel("手机连到与本机同一个 WiFi，用浏览器打开下面的地址，"
                        "从相册选照片上传。\n上传的是原始文件字节，不压缩、不转码——"
                        "EXIF 里的拍摄时间与 GPS 都会完整保留，有 GPS 的会自动落到地块地图上。", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        row = QHBoxLayout()
        self.url = LineEdit(self)
        self.url.setReadOnly(True)
        self.url.setText("http://%s:%d/" % (lan_ip(), PHONE_UPLOAD_PORT))
        self.url.setMinimumHeight(38)
        row.addWidget(self.url, 1)
        btn_copy = PushButton(FIF.COPY, "复制地址", self)
        btn_copy.clicked.connect(self._copy)
        row.addWidget(btn_copy)
        self.viewLayout.addLayout(row)
        # 二维码：手机扫一下直接打开上传页，省得手输 IP 与端口
        self.qr = QLabel(self)
        self.qr.setAlignment(Qt.AlignCenter)
        _pix = qr_pixmap(self.url.text(), 210)
        if _pix is not None:
            self.qr.setPixmap(_pix)
            self.viewLayout.addWidget(self.qr)
            _qtip = BodyLabel("用手机相机或微信「扫一扫」直接打开上传页"
                              "（手机要与本机连同一个 WiFi；不同网络请用「云端同步」）", self)
            _qtip.setWordWrap(True)
            _qtip.setAlignment(Qt.AlignCenter)
            self.viewLayout.addWidget(_qtip)
        else:
            self.qr.setText("（未随程序分发 qrcode，请手动输入上面的网址）")
            self.viewLayout.addWidget(self.qr)
        self.status = CaptionLabel("服务未启动", self)
        self.status.setWordWrap(True)
        self.viewLayout.addWidget(self.status)
        self.btn_toggle = PrimaryPushButton(FIF.ROBOT, "启动接收服务", self)
        self.btn_toggle.clicked.connect(self.toggle)
        self.viewLayout.addWidget(self.btn_toggle)
        self.yesButton.setText("关闭")
        self.cancelButton.hide()
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(520)

    def _copy(self):
        QApplication.clipboard().setText(self.url.text())
        InfoBar.success("已复制", "把它粘贴到手机浏览器打开", duration=2500,
                        position=InfoBarPosition.TOP, parent=self)

    def toggle(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(3000)
            self.thread = None
            self.btn_toggle.setText("启动接收服务")
            self.status.setText("服务已停止（已接收 %d 张）" % self.count)
            return
        self.thread = _track_thread(PhoneUploadThread(self.folder))
        self.thread.saved.connect(self._on_saved)
        self.thread.start()
        self.btn_toggle.setText("停止服务")
        self.status.setText("服务已启动，等待手机上传…（同一 WiFi 下打开上面地址）")

    def _on_saved(self, path, gps):
        self.count += 1
        name = os.path.basename(path)
        if gps.get("has_gps"):
            self.status.setText("已接收 %d 张｜最新：%s ✅ 含 GPS（%.5f, %.5f）→ 会自动落点"
                                % (self.count, name, gps.get("lon") or 0, gps.get("lat") or 0))
        else:
            self.status.setText("已接收 %d 张｜最新：%s（无 GPS，稍后在地图上点一下落位）"
                                % (self.count, name))

    def closeEvent(self, e):
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(2000)
            self.thread = None
        super().closeEvent(e)


class RelaySyncThread(QThread):
    """后台从中转服务拉取小程序上传的图片与结论（要联网，不能卡界面）"""
    done = pyqtSignal(dict)

    def __init__(self, base, token="", reset=False, parent=None):
        super().__init__(parent)
        self.base, self.token, self.reset = base, token, reset

    def run(self):
        try:
            self.done.emit(relay_pull(self.base, token=self.token, reset=self.reset))
        except Exception as e:
            log.exception("云端同步失败")
            self.done.emit({"ok": False, "error": str(e)})


class RelaySyncDialog(MessageBoxBase):
    """云端同步：小程序拍完传中转，电脑端一键拉进收件箱"""

    def __init__(self, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.thread = None
        self.viewLayout.addWidget(SubtitleLabel("云端同步（小程序 → 收件箱）", self))
        tip = BodyLabel("手机不在同一网络时走这条路：小程序把图和识别结果传到中转服务，"
                        "电脑端主动拉取——不需要公网 IP，也不用做端口映射。\n"
                        "拉下来的图与原字节一致，手机端结论会写进伴随文件，自动巡检照常处理。", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.base = LineEdit(self)
        self.base.setPlaceholderText("中转服务地址，例如 https://your-domain.com 或 http://1.2.3.4:9000")
        self.base.setText(str(CONFIG.get("relay_base") or ""))
        self.viewLayout.addWidget(self.base)
        self.token = PasswordLineEdit(self)
        self.token.setPlaceholderText("拉取口令（中转服务设了 RELAY_TOKEN 才需要）")
        self.token.setText(str(CONFIG.get("relay_token") or ""))
        self.viewLayout.addWidget(self.token)
        row = QHBoxLayout()
        self.btn_sync = PrimaryPushButton(FIF.SYNC, "立即同步", self)
        self.btn_sync.clicked.connect(lambda: self.start_sync(False))
        row.addWidget(self.btn_sync)
        self.btn_reset = PushButton("从头拉一遍", self)
        self.btn_reset.setToolTip("忽略上次同步位置，把中转上的记录重新拉一次（已存在的图会跳过）")
        self.btn_reset.clicked.connect(lambda: self.start_sync(True))
        row.addWidget(self.btn_reset)
        self.viewLayout.addLayout(row)
        self.status = CaptionLabel("", self)
        self.status.setWordWrap(True)
        self.viewLayout.addWidget(self.status)
        rep = relay_sync_report()
        self.status.setText("上次同步：%s｜累计 %s 张｜收件箱里来自手机端 %s 张"
                            % (rep["上次同步"], rep["累计同步"], rep["收件箱里来自手机端的图"]))
        self.yesButton.setText("关闭")
        self.cancelButton.hide()
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(560)

    def start_sync(self, reset):
        if self.thread is not None and self.thread.isRunning():
            return
        base = self.base.text().strip()
        if not base:
            self.status.setText("请先填写中转服务地址")
            return
        # 记住配置，下次不用再输
        CONFIG["relay_base"] = base
        CONFIG["relay_token"] = self.token.text().strip()
        try:
            save_config(CONFIG)
        except Exception as e:
            log.debug("保存中转配置失败：%s", e)
        self.btn_sync.setEnabled(False)
        self.status.setText("正在同步…")
        self.thread = _track_thread(RelaySyncThread(base, self.token.text().strip(), reset))
        self.thread.done.connect(self._on_done)
        self.thread.start()

    def _on_done(self, res):
        self.btn_sync.setEnabled(True)
        if not res.get("ok"):
            self.status.setText("同步失败：%s" % res.get("error", "未知错误"))
            return
        n = res.get("新增", 0)
        self.status.setText("同步完成：新增 %d 张（已存在跳过 %d 张）。"
                            "回到「自动巡检」点「处理现有图片」即可开始检测。"
                            % (n, res.get("已存在跳过", 0)))
        if n:
            InfoBar.success("已拉取 %d 张" % n, "正在交给自动巡检处理…", duration=4000,
                            position=InfoBarPosition.TOP, parent=self)


class _HoverTextButton(TransparentPushButton):
    """默认只露图标、悬停展开文字的按钮，宽度自己算。

    为什么要自己算：qfluentwidgets 的 PushButton 自带最小宽度一百多像素
    （实测 sizeHint=104），两个并排 200+，反而把文件名挤没了。
    这里收成「图标 + 内边距」和「文字 + 内边距」两种宽度。

    为什么不重写 __init__：它的构造参数是**重载分派**的，子类换签名会被绕进
    递归（实测 TypeError: missing 2 required positional arguments）。
    不碰构造函数，只改尺寸方法就没事。
    """

    _W_ICON = 52            # 只有图标时：36px 左内边距（留给图标）+ 右内边距

    def sizeHint(self):
        base = super().sizeHint()
        if self.text():
            w = self._W_ICON + 8 + self.fontMetrics().horizontalAdvance(self.text())
        else:
            w = self._W_ICON
        return QSize(w, base.height())

    def minimumSizeHint(self):
        return self.sizeHint()


def hover_text_button(icon, text, parent=None):
    """默认只露图标的按钮：鼠标移上去才展开文字。

    列表里每条都挂两个按钮，全写文字会把文件名挤没（窗口一窄尤其明显），
    只给图标又不好认，所以折中成「悬停展开」。

    两个坑：
    1) 不要写死宽度。按钮自带内边距，写死 38px 的话留给图标的空间不够，
       图标会被裁成一条缝（实测踩过）。让它按 sizeHint 自己算。
    2) 不要用子类改构造签名：qfluentwidgets 的 PushButton 靠重载分派构造参数，
       子类换签名会被它绕进递归（TypeError: missing 2 required positional
       arguments）。用事件过滤器就不碰构造函数。
    """
    btn = _HoverTextButton(icon, "", parent)
    btn.setToolTip(text)

    class _HoverFilter(QObject):
        def eventFilter(self, obj, event):
            t = event.type()
            if t == QEvent.Enter:
                btn.setText(text)
            elif t == QEvent.Leave:
                btn.setText("")
            return False

    btn._hover_filter = _HoverFilter(btn)      # 挂在按钮上，免得被回收
    btn.installEventFilter(btn._hover_filter)
    return btn


class WatchInterface(QWidget):
    """自动巡检页：把「人拍照→人点按钮」变成「新照片自己进来→agent 自己处理」。"""

    POLL_SECONDS = 5
    RECENT_ROWS = 30          # 列表最多显示多少条历史记录

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("watchInterface")
        self.folder = WATCH_DIR
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("创建收件箱失败: %s", e)
        self.thread = None
        self.count = 0
        self.disagree = 0
        self._thumb_cache = {}       # 文件名 → 缩略图（避免每次刷新重新解码）

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(page_title("自动巡检", self))
        head.addStretch()
        self.btn_relay = PushButton(FIF.SYNC, "云端同步", self)
        self.btn_relay.setToolTip("小程序/手机不在同一网络时用它：从公网中转服务拉取图片与识别结果")
        self.btn_relay.clicked.connect(self._open_relay_sync)
        head.addWidget(self.btn_relay)
        self.btn_phone = PrimaryPushButton(FIF.CLOUD, "手机传图", self)
        self.btn_phone.setToolTip("手机连同一 WiFi，浏览器打开一个地址就能上传原图——"
                                  "不转码，EXIF 里的 GPS 完整保留")
        self.btn_phone.clicked.connect(self._open_phone_upload)
        head.addWidget(self.btn_phone)
        self.btn_ledger = PushButton(FIF.SAVE, "打开台账", self)
        self.btn_ledger.clicked.connect(self._open_ledger)
        head.addWidget(self.btn_ledger)
        self.btn_reload = PushButton(getattr(FIF, "UPDATE", FIF.SYNC), "刷新列表", self)
        self.btn_reload.setToolTip("重新读一遍台账，把最近的处理结果显示出来")
        self.btn_reload.clicked.connect(self._load_recent_rows)
        head.addWidget(self.btn_reload)
        self.btn_goto_map = PushButton(getattr(FIF, "GLOBE", FIF.FOLDER), "地块地图", self)
        self.btn_goto_map.setToolTip("跳到地块地图，看这些点位落在哪")
        self.btn_goto_map.clicked.connect(
            lambda: self.main.go(self.main.mapInterface))
        head.addWidget(self.btn_goto_map)
        root.addLayout(head)
        root.addWidget(CaptionLabel(
            "把手机/相机拍的图丢进收件箱，程序自动检测、（低置信度或病害类时）请第二个模型复核、"
            "并把结论落库；出现分歧或重度检出会主动提醒。", self))

        # 来源图例：三种来源用颜色区分，省得猜
        legend = QHBoxLayout()
        legend.setSpacing(16)
        legend.addWidget(t_caption("来源：", self, color=NAU["subtext"]))
        for _name in ("小程序", "手机直传", "电脑本地"):
            _m = SOURCE_META[_name]
            legend.addWidget(t_caption("%s %s" % (_m["dot"], _name), self, color=_m["color"]))
        legend.addStretch()
        root.addLayout(legend)

        bar = QHBoxLayout()
        self.path_label = LineEdit(self)
        self.path_label.setReadOnly(True)
        self.path_label.setText(str(self.folder))
        bar.addWidget(self.path_label, 1)
        btn_dir = PushButton(FIF.FOLDER, "选择目录", self)
        btn_dir.clicked.connect(self.choose_folder)
        bar.addWidget(btn_dir)
        self.btn_toggle = PrimaryPushButton(FIF.ROBOT, "开始监控", self)
        self.btn_toggle.setMinimumHeight(40)
        self.btn_toggle.clicked.connect(self.toggle)
        bar.addWidget(self.btn_toggle)
        self.btn_once = PushButton(FIF.PLAY, "处理现有图片", self)
        self.btn_once.setMinimumHeight(40)
        self.btn_once.clicked.connect(self.run_once)
        bar.addWidget(self.btn_once)
        root.addLayout(bar)

        self.status = CaptionLabel("未开始", self)
        root.addWidget(self.status)
        scroll, self.box = _scroll_body(self)
        root.addWidget(scroll, 1)
        # 先把台账里已有的记录显示出来：列表原来是纯内存的，程序一重启就空白，
        # 但手机上早就看到结果了，看着像电脑没干活
        self._load_recent_rows()

        # ── 定时任务：到点自己干（跨会话的自主目标）──
        self.sched_card = HeaderCardWidget("定时任务（程序开着就到点自动执行）", self)
        self.sched_result = CaptionLabel("", self)
        self.sched_result.setWordWrap(True)
        self.sched_card.viewLayout.addWidget(self.sched_result)
        root.addWidget(self.sched_card)
        self._reload_schedules()

    def _show_empty(self):
        tip = t_body("收件箱为空。把病叶照片复制到上面这个目录，程序会自动处理。", self, muted=True)
        tip.setAlignment(Qt.AlignCenter)
        self.box.addWidget(tip)
        self.box.addStretch()

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择要巡检的目录", str(self.folder))
        if d:
            self.folder = Path(d)
            self.path_label.setText(d)

    def toggle(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
            self.btn_toggle.setText("开始监控")
            self.status.setText("已停止 · 本次共处理 %d 张" % self.count)
            return
        self._load_recent_rows()      # 保留台账历史，新结果会插到最上面
        self.count = 0
        self.disagree = 0
        self.thread = _track_thread(WatchThread(self.folder, self.POLL_SECONDS))
        self.thread.processed.connect(self._on_processed)
        self.thread.failed.connect(lambda m: self.status.setText("巡检异常：%s" % m))
        self.thread.start()
        self.btn_toggle.setText("停止监控")
        self.status.setText("监控中 · 每 %d 秒检查一次 %s" % (self.POLL_SECONDS, self.folder))
        log_event("watch_start", folder=str(self.folder))

    def run_once(self):
        if self.thread is not None and self.thread.isRunning():
            InfoBar.info("正在运行", "先停止当前监控再执行单次处理", duration=2500,
                         position=InfoBarPosition.TOP, parent=self)
            return
        self._load_recent_rows()
        self.count = 0
        self.disagree = 0
        self.status.setText("正在处理现有图片…")
        self.thread = _track_thread(WatchThread(self.folder, run_once=True))
        self.thread.processed.connect(self._on_processed)
        self.thread.finished_once.connect(
            lambda n: self.status.setText("处理完成 · 本次共 %d 张" % n))
        self.thread.failed.connect(lambda m: self.status.setText("处理异常：%s" % m))
        self.thread.start()

    def _clear_rows(self):
        while self.box.count():
            item = self.box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _source_of(self, rec):
        """来源：实时记录带 phone_from；从台账读回来的没有这个字段，回查伴随文件"""
        src = str(rec.get("phone_from") or rec.get("_source") or "")
        if src in SOURCE_META:
            return src
        if "小程序" in src or "中转" in src:
            return "小程序"
        if "手机" in src:
            return "手机直传"
        return watch_source(self.folder / str(rec.get("file") or ""))

    def _thumb_for(self, rec):
        """缩略图（带缓存，刷新列表时不必重新解码）"""
        name = str(rec.get("file") or "")
        if not name:
            return None
        if name not in self._thumb_cache:
            self._thumb_cache[name] = watch_thumb_pixmap(self.folder / name)
        return self._thumb_cache[name]

    def _add_row_card(self, rec):
        """把一条结果渲染成卡片插到最上面（实时处理和读台账共用同一套渲染）"""
        card = SimpleCardWidget(self)
        card.setFixedHeight(74)          # 比原来高一点，放得下缩略图
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(12)

        # 缩略图：列表原来只有文件名，「这条是哪张图」得点开才知道
        thumb = self._thumb_for(rec)
        if thumb is not None:
            pic = ImageLabel(card)
            pic.setPixmap(thumb)
            pic.setFixedSize(thumb.width(), thumb.height())
            lay.addWidget(pic)

        col = QVBoxLayout()
        col.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(8)
        # 来源色标：小程序=蓝 / 手机直传=绿 / 电脑本地=灰
        src = self._source_of(rec)
        meta = SOURCE_META.get(src) or SOURCE_META["电脑本地"]
        head.addWidget(t_body("%s %s" % (meta["dot"], src), card, color=meta["color"]))
        stamp = str(rec.get("time") or "")[:16]      # 识别时间：2026-10-14 11:01
        if stamp:
            head.addWidget(t_caption(stamp, card, color=NAU["muted"]))
        mark = {True: "✅", False: "⚠", "": ""}.get(rec.get("verify_agree"), "")
        title = "%s　%s%s" % (rec.get("file", ""), rec.get("cls") or "未检出",
                             ("（%.0f%%）" % (rec["conf"] * 100)) if rec.get("conf") else "")
        head.addWidget(StrongBodyLabel(title, card), 1)
        col.addLayout(head)

        detail = "强度 %s" % (rec.get("severity") or "—")
        if rec.get("verify"):
            detail += "　｜　复核：%s %s" % (rec["verify"], mark)
        if rec.get("note"):
            detail += "　｜　%s" % rec["note"]
        col.addWidget(t_caption(detail, card, color=NAU["subtext"]))
        lay.addLayout(col, 1)

        # 右侧两个图标按钮：先「标记」再「跳过去看」。
        # 用图标而不是文字：窄窗口下也不会把文件名挤没。
        btn_map = hover_text_button(FIF.PIN, "标记到地图", card)
        btn_map.setToolTip("标记到地块地图\n"
                           "· 照片有 GPS → 直接按拍摄坐标落点\n"
                           "· 没有 GPS → 落成「待落点」，去地图上点一下位置")
        btn_map.clicked.connect(lambda _=None, r=dict(rec): self._mark_to_map(r))
        lay.addWidget(btn_map)

        btn_goto = hover_text_button(FIF.GLOBE, "跳转到地图", card)
        btn_goto.setToolTip("跳转到地块地图，看这个点位落在哪")
        btn_goto.clicked.connect(lambda _=None, r=dict(rec): self._goto_map(r))
        lay.addWidget(btn_goto)
        self.box.insertWidget(0, card)          # 新的放最上面
        while self.box.count() > self.RECENT_ROWS:
            item = self.box.takeAt(self.box.count() - 1)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    @staticmethod
    def _row_from_csv(r):
        """CSV 行 → 卡片用的记录（台账里 verify_agree 存的是 True/False/空 三种字符串）"""
        raw = (r.get("verify_agree") or "").strip().lower()
        agree = True if raw == "true" else (False if raw == "false" else "")
        try:
            conf = float(r.get("conf") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            boxes = int(r.get("boxes") or 0)
        except (TypeError, ValueError):
            boxes = 0
        return {"file": r.get("file", ""), "time": r.get("time", ""),
                "cls": r.get("cls", ""), "conf": conf, "boxes": boxes,
                "severity": r.get("severity", ""), "verify": r.get("verify", ""),
                "verify_agree": agree, "note": r.get("note", "")}

    def _load_recent_rows(self):
        """把台账（watch/results.csv）里最近的记录读回列表。

        台账一直有记录，之前只是没人读回界面 —— 于是「手机上能看到结果、电脑上空的」。
        """
        self._clear_rows()
        rows = []
        try:
            if WATCH_RESULTS.exists():
                with open(WATCH_RESULTS, "r", encoding="utf-8-sig", newline="") as f:
                    rows = [r for r in csv.DictReader(f) if (r.get("file") or "").strip()]
        except Exception as e:
            log.warning("读巡检台账失败：%s", e)
        if not rows:
            self._show_empty()
            return
        # 正序插入 + 每次插到 0 号位 = 最新的在最上面
        # （之前用了 reversed，结果最旧的跑到了最上面）
        for r in rows[-self.RECENT_ROWS:]:
            rec = self._row_from_csv(r)
            rec["_source"] = watch_source(self.folder / str(rec.get("file") or ""))
            self._add_row_card(rec)
        self.status.setText("台账最近 %d 条（共 %d 条）"
                            % (min(len(rows), self.RECENT_ROWS), len(rows)))

    def _goto_map(self, rec):
        """跳到地块地图看这条结果（先刷新，保证刚标记的点立刻可见）"""
        try:
            self.main.go(self.main.mapInterface)
            self.main.mapInterface.refresh()
        except Exception:
            log.warning("跳转地块地图失败", exc_info=True)

    def _mark_to_map(self, rec):
        """把这一条标到地块地图。

        照片有 GPS → 直接按拍摄坐标落点；没有 → 落成「待落点」，
        去地图上点一下位置（和「导入照片并落点」是同一套流程）。
        来源（手机直传/小程序/电脑本地）会一起写进去，地图上按它描边。
        """
        name = str(rec.get("file") or "")
        path = self.folder / name
        if not name or not path.exists():
            InfoBar.warning("找不到图片", "收件箱里没有 %s" % (name or "（空文件名）"),
                            duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        st = geo_store()
        if st is None:
            InfoBar.warning("不可用", "地块地图模块未加载", duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        src = SOURCE_CODES.get(self._source_of(rec), "pc")
        try:
            import rice_geo
            meta = rice_geo.read_photo_meta(str(path))
            stamp = (meta.get("taken_at") or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
            has_gps = bool(meta.get("has_gps")) and meta.get("lon") is not None
            if has_gps:
                sha = rice_geo.sha1_of_file(str(path))
                if sha and st.has_sha1(sha):
                    InfoBar.info("已经标过了", "这条记录早就在地图上", duration=3000,
                                 position=InfoBarPosition.TOP, parent=self)
                else:
                    st.add_record(shot_at=stamp, lon=meta["lon"], lat=meta["lat"],
                                  coord_src="exif", cls=rec.get("cls") or "",
                                  conf=float(rec.get("conf") or 0),
                                  boxes=rec.get("boxes") or 0,
                                  note="巡检结果标记到地图（含 GPS）",
                                  source_path=str(path), device=meta.get("device") or "",
                                  source=src)
                    log_event("watch_geo_add", file=name, lon=meta["lon"], lat=meta["lat"],
                              by="manual", source=src)
                    InfoBar.success("已标到地图",
                                    "按拍摄坐标落点：%.5f, %.5f" % (meta["lon"], meta["lat"]),
                                    duration=4000, position=InfoBarPosition.TOP, parent=self)
            else:
                st.add_record(shot_at=stamp, lon=None, lat=None, coord_src="none",
                              cls=rec.get("cls") or "",
                              conf=float(rec.get("conf") or 0),
                              boxes=rec.get("boxes") or 0,
                              note="待落点：照片没有 GPS，请在地图上点一下位置",
                              source_path=str(path), device=meta.get("device") or "",
                              source=src)
                log_event("watch_geo_add", file=name, by="manual", pending=True, source=src)
                InfoBar.info("已加入待落点",
                             "这张照片没有 GPS：去「地块地图」点一下位置就能落点",
                             duration=6000, position=InfoBarPosition.TOP, parent=self)
            self.main.mapInterface.refresh()      # 地图实时更新
        except Exception as e:
            log.exception("标记到地图失败")
            InfoBar.error("标记失败", str(e)[:120], duration=5000,
                          position=InfoBarPosition.TOP, parent=self)

    def _on_processed(self, rec):
        self.count += 1
        if rec.get("verify_agree") is False:
            self.disagree += 1
        self.status.setText("已处理 %d 张 · 复核分歧 %d 次" % (self.count, self.disagree))
        self._add_row_card(rec)
        # 有 GPS 的图会被自动落库，地图要跟着实时更新（不然得手动点刷新）
        try:
            self.main.mapInterface.refresh()
        except Exception:
            log.debug("巡检后刷新地图失败", exc_info=True)
        alert = alert_level(rec)
        if alert:
            InfoBar.warning("巡检提醒：%s" % rec.get("file"), alert, duration=6000,
                            position=InfoBarPosition.TOP, parent=self)
            log_event("watch_alert", file=rec.get("file"), reason=alert,
                      cls=rec.get("cls"))
        self.main.reportInterface.reload()

    # ── 定时任务 ──
    def _reload_schedules(self):
        """重建定时任务列表（保留标题那一行）"""
        while self.sched_card.viewLayout.count() > 1:
            item = self.sched_card.viewLayout.takeAt(1)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
        for it in load_schedules():
            row = QHBoxLayout()
            when = ("每周%s %s" % ("一二三四五六日"[(int(it.get("weekday", 1)) - 1) % 7],
                                  it.get("time", ""))
                    if it.get("freq") == "weekly" else "每天 %s" % it.get("time", ""))
            row.addWidget(BodyLabel("%s　·　%s　·　%s"
                                    % (it.get("name", ""), when,
                                       ACTION_LABELS.get(it.get("action"), it.get("action"))),
                                    self.sched_card), 1)
            if it.get("last_run"):
                row.addWidget(t_caption("上次 %s" % it["last_run"][5:16], self.sched_card,
                                        color=NAU["subtext"]))
            b_run = PushButton("立即运行", self.sched_card)
            b_run.clicked.connect(lambda _=None, x=dict(it): self.run_now(x))
            row.addWidget(b_run)
            b_tog = TransparentPushButton("已启用" if it.get("enabled") else "已停用",
                                          self.sched_card)
            b_tog.clicked.connect(lambda _=None, x=it.get("id"): self._toggle_schedule(x))
            row.addWidget(b_tog)
            self.sched_card.viewLayout.addLayout(row)

    def _toggle_schedule(self, sid):
        items = load_schedules()
        for it in items:
            if it.get("id") == sid:
                it["enabled"] = not it.get("enabled")
        save_schedules(items)
        log_event("schedule_toggle", id=sid)
        self._reload_schedules()

    def run_now(self, item):
        """手动立刻跑一次（不等到点）"""
        self.sched_result.setText("正在执行：%s…" % item.get("name", ""))
        self.sched_thread = _track_thread(ScheduleWorker(item.get("action")))
        self.sched_thread.done.connect(self.show_schedule_result)
        self.sched_thread.failed.connect(
            lambda a, e: self.sched_result.setText("定时任务失败（%s）：%s" % (a, e)))
        self.sched_thread.start()

    def show_schedule_result(self, action, text):
        """展示定时任务结果（完整文本同时写进 watch/schedule_log.txt）"""
        first = (text or "").split("\n")[0][:100]
        self.sched_result.setText("最近结果：%s" % first)
        InfoBar.success("定时任务完成", first, duration=6000,
                        position=InfoBarPosition.TOP, parent=self)

    def _open_relay_sync(self):
        dlg = RelaySyncDialog(parent=self.window())
        dlg.exec()
        try:
            self.refresh_candidates()
        except Exception as e:
            log.debug("刷新待处理列表失败：%s", e)
        log_event("relay_sync_dialog")

    def _open_phone_upload(self):
        """手机传图：上传的原图直接落进收件箱，顺带刷新待处理列表"""
        dlg = PhoneUploadDialog(self.folder, parent=self.window())
        dlg.exec()
        try:
            self.refresh_candidates()
        except Exception as e:
            log.debug("刷新待处理列表失败：%s", e)
        log_event("phone_upload_dialog", folder=str(self.folder))

    def _open_ledger(self):
        try:
            WATCH_RESULTS.parent.mkdir(parents=True, exist_ok=True)
            if not WATCH_RESULTS.exists():
                WATCH_RESULTS.write_text("", encoding="utf-8")
            os.startfile(str(WATCH_RESULTS))
        except Exception as e:
            InfoBar.error("打不开台账", str(e), duration=4000,
                          position=InfoBarPosition.TOP, parent=self)

    def leave(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)


# ══════════════════════════════════════════════════════════════
# 教学资料：题库与案例库（导入 → 结构化 → 可追溯）
# --------------------------------------------------------------
# 来源合规是硬要求：只处理「用户自己提供的资料 / 官方公开资料 / 开放许可内容」，
# 不做任何盗版书库的抓取。每条题目与案例都必须带 source 字段，
# 界面上、成绩单里、台账里都能追到出处——这也是评审要看的技术规范性。
TEACHING_DIR = APP_DIR / "teaching"
BANK_PATH = TEACHING_DIR / "question_bank.json"
CASES_PATH = TEACHING_DIR / "cases.json"
IMPORT_LOG = TEACHING_DIR / "imports.jsonl"
SOURCE_HINT = "本地防治库 / 官方公开资料 / 用户自有资料（请注明出处，勿导入盗版内容）"


def _load_json_list(path):
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("读取 %s 失败: %s", path, e)
    return []


def _save_json_list(path, items):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except OSError as e:
        log.warning("写入 %s 失败: %s", path, e)
        return False


def load_question_bank():
    return _load_json_list(BANK_PATH)


def save_question_bank(items):
    return _save_json_list(BANK_PATH, items)


def load_case_library():
    return _load_json_list(CASES_PATH)


def save_case_library(items):
    return _save_json_list(CASES_PATH, items)


# ── 文本提取：txt/md/csv/json/docx/pdf/html ──────────────────
def _extract_docx(path):
    """docx 就是一个 zip：直接读 word/document.xml，不依赖 python-docx"""
    import zipfile
    with zipfile.ZipFile(str(path)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p\s*>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    return re.sub(r"<[^>]+>", "", xml)


def _extract_pdf(path):
    """PDF 用随程序分发的 pypdf（BSD 许可，见 vendor/pypdf_LICENSE.txt）"""
    vp = BASE_DIR / "vendor"
    if vp.is_dir() and str(vp) not in sys.path:
        sys.path.insert(0, str(vp))
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "缺少 PDF 解析库（应随程序分发 vendor/pypdf）"
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages), ""


def _extract_html(text):
    text = re.sub(r"<script[\s\S]*?</script>", " ", text)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"\n{2,}", "\n", text)


def extract_text_from_file(path):
    """返回 (文本, 错误)。支持 txt / md / csv / json / docx / pdf / html"""
    p = Path(path)
    if not p.exists():
        return "", "文件不存在：%s" % p
    suffix = p.suffix.lower()
    try:
        if suffix in (".txt", ".md", ".csv", ".tsv", ".json"):
            return p.read_text(encoding="utf-8", errors="ignore"), ""
        if suffix in (".html", ".htm"):
            return _extract_html(p.read_text(encoding="utf-8", errors="ignore")), ""
        if suffix == ".docx":
            return _extract_docx(p), ""
        if suffix == ".pdf":
            return _extract_pdf(p)
        return "", "暂不支持的文件类型：%s（支持 txt/md/csv/json/docx/pdf/html）" % suffix
    except Exception as e:
        log.exception("提取文本失败: %s", path)
        return "", "解析失败：%s" % e


def fetch_url_text(url, limit_chars=20000):
    """抓取网页正文（用户自己给链接；请确认来源合法）"""
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "rice-agent/1.0"})
        if resp.status_code != 200:
            return "", "HTTP %s" % resp.status_code
        resp.encoding = resp.apparent_encoding or "utf-8"
        return _extract_html(resp.text)[:limit_chars], ""
    except Exception as e:
        return "", "抓取失败：%s" % e


# ── 结构化解析：CSV / Markdown / JSON 直接能读的格式 ──────────
def normalize_question(d):
    """把一条外部数据规整成题库条目；缺关键字段就返回 None"""
    if not isinstance(d, dict):
        return None
    q = str(d.get("question") or d.get("题目") or d.get("题干") or "").strip()
    if not q:
        return None
    opts = d.get("options") or d.get("选项") or []
    if isinstance(opts, str):
        opts = [x.strip() for x in re.split(r"[|;；]", opts) if x.strip()]
    opts = [str(x).strip() for x in opts if str(x).strip()]
    ans = str(d.get("answer") or d.get("答案") or "").strip()
    return {
        "id": str(d.get("id") or hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]),
        "type": str(d.get("type") or ("choice" if opts else "short")),
        "question": q,
        "options": opts,
        "answer": ans,
        "explain": str(d.get("explain") or d.get("解析") or "").strip(),
        "source": str(d.get("source") or d.get("来源") or SOURCE_HINT).strip(),
        "tags": d.get("tags") or d.get("标签") or [],
        "reviewed": bool(d.get("reviewed", False)),
    }


def parse_questions_delimited(text, delimiter=","):
    """CSV/TSV：题干,选项A,选项B,选项C,选项D,答案,解析,来源"""
    out = []
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any(str(c).strip() for c in r)]
    if not rows:
        return out
    head = [str(c).strip() for c in rows[0]]
    has_head = any(h in ("题目", "题干", "question") for h in head)
    body = rows[1:] if has_head else rows
    for r in body:
        r = [str(c).strip() for c in r]
        if len(r) < 2:
            continue
        d = {"question": r[0], "options": [x for x in r[1:5] if x],
             "answer": r[5] if len(r) > 5 else "", "explain": r[6] if len(r) > 6 else "",
             "source": r[7] if len(r) > 7 else SOURCE_HINT}
        q = normalize_question(d)
        if q:
            out.append(q)
    return out


def parse_questions_markdown(text):
    """宽松 Markdown 约定：

        题干（可多行）
        A. 选项   B. 选项   C. 选项   D. 选项
        答案：A
        解析：...
        来源：...
    """
    out, cur = [], {}
    def flush():
        if cur.get("question"):
            q = normalize_question(cur)
            if q:
                out.append(q)
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^([A-Da-d])[.、)．]\s*(.+)$", line)
        if m:
            cur.setdefault("options", []).append(m.group(2).strip())
            continue
        m = re.match(r"^(答案|answer)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            cur["answer"] = m.group(2).strip()
            continue
        m = re.match(r"^(解析|explain|说明)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            cur["explain"] = m.group(2).strip()
            continue
        m = re.match(r"^(来源|source)\s*[:：]\s*(.+)$", line, re.I)
        if m:
            cur["source"] = m.group(2).strip()
            continue
        if re.match(r"^(题|Q\d*|\d+[.、])\s*", line, re.I) and cur.get("question"):
            flush(); cur = {"question": re.sub(r"^(题|Q\d*|\d+[.、])\s*", "", line)}
            continue
        cur["question"] = (cur.get("question", "") + " " + line).strip()
    flush()
    return out


def parse_questions_json(text):
    """JSON：数组，或 {"questions": [...]}"""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("questions") or data.get("items") or []
    if not isinstance(data, list):
        return []
    return [q for q in (normalize_question(x) for x in data) if q]


def parse_cases_json(text):
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("cases") or data.get("items") or []
    out = []
    for c in (data if isinstance(data, list) else []):
        if not isinstance(c, dict):
            continue
        title = str(c.get("title") or c.get("标题") or "").strip()
        if not title:
            continue
        out.append({"id": str(c.get("id") or hashlib.sha1(title.encode()).hexdigest()[:12]),
                    "title": title,
                    "symptom": str(c.get("symptom") or c.get("症状") or ""),
                    "condition": str(c.get("condition") or c.get("发生条件") or ""),
                    "action": str(c.get("action") or c.get("处置") or ""),
                    "outcome": str(c.get("outcome") or c.get("结果") or ""),
                    "source": str(c.get("source") or c.get("来源") or SOURCE_HINT),
                    "tags": c.get("tags") or []})
    return out


# ── 从本地防治库生成题库与案例（自有内容，零版权风险） ──────────
def seed_teaching_from_local():
    """用本地防治库生成一套可直接用的题目与案例。返回 (questions, cases)"""
    rng = random.Random(20261013)          # 固定种子：每次生成结果一致，便于比对
    questions, cases = [], []
    local_src = "本地防治库（TREATMENT_DB，项目自有）"
    for cls, tr in TREATMENT_DB.items():
        others = [c for c in TREATMENT_DB if c != cls]

        # 1) 症状识别（四选一）
        wrong = [TREATMENT_DB[o].get("symptom", "")[:70] for o in others[:3]]
        opts = [tr.get("symptom", "")[:70]] + wrong
        rng.shuffle(opts)
        questions.append(normalize_question({
            "question": "下面哪一段症状描述对应「%s」？" % cls,
            "options": opts, "answer": tr.get("symptom", "")[:70],
            "explain": "该描述取自本地防治库中「%s」的症状字段。" % cls,
            "source": local_src, "tags": [cls], "type": "choice"}))

        # 2) 防治指标（简答）
        questions.append(normalize_question({
            "question": "「%s」的防治指标是什么？" % cls,
            "answer": tr.get("threshold", ""),
            "explain": "防治指标是决定是否施药的依据，各地植保站可能另有细化标准。",
            "source": local_src, "tags": [cls], "type": "short"}))

        # 3) 药剂归属（四选一，用其他类别的药剂做干扰项）
        drugs = tr.get("drugs", [])
        if drugs:
            answer = drugs[0].get("name", "")
            pool = [d.get("name", "") for o in others for d in TREATMENT_DB[o].get("drugs", [])]
            rng.shuffle(pool)
            opts = [answer] + pool[:3]
            rng.shuffle(opts)
            questions.append(normalize_question({
                "question": "下面哪种药剂是本地防治库为「%s」推荐的？" % cls,
                "options": opts, "answer": answer,
                "explain": "库内推荐：%s。" % "、".join(d.get("name", "") for d in drugs),
                "source": local_src, "tags": [cls], "type": "choice"}))

        # 4) 是非题（从农业防治要点里取一条做判断）
        agri = tr.get("agri", [])
        if agri:
            right = agri[0]
            wrong_pick = TREATMENT_DB[others[0]].get("agri", [""])[0] if others else ""
            use_wrong = bool(wrong_pick) and (len(questions) % 2 == 0)
            questions.append(normalize_question({
                "question": "判断：%s" % (wrong_pick if use_wrong else right),
                "options": ["正确", "错误"],
                "answer": "错误" if use_wrong else "正确",
                "explain": ("这条其实是「%s」的措施，不是「%s」的。"
                            % (others[0], cls)) if use_wrong else
                           ("该措施来自本地防治库「%s」的农业防治条目。" % cls),
                "source": local_src, "tags": [cls], "type": "judge"}))

        # 案例卡：症状 / 发生条件 / 处置 / 结果
        cases.append({
            "id": hashlib.sha1(("case-" + cls).encode("utf-8")).hexdigest()[:12],
            "title": "%s：识别与处置要点" % cls,
            "symptom": tr.get("symptom", ""),
            "condition": tr.get("threshold", ""),
            "action": "；".join(tr.get("agri", []) + tr.get("bio", [])),
            "outcome": "推荐药剂：" + "、".join(
                "%s（%s，%s）" % (d.get("name", ""), d.get("spec", ""), d.get("dose", ""))
                for d in tr.get("drugs", [])),
            "source": local_src, "tags": [cls]})

    questions = [q for q in questions if q]
    return questions, cases


def merge_into_bank(new_questions, new_cases=None):
    """合并进题库/案例库（按题干/标题去重），返回新增数量"""
    bank = load_question_bank()
    have_q = {q.get("question", "").strip() for q in bank}
    added_q = 0
    for q in new_questions or []:
        if q and q.get("question", "").strip() not in have_q:
            bank.append(q)
            have_q.add(q.get("question", "").strip())
            added_q += 1
    if added_q:
        save_question_bank(bank)
    added_c = 0
    if new_cases:
        cases = load_case_library()
        have_c = {c.get("title", "").strip() for c in cases}
        for c in new_cases:
            if c and c.get("title", "").strip() not in have_c:
                cases.append(c)
                have_c.add(c.get("title", "").strip())
                added_c += 1
        if added_c:
            save_case_library(cases)
    return added_q, added_c


# ── 用大模型把原始资料结构化成题目/案例 ──────────────────────
LLM_QUESTION_PROMPT = (
    "下面是一份水稻病虫害防治资料。请把它整理成适合教学的题目。\n"
    "要求：\n"
    "1. 只依据原文，不要补充原文没有的内容；原文没写的不要编。\n"
    "2. 只输出 JSON：{\"questions\":[{\"question\":\"...\",\"options\":[\"...\"],"
    "\"answer\":\"...\",\"explain\":\"...\",\"tags\":[\"稻瘟病\"]}]}\n"
    "3. 选项 3-4 个，答案必须与其中一个选项完全一致；简答题 options 给空数组。\n"
    "4. 最多 %d 题；每题都在 explain 里写清楚依据原文的哪一句。\n"
    "5. 来源统一标注为：%s\n\n资料原文：\n%s")

LLM_CASE_PROMPT = (
    "下面是一份水稻病虫害防治资料。请整理成案例卡（用于教学阅读）。\n"
    "只依据原文，只输出 JSON：{\"cases\":[{\"title\":\"...\",\"symptom\":\"...\","
    "\"condition\":\"...\",\"action\":\"...\",\"outcome\":\"...\",\"tags\":[\"...\"]}]}\n"
    "最多 %d 条；来源统一标注为：%s\n\n资料原文：\n%s")


def structure_text_with_llm(text, source, want="questions", limit=12):
    """把资料交给模型结构化；返回 (条目列表, 错误)"""
    if not CONFIG.get("enable_llm", False) or not get_api_key():
        return [], "未启用云端 AI"
    body = (text or "")[:12000]
    if not body.strip():
        return [], "资料没有可用文本"
    if want == "cases":
        prompt = LLM_CASE_PROMPT % (limit, source, body)
    else:
        prompt = LLM_QUESTION_PROMPT % (limit, source, body)
    data, err = call_llm_json(prompt, max_tokens=3000, timeout=(10, 180))
    if data is None:
        return [], "结构化失败：%s" % err
    if want == "cases":
        items = parse_cases_json(json.dumps(data, ensure_ascii=False))
        for c in items:
            if not c.get("source") or c["source"] == SOURCE_HINT:
                c["source"] = source
        return items, ""
    raw = data.get("questions") if isinstance(data, dict) else data
    items = [q for q in (normalize_question(x) for x in (raw or [])) if q]
    for q in items:
        if not q.get("source") or q["source"] == SOURCE_HINT:
            q["source"] = source
        # 简单质检：选择题的答案必须落在选项里，否则降级为简答，避免出现"答案不在选项"的坏题
        if q["options"] and q["answer"] and q["answer"] not in q["options"]:
            q["options"] = []
            q["type"] = "short"
    return items[:limit], ""


def import_teaching_file(path=None, url=None, with_llm=False, limit=12):
    """统一导入入口：提取文本 → 先试直接解析 → 再按需调用模型结构化。"""
    if url:
        text, err = fetch_url_text(url)
        source = url
    else:
        text, err = extract_text_from_file(path)
        source = "本地资料：%s" % Path(path).name if path else "未知来源"
    if err:
        return {"ok": False, "error": err, "source": source}

    suffix = Path(path).suffix.lower() if path else ""
    questions, cases = [], []
    if suffix == ".json":
        questions = parse_questions_json(text)
        cases = parse_cases_json(text)
    elif suffix in (".csv", ".tsv"):
        questions = parse_questions_delimited(text, "\t" if suffix == ".tsv" else ",")
    elif suffix in (".md", ".txt", ".html", ".htm", ".docx", ".pdf") or url:
        # 碰巧符合 Markdown 约定就能直接读；但要过滤掉"整段散文被当成一道题"的伪题目
        # （既没选项也没答案的，交给模型结构化更靠谱）
        questions = [q for q in parse_questions_markdown(text)
                     if q.get("options") or q.get("answer")]

    from_llm = False
    if not questions and not cases and with_llm:
        questions, e1 = structure_text_with_llm(text, source, "questions", limit)
        cases, _ = structure_text_with_llm(text, source, "cases", max(3, limit // 3))
        from_llm = True
        if e1 and not questions and not cases:
            return {"ok": False, "error": e1, "source": source}

    added_q, added_c = merge_into_bank(questions, cases)
    record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": source,
              "chars": len(text or ""), "questions": added_q, "cases": added_c,
              "by_llm": from_llm}
    try:
        TEACHING_DIR.mkdir(parents=True, exist_ok=True)
        with open(IMPORT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("写导入记录失败: %s", e)
    log_event("teaching_import", **{k: v for k, v in record.items() if k != "time"})
    return {"ok": True, "questions": added_q, "cases": added_c, "source": source,
            "chars": len(text or ""), "by_llm": from_llm,
            "parsed": len(questions) + len(cases)}


class TeachingImportThread(QThread):
    """后台导入教学资料（提取文本 → 直接解析 → 必要时调模型结构化）"""
    item_done = pyqtSignal(dict)
    all_done = pyqtSignal(int)

    def __init__(self, jobs, with_llm=None, parent=None):
        super().__init__(parent)
        self.jobs = jobs or []
        self.with_llm = (CONFIG.get("enable_llm", False) and bool(get_api_key()))             if with_llm is None else bool(with_llm)

    def run(self):
        total = 0
        for job in self.jobs:
            try:
                res = import_teaching_file(path=job.get("path"), url=job.get("url"),
                                           with_llm=self.with_llm)
            except Exception as e:
                log.exception("导入教学资料失败")
                res = {"ok": False, "error": str(e), "source": job.get("path") or job.get("url")}
            total += int(res.get("questions", 0)) + int(res.get("cases", 0))
            self.item_done.emit(res)
        self.all_done.emit(total)


# ── 题目复核（人工把关，避免坏题进考核） ──────────────────────
def pending_questions():
    """还没人工复核过的题目"""
    return [q for q in load_question_bank() if not q.get("reviewed")]


def reviewed_questions():
    return [q for q in load_question_bank() if q.get("reviewed")]


def mark_question_reviewed(qid, reviewed=True):
    """复核结论：通过（reviewed=True）或打回（reviewed=False）"""
    bank = load_question_bank()
    hit = False
    for q in bank:
        if q.get("id") == qid:
            q["reviewed"] = bool(reviewed)
            q["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            hit = True
    if hit:
        save_question_bank(bank)
        log_event("question_reviewed", qid=qid, reviewed=bool(reviewed))
    return hit


def delete_question(qid):
    bank = load_question_bank()
    keep = [q for q in bank if q.get("id") != qid]
    if len(keep) == len(bank):
        return False
    save_question_bank(keep)
    log_event("question_deleted", qid=qid)
    return True


def quiz_questions(prefer_reviewed=True):
    """出题用的题池：优先只用已复核的题；一道都没复核过才退回全部（并在界面提示）"""
    bank = load_question_bank()
    rev = [q for q in bank if q.get("reviewed")]
    if prefer_reviewed and rev:
        return rev, True
    return bank, False


# ── 错题本与间隔重复（SM-2 简化版：答对就拉长间隔，答错回到 1 天） ──
REVIEW_PATH = TEACHING_DIR / "review_schedule.json"
REVIEW_INTERVALS = [1, 3, 7, 15, 30, 60]        # 单位：天
MASTERED_STREAK = 5                              # 连续答对 5 次算掌握


def load_review_schedule():
    try:
        if REVIEW_PATH.exists():
            data = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning("读取复习计划失败: %s", e)
    return {}


def save_review_schedule(data):
    try:
        REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except OSError as e:
        log.warning("保存复习计划失败: %s", e)
        return False


def record_answer_result(qid, correct, now=None):
    """记录一次作答：更新错题本与下次复习时间。返回该题的最新记录。"""
    now = now or datetime.now()
    data = load_review_schedule()
    rec = data.get(qid) or {"qid": qid, "wrong": 0, "streak": 0, "interval": 0,
                            "due": now.strftime("%Y-%m-%d"), "mastered": False}
    if correct:
        rec["streak"] = int(rec.get("streak", 0)) + 1
        # 答对一次就跳到 3 天（刚看过的东西没必要第二天再看），之后 7/15/30/60
        idx = min(max(0, int(rec["streak"])), len(REVIEW_INTERVALS) - 1)
        rec["interval"] = REVIEW_INTERVALS[idx]
        rec["mastered"] = rec["streak"] >= MASTERED_STREAK
    else:
        rec["wrong"] = int(rec.get("wrong", 0)) + 1
        rec["streak"] = 0
        rec["interval"] = REVIEW_INTERVALS[0]
        rec["mastered"] = False
    rec["due"] = (now + timedelta(days=int(rec["interval"]))).strftime("%Y-%m-%d")
    rec["last"] = now.strftime("%Y-%m-%d %H:%M")
    data[qid] = rec
    save_review_schedule(data)
    return rec


def due_review_items(now=None, include_mastered=False):
    """到期待复习的错题（默认不含已掌握的）"""
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    bank = {q.get("id"): q for q in load_question_bank()}
    out = []
    for qid, rec in load_review_schedule().items():
        if not include_mastered and rec.get("mastered"):
            continue
        if str(rec.get("due", "")) <= today:
            q = bank.get(qid)
            if not q:
                continue                     # 题目已被删掉
            out.append({"qid": qid, "question": q.get("question", ""),
                        "source": q.get("source", ""), "wrong": rec.get("wrong", 0),
                        "streak": rec.get("streak", 0), "due": rec.get("due", ""),
                        "mastered": bool(rec.get("mastered"))})
    out.sort(key=lambda x: (-int(x["wrong"]), x["due"]))
    return out


def review_stats(now=None):
    now = now or datetime.now()
    sched = load_review_schedule()
    due = due_review_items(now)
    wrong_total = sum(1 for r in sched.values() if int(r.get("wrong", 0)) > 0)
    mastered = sum(1 for r in sched.values() if r.get("mastered"))
    return {"有记录": len(sched), "错题数": wrong_total, "今日到期": len(due),
            "已掌握": mastered}


def clear_review_item(qid):
    data = load_review_schedule()
    if qid in data:
        data.pop(qid)
        save_review_schedule(data)
        return True
    return False


def clear_all_review():
    save_review_schedule({})
    log_event("review_cleared")


def teaching_stats():
    bank, cases = load_question_bank(), load_case_library()
    src_q = Counter(q.get("source", "未标注") for q in bank)
    imported = read_jsonl(IMPORT_LOG)
    return {"题库": len(bank), "案例": len(cases),
            "待复核": sum(1 for q in bank if not q.get("reviewed")),
            "已复核": sum(1 for q in bank if q.get("reviewed")),
            "来源数": len(src_q), "主要来源": dict(src_q.most_common(3)),
            "导入次数": len(imported), "错题本": review_stats()}


def search_cases(keyword):
    kw = (keyword or "").strip()
    cases = load_case_library()
    if not kw:
        return cases
    return [c for c in cases
            if kw in " ".join(str(c.get(k, "")) for k in
                              ("title", "symptom", "condition", "action", "outcome", "source"))]


# ══════════════════════════════════════════════════════════════
# 地块地图页：看点位 + 让 Agent 分析这片区域
# --------------------------------------------------------------
# 地图控件直接复用 rice_geo.MapCanvas（高德瓦片，不需要 Key，不需要 QtWebEngine）；
# 模块缺失时页面降级为提示文案，不影响主程序其它功能。
# ══════════════════════════════════════════════════════════════
MAP_AGENT_PROMPT = (
    "请分析我这几块稻田的情况，用中文给出结论，控制在 250 字以内。要求：\n"
    "1. 先说哪块地风险最高、主要问题是什么（用工具查，不要凭空猜）；\n"
    "2. 给出巡田建议顺序与每一站要看什么；\n"
    "3. 指出需要尽快处理的地块和理由；\n"
    "4. 如果记录太少不足以判断，就直说还需要补哪些数据。\n"
    "可用工具：map_digest（区域简报）、plot_risk（地块风险排序）、patrol_plan（巡田路线）、"
    "map_hotspots（密集区域）、nearby_records（附近历史）、map_overview（数据总览）、"
    "get_treatment（防治方案）、check_pesticide（农药合规）、get_weather（天气）。")


# ══════════════════════════════════════════════════════════════
# 网页版地图：QWebEngineView + Leaflet + 高德瓦片
# --------------------------------------------------------------
# 两个坑决定了这里的写法：
# 1) QtWebEngine 需要 GL 上下文，offscreen 平台会直接崩（0xC0000005），
#    所以网页地图必须「用到才创建」，并且离屏自检里一律不创建；
# 2) Leaflet 抓到 vendor/leaflet 本地加载，断网也能出图（只有瓦片要联网，
#    WebEngine 自带的磁盘缓存会自动复用看过的区域）。
# ══════════════════════════════════════════════════════════════
LEAFLET_DIR = BASE_DIR / "vendor" / "leaflet"
_WEBENGINE_STATE = {"checked": False, "ok": False}


def webengine_available():
    """QtWebEngine 能不能用：装了库 + 不是离屏平台（离屏没有 GL，创建视图会直接崩）

    注意：平台名要等 QApplication 起来才读得到。还没创建 QApplication 时先判为不可用，
    并且**不缓存**这个结论——否则单元自检阶段（还没有 QApplication）会把结果固化错。
    """
    if _WEBENGINE_STATE["checked"]:
        return _WEBENGINE_STATE["ok"]
    ok = False
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
    except Exception:
        app = None
    platform = (app.platformName() if app is not None
                else os.environ.get("QT_QPA_PLATFORM", "")).lower()
    if app is None and not platform:
        return False                      # 判断不了，先按不可用且不缓存
    try:
        if os.environ.get("RICE_NO_WEBENGINE"):
            raise RuntimeError("已通过 RICE_NO_WEBENGINE 禁用")
        if platform in ("offscreen", "minimal", "minimalegl"):
            raise RuntimeError("离屏平台 %s 没有 GL 上下文，网页地图不可用" % platform)
        import PyQt5.QtWebEngineWidgets  # noqa: F401
        ok = (LEAFLET_DIR / "leaflet.js").exists()
        if not ok:
            log.warning("网页地图缺少 vendor/leaflet/leaflet.js")
    except Exception as e:
        log.info("网页地图不可用：%s", e)
    _WEBENGINE_STATE["checked"] = True
    _WEBENGINE_STATE["ok"] = ok
    return ok


MAP_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>稻视界 · 地块地图</title>
<link rel="stylesheet" href="__LEAFLET__/leaflet.css">
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script src="__LEAFLET__/leaflet.js"></script>
<script src="__LEAFLET__/leaflet-heat.js"></script>
<style>
  html,body,#map{height:100%;margin:0;background:#eef2ef}
  .leaflet-popup-content{font:12px/1.6 "Microsoft YaHei",sans-serif;margin:8px 10px}
  .legend{position:absolute;right:10px;bottom:18px;z-index:999;background:rgba(255,255,255,.92);
          border-radius:6px;padding:8px 10px;font:12px/1.7 "Microsoft YaHei",sans-serif;
          box-shadow:0 1px 6px rgba(0,0,0,.2)}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
  #hint{position:absolute;left:10px;top:10px;z-index:999;background:rgba(255,255,255,.92);
        border-radius:6px;padding:6px 10px;font:12px/1.5 "Microsoft YaHei",sans-serif;
        box-shadow:0 1px 6px rgba(0,0,0,.2)}
</style></head>
<body>
<div id="map"></div>
<div id="hint">点地图任意位置可把「待落点」记录落到这里</div>
<div class="legend"><b>风险</b><br>
  <i style="background:#c62828"></i>高（≥70）<br>
  <i style="background:#ef6c00"></i>中（40-69）<br>
  <i style="background:#f9a825"></i>低（15-39）<br>
  <i style="background:#2e7d32"></i>轻微（&lt;15）
  <div id="sourceLegend"><b>点色</b>按风险 · <b>边框</b>按来源<br>
  <i style="background:#007A3F"></i>手机直传<br>
  <i style="background:#2C6E8F"></i>小程序<br>
  <i style="background:#6E7873"></i>电脑本地</div>
  <div id="sprayLegend" style="display:none"><b>施药窗口</b><br>
  <i style="background:#2e7d32"></i>今天可施药<br>
  <i style="background:#f9a825"></i>窗口很窄<br>
  <i style="background:#c62828"></i>今天不宜<br>
  <i style="background:#607d8b"></i>天气未知</div></div>
<script>
var bridge = null, map, road, satellite, layers = {markers: [], route: null, hotspots: [], plots: []};
var lastPoints = [], heatLayer = null, heatOn = false;
var lastPayload = null, colorBy = 'risk';   // 'risk'=按风险着色 'source'=按来源着色
function initMap(){
  map = L.map('map', {zoomControl: true, attributionControl: true}).setView([32.0510, 118.8620], 12);
  road = L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                     {subdomains: '1234', maxZoom: 18, attribution: '底图 © 高德地图'});
  satellite = L.tileLayer('https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
                          {subdomains: '1234', maxZoom: 18, attribution: '底图 © 高德地图'});
  road.addTo(map);
  L.control.layers({'路网': road, '卫星': satellite}, {}, {position: 'topleft'}).addTo(map);
  map.on('click', function(e){
    if (bridge) { bridge.pick(e.latlng.lng, e.latlng.lat); }
  });
}
function riskColor(score){
  if (score === null || score === undefined) return '#607d8b';
  if (score >= 70) return '#c62828';
  if (score >= 40) return '#ef6c00';
  if (score >= 15) return '#f9a825';
  return '#2e7d32';
}
function clearLayers(){
  layers.markers.forEach(function(m){ map.removeLayer(m); }); layers.markers = [];
  if (layers.route) { map.removeLayer(layers.route); layers.route = null; }
  layers.hotspots.forEach(function(c){ map.removeLayer(c); }); layers.hotspots = [];
  layers.plots.forEach(function(c){ map.removeLayer(c); }); layers.plots = [];
}
function sourceColor(s){
  if (s === 'phone') return '#007A3F';      // 手机直传（局域网）
  if (s === 'mini')  return '#2C6E8F';      // 小程序（公网中转）
  if (s === 'pc')    return '#6E7873';      // 电脑本地
  return '#9E9E9E';                          // 未标注来源
}
function sourceLabel(s){
  if (s === 'phone') return '手机直传';
  if (s === 'mini')  return '小程序';
  if (s === 'pc')    return '电脑本地';
  return '未标注';
}
// 两种着色模式：填充和边框各承担一层信息，所以怎么切都不会丢信息
function setColorBy(mode){
  colorBy = (mode === 'source') ? 'source' : 'risk';
  if (lastPayload) { renderMap(lastPayload); }
}
function sprayColor(s){
  if (s === 'ok') return '#2e7d32';
  if (s === 'narrow') return '#f9a825';
  if (s === 'no') return '#c62828';
  return '#607d8b';
}
function fmtNum(v, unit){
  return (v === null || v === undefined) ? '—' : (Math.round(v * 10) / 10) + unit;
}
function drawHeat(){
  if (heatLayer) { map.removeLayer(heatLayer); heatLayer = null; }
  // leaflet-heat 是可选插件：没加载就安静跳过，绝不影响点位渲染
  if (!heatOn || typeof L.heatLayer !== 'function' || !lastPoints.length) { return; }
  heatLayer = L.heatLayer(lastPoints.map(function(p){
    var w = (p.risk === null || p.risk === undefined) ? 0.45
            : Math.max(0.2, Math.min(1, p.risk / 100));
    return [p.lat, p.lon, w];
  }), {radius: 32, blur: 24, maxZoom: 17, minOpacity: 0.25}).addTo(map);
}
function setHeat(on){ heatOn = !!on; if (map) { drawHeat(); } }
function refreshSize(){
  // 容器可能是在隐藏状态下初始化的（QStackedWidget 里先建后显示），
  // 尺寸为 0 时 fitBounds 会算错，显示出来必须重新量一次
  if (map) { map.invalidateSize(); }
}
function renderMap(payload){
  if (!map) { initMap(); }
  lastPayload = payload;
  refreshSize();
  clearLayers();
  // 地块级施药窗口徽章：比点位大一圈，画在点位下面（点位要能盖住它）
  ((payload && payload.plots) || []).forEach(function(p){
    var color = sprayColor(p.status);
    var mk = L.circleMarker([p.lat, p.lon], {radius: 13, color: '#fff', weight: 3,
                                             fillColor: color, fillOpacity: 0.9});
    var html = '<b>' + (p.plot || '未命名地块') + '</b><br>'
             + '<span style="color:' + color + ';font-weight:bold">' + (p.label || '') + '</span>';
    if (p.reason) { html += '<br>主要原因：' + p.reason; }
    if (p.now) {
      html += '<br>当前：' + fmtNum(p.now['温度'], '℃') + ' · 风 ' + fmtNum(p.now['风速'], 'm/s')
            + ' · 雨 ' + fmtNum(p.now['降水'], 'mm');
    }
    if (p.window) {
      html += '<br>窗口 ' + String(p.window['开始']).slice(11, 16) + '–'
            + String(p.window['结束']).slice(11, 16)
            + '（' + p.window['时长小时'] + ' 小时 · 均分 ' + p.window['平均评分'] + '）';
      if (p.window['建议']) { html += '<br>' + p.window['建议']; }
    }
    if (p.window && p.window['最大风速'] !== undefined) {
      html += '<br><span style="color:#666">窗口内最大风速 ' + p.window['最大风速']
            + ' m/s · 最大降雨概率 ' + p.window['最大降雨概率'] + '%</span>';
    }
    mk.bindPopup(html);
    mk.bindTooltip((p.plot || '未命名地块') + ' · ' + (p.label || ''),
                   {direction: 'top', offset: [0, -8]});
    mk.addTo(map); layers.plots.push(mk);
  });
  // 没查天气就没有徽章，图例也跟着收起来（否则图例上的颜色在地图上找不到对应）
  var plotRows = (payload && payload.plots) || [];
  var _lg = document.getElementById('sprayLegend');
  if (_lg) { _lg.style.display = plotRows.length ? 'block' : 'none'; }
  var pts = (payload && payload.points) || [];
  pts.forEach(function(p){
    var riskC = riskColor(p.risk);
    var srcC = sourceColor(p.source);
    // 按来源着色时：填充=来源、边框=风险；否则填充=风险、边框=来源
    var fill = (colorBy === 'source') ? srcC : riskC;
    var edge = (colorBy === 'source') ? riskC : srcC;
    var mk = L.circleMarker([p.lat, p.lon], {radius: 8, color: edge, weight: 3,
                                             fillColor: fill, fillOpacity: 0.95});
    var html = '<b>' + (p.plot || '未命名地块') + '</b><br>' + (p.time || '');
    html += '<br>来源：<span style="color:' + srcC + ';font-weight:bold">'
          + sourceLabel(p.source) + '</span>';
    if (p.cls) { html += '<br>类别：' + p.cls; }
    if (p.risk !== null && p.risk !== undefined) { html += '<br>风险分：' + p.risk; }
    mk.bindPopup(html);
    mk.on('click', function(){ if (bridge) { bridge.select(p.plot || ''); } });
    mk.addTo(map); layers.markers.push(mk);
  });
  if (payload && payload.route && payload.route.length > 1) {
    layers.route = L.polyline(payload.route.map(function(s){ return [s.lat, s.lon]; }),
                              {color: '#1565c0', weight: 3, dashArray: '7 6'}).addTo(map);
    payload.route.forEach(function(s){
      var lbl = L.divIcon({className: '', html: '<div style="background:#1565c0;color:#fff;'
        + 'border-radius:50%;width:20px;height:20px;text-align:center;line-height:20px;'
        + 'font:12px sans-serif">' + s.order + '</div>', iconSize: [20, 20]});
      var mk = L.marker([s.lat, s.lon], {icon: lbl}).addTo(map);
      mk.bindPopup('<b>第 ' + s.order + ' 站</b><br>' + (s.plot || '') + '<br>风险分 ' + (s.risk || ''));
      layers.markers.push(mk);
    });
  }
  ((payload && payload.hotspots) || []).forEach(function(h){
    var c = L.circle([h.lat, h.lon], {radius: h.radius || 400, color: '#c62828',
                                      weight: 1, fillColor: '#c62828', fillOpacity: 0.10});
    c.bindPopup('密集区域：' + h.count + ' 个点位<br>主要类别：' + (h.cls || ''));
    c.addTo(map); layers.hotspots.push(c);
  });
  lastPoints = pts;
  drawHeat();
  var all = pts.map(function(p){ return [p.lat, p.lon]; });
  ((payload && payload.route) || []).forEach(function(s){ all.push([s.lat, s.lon]); });
  ((payload && payload.plots) || []).forEach(function(p){ all.push([p.lat, p.lon]); });
  if (all.length) {
    refreshSize();
    map.fitBounds(L.latLngBounds(all), {padding: [36, 36], maxZoom: 17});
  }
  return pts.length;
}
function focusPlot(lon, lat, zoom){ if (map) { map.setView([lat, lon], zoom || 16); } }
initMap();
new QWebChannel(qt.webChannelTransport, function(channel){
  bridge = channel.objects.bridge;
  if (bridge && bridge.ready) { bridge.ready(); }
});
</script></body></html>
"""


def map_html_text(leaflet_base=None):
    """把 Leaflet 的路径填进去。

    默认用 file:// 本地路径（页面本身也是 file:// 打开的，离线可用）；
    自检/预览时可以用相对路径，方便在普通浏览器里验证。
    """
    if leaflet_base is None:
        leaflet_base = LEAFLET_DIR.as_uri() if LEAFLET_DIR.exists() else ""
    return MAP_HTML.replace("__LEAFLET__", leaflet_base)


class MapBridge(QObject):
    """网页地图 → Python 的回调（点击落点、点选地块）"""
    picked = pyqtSignal(float, float)      # GCJ-02 经纬度
    plotSelected = pyqtSignal(str)

    @pyqtSlot(float, float)
    def pick(self, lon, lat):
        self.picked.emit(float(lon), float(lat))

    @pyqtSlot(str)
    def select(self, plot):
        self.plotSelected.emit(str(plot))

    @pyqtSlot()
    def ready(self):
        log.info("网页地图已就绪（Leaflet + 高德瓦片）")


def wgs84_to_gcj02_safe(lon, lat):
    """WGS-84 → GCJ-02（高德底图用的火星坐标）。

    工具返回的一律是 WGS-84（数据真相，也便于算距离）；只有往地图上画的时候才转，
    否则点位会在高德底图上偏移 300~600 米——这个坑记录在 docs/地图溯源接入说明.md 里。
    """
    try:
        import rice_geo
        return rice_geo.wgs84_to_gcj02(float(lon), float(lat))
    except Exception as e:
        log.debug("坐标转换失败（按原值使用）：%s", e)
        return float(lon), float(lat)


def map_web_payload(days=None, cls_filter=None, plots=None):
    """给网页地图的数据：点位（按地块带风险色）+ 巡田路线 + 密集区域（全部转成 GCJ-02）

    days / cls_filter 只决定「画哪些」，不改变载荷的三个键——
    checks/fluent_smoke.py 断言载荷恒为 points/route/hotspots；
    热力层走单独的 setHeat() 开关，不往载荷里塞第四个键。
    """
    st = geo_store()
    if st is None:
        return {"points": [], "route": [], "hotspots": []}
    risk = {}
    for r in (plot_risk_rows(days) or []):
        risk[r["plot"]] = r["风险分"]
    risk.setdefault("未命名地块", None)
    points = []
    # to_points 支持传入记录：时间筛选才能真正落到点位上，而不只是影响风险分
    for p in st.to_points(_geo_records(days)):
        cls = p.get("cls") or ""
        if cls_filter and cls != cls_filter:
            continue
        plot = p.get("plot") or "未命名地块"
        points.append({"lon": float(p["lon"]), "lat": float(p["lat"]),
                       "plot": plot, "cls": cls,
                       "time": p.get("time") or "", "risk": risk.get(plot),
                       "source": p.get("source") or ""})
    route = []
    plan = patrol_plan_data(max_stops=6, days=days)
    for i, stop in enumerate(plan.get("顺序") or []):
        glon, glat = wgs84_to_gcj02_safe(stop["坐标"][0], stop["坐标"][1])
        route.append({"order": i + 1, "plot": stop["地块"], "risk": stop["风险分"],
                      "lon": glon, "lat": glat})
    hotspots = []
    hs = map_hotspots_data(days=days or 60, top=3)
    for h in hs.get("热点") or []:
        # 选了具体类别时，别的病虫害的密集区不该还画在上面
        if cls_filter and (h.get("主要类别") or "") != cls_filter:
            continue
        glon, glat = wgs84_to_gcj02_safe(h["中心"][0], h["中心"][1])
        hotspots.append({"lon": glon, "lat": glat,
                         "radius": h.get("点数", 0) and max(300, 120 * h["点数"]) or 300,
                         "count": h["点数"], "cls": h["主要类别"]})
    return {"points": points, "route": route, "hotspots": hotspots,
            "plots": list(plots or [])}


class MapWebWindow(QWidget):
    """交互地图独立窗口（Leaflet + 高德瓦片）。

    为什么不做成页内切换：QtWebEngine 一创建就会改掉整个窗口的合成路径，而主窗口的
    导航栏靠透明背景（WA_TranslucentBackground）做亚克力效果 —— 两者冲突会把导航栏
    渲染成黑色，而且 GL 上下文不释放，切回轻量地图也恢复不了。
    放进独立顶层窗口，主界面外观就完全不受影响。
    """

    def __init__(self, owner):
        super().__init__(owner.window(), Qt.Window)
        self.owner = owner
        self.setObjectName("mapWebWindow")
        self.setWindowTitle("稻视界 · 交互地图")
        self.resize(1200, 780)
        self.web_view = None
        self.bridge = None
        self._loaded = False           # 页面加载完才允许推送数据

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(BodyLabel("筛选", self))
        self.cmb_days = ComboBox(self)
        for _label, _val in (("全部时间", None), ("最近 7 天", 7),
                             ("最近 30 天", 30), ("最近 90 天", 90)):
            self.cmb_days.addItem(_label, userData=_val)
        self.cmb_days.setCurrentIndex(max(0, owner.cmb_days.currentIndex()))
        bar.addWidget(self.cmb_days)
        self.cmb_cls = ComboBox(self)
        self.cmb_cls.addItem("全部类别", userData=None)
        for _name in CLASS_NAMES:
            self.cmb_cls.addItem(_name, userData=_name)
        self.cmb_cls.setCurrentIndex(max(0, owner.cmb_cls.currentIndex()))
        bar.addWidget(self.cmb_cls)
        self.chk_heat = CheckBox("热力图", self)
        self.chk_heat.setChecked(owner.chk_heat.isChecked())
        self.chk_heat.setToolTip("叠一层热力，快速看出病虫害集中在哪片")
        bar.addWidget(self.chk_heat)
        self.chk_by_source = CheckBox("按来源着色", self)
        self.chk_by_source.setToolTip(
            "按「手机直传 / 小程序 / 电脑本地」给点上色，一眼看出哪些是手机传的\n"
            "（关掉则按风险着色；两种模式下填充和边框各承担一层信息）")
        self.chk_by_source.stateChanged.connect(lambda *_: self.push())
        bar.addWidget(self.chk_by_source)
        bar.addStretch()
        btn_refresh = PushButton(FIF.SYNC, "刷新", self)
        btn_refresh.clicked.connect(self.push)
        bar.addWidget(btn_refresh)
        root.addLayout(bar)

        self.host = QWidget(self)
        self.host_layout = QVBoxLayout(self.host)
        self.host_layout.setContentsMargins(0, 0, 0, 0)
        # 视图是延后创建的，先放一句提示，别让窗口空着像卡死
        self._loading_lbl = BodyLabel("正在加载交互地图…", self.host)
        self._loading_lbl.setAlignment(Qt.AlignCenter)
        self.host_layout.addWidget(self._loading_lbl)
        root.addWidget(self.host, 1)
        root.addWidget(CaptionLabel(
            "滚轮缩放 / 拖动平移；点地图任意位置可给「待落点」记录落位", self))

        self.cmb_days.currentIndexChanged.connect(lambda *_: self.push())
        self.cmb_cls.currentIndexChanged.connect(lambda *_: self.push())
        self.chk_heat.stateChanged.connect(lambda *_: self.push())
        # 这里**不要**建视图：WebEngine 首次初始化会阻塞一两秒，
        # 构造时同步建 = 点一下界面就冻住。交给 ensure_view()，窗口画出来之后再建。

    def ensure_view(self):
        """真正创建 WebEngine 视图（延后调用）。

        首次创建要初始化 QtWebEngine：加载一堆 DLL、拉起 QtWebEngineProcess 子进程，
        会阻塞一两秒。放在窗口显示之后做 —— 点一下先出窗口，不至于让人以为程序卡死。
        """
        if self.web_view is not None:
            return
        try:
            self._create_view()
        except Exception as e:
            log.exception("创建交互地图视图失败")
            InfoBar.error("交互地图打开失败", str(e)[:150], duration=8000,
                          position=InfoBarPosition.TOP, parent=self)

    def _create_view(self):
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        from PyQt5.QtWebEngineWidgets import QWebEngineSettings
        from PyQt5.QtWebChannel import QWebChannel
        view = QWebEngineView(self.host)
        page = view.page()
        # 关键：页面是 file:// 打开的，QtWebEngine 默认禁止本地页面访问远程 URL，
        # 不打开这个开关，高德瓦片一张都加载不出来（点位能画，底图全白）。
        try:
            st = page.settings()
            st.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            st.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        except Exception as e:
            log.warning("设置 WebEngine 本地访问权限失败：%s", e)
        self.bridge = MapBridge(self)
        self.bridge.picked.connect(self.owner._on_web_pick)
        self.bridge.plotSelected.connect(self.owner._on_plot_selected)
        channel = QWebChannel(page)
        channel.registerObject("bridge", self.bridge)
        page.setWebChannel(channel)
        (APP_DIR / "maps").mkdir(parents=True, exist_ok=True)
        html_path = APP_DIR / "maps" / "map.html"
        html_path.write_text(map_html_text(), encoding="utf-8")
        # 加载完再推数据：load 之前 runJavaScript 打在空白页上，renderMap 还没定义，
        # 结果就是窗口开着但地图全空 —— 看着像卡住了
        view.loadFinished.connect(self._on_load_finished)
        view.load(QUrl.fromLocalFile(str(html_path)))
        if self._loading_lbl is not None:
            self.host_layout.removeWidget(self._loading_lbl)
            self._loading_lbl.deleteLater()
            self._loading_lbl = None
        self.host_layout.addWidget(view)
        self.web_view = view
        log_event("map_web_window", html=str(html_path))
        log.info("交互地图窗口已创建：%s", html_path)

    def _on_load_finished(self, ok):
        if not ok:
            log.warning("交互地图页面加载失败：%s", APP_DIR / "maps" / "map.html")
            InfoBar.error("交互地图加载失败", "页面没能加载，请重开一次",
                          duration=6000, position=InfoBarPosition.TOP, parent=self)
            return
        self._loaded = True
        if self._loading_lbl is not None:
            self._loading_lbl.hide()
        self.push()

    def push(self):
        """把当前筛选结果推给网页渲染"""
        if self.web_view is None or not self._loaded:
            return                      # 还没加载完，_on_load_finished 里会补推
        try:
            payload = map_web_payload(self.cmb_days.currentData(), self.cmb_cls.currentData(),
                                      plots=self.owner.spray_payload())
            if not self.owner.chk_route.isChecked():
                payload["route"] = []
            js = "colorBy=%s; renderMap(%s); setHeat(%s);" % (
                json.dumps("source" if self.chk_by_source.isChecked() else "risk"),
                json.dumps(payload, ensure_ascii=False),
                "true" if self.chk_heat.isChecked() else "false")
            self.web_view.page().runJavaScript(
                js, lambda n: log.debug("交互地图渲染 %s 个点位", n))
        except Exception as e:
            log.warning("推送地图数据失败：%s", e)

    def closeEvent(self, e):
        # 关窗就把 WebEngine 视图销毁：别让它的 GL 上下文一直挂着
        try:
            if self.web_view is not None:
                self.web_view.setParent(None)
                self.web_view.deleteLater()
                self.web_view = None
        except Exception:
            pass
        self.owner._web_window = None
        super().closeEvent(e)


def _parse_one_coord(part):
    """解析单个坐标分量（十进制或度分秒），返回 (值, 方向字母或 None)"""
    part = part.strip().replace("　", " ")
    d = None
    m = re.search(r"([NSEWnsew南北东西])\s*$", part)
    if m:
        d = m.group(1).upper()
        part = part[:m.start()]
    m = re.match(r"^\s*([NSEWnsew南北东西])\s*(.+)$", part)
    if m:
        d = d or m.group(1).upper()
        part = m.group(2)
    nums = re.findall(r"\d+(?:\.\d+)?", part)
    if not nums:
        raise ValueError("认不出这个坐标：%s" % part.strip())
    val = float(nums[0])
    if len(nums) >= 2:
        val += float(nums[1]) / 60.0
    if len(nums) >= 3:
        val += float(nums[2]) / 3600.0
    if part.strip().startswith("-") or val < 0:
        val = -abs(val)
    _dir = {"北": "N", "南": "S", "东": "E", "西": "W"}.get(d, d)
    return (val if _dir not in ("S", "W") else -abs(val)), _dir


def parse_coord_text(text):
    """把用户粘贴的坐标文本解析成经纬度。

    支持（都是田间实际会遇到的写法）：
        32°0'38"N 118°37'48"E      手机水印（度分秒 + 方向）
        32°0′38″N, 118°37′48″E     全角/弯引号
        32度0分38秒N 118度37分48秒E
        32.010556, 118.63          十进制（纬,经）
        118.63, 32.010556          十进制（经,纬）——按 |值|>90 判断
        N32.010556 E118.63         方向在前

    返回 {"lon":, "lat":, "fmt": "dms"/"decimal", "order": "latlon"/"lonlat"}；
    认不出就抛 ValueError，消息里说明支持的写法。
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("没有输入坐标")
    # 统一全角与各种引号写法
    for a, b in (("，", ","), ("；", ";"), ("：", ":"), ("　", " "), ("度", "°"),
                 ("分", "'"), ("秒", '"'), ("′", "'"), ("’", "'"), ("″", '"'),
                 ("”", '"'), ("“", '"')):
        s = s.replace(a, b)

    # ① 带方向字母：分开找两组（方向可能在数字前或后）
    pat = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°\s*(\d+(?:\.\d+)?)\s*(?:'\s*(\d+(?:\.\d+)?)\s*\"?)?)?"
                     r"\s*([NSEWnsew])")
    hits = []
    for m in pat.finditer(s):
        deg = float(m.group(1))
        if m.group(2):
            deg += float(m.group(2)) / 60.0
        if m.group(3):
            deg += float(m.group(3)) / 3600.0
        d = m.group(4).upper()
        hits.append((deg if d in ("N", "E") else -deg, d))
    if len(hits) >= 2:
        lat = lon = None
        for val, d in hits:
            if d in ("N", "S"):
                lat = val
            else:
                lon = val
        if lat is None or lon is None:
            raise ValueError("方向字母不完整：需要一组成员里既有 N/S 又有 E/W")
        fmt = "dms" if ("°" in s or "'" in s or '"' in s) else "decimal"
        return {"lon": lon, "lat": lat, "fmt": fmt,
                "order": "latlon" if s.index(str(abs(lat)).split(".")[0]) < s.index(
                    str(abs(lon)).split(".")[0]) else "lonlat"}

    # ② 没有方向字母：按分隔符切成两段
    parts = [p for p in re.split(r"[,;\s]+", s) if p.strip()]
    if len(parts) != 2:
        raise ValueError("需要两个数字（经度和纬度）。支持：32°0'38\"N 118°37'48\"E "
                         "或 32.010556, 118.63")
    vals = [_parse_one_coord(p)[0] for p in parts]
    first, second = vals
    # 经度范围 ±180、纬度 ±90：|值|>90 的那个一定是经度
    if abs(first) > 90 and abs(second) <= 90:
        lon, lat, order = first, second, "lonlat"
    elif abs(second) > 90 and abs(first) <= 90:
        lon, lat, order = second, first, "latlon"
    else:
        # 两个都在 ±90 内，无法判断——按国内地图 App 复制出来的「经度,纬度」处理
        lon, lat, order = first, second, "lonlat"
    fmt = "dms" if ("°" in s or "'" in s or '"' in s) else "decimal"
    return {"lon": lon, "lat": lat, "fmt": fmt, "order": order}


class ManualPinDialog(MessageBoxBase):
    """粘贴坐标落点：水印/地图 App 复制的坐标直接填进来，不用在地图上找"""

    def __init__(self, geo, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.geo = geo
        self.viewLayout.addWidget(SubtitleLabel("粘贴坐标落点", self))
        tip = BodyLabel("手机水印、高德/腾讯地图「复制坐标」得到的都能直接粘进来，"
                        "支持度分秒和十进制两种写法：\n"
                        "32°0'38\"N 118°37'48\"E ／ 32.010556, 118.63 ／ 118.63, 32.010556", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.text = LineEdit(self)
        self.text.setPlaceholderText("在这里粘贴坐标，例如 32°0'38\"N 118°37'48\"E")
        self.text.setMinimumHeight(38)
        self.text.textChanged.connect(self.preview)
        self.viewLayout.addWidget(self.text)
        row = QHBoxLayout()
        row.addWidget(BodyLabel("坐标系：", self))
        self.sys = ComboBox(self)
        self.sys.addItems(["GCJ-02（手机水印 / 高德、腾讯地图复制）",
                           "WGS-84（GPS 设备 / 照片 EXIF）"])
        self.sys.currentIndexChanged.connect(self.preview)
        row.addWidget(self.sys, 1)
        self.viewLayout.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(BodyLabel("落到哪：", self))
        self.target = ComboBox(self)
        self.target.addItem("不关联照片，只落一个点")
        self.recs = [r for r in self.geo.all_records() if r.get("lon") is None][-40:]
        for r in self.recs:
            self.target.addItem("%s  %s" % (r.get("shot_at"), r.get("cls") or "未识别"))
        row2.addWidget(self.target, 1)
        self.viewLayout.addLayout(row2)
        self.note = LineEdit(self)
        self.note.setPlaceholderText("备注（可选），例如：东片1号田 水印坐标")
        self.viewLayout.addWidget(self.note)
        self.prev = CaptionLabel("", self)
        self.prev.setWordWrap(True)
        self.viewLayout.addWidget(self.prev)
        self.btn_ok = PrimaryPushButton(FIF.PIN, "落点", self)
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.apply)
        self.viewLayout.addWidget(self.btn_ok)
        self.yesButton.setText("关闭")
        self.cancelButton.hide()
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(560)

    def _parsed(self):
        """返回 (lon_wgs, lat_wgs, 说明)；解析失败抛 ValueError"""
        r = parse_coord_text(self.text.text())
        lon, lat = r["lon"], r["lat"]
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("超出经纬度范围：经度 %.4f 纬度 %.4f" % (lon, lat))
        if self.sys.currentIndex() == 0:          # GCJ-02 → WGS-84
            import rice_geo                       # 与程序其它地方一致：函数内局部导入
            lon, lat = rice_geo.from_gcj02(lon, lat, "wgs84")
            extra = "（水印/地图坐标 GCJ-02 → 已换算成 WGS-84 存档）"
        else:
            extra = "（按 WGS-84 存档）"
        return lon, lat, "经度 %.6f，纬度 %.6f %s" % (lon, lat, extra)

    def preview(self):
        try:
            lon, lat, desc = self._parsed()
            self.prev.setText("✅ " + desc)
            self.btn_ok.setEnabled(True)
        except ValueError as e:
            self.prev.setText("⚠ " + str(e))
            self.btn_ok.setEnabled(False)

    def apply(self):
        try:
            lon, lat, _ = self._parsed()
        except ValueError as e:
            InfoBar.warning("坐标有问题", str(e), duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        idx = self.target.currentIndex()
        try:
            if idx > 0:
                rec = self.recs[idx - 1]
                self.geo.set_location(rec["id"], lon, lat, coord_src="manual")
                msg = "已给记录 %s 补上坐标" % rec.get("shot_at")
            else:
                # 注意 add_record 返回的是记录 id（int），不是记录字典
                rid = self.geo.add_record(shot_at=datetime.now(), lon=lon, lat=lat,
                                          coord_src="manual", note=self.note.text().strip())
                msg = "已新增落点（记录 #%s）" % rid
            log_event("manual_pin", lon=lon, lat=lat, note=self.note.text().strip())
            InfoBar.success("落点成功", msg, duration=3500,
                            position=InfoBarPosition.TOP, parent=self)
            self.done(1)
        except Exception as e:
            log.exception("落点失败")
            InfoBar.error("落点失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)


class GeoManageDialog(MessageBoxBase):
    """记录管理：逐条看识别结果与来源（📱手机 / 💻电脑），可随时删除"""

    def __init__(self, geo, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.geo = geo
        self.viewLayout.addWidget(SubtitleLabel("地块记录管理", self))
        tip = BodyLabel("📱 = 手机传上来后识别的，💻 = 电脑本地识别的。"
                        "选中不需要的记录点「删除选中」即可（会同时删掉档案里的照片副本）。", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(tip)
        self.stat = CaptionLabel("", self)
        self.viewLayout.addWidget(self.stat)
        self.lst = QListWidget(self)
        self.lst.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst.setMinimumHeight(320)
        self.viewLayout.addWidget(self.lst)
        row = QHBoxLayout()
        btn_r = PushButton(FIF.SYNC, "刷新", self)
        btn_r.clicked.connect(self.reload)
        row.addWidget(btn_r)
        self.btn_pin = PushButton(FIF.PIN, "补坐标", self)
        self.btn_pin.setToolTip("给选中的「待落点」记录补上坐标（粘水印坐标或地图复制的都行）")
        self.btn_pin.clicked.connect(self.pin_selected)
        row.addWidget(self.btn_pin)
        self.btn_del = PushButton(FIF.DELETE, "删除选中", self)
        self.btn_del.clicked.connect(self.delete_selected)
        row.addWidget(self.btn_del)
        row.addStretch()
        self.viewLayout.addLayout(row)
        self.yesButton.setText("关闭")
        self.cancelButton.hide()
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(640)
        self.reload()

    def reload(self):
        self.lst.clear()
        rows = list(reversed(self.geo.all_records() or []))
        n_phone = sum(1 for r in rows if (r.get("source") or "") == "phone")
        no_loc = sum(1 for r in rows if r.get("lon") is None)
        self.stat.setText("共 %d 条（📱 手机 %d 条）· 待落点 %d 条" % (len(rows), n_phone, no_loc))
        for r in rows:
            src = "📱 手机" if (r.get("source") or "") == "phone" else "💻 电脑"
            loc = ("%.5f, %.5f" % (r.get("lon"), r.get("lat"))) if r.get("lon") is not None \
                else "待落点"
            item = QListWidgetItem("%s　%s　%s　%.0f%%　%s"
                                   % (src, r.get("shot_at") or "—",
                                      r.get("cls") or "未识别",
                                      (r.get("conf") or 0) * 100, loc))
            item.setData(Qt.UserRole, r.get("id"))
            item.setToolTip("坐标来源：%s｜备注：%s｜UID：%s"
                            % (r.get("coord_src") or "-", r.get("note") or "-",
                               r.get("uid") or "-"))
            self.lst.addItem(item)

    def pin_selected(self):
        """给选中的待落点记录补坐标（水印坐标直接粘）"""
        items = self.lst.selectedItems()
        if len(items) != 1:
            InfoBar.info("请只选一条", "每次给一条记录补坐标（可按住 Ctrl 取消多选）",
                         duration=3000, position=InfoBarPosition.TOP, parent=self)
            return
        text, ok = QInputDialog.getText(
            self, "补坐标", "粘贴坐标（支持 32°0'38\"N 118°37'48\"E 或 32.010556, 118.63）：")
        if not ok or not text.strip():
            return
        try:
            pr = parse_coord_text(text)
        except ValueError as e:
            InfoBar.warning("坐标解析失败", str(e), duration=4500,
                            position=InfoBarPosition.TOP, parent=self)
            return
        try:
            import rice_geo
            lon_w, lat_w = rice_geo.from_gcj02(pr["lon"], pr["lat"], "wgs84")
            rid = items[0].data(Qt.UserRole)
            self.geo.set_location(rid, lon_w, lat_w, coord_src="manual")
            log_event("manual_pin", rec_id=rid, lon=lon_w, lat=lat_w, via="manage_dialog")
            InfoBar.success("补坐标成功", "记录已落点，地图上刷新即可看到",
                            duration=3500, position=InfoBarPosition.TOP, parent=self)
            self.reload()
        except Exception as e:
            log.exception("补坐标失败")
            InfoBar.error("补坐标失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)

    def delete_selected(self):
        items = self.lst.selectedItems()
        if not items:
            InfoBar.info("没选记录", "先在列表里点一条（可按住 Ctrl 多选）", duration=3000,
                         position=InfoBarPosition.TOP, parent=self)
            return
        if MessageBox("确认删除", "确定删除选中的 %d 条记录吗？档案里的照片副本也会一起删除，"
                                 "此操作不可恢复。" % len(items),
                      self.window()).exec():
            n = 0
            for it in items:
                rid = it.data(Qt.UserRole)
                try:
                    self.geo.delete(rid)
                    log_event("geo_delete", rec_id=rid, by="manual")
                    n += 1
                except Exception as e:
                    log.warning("删除记录 %s 失败：%s", rid, e)
            InfoBar.success("已删除 %d 条" % n, "列表已刷新", duration=3000,
                            position=InfoBarPosition.TOP, parent=self)
            self.reload()


class MapInterface(QWidget):
    """地块地图：点位可视化 + 风险列表 + Agent 区域分析"""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("mapInterface")
        self._agent_busy = False

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(page_title("地块地图", self))
        head.addStretch()
        self.lbl_stat = CaptionLabel("", self)
        head.addWidget(self.lbl_stat)
        self.btn_web_map = PushButton(FIF.GLOBE, "交互地图", self)
        self.btn_web_map.setToolTip("在独立窗口里打开 Leaflet 交互地图（支持热力图）")
        self.btn_web_map.clicked.connect(self.open_web_window)
        head.addWidget(self.btn_web_map)
        self.chk_route = CheckBox("显示巡田路线", self)
        self.chk_route.setChecked(True)
        self.chk_route.stateChanged.connect(lambda _=None: self._push_to_web())
        head.addWidget(self.chk_route)
        self.btn_refresh = PushButton(FIF.SYNC, "刷新", self)
        self.btn_refresh.clicked.connect(self.refresh)
        head.addWidget(self.btn_refresh)
        self.btn_import = PushButton(FIF.FOLDER_ADD, "导入照片并落点", self)
        self.btn_import.setToolTip("选择带 GPS 的田间照片，导入档案库并（可选）跑一遍识别")
        self.btn_import.clicked.connect(self.import_photos)
        head.addWidget(self.btn_import)
        self.btn_pin = PushButton(FIF.PIN, "粘贴坐标落点", self)
        self.btn_pin.setToolTip("手机水印/地图 App 复制的坐标直接粘贴落点——"
                                "照片没有 GPS 时最省事的补法")
        self.btn_pin.clicked.connect(self.manual_pin)
        head.addWidget(self.btn_pin)
        self.btn_manage = PushButton(FIF.LIBRARY, "记录管理", self)
        self.btn_manage.setToolTip("逐条查看识别结果与来源（📱手机 / 💻电脑），可随时删除")
        self.btn_manage.clicked.connect(self.manage_records)
        head.addWidget(self.btn_manage)
        self.btn_export = PushButton(FIF.SAVE, "导出点位", self)
        self.btn_export.clicked.connect(self.export_points)
        head.addWidget(self.btn_export)
        root.addLayout(head)

        # 筛选行：只决定「画哪些」，不改右侧 Agent 工具的统计口径
        filt = QHBoxLayout()
        filt.setSpacing(10)
        filt.addWidget(BodyLabel("筛选", self))
        self.cmb_days = ComboBox(self)
        for _label, _val in (("全部时间", None), ("最近 7 天", 7),
                             ("最近 30 天", 30), ("最近 90 天", 90)):
            self.cmb_days.addItem(_label, userData=_val)
        self.cmb_days.setMinimumWidth(120)
        self.cmb_days.setToolTip("按拍摄时间筛选点位；风险分、巡田路线、密集区域会一起跟着重算")
        filt.addWidget(self.cmb_days)
        self.cmb_cls = ComboBox(self)
        self.cmb_cls.addItem("全部类别", userData=None)
        for _name in CLASS_NAMES:
            self.cmb_cls.addItem(_name, userData=_name)
        self.cmb_cls.setMinimumWidth(140)
        self.cmb_cls.setToolTip("只看某一类病虫害的点位与密集区域")
        filt.addWidget(self.cmb_cls)
        self.chk_heat = CheckBox("热力图", self)
        self.chk_heat.setToolTip("叠一层热力，快速看出病虫害集中在哪片（交互地图下生效）")
        filt.addWidget(self.chk_heat)
        filt.addStretch()
        self.btn_spray = PushButton(FIF.GLOBE, "查施药窗口", self)
        self.btn_spray.setToolTip("逐地块查天气，算今天能不能打药；结果标在地图上（30 分钟缓存）")
        self.btn_spray.clicked.connect(self.query_spray)
        filt.addWidget(self.btn_spray)
        root.addLayout(filt)
        self.cmb_days.currentIndexChanged.connect(self._on_filter_changed)
        self.cmb_cls.currentIndexChanged.connect(self._on_filter_changed)
        self.chk_heat.stateChanged.connect(self._on_filter_changed)

        body = QHBoxLayout()
        body.setSpacing(12)

        # 左：地图
        left = SimpleCardWidget(self)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        self.map_stack = QStackedWidget(left)
        native_page = QWidget(left)
        self.map_holder = QVBoxLayout(native_page)
        self.map_holder.setContentsMargins(0, 0, 0, 0)
        self.web_page = QWidget(left)
        self.web_layout = QVBoxLayout(self.web_page)
        self.web_layout.setContentsMargins(0, 0, 0, 0)
        self.map_stack.addWidget(native_page)      # 0：轻量（原生瓦片，无需 GL/Key）
        self.map_stack.addWidget(self.web_page)    # 1：交互（Leaflet + QtWebEngine）
        lv.addWidget(self.map_stack, 1)
        self.lbl_map_tip = CaptionLabel("", left)
        lv.addWidget(self.lbl_map_tip)
        body.addWidget(left, 7)
        self.web_view = None      # 页内不再创建 WebEngine 视图（原因见 MapWebWindow 说明）
        self.bridge = None
        self._web_window = None
        self._spray = {}              # 地块 → 施药窗口状态（查过才有）
        self._spray_at = ""
        self._spray_busy = False
        self._days_filter = None
        self._cls_filter = None
        self._filter_note = ""
        self.canvas = self._build_canvas()
        self._pt_records = []
        if self.canvas is not None:
            try:
                # markerClicked 给了序号：点一个点就弹这条记录的详情
                self.canvas.markerClicked.connect(self._on_marker_clicked)
            except Exception as e:
                log.debug("接地图点位点击失败：%s", e)
        self.map_stack.setCurrentIndex(0)

        # 右：风险列表 + Agent 分析
        right = SimpleCardWidget(self)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 14, 16, 14)
        rv.setSpacing(8)
        rv.addWidget(brand_title("地块风险", right))
        self.risk_list = QListWidget(right)
        self.risk_list.setMinimumHeight(150)
        rv.addWidget(self.risk_list, 1)
        row_btn = QHBoxLayout()
        self.btn_analyze = PrimaryPushButton(FIF.ROBOT, "让 Agent 分析这片区域", right)
        self.btn_analyze.clicked.connect(self.analyze_region)
        row_btn.addWidget(self.btn_analyze)
        self.btn_stop = PushButton("停止", right)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        row_btn.addWidget(self.btn_stop)
        rv.addLayout(row_btn)
        self.agent_out = TextEdit(right)
        self.agent_out.setReadOnly(True)
        self.agent_out.setPlaceholderText(
            "点上面的按钮，Agent 会自己调用地图工具（风险排序、巡田路线、密集区域…）"
            "并给出结论；调用过程会显示在这里。")
        rv.addWidget(self.agent_out, 2)
        body.addWidget(right, 4)
        root.addLayout(body, 1)

        self.map_agent = None
        QTimer.singleShot(300, self.refresh)

    # ── 地图控件（惰性创建：模块不可用就显示提示） ──
    def _on_marker_clicked(self, idx):
        """点地图上的点位 → 直接弹出这条记录的详情（识别结果/来源/坐标/照片）"""
        recs = getattr(self, "_pt_records", None) or []
        if idx < 0 or idx >= len(recs):
            return
        r = recs[idx]
        src = "📱 手机传图" if (r.get("source") or "") == "phone" else "💻 电脑本地"
        loc = ("%.6f, %.6f" % (r.get("lon"), r.get("lat"))) if r.get("lon") is not None \
            else "尚未落点"
        lines = ["识别结果：%s　置信度 %.0f%%" % (r.get("cls") or "未识别",
                                              (r.get("conf") or 0) * 100),
                 "拍摄时间：%s" % (r.get("shot_at") or "—"),
                 "来源：%s" % src,
                 "坐标：%s（%s）" % (loc, r.get("coord_src") or "-"),
                 "地块：%s" % (r.get("plot") or "未指定"),
                 "备注：%s" % (r.get("note") or "—")]
        dlg = MessageBox("点位详情", "\n".join(lines), self.window())
        dlg.exec()

    def _build_canvas(self):
        try:
            import rice_geo
            self.geo = rice_geo
            canvas = rice_geo.MapCanvas(rice_geo.writable_root() / "tiles")
            self.map_holder.addWidget(canvas)
            self.lbl_map_tip.setText("底图：高德瓦片（本地缓存，无需 Key）。点选点位可在控制台看详情。")
            return canvas
        except Exception as e:
            self.geo = None
            hint = QLabel("地块地图模块不可用：%s\n\n"
                          "请确认项目根目录存在 rice_geo.py，且已安装 PyQt5 / Pillow / requests。"
                          % e, self)
            hint.setWordWrap(True)
            self.map_holder.addWidget(hint)
            self.lbl_map_tip.setText("地图控件未加载；风险分析与 Agent 工具仍可用。")
            return None

    def _store(self):
        return geo_store()

    # ── 交互地图（QtWebEngine + Leaflet）──
    def set_map_mode(self, mode):
        """保留的入口：web=打开独立交互地图窗口；native=页内轻量地图"""
        if mode == "web":
            self.open_web_window()
            return
        self.map_stack.setCurrentIndex(0)
        self.lbl_map_tip.setText("轻量地图：高德瓦片本地缓存，无需 Key 与 GL；点选点位可在右侧看风险")

    def open_web_window(self):
        """在独立窗口里打开交互地图。

        页内开会把主窗口导航栏渲染成黑色且不可逆（QtWebEngine 改掉窗口合成路径，
        而导航栏靠透明背景做亚克力），所以这里一律开独立顶层窗口。
        """
        if not webengine_available():
            InfoBar.warning("交互地图不可用",
                            "需要 PyQtWebEngine，且不能在离屏环境使用；已保持轻量地图",
                            duration=5000, position=InfoBarPosition.TOP, parent=self)
            return
        try:
            if self._web_window is None:
                self._web_window = MapWebWindow(self)
            self._web_window.show()
            self._web_window.raise_()
            self._web_window.activateWindow()
            # 视图延后一帧再建：让窗口先画出来，别被 QtWebEngine 的初始化卡住界面
            QTimer.singleShot(0, self._web_window.ensure_view)
            self.lbl_map_tip.setText(
                "交互地图已在独立窗口打开：滚轮缩放 / 拖动平移；点地图可给「待落点」落位")
        except Exception as e:
            log.exception("打开交互地图窗口失败")
            self._web_window = None
            InfoBar.error("交互地图启动失败", str(e)[:140], duration=6000,
                          position=InfoBarPosition.TOP, parent=self)

    # 交互地图的创建/加载逻辑已整体移到 MapWebWindow（页内创建会毁掉主窗口外观）

    def _on_filter_changed(self, *_):
        """筛选变化：时间/类别影响画面与风险列表，热力图只影响叠加层"""
        self._days_filter = self.cmb_days.currentData()
        self._cls_filter = self.cmb_cls.currentData()
        parts = []
        if self._days_filter:
            parts.append("最近 %d 天" % self._days_filter)
        if self._cls_filter:
            parts.append(self._cls_filter)
        if self.chk_heat.isChecked():
            parts.append("热力图")
        self._filter_note = ("　·　筛选：" + " / ".join(parts)) if parts else ""
        self.refresh()

    def _push_to_web(self):
        """页内筛选变了 → 同步给已打开的交互地图窗口"""
        win = self._web_window
        if win is not None and win.isVisible():
            win.push()

    def _on_web_pick(self, lon, lat):
        """在网页地图上点一下：给最近的「待落点」记录落位；没有待落点就新建一条"""
        st = self._store()
        if st is None:
            return
        try:
            import rice_geo
            wlon, wlat = rice_geo.gcj02_to_wgs84(lon, lat)
        except Exception:
            wlon, wlat = lon, lat
        pending = [r for r in st.all_records() if r.get("lon") is None]
        if pending:
            rec = pending[0]
            st.set_location(rec["id"], wlon, wlat, coord_src="map-pick")
            msg = "「%s」已落位到 %.5f, %.5f" % (
                os.path.basename(rec.get("orig_path") or "") or ("记录 #%s" % rec["id"]),
                wlon, wlat)
            log_event("map_pick", rec_id=rec["id"], lon=round(wlon, 6), lat=round(wlat, 6))
            self.refresh()
        else:
            st.add_record(shot_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          lon=wlon, lat=wlat, coord_src="map-pick",
                          cls="", conf=0.0, note="在地图上手动添加的点位（无照片）")
            msg = "已新增一个点位：%.5f, %.5f" % (wlon, wlat)
            log_event("map_pick_new", lon=round(wlon, 6), lat=round(wlat, 6))
            self.refresh()
        InfoBar.success("落位成功", msg, duration=4000,
                        position=InfoBarPosition.TOP, parent=self)

    def _on_plot_selected(self, plot):
        """网页地图上点了某个点位 → 右侧列表同步选中"""
        if not plot:
            return
        for i in range(self.risk_list.count()):
            if self.risk_list.item(i).text().startswith(plot):
                self.risk_list.setCurrentRow(i)
                self.risk_list.scrollToItem(self.risk_list.item(i))
                break

    def showEvent(self, event):
        # 切到本页就自动刷新：识别/落点后切过来立刻能看到新点位，不用手点「刷新」
        super().showEvent(event)
        try:
            self.refresh()
        except Exception as e:
            log.debug("切页刷新地图失败：%s", e)

    # ── 数据刷新 ──
    def refresh(self):
        st = self._store()
        if st is None:
            self.lbl_stat.setText("地块地图模块不可用")
            self.risk_list.clear()
            self.risk_list.addItem("请检查 rice_geo.py 是否存在")
            return
        ov = map_overview_data()
        self.lbl_stat.setText("记录 %d 条（待落点 %d）· 地块 %d 个 · %s%s"
                              % (ov["记录总数"], ov["待落点"], ov["地块数"],
                                 "～".join(ov["时间范围"]) if ov["时间范围"] else "无时间信息",
                                 self._filter_note))
        rows = plot_risk_rows(self._days_filter) or []
        self.risk_list.clear()
        if not rows:
            self.risk_list.addItem("当前筛选下没有记录，换个时间范围试试"
                                   if (self._days_filter or self._cls_filter)
                                   else "还没有已落点的记录：点「导入照片并落点」开始")
        for r in rows[:20]:
            item = QListWidgetItem("%s　风险 %.0f　%s　%d 个点位　最近 %s%s"
                                   % (r["plot"], r["风险分"], r["主要问题"] or "—",
                                      r["点数"], r["最近时间"], self._spray_suffix(r["plot"])))
            item.setToolTip("类别分布：%s" % json.dumps(r["类别分布"], ensure_ascii=False))
            self.risk_list.addItem(item)
        if self.canvas is not None:
            try:
                # 与网页地图同一口径：时间和类别筛选都生效
                _recs = _geo_records(self._days_filter)
                if self._cls_filter:
                    _recs = [r for r in _recs if (r.get("cls") or "") == self._cls_filter]
                pts = st.to_points(_recs)
                # 记住映射：markerClicked 给的是序号，靠它找回是哪条记录
                self._pt_records = _recs
                self.canvas.set_points(pts)
                if pts:
                    self.canvas.fit_points(pts)
            except Exception as e:
                log.warning("地图点位刷新失败：%s", e)
        self._push_to_web()

    # ── 地块施药窗口（天气叠加） ──
    def _spray_suffix(self, plot):
        """风险列表行尾的施药状态；没查过就不加，不硬凑"""
        sp = self._spray.get(plot)
        if not sp:
            return ""
        st = sp.get("status")
        if st == "ok":
            return "　·　今日可施药 %s" % sp.get("label", "")
        if st == "narrow":
            return "　·　窗口窄 %s" % sp.get("label", "")
        if st == "no":
            return "　·　今日不宜（%s）" % (sp.get("reason") or "无窗口")
        return "　·　天气未知"

    def spray_payload(self):
        """给地图徽章的数据：只含有坐标的地块"""
        out = []
        for r in (plot_risk_rows(self._days_filter) or []):
            sp = self._spray.get(r["plot"])
            if sp and sp.get("lon") is not None:
                out.append(sp)
        return out

    def query_spray(self):
        """逐地块查今天能不能打药（网络慢，必须后台跑）"""
        if self._spray_busy:
            return
        plots = []
        for r in (plot_risk_rows(self._days_filter) or []):
            if r.get("lon") is not None and r.get("lat") is not None:
                plots.append((r["plot"], r["lon"], r["lat"]))
        if not plots:
            InfoBar.info("还没有可查的地块",
                         "先导入带 GPS 的照片并落点，或在地图上手动落位",
                         duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        self._spray_busy = True
        self.btn_spray.setEnabled(False)
        self.lbl_map_tip.setText("正在逐个地块查天气（共 %d 块）…" % len(plots))
        self._spray_thread = _track_thread(PlotWeatherThread(plots))
        self._spray_thread.progress.connect(self._on_spray_progress)
        self._spray_thread.done.connect(self._on_spray_done)
        self._spray_thread.start()

    def _on_spray_progress(self, i, total, name):
        self.lbl_map_tip.setText("查天气 %d/%d：%s" % (i, total, name))

    def _on_spray_done(self, rows):
        self._spray_busy = False
        self.btn_spray.setEnabled(True)
        self._spray = {r.get("plot"): r for r in rows if r.get("plot")}
        self._spray_at = datetime.now().strftime("%H:%M")
        cnt = Counter(r.get("status") for r in rows)
        self.lbl_map_tip.setText(
            "施药窗口（%s 查）：可施药 %d · 窗口窄 %d · 不宜 %d · 未知 %d"
            % (self._spray_at, cnt.get("ok", 0), cnt.get("narrow", 0),
               cnt.get("no", 0), cnt.get("unknown", 0)))
        try:
            log_event("spray_overview", plots=len(rows), ok=cnt.get("ok", 0),
                      narrow=cnt.get("narrow", 0), no=cnt.get("no", 0),
                      unknown=cnt.get("unknown", 0))
        except Exception:
            log.warning("施药窗口事件写入失败", exc_info=True)
        InfoBar.success("施药窗口已更新",
                        "可施药 %d 块 · 不宜 %d 块" % (cnt.get("ok", 0) + cnt.get("narrow", 0),
                                                   cnt.get("no", 0)),
                        duration=4000, position=InfoBarPosition.TOP, parent=self)
        self.refresh()

    # ── Agent 区域分析 ──
    def analyze_region(self):
        if self._agent_busy:
            return
        if self._store() is None:
            InfoBar.warning("不可用", "地块地图模块未加载", duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        ov = map_overview_data()
        if not ov.get("已落点"):
            InfoBar.info("还没有点位", "先导入带 GPS 的照片，Agent 才有数据可分析",
                         duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        self._agent_busy = True
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.agent_out.setPlainText("正在分析…（Agent 会自己决定调用哪些地图工具）\n")
        log_event("map_agent_start", points=ov.get("已落点"), plots=ov.get("地块数"))
        self.map_agent = _track_thread(LLMAnalysisThread("地块地图", None, None,
                                                        prompt=MAP_AGENT_PROMPT))
        self.map_agent.tool_used.connect(self._on_tool)
        self.map_agent.chunk.connect(self._on_chunk)
        self.map_agent.result_ready.connect(self._on_done)
        self.map_agent.failed.connect(self._on_fail)
        self.map_agent.start()

    def _on_tool(self, name, label):
        self.agent_out.append("🔧 %s…" % label)
        log_event("map_agent_tool", tool=name)

    def _on_chunk(self, text):
        cur = self.agent_out.toPlainText()
        head = cur.split("\n\n", 1)[0] if "\n\n" in cur else ""
        # 流式内容整体替换「正文」部分，保留前面的工具调用轨迹
        self.agent_out.setPlainText((head + "\n\n" + text) if head.startswith("正在分析") else text)

    def _on_done(self, text):
        self._agent_busy = False
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.agent_out.setPlainText(text or "")
        log_event("map_agent_done", chars=len(text or ""))
        try:
            out = TEACHING_DIR.parent / "maps"
            out.mkdir(parents=True, exist_ok=True)
            (out / ("区域分析_%s.md" % datetime.now().strftime("%Y%m%d_%H%M%S"))
             ).write_text(text or "", encoding="utf-8")
        except OSError as e:
            log.warning("区域分析落盘失败：%s", e)

    def _on_fail(self, msg):
        self._agent_busy = False
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.agent_out.setPlainText("分析失败：%s" % msg)
        log_event("map_agent_failed", error=str(msg)[:120])

    def stop_analysis(self):
        if self.map_agent is not None:
            self.map_agent.stop()
            self.btn_stop.setEnabled(False)
            self.agent_out.append("（已请求停止）")

    # ── 导入 / 导出（写操作，人点触发） ──
    def manage_records(self):
        """记录管理：看来源、删记录"""
        dlg = GeoManageDialog(geo_store(), parent=self.window())
        dlg.exec()
        self.refresh()

    def manual_pin(self):
        """粘贴坐标落点（水印坐标、地图复制的坐标都能用）"""
        dlg = ManualPinDialog(geo_store(), parent=self.window())   # 传档案库实例，不是模块
        if dlg.exec():
            try:
                self.reload()
            except Exception:
                try:
                    self.refresh()
                except Exception as e:
                    log.debug("刷新地图失败：%s", e)

    def import_photos(self):
        if self.geo is None:
            InfoBar.warning("不可用", "地块地图模块未加载", duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        files, _ = QFileDialog.getOpenFileNames(self, "选择田间照片（带 GPS 的会自动落点）", "",
                                                "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        self.btn_import.setEnabled(False)
        self.lbl_map_tip.setText("正在导入 %d 张照片…" % len(files))
        model = getattr(self.main, "model", None)
        device = getattr(self.main, "device", None)
        if model is None:
            InfoBar.info("只落点不识别", "模型未加载，将只读取时间与坐标",
                         duration=3000, position=InfoBarPosition.TOP, parent=self)
        self.import_thread = _track_thread(self.geo.ImportThread(
            self._store(), files, model=model, device=device))
        self.import_thread.done.connect(self._on_import_done)
        self.import_thread.start()

    def _on_import_done(self, *a):
        self.btn_import.setEnabled(True)
        self.lbl_map_tip.setText("导入完成。")
        InfoBar.success("导入完成", "已写入地块档案库", duration=4000,
                        position=InfoBarPosition.TOP, parent=self)
        self.refresh()

    def export_points(self):
        st = self._store()
        if st is None:
            return
        rows = st.all_records()
        if not rows:
            InfoBar.info("没有数据", "先导入照片", duration=3000,
                         position=InfoBarPosition.TOP, parent=self)
            return
        out = st.export_dir / ("点位表_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        try:
            self.geo.export_csv(st, out)
            InfoBar.success("已导出", str(out), duration=5000,
                            position=InfoBarPosition.TOP, parent=self)
            log_event("map_export", path=str(out), n=len(rows))
        except Exception as e:
            InfoBar.error("导出失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)


class AuditDialog(MessageBoxBase):
    """库体检结果"""

    def __init__(self, text, parent=None):
        super().__init__(parent or QApplication.activeWindow())
        self.viewLayout.addWidget(SubtitleLabel("本地防治库体检", self))
        body = TextEdit(self)
        body.setReadOnly(True)
        body.setPlainText(text)
        body.setMinimumHeight(380)
        self.viewLayout.addWidget(body)
        self.yesButton.setText("知道了")
        self.cancelButton.hide()
        box = getattr(self, "widget", None)
        if box is not None:
            box.setMinimumWidth(560)


class TreatmentInterface(QWidget):
    """防治库维护：查看内置条目 + 增删改自己的当地农药标签（不动代码）"""

    FIELDS = [("name", "药剂名称", True), ("spec", "规格剂型", False),
              ("dose", "亩用量", False), ("safe_days", "安全间隔期(天)", False),
              ("reg_no", "登记证号", False), ("method", "施药方法", False),
              ("max_per_season", "每季最多(次)", False), ("target", "登记对象", False),
              ("type", "类型(化学/生物)", False), ("timing", "防治适期", False),
              ("caution", "注意事项", False), ("source", "来源依据", False)]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("treatmentInterface")
        self.current_cls = CLASS_NAMES[0]
        self.current_drug = None

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(page_title("防治库维护", self))
        head.addStretch()
        self.lbl_stat = CaptionLabel("", self)
        head.addWidget(self.lbl_stat)
        for label, icon, slot in (("库体检", FIF.DOCUMENT, self.show_audit),
                                  ("导入", FIF.FOLDER_ADD, self.import_db),
                                  ("导出", FIF.SAVE, self.export_db),
                                  ("新建药剂", FIF.ADD, self.new_drug)):
            b = PushButton(icon, label, self)
            b.clicked.connect(slot)
            head.addWidget(b)
        root.addLayout(head)
        root.addWidget(t_caption(
            "内置条目来自农药登记标签（只读，不可删改）；你补充的条目可随时增删改，"
            "同名时以你的为准。数据存在 treatment/user_db.json，升级程序不会丢。", self))

        body = QHBoxLayout()
        body.setSpacing(12)
        # 左：类别 + 药剂列表
        left = SimpleCardWidget(self)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 12, 14, 12)
        lv.setSpacing(8)
        self.cmb_cls = ComboBox(left)
        self.cmb_cls.addItems(CLASS_NAMES)
        self.cmb_cls.currentTextChanged.connect(self._on_cls)
        lv.addWidget(self.cmb_cls)
        self.lst = QListWidget(left)
        self.lst.currentRowChanged.connect(self._on_drug)
        lv.addWidget(self.lst, 1)
        body.addWidget(left, 4)

        # 右：编辑表单 + 类别指标
        right = SimpleCardWidget(self)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 14, 16, 14)
        rv.setSpacing(8)
        rv.addWidget(brand_title("药剂信息", right))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        self.inputs = {}
        for i, (key, label, required) in enumerate(self.FIELDS):
            row, col = divmod(i, 2)
            cell = QWidget(right)
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(2)
            cv.addWidget(t_caption(label + ("（必填）" if required else ""), cell))
            ed = LineEdit(cell)
            cv.addWidget(ed)
            self.inputs[key] = ed
            grid.addWidget(cell, row, col)
        rv.addLayout(grid)
        row_op = QHBoxLayout()
        self.btn_save = PrimaryPushButton(FIF.SAVE, "保存到我的库", right)
        self.btn_save.clicked.connect(self.save_drug)
        row_op.addWidget(self.btn_save)
        self.btn_del = PushButton(FIF.DELETE, "删除这条", right)
        self.btn_del.clicked.connect(self.delete_drug)
        row_op.addWidget(self.btn_del)
        row_op.addStretch()
        rv.addLayout(row_op)
        rv.addWidget(t_caption("类别量化指标（覆盖内置，用于「什么时候该打药」的判断）", right))
        self.th_inputs = {}
        for label in ("量化指标", "施药适期", "指标来源"):
            row = QHBoxLayout()
            lab = BodyLabel(label, right)
            lab.setFixedWidth(70)
            row.addWidget(lab)
            ed = LineEdit(right)
            self.th_inputs[label] = ed
            row.addWidget(ed, 1)
            rv.addLayout(row)
        btn_th = PushButton(FIF.SAVE, "保存类别指标", right)
        btn_th.clicked.connect(self.save_threshold)
        rv.addWidget(btn_th)
        rv.addStretch()
        body.addWidget(right, 6)
        root.addLayout(body, 1)
        self.reload()

    # ── 数据 ──
    def reload(self):
        st = user_db_stats()
        a = treatments_audit()
        self.lbl_stat.setText("内置+用户共 %d 条（我补充的 %d 条）· 类别指标 %d 个 · 登记证号完整度 %d%%"
                              % (a["药剂条目"], st["用户药剂"], st["用户指标"],
                                 a["有登记证号"] * 100 // max(1, a["药剂条目"])))
        self._fill_list()

    def _on_cls(self, cls):
        self.current_cls = cls
        self._fill_list()

    def _fill_list(self):
        self.lst.clear()
        for d in TREATMENT_DB.get(self.current_cls, {}).get("drugs", []):
            mine = d.get("origin") == "user"
            label = "%s%s　%s" % ("✎ " if mine else "　", d.get("name", ""),
                                  (d.get("dose") or ""))
            item = QListWidgetItem(label)
            item.setToolTip("我补充的" if mine else "内置（只读）")
            if mine:
                item.setForeground(QColor(NAU["primary"]))
            self.lst.addItem(item)
        meta = TREATMENT_DB.get(self.current_cls, {}).get("meta", {})
        for k, ed in self.th_inputs.items():
            ed.setText(str(meta.get(k, "")))
        if self.lst.count():
            self.lst.setCurrentRow(0)

    def _on_drug(self, row):
        drugs = TREATMENT_DB.get(self.current_cls, {}).get("drugs", [])
        if row < 0 or row >= len(drugs):
            return
        d = drugs[row]
        self.current_drug = d
        for key, ed in self.inputs.items():
            ed.setText("" if d.get(key) is None else str(d.get(key)))
        deletable = d.get("origin") == "user"
        self.btn_del.setEnabled(deletable)
        self.btn_del.setToolTip("" if deletable else "内置条目不可删除（可另存一条同名条目覆盖）")

    # ── 操作 ──
    def new_drug(self):
        self.current_drug = None
        for ed in self.inputs.values():
            ed.clear()
        self.lst.setCurrentRow(-1)
        self.inputs["name"].setFocus()
        InfoBar.info("新建", "填好名称后点「保存到我的库」", duration=3000,
                     position=InfoBarPosition.TOP, parent=self)

    def save_drug(self):
        drug = {k: ed.text().strip() for k, ed in self.inputs.items()}
        rec, msg = user_add_drug(self.current_cls, drug)
        if rec is None:
            InfoBar.warning("保存失败", msg, duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success(msg, "%s · %s" % (self.current_cls, rec.get("name")),
                        duration=3000, position=InfoBarPosition.TOP, parent=self)
        self.reload()

    def delete_drug(self):
        if not self.current_drug:
            return
        ok, msg = user_delete_drug(self.current_cls, self.current_drug.get("name"))
        (InfoBar.success if ok else InfoBar.warning)("删除" if ok else "不能删除", msg,
                                                     duration=4000,
                                                     position=InfoBarPosition.TOP,
                                                     parent=self)
        self.reload()

    def save_threshold(self):
        ok, msg = user_set_threshold(self.current_cls,
                                     {k: ed.text().strip() for k, ed in self.th_inputs.items()})
        (InfoBar.success if ok else InfoBar.warning)("已保存" if ok else "失败", msg,
                                                     duration=3000,
                                                     position=InfoBarPosition.TOP, parent=self)
        self.reload()

    def show_audit(self):
        a = treatments_audit()
        lines = ["类别 %d 个 · 药剂条目 %d 条（生物农药 %d 条）"
                 % (a["类别数"], a["药剂条目"], a["生物农药"]),
                 "有登记证号 %d · 有安全间隔期 %d · 有防治适期 %d"
                 % (a["有登记证号"], a["有安全间隔期"], a["有防治适期"]), ""]
        for key in ("缺登记证号", "缺安全间隔期", "缺防治适期"):
            lines.append("%s（%d 条）：%s" % (key, len(a[key]), "、".join(a[key][:6]) or "无"))
        lines.append("")
        lines.append("缺的条目在界面与回答里都显示「待补，以标签为准」——程序不会用猜测值替代。")
        AuditDialog("\n".join(lines), parent=self.window()).exec()

    def export_db(self):
        default = str(Path.home() / "Desktop" / "稻视界_我的防治库.json")
        fp, _ = QFileDialog.getSaveFileName(self, "导出我的防治库", default, "JSON (*.json)")
        if not fp:
            return
        ok, msg = export_user_db(fp)
        (InfoBar.success if ok else InfoBar.error)("已导出" if ok else "导出失败", msg,
                                                   duration=5000,
                                                   position=InfoBarPosition.TOP, parent=self)

    def import_db(self):
        fp, _ = QFileDialog.getOpenFileName(self, "导入防治库（与他人合并）", "", "JSON (*.json)")
        if not fp:
            return
        ok, msg = import_user_db(fp, merge=True)
        (InfoBar.success if ok else InfoBar.error)("导入成功" if ok else "导入失败", msg,
                                                   duration=5000,
                                                   position=InfoBarPosition.TOP, parent=self)
        self.reload()


class TrainInterface(QWidget):
    """教学页：三种模式 —— 图片考核（人机对照）/ 题库答题 / 案例学习。

    题库与案例都可从外部资料导入（自己买的教材、官方公开资料、开放许可内容），
    每条都带来源标注；也可以一键用本地防治库生成一套现成的题。
    """

    SESSION_SIZE = 10

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("trainInterface")
        self.pool = collect_question_pool()
        self.session = []
        self.records = []
        self.index = 0
        self.current = None
        self.model_guess = None
        self._q_t0 = datetime.now()
        self._answered = False
        self.mode = "image"          # image=图片考核 / bank=题库答题 / case=案例学习
        self.bank_session = []
        self.bank_index = 0
        self.case_index = 0
        self.case_hits = []

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(page_title("教学与考核", self))
        head.addStretch()
        self.btn_seed = PushButton(FIF.EDUCATION, "用本地库生成题库", self)
        self.btn_seed.setToolTip("按本地防治库自动出一套题与案例（内容自有，来源标注为本地防治库）")
        self.btn_seed.clicked.connect(self.seed_from_local)
        head.addWidget(self.btn_seed)
        self.btn_import = PushButton(FIF.FOLDER_ADD, "导入资料", self)
        self.btn_import.setToolTip("导入 PDF / Word / 文本 / CSV / JSON（请使用自有或公开合法资料）")
        self.btn_import.clicked.connect(self.import_material)
        head.addWidget(self.btn_import)
        self.btn_import_url = PushButton(FIF.GLOBE, "从网址导入", self)
        self.btn_import_url.clicked.connect(self.import_from_url)
        head.addWidget(self.btn_import_url)
        self.btn_export = PushButton(FIF.SAVE, "导出成绩单", self)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_sheet)
        head.addWidget(self.btn_export)
        root.addLayout(head)

        mode_row = QHBoxLayout()
        self.mode_bar = SegmentedWidget(self)
        self.mode_bar.addItem("image", "图片考核", lambda: self._switch_mode("image"))
        self.mode_bar.addItem("bank", "题库答题", lambda: self._switch_mode("bank"))
        self.mode_bar.addItem("case", "案例学习", lambda: self._switch_mode("case"))
        self.mode_bar.addItem("review", "题目复核", lambda: self._switch_mode("review"))
        self.mode_bar.addItem("wrong", "错题本", lambda: self._switch_mode("wrong"))
        self.mode_bar.setCurrentItem("image")
        mode_row.addWidget(self.mode_bar)
        mode_row.addStretch()
        self.lbl_teach = CaptionLabel("", self)
        mode_row.addWidget(self.lbl_teach)
        root.addLayout(mode_row)
        self._refresh_teach_status()
        self.status = CaptionLabel(
            "图片题库 %d 张（来自验证集与样例图）· 先自己判断，再看模型与知识库"
            % len(self.pool), self)
        root.addWidget(self.status)

        body = QHBoxLayout()
        body.setSpacing(12)
        left = SimpleCardWidget(self)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 14, 14, 14)
        self.img = ImageLabel(left)
        self.img.setMinimumSize(460, 340)
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setText("点「开始考核」随机出题")
        lv.addWidget(self.img, 1)
        self.progress = CaptionLabel("", left)
        lv.addWidget(self.progress)
        body.addWidget(left, 5)

        right = SimpleCardWidget(self)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 14, 16, 14)
        rv.setSpacing(8)
        rv.addWidget(brand_title("请判断这是什么病虫害", right))
        self.opts = []
        for name in CLASS_NAMES:
            b = PushButton(name, right)
            b.setMinimumHeight(38)
            b.clicked.connect(lambda _=None, n=name: self.answer(n))
            rv.addWidget(b)
            self.opts.append(b)
        b_unsure = TransparentPushButton("不确定 / 看不清", right)
        b_unsure.clicked.connect(lambda: self.answer("不确定"))
        rv.addWidget(b_unsure)
        self.opts.append(b_unsure)
        self.feedback = TextEdit(right)
        self.feedback.setReadOnly(True)
        self.feedback.setPlaceholderText("判定后会在这里给出正确答案、模型判断和知识库要点…")
        rv.addWidget(self.feedback, 1)
        body.addWidget(right, 4)
        self.left_card, self.right_card = left, right

        # ── 题库答题面板 ──
        self.bank_card = SimpleCardWidget(self)
        bv = QVBoxLayout(self.bank_card)
        bv.setContentsMargins(20, 16, 20, 16)
        bv.setSpacing(10)
        bv.addWidget(brand_title("题库答题", self.bank_card))
        self.bank_question = BodyLabel("", self.bank_card)
        self.bank_question.setWordWrap(True)
        bv.addWidget(self.bank_question)
        self.bank_opts_box = QVBoxLayout()
        self.bank_opts_box.setSpacing(6)
        bv.addLayout(self.bank_opts_box)
        self.bank_source = t_caption("", self.bank_card, color=NAU["subtext"])
        self.bank_source.setWordWrap(True)
        bv.addWidget(self.bank_source)
        self.bank_feedback = TextEdit(self.bank_card)
        self.bank_feedback.setReadOnly(True)
        self.bank_feedback.setPlaceholderText("作答后这里显示解析与来源…")
        bv.addWidget(self.bank_feedback, 1)
        row_b = QHBoxLayout()
        self.bank_progress = CaptionLabel("", self.bank_card)
        row_b.addWidget(self.bank_progress)
        row_b.addStretch()
        btn_next = PushButton("下一题", self.bank_card)
        btn_next.clicked.connect(lambda: self._next_bank(1))
        row_b.addWidget(btn_next)
        bv.addLayout(row_b)
        body.addWidget(self.bank_card, 9)

        # ── 案例学习面板 ──
        self.case_card = SimpleCardWidget(self)
        cv = QVBoxLayout(self.case_card)
        cv.setContentsMargins(20, 16, 20, 16)
        cv.setSpacing(10)
        cv.addWidget(brand_title("案例学习", self.case_card))
        row_s = QHBoxLayout()
        self.case_search = LineEdit(self.case_card)
        self.case_search.setPlaceholderText("按关键词搜索（类别、症状、措施、来源）…")
        self.case_search.returnPressed.connect(self._search_cases_ui)
        row_s.addWidget(self.case_search, 1)
        btn_search = PushButton("搜索", self.case_card)
        btn_search.clicked.connect(self._search_cases_ui)
        row_s.addWidget(btn_search)
        cv.addLayout(row_s)
        self.case_view = TextEdit(self.case_card)
        self.case_view.setReadOnly(True)
        cv.addWidget(self.case_view, 1)
        row_c = QHBoxLayout()
        self.case_progress = CaptionLabel("", self.case_card)
        row_c.addWidget(self.case_progress)
        row_c.addStretch()
        b_prev = PushButton("上一条", self.case_card)
        b_prev.clicked.connect(lambda: self._next_case(-1))
        row_c.addWidget(b_prev)
        b_nextc = PushButton("下一条", self.case_card)
        b_nextc.clicked.connect(lambda: self._next_case(1))
        row_c.addWidget(b_nextc)
        cv.addLayout(row_c)
        body.addWidget(self.case_card, 9)

        # ── 题目复核面板 ──
        self.review_card = SimpleCardWidget(self)
        rvv = QVBoxLayout(self.review_card)
        rvv.setContentsMargins(20, 16, 20, 16)
        rvv.setSpacing(10)
        rvv.addWidget(brand_title("题目复核（人工把关）", self.review_card))
        rvv.addWidget(t_caption(
            "导入或自动生成的题目先到这里过一遍：确认题干、答案、解析、来源都没问题再放进出题池。"
            "未复核的题不会被考核使用。", self.review_card, color=NAU["subtext"]))
        self.review_view = TextEdit(self.review_card)
        self.review_view.setReadOnly(True)
        rvv.addWidget(self.review_view, 1)
        self.review_progress = CaptionLabel("", self.review_card)
        rvv.addWidget(self.review_progress)
        row_r = QHBoxLayout()
        b_ok = PrimaryPushButton("通过并进题库", self.review_card)
        b_ok.clicked.connect(lambda: self._review_action("approve"))
        row_r.addWidget(b_ok)
        b_skip = PushButton("跳过", self.review_card)
        b_skip.clicked.connect(lambda: self._review_action("skip"))
        row_r.addWidget(b_skip)
        b_del = PushButton("废弃这道题", self.review_card)
        b_del.clicked.connect(lambda: self._review_action("delete"))
        row_r.addWidget(b_del)
        row_r.addStretch()
        rvv.addLayout(row_r)
        body.addWidget(self.review_card, 9)

        # ── 错题本面板 ──
        self.wrong_card = SimpleCardWidget(self)
        wv = QVBoxLayout(self.wrong_card)
        wv.setContentsMargins(20, 16, 20, 16)
        wv.setSpacing(10)
        wv.addWidget(brand_title("错题本（间隔重复）", self.wrong_card))
        self.wrong_stats = CaptionLabel("", self.wrong_card)
        wv.addWidget(self.wrong_stats)
        self.wrong_view = TextEdit(self.wrong_card)
        self.wrong_view.setReadOnly(True)
        wv.addWidget(self.wrong_view, 1)
        row_w = QHBoxLayout()
        b_go = PrimaryPushButton("开始复习（到期题目）", self.wrong_card)
        b_go.clicked.connect(self._start_wrong_review)
        row_w.addWidget(b_go)
        b_clear = PushButton("清空错题本", self.wrong_card)
        b_clear.clicked.connect(self._clear_wrong_book)
        row_w.addWidget(b_clear)
        row_w.addStretch()
        wv.addLayout(row_w)
        body.addWidget(self.wrong_card, 9)

        self.bank_card.setVisible(False)
        self.case_card.setVisible(False)
        self.review_card.setVisible(False)
        self.wrong_card.setVisible(False)
        root.addLayout(body, 1)

        row = QHBoxLayout()
        self.btn_start = PrimaryPushButton(FIF.EDUCATION, "开始考核", self)
        self.btn_start.setMinimumHeight(42)
        self.btn_start.clicked.connect(self.start_session)
        row.addWidget(self.btn_start)
        self.btn_pool = PushButton(FIF.FOLDER, "换个题库", self)
        self.btn_pool.setMinimumHeight(42)
        self.btn_pool.clicked.connect(self.choose_pool)
        row.addWidget(self.btn_pool)
        row.addStretch()
        root.addLayout(row)
        self._set_options_enabled(False)

    # ── 模式切换 ──
    def _switch_mode(self, mode):
        self.mode = mode
        is_img = (mode == "image")
        self.left_card.setVisible(is_img)
        self.right_card.setVisible(is_img)
        self.bank_card.setVisible(mode == "bank")
        self.case_card.setVisible(mode == "case")
        self.review_card.setVisible(mode == "review")
        self.wrong_card.setVisible(mode == "wrong")
        if mode == "bank":
            self._start_bank_quiz()
        elif mode == "case":
            self.case_hits = search_cases(self.case_search.text())
            self.case_index = 0
            self._show_case()
        elif mode == "review":
            self._review_skip = set()
            self._load_review_question()
        elif mode == "wrong":
            self._refresh_wrong_panel()
        self._refresh_teach_status()
        log_event("teach_mode", mode=mode)

    def _refresh_teach_status(self):
        st = teaching_stats()
        rv = st.get("错题本", {})
        self.lbl_teach.setText(
            "题库 %d 题（待复核 %d）　·　案例 %d 条　·　错题 %d（今日到期 %d）　·　导入 %d 次"
            % (st["题库"], st.get("待复核", 0), st["案例"], rv.get("错题数", 0),
               rv.get("今日到期", 0), st["导入次数"]))

    # ── 题库答题 ──
    def _start_bank_quiz(self):
        bank, only_reviewed = quiz_questions()
        self._quiz_note = ("题目池：仅已复核题目" if only_reviewed else
                           "题目池：全部题目（还没有复核过的题，建议先去「题目复核」过一遍）")
        self._clear_bank_opts()
        if not bank:
            self.bank_question.setText(
                "题库还是空的。点右上角「用本地库生成题库」一键生成，"
                "或用「导入资料」把 PDF / Word / 文本 / CSV / JSON 变成题目。")
            self.bank_source.setText("")
            self.bank_feedback.setPlainText("")
            self.bank_progress.setText("")
            return
        n = min(self.SESSION_SIZE, len(bank))
        self.bank_session = random.sample(bank, n)
        self.bank_index = 0
        self._load_bank_question()

    def _clear_bank_opts(self):
        while self.bank_opts_box.count():
            item = self.bank_opts_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _load_bank_question(self):
        """渲染当前题目：选项按钮按题库动态生成（不再限于六类）"""
        self._clear_bank_opts()
        self._bank_answered = False
        q = self.bank_session[self.bank_index]
        self.bank_question.setText(q.get("question", ""))
        self.bank_source.setText("来源：%s　·　%s"
                                 % (q.get("source", "未标注"), getattr(self, "_quiz_note", "")))
        self.bank_feedback.setPlainText("")
        self.bank_progress.setText("第 %d / %d 题　·　%s"
                                   % (self.bank_index + 1, len(self.bank_session),
                                      {"choice": "单选", "judge": "判断",
                                       "short": "简答"}.get(q.get("type"), "题目")))
        opts = q.get("options") or []
        if opts:
            for i, opt in enumerate(opts):
                b = PushButton("%s. %s" % ("ABCD"[i] if i < 4 else str(i + 1), opt),
                               self.bank_card)
                b.setMinimumHeight(36)
                b.clicked.connect(lambda _=None, x=opt: self._answer_bank(x))
                self.bank_opts_box.addWidget(b)
        else:
            b = PushButton("显示参考答案", self.bank_card)
            b.setMinimumHeight(36)
            b.clicked.connect(lambda: self._answer_bank(None))
            self.bank_opts_box.addWidget(b)
        self._q_t0 = datetime.now()

    def _answer_bank(self, choice):
        if getattr(self, "_bank_answered", False):
            return
        self._bank_answered = True
        q = self.bank_session[self.bank_index]
        answer = str(q.get("answer", "")).strip()
        correct = (choice is not None and choice.strip() == answer)
        seconds = (datetime.now() - self._q_t0).total_seconds()
        rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "session": getattr(self, "_session_id", ""), "kind": "bank",
               "image": "题库", "label": answer, "answer": choice or "（看答案）",
               "correct": correct, "seconds": round(seconds, 1),
               "model_guess": "—", "model_conf": 0.0, "model_correct": None,
               "source": q.get("source", "")}
        self.records.append(rec)
        append_train_record(**rec)
        # 接进间隔重复：答错 1 天后复习，答对则拉长间隔
        srs = record_answer_result(q.get("id", ""), correct)
        self._refresh_wrong_panel()
        self._refresh_teach_status()
        # 注意：log_event 的第一个参数就是 kind，额外字段不能再用 kind 这个名字
        log_event("training_answer", mode="bank", cls=(q.get("tags") or ["未标注"])[0],
                  correct=correct, seconds=round(seconds, 1),
                  next_due=srs.get("due"), interval=srs.get("interval"))
        kind = {"choice": "单选", "judge": "判断", "short": "简答"}.get(q.get("type"), "题目")
        head = ("✅ 答对了。" if correct else
                ("参考答案已显示，请对照上面的解析自行判断。" if choice is None
                 else "❌ 答错了。"))
        self.bank_feedback.setPlainText("\n\n".join([
            "%s%s" % (head, "" if choice is None else "　你选的是：%s" % choice),
            "正确答案：%s" % answer,
            "解析：%s" % (q.get("explain") or "（无）"),
            "来源：%s" % q.get("source", "未标注"),
            "题型：%s" % kind,
            "复习计划：%s（间隔 %d 天，答错 %d 次）"
            % (srs.get("due"), int(srs.get("interval", 0)), int(srs.get("wrong", 0)))]))
        QTimer.singleShot(600, lambda: self._next_bank(1))

    def _next_bank(self, delta):
        if not self.bank_session:
            return
        nxt = self.bank_index + delta
        if nxt >= len(self.bank_session):
            s = training_summary([r for r in self.records if r.get("kind") == "bank"])
            self.bank_feedback.setPlainText(
                "本轮题库答题结束：共 %s 题，正确率 %s%%，平均用时 %s 秒。\n"
                "错题已记进 teaching 台账，可导出成绩单。"
                % (s.get("题数"), s.get("人的正确率"), s.get("平均用时秒")))
            self.bank_progress.setText("本轮结束")
            self.btn_export.setEnabled(True)
            return
        self.bank_index = max(0, nxt)
        self._load_bank_question()

    # ── 案例学习 ──
    def _search_cases_ui(self):
        self.case_hits = search_cases(self.case_search.text())
        self.case_index = 0
        self._show_case()

    def _show_case(self):
        if not self.case_hits:
            kw = self.case_search.text().strip()
            self.case_view.setPlainText(
                ("没有匹配「%s」的案例。" % kw) if kw else
                "案例库为空。点右上角「用本地库生成题库」或用「导入资料」把教材/官方资料变成案例卡。")
            self.case_progress.setText("")
            return
        self.case_index = max(0, min(self.case_index, len(self.case_hits) - 1))
        c = self.case_hits[self.case_index]
        self.case_view.setPlainText("\n\n".join(
            x for x in ["【%s】" % c.get("title", ""),
                        "症状：%s" % c.get("symptom", ""),
                        "发生条件 / 防治指标：%s" % c.get("condition", ""),
                        "处置措施：%s" % c.get("action", ""),
                        "用药与结果：%s" % c.get("outcome", ""),
                        "来源：%s" % c.get("source", "未标注")] if x.split("：", 1)[-1]))
        self.case_progress.setText("第 %d / %d 条　·　来源：%s"
                                   % (self.case_index + 1, len(self.case_hits),
                                      c.get("source", "未标注")))

    def _next_case(self, delta):
        if not self.case_hits:
            return
        self.case_index = (self.case_index + delta) % len(self.case_hits)
        self._show_case()

    # ── 题目复核 ──
    def _load_review_question(self):
        skip = getattr(self, "_review_skip", set())
        pend = [q for q in pending_questions() if q.get("id") not in skip]
        total_pend = len(pending_questions())
        self.review_progress.setText("待复核 %d 题（本次已跳过 %d）　·　已复核 %d 题"
                                     % (total_pend, len(skip), len(reviewed_questions())))
        if not pend:
            self._review_current = None
            self.review_view.setPlainText(
                "没有待复核的题目了。\n\n导入或生成题库后，新题会先出现在这里；"
                "只有「通过」过的题才会被考核使用。")
            return
        q = pend[0]
        self._review_current = q
        lines = [q.get("question", ""), ""]
        for i, o in enumerate(q.get("options") or []):
            lines.append("  %s. %s" % ("ABCD"[i] if i < 4 else str(i + 1), o))
        if not q.get("options"):
            lines.append("  （简答题）")
        lines += ["", "答案：%s" % (q.get("answer") or "（未给出）"),
                  "解析：%s" % (q.get("explain") or "（无）"),
                  "来源：%s" % (q.get("source") or "未标注"),
                  "题型：%s" % {"choice": "单选", "judge": "判断",
                                "short": "简答"}.get(q.get("type"), "题目")]
        self.review_view.setPlainText("\n".join(lines))

    def _review_action(self, action):
        q = getattr(self, "_review_current", None)
        if not q:
            return
        if action == "approve":
            mark_question_reviewed(q["id"], True)
            InfoBar.success("已通过", "这道题现在可以用于考核了", duration=2500,
                            position=InfoBarPosition.TOP, parent=self)
        elif action == "delete":
            delete_question(q["id"])
            clear_review_item(q["id"])
            InfoBar.info("已废弃", "题目已从题库删除", duration=2500,
                         position=InfoBarPosition.TOP, parent=self)
        else:
            self._review_skip = getattr(self, "_review_skip", set()) | {q["id"]}
        self._refresh_teach_status()
        self._load_review_question()

    # ── 错题本（间隔重复） ──
    def _refresh_wrong_panel(self):
        st = review_stats()
        due = due_review_items()
        self.wrong_stats.setText(
            "错题 %d 道　·　今日到期 %d 道　·　已掌握 %d 道　·　规则：答错 1 天后复习，"
            "连续答对间隔拉长（1/3/7/15/30/60 天）"
            % (st["错题数"], st["今日到期"], st["已掌握"]))
        if not due:
            self.wrong_view.setPlainText(
                "今天没有到期需要复习的题。\n\n"
                "答错的题会自动进来：答错后 1 天复习；连续答对，间隔按 1/3/7/15/30/60 天拉长；"
                "连续答对 5 次标记为已掌握。")
            return
        lines = []
        for d in due[:30]:
            lines.append("· %s\n    错 %d 次　·　连续答对 %d 次　·　到期 %s\n    来源：%s"
                         % (d["question"][:80], d["wrong"], d["streak"], d["due"],
                            (d["source"] or "未标注")[:60]))
        self.wrong_view.setPlainText("\n".join(lines))

    def _start_wrong_review(self):
        due = due_review_items()
        if not due:
            InfoBar.info("今天没有到期错题", "可以先做一套题库答题，错题会自动进错题本",
                         duration=4000, position=InfoBarPosition.TOP, parent=self)
            return
        bank = {q.get("id"): q for q in load_question_bank()}
        qs = [bank[d["qid"]] for d in due if d["qid"] in bank]
        if not qs:
            return
        self.mode_bar.setCurrentItem("bank")
        self._switch_mode("bank")
        self.bank_session = qs[:self.SESSION_SIZE]
        self.bank_index = 0
        self._load_bank_question()
        log_event("wrong_review_start", n=len(self.bank_session))
        InfoBar.success("开始复习", "本轮 %d 道到期错题" % len(self.bank_session),
                        duration=3000, position=InfoBarPosition.TOP, parent=self)

    def _clear_wrong_book(self):
        clear_all_review()
        self._refresh_wrong_panel()
        self._refresh_teach_status()
        InfoBar.info("已清空错题本", "复习计划已重置", duration=2500,
                     position=InfoBarPosition.TOP, parent=self)

    # ── 生成与导入 ──
    def seed_from_local(self):
        """用本地防治库生成一套题库与案例（自有内容）"""
        qs, cs = seed_teaching_from_local()
        added_q, added_c = merge_into_bank(qs, cs)
        self._refresh_teach_status()
        log_event("teaching_seed", questions=added_q, cases=added_c)
        InfoBar.success("已生成教学资源",
                        "新增 %d 题、%d 条案例（来源：本地防治库）" % (added_q, added_c),
                        duration=5000, position=InfoBarPosition.TOP, parent=self)
        if self.mode == "bank":
            self._start_bank_quiz()
        elif self.mode == "case":
            self._search_cases_ui()

    def import_material(self):
        """从本地文件导入（PDF / Word / 文本 / CSV / JSON）"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择教学资料（请使用自有或公开合法资料）", "",
            "资料文件 (*.pdf *.docx *.txt *.md *.csv *.tsv *.json *.html);;所有文件 (*)")
        if not files:
            return
        self._run_import([{"path": f} for f in files])

    def import_from_url(self):
        """从网址导入（请确认来源合法并保留出处）"""
        url, ok = QInputDialog.getText(self, "从网址导入",
                                       "资料网址（请确认来源合法，题目会标注该网址）：")
        if not ok or not url.strip():
            return
        self._run_import([{"url": url.strip()}])

    def _run_import(self, jobs):
        self.status.setText("正在导入 %d 份资料…" % len(jobs))
        self.import_thread = _track_thread(TeachingImportThread(jobs))
        self.import_thread.item_done.connect(self._on_import_item)
        self.import_thread.all_done.connect(self._on_import_done)
        self.import_thread.start()

    def _on_import_item(self, res):
        src = (res.get("source") or "")[:48]
        if res.get("ok"):
            InfoBar.success("导入成功", "%s：新增 %d 题、%d 案例"
                            % (src, res.get("questions", 0), res.get("cases", 0)),
                            duration=4000, position=InfoBarPosition.TOP, parent=self)
        else:
            InfoBar.warning("导入未完成", "%s：%s" % (src, res.get("error", "")),
                            duration=6000, position=InfoBarPosition.TOP, parent=self)

    def _on_import_done(self, total):
        self._refresh_teach_status()
        self.status.setText("导入完成：共新增 %d 条教学资源" % total)
        if self.mode == "bank":
            self._start_bank_quiz()
        elif self.mode == "case":
            self._search_cases_ui()

    # ── 会话控制 ──
    def _set_options_enabled(self, on):
        for b in self.opts:
            b.setEnabled(on)

    def choose_pool(self):
        d = QFileDialog.getExistingDirectory(self, "选择题库文件夹（文件名需以类别英文名开头）")
        if not d:
            return
        self.pool = collect_question_pool([d])
        self.status.setText("题库 %d 张（来自 %s）" % (len(self.pool), os.path.basename(d)))

    def start_session(self):
        if not self.pool:
            InfoBar.warning("没有可用题目", "未找到带类别命名的图片，请用「换个题库」指定目录",
                            duration=5000, position=InfoBarPosition.TOP, parent=self)
            return
        n = min(self.SESSION_SIZE, len(self.pool))
        self._session_id = datetime.now().strftime("%m%d_%H%M")
        self.session = random.sample(self.pool, n)
        self.records = []
        self.index = 0
        self.btn_export.setEnabled(False)
        self.btn_start.setText("重新开始")
        log_event("training_start", questions=n, pool=len(self.pool))
        self._load_question()

    def _load_question(self):
        self._answered = False
        self.current = self.session[self.index]
        self.model_guess = None
        img = cv2.imread(self.current["path"])
        if img is None:
            self._next_question()
            return
        self.img.setPixmap(bgr_to_qpixmap(img).scaled(
            440, 330, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.progress.setText("第 %d / %d 题　·　来源：%s"
                              % (self.index + 1, len(self.session), self.current["source"]))
        self.feedback.setPlainText("")
        self._q_t0 = datetime.now()
        self._set_options_enabled(True)
        # 模型判断在后台算，判定后才展示，避免先看答案
        self.thread = _track_thread(InferenceThread(self.main.model, img))
        self.thread.result_ready.connect(self._on_model_result)
        self.thread.failed.connect(lambda m: log.warning("实训页模型推理失败: %s", m))
        self.thread.start()

    def _on_model_result(self, result_list):
        best = max(result_list, key=lambda x: x["conf"]) if result_list else None
        self.model_guess = (best["name"], round(best["conf"], 3)) if best else ("(无检出)", 0.0)

    def answer(self, choice):
        if self._answered or not self.current:
            return
        self._answered = True
        self._set_options_enabled(False)
        truth = self.current["label"]
        correct = (choice == truth)
        seconds = (datetime.now() - self._q_t0).total_seconds()
        guess, conf = self.model_guess or ("(模型未算完)", 0.0)
        model_correct = (guess == truth)
        rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "session": getattr(self, "_session_id", ""),
               "image": os.path.basename(self.current["path"]),
               "label": truth, "answer": choice, "correct": correct,
               "seconds": round(seconds, 1),
               "model_guess": guess, "model_conf": conf, "model_correct": model_correct}
        self.records.append(rec)
        append_train_record(**rec)
        log_event("training_answer", cls=truth, answer=choice, correct=correct,
                  seconds=round(seconds, 1), model=guess, model_correct=model_correct)

        tr = TREATMENT_DB.get(truth, {})
        parts = ["%s 你选的是「%s」，正确答案是「%s」" % ("✅ 答对了。" if correct else "❌ 答错了。",
                                                        choice, truth),
                 "模型判断：%s（%.0f%%）%s" % (guess, conf * 100,
                                              "，与正确答案一致" if model_correct else "，同样判断错误"),
                 "典型症状：%s" % tr.get("symptom", "（本地库暂无）"),
                 "防治指标：%s" % tr.get("threshold", "（本地库暂无）")]
        if tr.get("agri"):
            parts.append("农业防治要点：" + "；".join(tr["agri"][:2]))
        self.feedback.setPlainText("\n\n".join(parts))
        InfoBar.success("已记录", "第 %d/%d 题" % (self.index + 1, len(self.session)),
                        duration=1200, position=InfoBarPosition.TOP, parent=self)
        QTimer.singleShot(700, self._next_question)

    def _next_question(self):
        if self.index + 1 < len(self.session):
            self.index += 1
            self._load_question()
        else:
            self._finish()

    def _finish(self):
        self._set_options_enabled(False)
        s = training_summary(self.records)
        self.progress.setText("本轮结束：共 %d 题" % s.get("题数", 0))
        self.feedback.setPlainText(
            "本轮成绩单\n\n"
            "题数：%s\n人的正确率：%s%%\n模型正确率：%s%%\n平均用时：%s 秒\n"
            "错题类别：%s\n人和模型都错：%s 题\n\n"
            "教学提示：模型与人都错的那几类，正是最需要补数据/补实训的方向。"
            % (s.get("题数"), s.get("人的正确率"), s.get("模型正确率"), s.get("平均用时秒"),
               s.get("错题类别") or "无", s.get("人机都错")))
        self.btn_export.setEnabled(True)
        log_event("training_finish", **{k: v for k, v in s.items() if k != "错题类别"})
        InfoBar.success("本轮完成", "正确率 %s%%" % s.get("人的正确率"),
                        duration=4000, position=InfoBarPosition.TOP, parent=self)

    def export_sheet(self):
        """导出成绩单 CSV（可直接作为教学实践记录放进参赛支撑材料）"""
        if not self.records:
            return
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        path = TRAIN_DIR / ("实训成绩单_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        s = training_summary(self.records)
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["时间", "题型", "图片/来源", "正确答案", "学员选择", "是否正确",
                            "用时秒", "模型判断", "模型置信度", "模型是否正确", "出处"])
                for r in self.records:
                    mc = r.get("model_correct")
                    w.writerow([r["time"], r.get("kind", "image"), r["image"], r["label"],
                                r["answer"], "是" if r["correct"] else "否", r["seconds"],
                                r["model_guess"], r["model_conf"],
                                "" if mc is None else ("是" if mc else "否"),
                                r.get("source", "")])
                w.writerow([])
                w.writerow(["汇总", "题数", s.get("题数"), "人的正确率%", s.get("人的正确率"),
                            "模型正确率%", s.get("模型正确率"), "平均用时秒", s.get("平均用时秒")])
        except OSError as e:
            InfoBar.error("导出失败", str(e), duration=4000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已导出", os.path.basename(str(path)), duration=4000,
                        position=InfoBarPosition.TOP, parent=self)


class SettingInterface(QWidget):
    """API 设置：服务商 / Key / 模型 / 接口地址 / 看图能力。

    模型名可自由填写（EditableComboBox），也能接任意 OpenAI 兼容服务——
    换模型、换服务商都不用改代码。
    """

    PROVIDERS = [("deepseek", "DeepSeek"), ("qwen", "千问（阿里云）"),
                 ("custom", "自定义（OpenAI 兼容）")]
    MODELS = {
        "deepseek": ["deepseek-flash", "deepseek-v4-pro"],
        "qwen": ["qwen-vl-max", "qwen-vl-plus", "qwen-plus", "qwen-max"],
        "custom": [],
    }
    VISION_CHOICES = [("auto", "自动判断"), ("on", "强制：可看图"), ("off", "强制：纯文本")]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("settingInterface")
        self._pending_provider = CONFIG.get("llm_provider", "deepseek")
        self._pending_keys = {"deepseek": CONFIG.get("deepseek_api_key", ""),
                              "qwen": CONFIG.get("dashscope_api_key", ""),
                              "custom": CONFIG.get("custom_api_key", "")}
        self._pending_models = {"deepseek": CONFIG.get("deepseek_model", "deepseek-flash"),
                                "qwen": CONFIG.get("qwen_model", "qwen-vl-max"),
                                "custom": CONFIG.get("custom_model", "")}
        self._pending_base = CONFIG.get("custom_base_url", "")

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(6)
        root.addWidget(page_title("API 设置", self))
        root.addWidget(CaptionLabel("配置大模型服务商，用于识别结果页的智能分析", self))
        root.addSpacing(14)

        card = HeaderCardWidget("服务商与密钥", self)
        self.combo_provider = ComboBox(self)
        self.combo_provider.addItems([label for _, label in self.PROVIDERS])
        self.combo_provider.currentIndexChanged.connect(self._switch)
        self.key_input = PasswordLineEdit(self)
        self.key_input.setPlaceholderText("sk-...")
        self.combo_model = EditableComboBox(self)      # 可直接输入任意模型名
        self.combo_model.setPlaceholderText("模型名，例如 deepseek-flash")
        self.base_input = LineEdit(self)
        self.base_input.setPlaceholderText("https://api.example.com/v1（仅自定义服务需要）")
        self.combo_vision = ComboBox(self)
        self.combo_vision.addItems([label for _, label in self.VISION_CHOICES])
        self.switch_thinking = SwitchButton(self)
        self.switch_thinking.setOnText("深度思考")
        self.switch_thinking.setOffText("快速回答")
        self.switch_thinking.setToolTip("DeepSeek 思考模式：开=先出思维链再回答（更准但更慢更贵），关=直接回答")
        self.switch_enable = SwitchButton(self)
        self.switch_enable.setOnText("已启用")
        self.switch_enable.setOffText("已关闭")
        for label, widget in [("服务商", self.combo_provider), ("API Key", self.key_input),
                              ("模型", self.combo_model), ("接口地址", self.base_input),
                              ("看图能力", self.combo_vision), ("回答模式", self.switch_thinking),
                              ("云端 AI 分析", self.switch_enable)]:
            row = QHBoxLayout()
            lbl = BodyLabel(label, card)
            lbl.setFixedWidth(90)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            card.viewLayout.addLayout(row)
        self.lbl_cap = CaptionLabel("", card)          # 直接写出「能不能看图」，不用用户猜
        self.lbl_cap.setWordWrap(True)
        card.viewLayout.addWidget(self.lbl_cap)
        root.addWidget(card)
        self.combo_model.currentTextChanged.connect(lambda _=None: self._update_caption())
        self.combo_vision.currentIndexChanged.connect(lambda _=None: self._update_caption())
        # 注意：qfluentwidgets 的 SwitchButton 没有 toggled 信号，用的是 checkedChanged
        self.switch_thinking.checkedChanged.connect(lambda _=None: self._update_caption())

        row = QHBoxLayout()
        btn_save = PrimaryPushButton(FIF.SAVE, "保存设置", self)
        btn_save.setMinimumHeight(40)
        btn_save.clicked.connect(self._save)
        row.addWidget(btn_save)
        row.addStretch()
        root.addLayout(row)
        root.addStretch()
        self._load()

    # ── 内部：上下标转换 ──
    def _provider_key(self):
        idx = max(0, self.combo_provider.currentIndex())
        return self.PROVIDERS[min(idx, len(self.PROVIDERS) - 1)][0]

    def _vision_key(self):
        idx = max(0, self.combo_vision.currentIndex())
        return self.VISION_CHOICES[min(idx, len(self.VISION_CHOICES) - 1)][0]

    def _load(self):
        keys = [k for k, _ in self.PROVIDERS]
        idx = keys.index(self._pending_provider) if self._pending_provider in keys else 0
        self.combo_provider.setCurrentIndex(idx)
        vkeys = [k for k, _ in self.VISION_CHOICES]
        v = CONFIG.get("vision_override", "auto")
        self.combo_vision.setCurrentIndex(vkeys.index(v) if v in vkeys else 0)
        self._refresh()
        self.switch_thinking.setChecked(CONFIG.get("llm_thinking", False))
        self.switch_enable.setChecked(CONFIG.get("enable_llm", False))

    def _switch(self, idx):
        """切服务商：先把当前页面填的内容存进暂存区，再刷新成新服务商的"""
        p = self._pending_provider
        self._pending_keys[p] = self.key_input.text().strip()
        self._pending_models[p] = self.combo_model.currentText().strip()
        if p == "custom":
            self._pending_base = self.base_input.text().strip()
        self._pending_provider = self._provider_key()
        self._refresh()

    def _refresh(self):
        p = self._pending_provider
        self.key_input.setText(self._pending_keys.get(p, ""))
        self.key_input.setPlaceholderText("sk-..." if p != "custom" else "服务商给的 Key")
        self.base_input.setText(self._pending_base if p == "custom" else "")
        self.base_input.setEnabled(p == "custom")
        items = self.MODELS.get(p, [])
        self.combo_model.clear()
        if items:
            self.combo_model.addItems(items)
        self.combo_model.setCurrentText(self._pending_models.get(p, "") or (items[0] if items else ""))
        self._update_caption()

    def _update_caption(self):
        """把「这个模型能不能看图」直接写出来，省得用户猜"""
        p = self._provider_key()
        model = self.combo_model.currentText().strip()
        can = model_supports_vision(p, model, self._vision_key())
        mode = "深度思考（先思维链再回答，更慢更贵）" if self.switch_thinking.isChecked() \
            else "快速回答（直接出结果）"
        try:
            _cost = usage_cost_summary()
            _cost_txt = "　·　已调用 %d 次 · 约 %.3f 元" % (_cost["调用次数"],
                                                      _cost["估算费用元"] or 0.0)
        except Exception:
            _cost_txt = ""
        self.lbl_cap.setText(
            "当前：%s · %s · %s · %s%s" % (
                PROVIDER_LABELS.get(p, p), model or "未填模型",
                "可看图（照片会发给服务商）" if can else "纯文本（不发照片）", mode, _cost_txt))
        self.lbl_cap.setTextColor(NAU["subtext"], NAU["subtext"])

    def _save(self):
        p = self._provider_key()
        self._pending_keys[p] = self.key_input.text().strip()
        self._pending_models[p] = self.combo_model.currentText().strip()
        if p == "custom":
            self._pending_base = self.base_input.text().strip()
        CONFIG["llm_provider"] = p
        CONFIG["deepseek_api_key"] = self._pending_keys.get("deepseek", "")
        CONFIG["deepseek_model"] = self._pending_models.get("deepseek", "deepseek-flash")
        CONFIG["dashscope_api_key"] = self._pending_keys.get("qwen", "")
        CONFIG["qwen_model"] = self._pending_models.get("qwen", "qwen-vl-max")
        CONFIG["custom_api_key"] = self._pending_keys.get("custom", "")
        CONFIG["custom_model"] = self._pending_models.get("custom", "")
        CONFIG["custom_base_url"] = self._pending_base
        CONFIG["vision_override"] = self._vision_key()
        CONFIG["llm_thinking"] = self.switch_thinking.isChecked()
        CONFIG["enable_llm"] = self.switch_enable.isChecked()
        save_config(CONFIG)
        InfoBar.success("已保存", "当前：%s" % provider_label(), duration=3000,
                        position=InfoBarPosition.TOP, parent=self)


# ── 主窗口 ────────────────────────────────────────────────────
class ReportInterface(QWidget):
    """报告记录：导出的报告都登记在这里，随时可查/可打开"""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("reportInterface")

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(page_title("报告记录", self))
        head.addStretch()
        btn_dir = PushButton(FIF.FOLDER, "打开报告目录", self)
        btn_dir.clicked.connect(self._open_dir)
        head.addWidget(btn_dir)
        btn_refresh = PushButton(FIF.SYNC, "刷新", self)
        btn_refresh.clicked.connect(self.reload)
        head.addWidget(btn_refresh)
        btn_ledger = PushButton(FIF.CHART if hasattr(FIF, "CHART") else FIF.DOCUMENT,
                                "导出运行台账", self)
        btn_ledger.setToolTip("把检测次数、AI 调用、反馈、实训答题等数据汇总成 CSV")
        btn_ledger.clicked.connect(self._export_ledger)
        head.addWidget(btn_ledger)
        root.addLayout(head)
        root.addWidget(t_caption("每次导出报告都会自动登记在这里，双击卡片可打开", self))

        scroll, self.box = _scroll_body(self)
        root.addWidget(scroll, 1)
        self.reload()

    # ── 列表 ──
    def _clear(self):
        while self.box.count():
            item = self.box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def reload(self):
        self._clear()
        self._add_session_section()          # 先把「历史会话」放上去（Agent 的记忆入口）
        items = load_report_index()
        if not items:
            art = brand_pixmap("主楼剪影_淡金_透明.png", height=120)
            if art is not None:
                pic = ImageLabel(self)
                pic.setPixmap(art)
                pic.setAlignment(Qt.AlignCenter)
                self.box.addSpacing(30)
                self.box.addWidget(pic)
            tip = t_body("还没有导出过报告。在「结果与治理」页点「导出报告」，记录会自动出现在这里。",
                         self, muted=True)
            tip.setAlignment(Qt.AlignCenter)
            self.box.addWidget(tip)
            self.box.addStretch()
            return
        for it in items:
            self.box.addWidget(self._row(it))
        self.box.addStretch()

    def _add_session_section(self):
        """历史会话：点「继续」可以把那次识别连同问答历史一起载入，接着追问"""
        sessions = load_session_index()[:8]
        if not sessions:
            return
        self.box.addWidget(t_block("历史会话（可载入继续追问）", self))
        for it in sessions:
            card = SimpleCardWidget(self)
            card.setFixedHeight(64)
            lay = QHBoxLayout(card)
            lay.setContentsMargins(12, 8, 12, 8)
            lay.setSpacing(12)
            sid = it.get("id", "")
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(_sized(StrongBodyLabel(
                "%s　·　%s" % (it.get("cls") or "未识别", it.get("severity") or "—"), card),
                "body", True))
            col.addWidget(t_caption("%s　·　问答 %s 轮%s" % (
                it.get("time", ""), it.get("turns", 0),
                ("　·　诊断卡药剂 " + "、".join(it.get("card_chemicals") or []))
                if it.get("card_chemicals") else ""), card, color=NAU["subtext"]))
            lay.addLayout(col, 1)
            b = PushButton(FIF.HISTORY, "载入", card)
            b.clicked.connect(lambda _, s=sid: self.main.load_session(s))
            lay.addWidget(b)
            self.box.addWidget(card)
        self.box.addSpacing(10)

    def _row(self, it):
        path = it.get("path", "")
        exists = os.path.exists(path)
        card = SimpleCardWidget(self)
        card.setFixedHeight(118)          # 不定高的话卡片会被拉伸得高矮不一
        lay = QHBoxLayout(card)
        lay.setContentsMargins(14, 12, 16, 12)
        lay.setSpacing(16)

        thumb = it.get("thumb") or ""
        pic = ImageLabel(card)
        pic.setFixedSize(64, 92)
        pic.setAlignment(Qt.AlignCenter)
        if thumb and os.path.exists(thumb):
            pic.setPixmap(QPixmap(thumb).scaled(64, 92, Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))
        else:
            pic.setText("PDF" if path.lower().endswith(".pdf") else "—")
        lay.addWidget(pic)

        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(_sized(StrongBodyLabel(it.get("cls_name") or "识别报告", card), "body", True))
        col.addWidget(t_caption(f"置信度 {it.get('conf', 0) * 100:.1f}%　·　{it.get('time', '')}"
                                f"　·　{it.get('device', '')}", card, color=NAU["subtext"]))
        pl = t_caption(path, card, color=NAU["muted"])
        pl.setWordWrap(True)
        col.addWidget(pl)
        col.addStretch()
        lay.addLayout(col, 1)

        if exists:
            b_open = PushButton(FIF.VIEW, "打开", card)
            b_open.clicked.connect(lambda _, p=path: self._open(p))
            b_dir = PushButton(FIF.FOLDER, "定位", card)
            b_dir.clicked.connect(lambda _, p=path: self._reveal(p))
        else:
            gone = t_caption("文件已被移动或删除", card, color=NAU["red"])
            lay.addWidget(gone)
        b_del = PushButton(getattr(FIF, "DELETE", FIF.CANCEL), "移除", card)
        b_del.clicked.connect(lambda _, p=path: self._forget(p))
        if exists:
            lay.addWidget(b_open)
            lay.addWidget(b_dir)
        lay.addWidget(b_del)
        return card

    # ── 操作 ──
    def _open(self, path):
        try:
            os.startfile(path)          # noqa: S606  Windows 默认程序打开
        except Exception as e:
            InfoBar.error("打开失败", str(e), duration=4000,
                          position=InfoBarPosition.TOP, parent=self)

    def _reveal(self, path):
        try:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        except Exception as e:
            InfoBar.error("定位失败", str(e), duration=4000,
                          position=InfoBarPosition.TOP, parent=self)

    def _open_dir(self):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self._open(str(REPORT_DIR))

    def _export_ledger(self):
        """运行台账：赛后统计与支撑材料直接引用"""
        try:
            path, s = export_usage_ledger()
        except Exception as e:
            log.exception("导出运行台账失败")
            InfoBar.error("导出失败", str(e), duration=5000,
                          position=InfoBarPosition.TOP, parent=self)
            return
        InfoBar.success("已导出", "检测 %s 次 · AI 分析 %s 次 · 实训 %s 题"
                        % (s["检测次数"], s["AI 分析次数"], s["实训答题数"]),
                        duration=5000, position=InfoBarPosition.TOP, parent=self)
        self._open(str(path))

    def _forget(self, key):
        """只移除记录，不删文件"""
        save_report_index([x for x in load_report_index() if x.get("path") != key])
        self.reload()


class ScienceInterface(QWidget):
    """病害科普：可旋转的 3D 水稻植株 + 按病害高亮受害部位。

    3D 部分是自己写的软件渲染器（rice_3d.py），零依赖 ——
    用的是程序化生成的通用植株模型，定位是科普讲解，不是诊断数据的三维重建。
    """

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("scienceInterface")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 12)
        root.setSpacing(8)

        head = QHBoxLayout()
        head.addWidget(page_title("病害科普", self))
        head.addStretch()
        root.addLayout(head)
        root.addWidget(t_caption(
            "拖动旋转 · 滚轮缩放 · 双击暂停自动旋转　　红色高亮处 = 该病害主要侵害的部位", self))

        body = QHBoxLayout()
        body.setSpacing(12)

        # 左：3D 视图
        left = SimpleCardWidget(self)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        from rice_3d import DISEASE_PARTS
        self.web = None
        self.view3d = None
        self.html_3d = APP_DIR / "3d_viewer.html"
        # QtWebEngine 内嵌默认关闭：它在没有 OpenGL 的环境里初始化 Chromium 时会
        # 直接访问冲突崩溃（C++ 层，try/except 拦不住），无法保证每台机器都安全。
        # 想试内嵌就把环境变量 RICE_EMBED_3D 设为 1；否则用已验证的软件渲染器，
        # 需要高清 Three.js 效果时点「高清 3D」按钮用系统浏览器打开。
        if os.environ.get("RICE_EMBED_3D") == "1" and self.html_3d.exists():
            try:
                from PyQt5.QtWebEngineWidgets import QWebEngineView
                from PyQt5.QtCore import QUrl
                self.web = QWebEngineView(left)
                self.web.load(QUrl.fromLocalFile(str(self.html_3d)))
                ll.addWidget(self.web, 1)
            except Exception as e:
                log.warning("QtWebEngine 不可用（%s），回退软件渲染", e)
                self.web = None
        if self.web is None:
            from rice_3d import Rice3DView
            self.view3d = Rice3DView(left)
            ll.addWidget(self.view3d, 1)
        body.addWidget(left, 3)

        # 右：病害选择 + 说明
        right = SimpleCardWidget(self)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(6)
        btn_hd = PushButton("高清 3D", right)
        btn_hd.setToolTip("用系统浏览器打开 Three.js 高清场景（模型来自 Sketchfab，CC-BY-4.0）")
        btn_hd.clicked.connect(self._open_3d)
        rl.addWidget(btn_hd)
        rl.addSpacing(4)
        rl.addWidget(t_block("选择病害", right))
        self.disease_parts = DISEASE_PARTS
        self.btns = {}
        for name in DISEASE_PARTS:
            b = PushButton(name, right)
            b.clicked.connect(lambda _=False, n=name: self._select(n))
            rl.addWidget(b)
            self.btns[name] = b
        rl.addSpacing(8)
        rl.addWidget(t_block("症状要点", right))
        self.lbl_sym = t_body("", right)
        self.lbl_sym.setWordWrap(True)
        rl.addWidget(self.lbl_sym)
        rl.addStretch()
        body.addWidget(right, 2)
        root.addLayout(body, 1)

        self._select(next(iter(DISEASE_PARTS)))

    def _open_3d(self):
        """用系统浏览器打开 Three.js 场景"""
        if not self.html_3d.exists():
            InfoBar.warning("找不到 3D 场景", "缺少 3d_viewer.html", duration=4000,
                            position=InfoBarPosition.TOP, parent=self)
            return
        try:
            os.startfile(str(self.html_3d))
        except Exception as e:
            InfoBar.error("打开失败", str(e), duration=4000,
                          position=InfoBarPosition.TOP, parent=self)

    def _select(self, name):
        parts, color, desc = self.disease_parts[name]
        if self.web is not None:
            js = "window.setDisease && window.setDisease(%s);" % (
                __import__("json").dumps(name, ensure_ascii=False))
            self.web.page().runJavaScript(js)
        if self.view3d is not None:
            self.view3d.set_highlight(list(parts), color)
        self.lbl_sym.setText(desc)
        for n, b in self.btns.items():
            b.setEnabled(n != name)


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.model, self.device = get_model()
        setTheme(Theme.LIGHT)
        setThemeColor(NAU["primary"])

        self.homeInterface = HomeInterface(self)
        self.imageInterface = ImageInterface(self)
        self.cameraInterface = CameraInterface(self)
        self.watchInterface = WatchInterface(self)
        self.mapInterface = MapInterface(self)
        self.treatmentInterface = TreatmentInterface(self)
        self.trainInterface = TrainInterface(self)
        self.resultInterface = ResultInterface(self)
        self.reportInterface = ReportInterface(self)
        self.scienceInterface = ScienceInterface(self)
        self.settingInterface = SettingInterface(self)

        self.addSubInterface(self.homeInterface, FIF.HOME, "首页")
        self.addSubInterface(self.imageInterface, FIF.PHOTO, "图片识别")
        self.addSubInterface(self.cameraInterface, FIF.CAMERA, "实时识别")
        self.addSubInterface(self.watchInterface, FIF.ROBOT, "自动巡检")
        self.addSubInterface(self.mapInterface, FIF.GLOBE, "地块地图")
        self.addSubInterface(self.treatmentInterface, getattr(FIF, "LIBRARY", FIF.DOCUMENT), "防治库")
        self.addSubInterface(self.trainInterface, FIF.EDUCATION, "教学与考核")
        self.addSubInterface(self.resultInterface, FIF.VIEW, "结果与治理")
        self.addSubInterface(self.reportInterface, FIF.DOCUMENT, "报告记录")
        _ico = getattr(FIF, "BOOK_SHELF", None) or FIF.EDUCATION
        self.addSubInterface(self.scienceInterface, _ico, "病害科普")
        self.addSubInterface(self.settingInterface, FIF.SETTING, "API 设置",
                             NavigationItemPosition.BOTTOM)

        self.navigationInterface.setExpandWidth(200)
        self.setMinimumSize(1120, 720)
        self.setWindowTitle("稻视界 · 水稻病虫害智能诊断系统")
        self._confirmed_exit = False
        self._restore_state()
        self._init_shortcuts()
        # 定时自主任务：每分钟检查一次；启动 5 秒后也查一次（补上运行期间错过的）
        self._schedule_timer = QTimer(self)
        self._schedule_timer.setInterval(60_000)
        self._schedule_timer.timeout.connect(self._check_schedules)
        self._schedule_timer.start()
        QTimer.singleShot(5000, self._check_schedules)

    def showEvent(self, event):
        """首次显示后再展开侧栏。

        在 __init__ 里 expand() 只会改侧栏宽度，内容区不会重新布局，
        结果整个页面被展开的侧栏盖住（实测踩过）。
        """
        super().showEvent(event)
        if not getattr(self, "_nav_expanded", False):
            self._nav_expanded = True
            QTimer.singleShot(0, self._expand_nav)

    def _expand_nav(self):
        """默认展开侧栏：页面名一直可见，不用悬停才想起来（Nielsen #6 识别优于回忆）"""
        try:
            self.navigationInterface.expand(useAni=False)
        except Exception:
            pass

    # ── 桌面程序规范：窗口大小/位置跨会话保持 ───────────────
    def _restore_state(self):
        geo = SETTINGS.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1380, 900)
        if SETTINGS.value("window/maximized", False, bool):
            self.showMaximized()

    def _save_state(self):
        SETTINGS.setValue("window/geometry", self.saveGeometry())
        SETTINGS.setValue("window/maximized", self.isMaximized())
        SETTINGS.sync()      # 立刻落盘，别等进程退出

    # ── 人因工程：全流程可纯键盘完成（Fitts/Hick 之外的硬指标）──
    def _init_shortcuts(self):
        pages = [("Ctrl+1", self.homeInterface), ("Ctrl+2", self.imageInterface),
                 ("Ctrl+3", self.cameraInterface), ("Ctrl+4", self.resultInterface),
                 ("Ctrl+5", self.settingInterface), ("Ctrl+6", self.trainInterface),
                 ("Ctrl+7", self.watchInterface),
            ("Ctrl+8", self.mapInterface),
            ("Ctrl+9", self.treatmentInterface)]
        for key, iface in pages:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda i=iface: self.go(i))
        for key, slot in [("Ctrl+O", self.imageInterface.pick),
                          ("Ctrl+R", self.imageInterface.run_inference),
                          ("Ctrl+Q", self.close),
                          ("F1", self.show_about),
                          ("Esc", self._on_escape)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)

    def _on_escape(self):
        """Esc = 取消/返回，桌面程序通用语义"""
        if self.cameraInterface.cap is not None:
            self.cameraInterface.close_camera()
            InfoBar.info("已停止", "摄像头已关闭", duration=2000,
                         position=InfoBarPosition.TOP, parent=self)
        elif self.stackedWidget.currentWidget() is not self.homeInterface:
            self.go(self.homeInterface)

    # ── 标准"关于"对话框：版本 / 模型 / 开源许可 / 快捷键 ────
    def show_about(self):
        dev = "GPU (CUDA)" if self.device != "cpu" else "CPU"
        text = (
            f"<b>水稻病虫害智能识别系统</b>　v{APP_VERSION}<br>"
            "南京农业大学 · 毕业设计<br><br>"
            f"检测模型：YOLOv8n（{MODEL_INFO['训练数据集']}）<br>"
            f"参数量 {MODEL_INFO['Params']}　计算量 {MODEL_INFO['FLOPs']}　设备 {dev}<br>"
            f"识别类别：{'、'.join(CLASS_NAMES)}<br><br>"
            "<b>开源组件与许可</b><br>"
            "· PyQt5 — GPL-3.0<br>"
            "· PyQt-Fluent-Widgets — GPL-3.0<br>"
            "· Lucide 图标 — ISC<br>"
            "· Noto Sans SC 字体 — SIL OFL 1.1<br><br>"
            "<b>农药合规</b><br>"
            f"· 禁用农药名单：农业农村部公告第 1039 号（2026-07-13 生效，"
            f"{len(PESTICIDE_CONTROL['banned']['names'])} 个品种）<br>"
            "· 停止使用品种：氟虫腈（三部门公告第 1157 号）<br>"
            "· AI 回答与本地防治库都会按该名单自动核验，命中即标红<br><br>"
            "<b>快捷键</b><br>"
            "Ctrl+1~7 切换页面　Ctrl+O 打开图片　Ctrl+R 开始识别<br>"
            "Esc 停止/返回　F1 关于　Ctrl+Q 退出"
        )
        box = MessageBox("关于", text, self)
        box.yesButton.setText("确定")
        box.cancelButton.hide()
        box.exec()

    def go(self, interface):
        """切页：离开摄像头页时释放设备（自动巡检在后台继续跑，不因切页停止）"""
        if self.stackedWidget.currentWidget() is self.cameraInterface \
                and interface is not self.cameraInterface:
            self.cameraInterface.leave()
        if hasattr(self, "switchTo"):
            self.switchTo(interface)
        else:
            self.stackedWidget.setCurrentWidget(interface)

    def show_result(self, result_list, image, mode):
        self.resultInterface.show_result(result_list, image, mode)
        self.go(self.resultInterface)

    # ── 定时任务调度 ──
    def _check_schedules(self):
        try:
            due = due_schedules()
        except Exception as e:
            log.warning("检查定时任务失败: %s", e)
            return
        for item, when in due:
            mark_schedule_run(item.get("id", ""))     # 先标记，避免重复触发
            log.info("定时任务触发：%s（%s，计划时刻 %s）",
                     item.get("name"), item.get("action"), when)
            log_event("schedule_trigger", id=item.get("id"), action=item.get("action"))
            worker = _track_thread(ScheduleWorker(item.get("action")))
            worker.done.connect(
                lambda a, t: self.watchInterface.show_schedule_result(a, t))
            worker.failed.connect(
                lambda a, e: InfoBar.error("定时任务失败", "%s：%s" % (a, e), duration=6000,
                                           position=InfoBarPosition.TOP, parent=self))
            worker.start()

    def load_session(self, sid):
        """从报告记录页载入一次历史会话（记忆）"""
        self.go(self.resultInterface)
        self.resultInterface.restore_session(sid)

    def closeEvent(self, event):
        # 防错原则（Nielsen #5）：可能丢失正在进行的工作时先确认
        busy = self.cameraInterface.cap is not None or self.resultInterface._ai_busy
        if busy and not self._confirmed_exit:
            box = MessageBox("确认退出", "摄像头仍在运行或 AI 正在分析，确定要退出吗？", self)
            box.yesButton.setText("退出")
            box.cancelButton.setText("取消")
            if not box.exec():
                event.ignore()
                return
            self._confirmed_exit = True
        self._save_state()
        self.cameraInterface.leave()
        self.watchInterface.leave()      # 关程序时停掉巡检线程
        _wait_active_threads()
        super().closeEvent(event)


# ── 主入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 全局异常捕获，防止闪退
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        detail = traceback.format_exc()
        log.error("未捕获异常: %s\n%s", exc_value, detail)
        # QMessageBox 只能在 GUI 线程创建；工作线程的异常由线程自己的 failed 信号处理
        app_inst = QApplication.instance()
        if app_inst is not None and QThread.currentThread() is app_inst.thread():
            QMessageBox.critical(None, "程序异常",
                f"{exc_type.__name__}: {exc_value}\n\n{detail[-500:]}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    log.info("程序启动 | 资源目录: %s | 数据目录: %s", BASE_DIR, APP_DIR)
    if not os.path.exists(MODEL_PATH):
        log.error("模型文件缺失: %s", MODEL_PATH)
    os.makedirs(APP_DIR / "resources", exist_ok=True)
    init_treatment_dirs()
    # 农药合规自检：本地防治库里不许出现国家禁用/停止使用的品种，启动时就喊出来
    _bad_drugs = audit_treatment_db()
    if _bad_drugs:
        for _cls, _drug, _hit in _bad_drugs:
            log.error("合规自检未通过：%s 的药剂「%s」命中禁用名单 %s", _cls, _drug, _hit)
    else:
        log.info("农药合规自检通过：本地防治库无禁用/停用品种")

    # 高 DPI 属性必须在 QApplication 之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    # QtWebEngine 要求：必须在 QApplication 之前设置共享 OpenGL 上下文
    try:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    except Exception:
        pass
    app = QApplication(sys.argv)
    _register_bundled_fonts()          # 注册随程序分发的开源字体
    app.setFont(ui_font(TYPE["body"]))
    try:                               # 让 Fluent 控件也用同一套字体
        from qfluentwidgets import qconfig
        qconfig.set(qconfig.fontFamilies, font_stack(ui_family()))
    except Exception as e:
        log.warning("设置 Fluent 字体失败（不影响运行）: %s", e)

    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    else:
        log.warning("图标文件缺失，使用系统默认图标: %s", ICON_PATH)
    app.aboutToQuit.connect(_wait_active_threads)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
