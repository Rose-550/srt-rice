import sys
import os
import json
import logging
import warnings
import base64
import re
import markdown
import cv2
import torch
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
                             QFileDialog, QMessageBox, QProgressBar, QTextEdit,
                             QGraphicsDropShadowEffect, QScrollArea, QFrame,
                             QLineEdit, QDialog, QComboBox, QCheckBox)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QDragEnterEvent, QDropEvent,
                         QColor, QPainterPath, QRegion, QIcon)
from PyQt5.QtCore import (Qt, QTimer, QThread, pyqtSignal, QVariantAnimation,
                          QEasingCurve, QRectF, QPoint)

warnings.filterwarnings("ignore")

# ── 地块地图 / 溯源记录模块（照片时间戳 + GPS → 地图点位） ──────
# rice_geo.py 与本文件同目录；缺失时只影响地图功能，不影响识别主流程
try:
    import rice_geo
    GEO_AVAILABLE = True
except Exception as _geo_err:      # pragma: no cover
    rice_geo = None
    GEO_AVAILABLE = False
    print("[rice] 地块地图模块加载失败：%s" % _geo_err)

# ── 路径解析（兼容 PyInstaller 打包 与 源码直接运行） ─────────────
# BASE_DIR：只读资源根目录。打包后指向解压临时目录 _MEIPASS，源码运行时指向脚本所在目录。
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# APP_DIR：可写目录（配置文件、用户后续自行补充的图片）。打包后指向 exe 所在目录。
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parent

# ── 日志系统 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rice_app")


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


# ====================== 【一键配置区】修改这里即可 ======================
MODEL_PATH = res_path("best.pt")
CLASS_NAMES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]
# 置信度阈值：0.4（2026-10-09 由 0.6 下调。干净 val 实测：0.6 时白叶枯病召回 0.506、稻瘟病 0.685；
# 0.4 时分别为 0.652 / 0.832，总体 F1 0.921（0.6 时 0.904）。漏检是本项目最危险的错误类型。）
CONF_THRESHOLD = 0.4
ICON_PATH = res_path("resources/icon.ico")
BG_HOME = res_path("resources/bg_home.png")
BG_RECOG = res_path("resources/bg_recog.png")
MODEL_INFO = {
    "FLOPs": "8.1 GFLOPs",
    "Params": "3.0 M",
    "Input Size": "640x640",
    "训练数据集": "NAU水稻病虫害数据集",
    "模型版本": "YOLOv8n"
}
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
FONT_SIZE = 30
# ===========================================================================

# 把识别类别与置信度阈值同步给地图模块（导入照片时用它做自动识别）
if GEO_AVAILABLE:
    rice_geo.CLASS_NAMES = list(CLASS_NAMES)
    rice_geo.CONF_THRESHOLD = CONF_THRESHOLD

# ── 治理方案数据库（农户友好版） ──────────────────────────────
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
            {"name": "氟虫腈（锐劲特）", "spec": "5% SC", "dose": "30-40ml/亩", "img": "resources/treatment/稻水象甲/drug_1.jpg"},
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


def make_image_card(image_path, label_text, w=180, h=140):
    """创建一个图片卡片组件（图片+标题）"""
    card = QWidget()
    card.setStyleSheet(f"background-color: {NAU['card']}; border-radius: 20px; border: 1px solid {NAU['border']};")
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
        placeholder = QLabel("📷\n待添加图片")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setFixedSize(w, h)
        placeholder.setFont(QFont("Microsoft YaHei", 11))
        placeholder.setStyleSheet(f"""
            background-color: {NAU['primary_light']};
            border-radius: 14px;
            color: {NAU['subtext']};
            border: 2px dashed {NAU['border']};
        """)
        layout.addWidget(placeholder, alignment=Qt.AlignCenter)

    name_label = QLabel(label_text)
    name_label.setFont(QFont("Microsoft YaHei", 11))
    name_label.setAlignment(Qt.AlignCenter)
    name_label.setWordWrap(True)
    name_label.setMaximumWidth(w + 20)
    name_label.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
    layout.addWidget(name_label, alignment=Qt.AlignCenter)

    return card

# ── 南农大官方配色（从校徽 & 辅助图形中提取） ──────────────────
NAU = {
    "primary": "#00783C",        # PDF主色 - 翠绿 (0,120,60)
    "primary_hover": "#006433",  # 深绿 - 悬停态 (0,100,60)
    "primary_light": "#E6F0EA",  # 浅绿 - 选中/背景高亮
    "accent_gold": "#F0A014",    # PDF金色 - 高亮/装饰 (240,160,20)
    "accent_blue": "#006080",    # 科技蓝 - 链接/次要强调
    "bg": "#F0F0F0",            # PDF背景 - 纯浅灰 (240,240,240)
    "card": "#FFFFFF",           # 白色卡片
    "dark": "#1A3A28",          # 深绿黑 - 标题文字
    "text": "#2D3A30",          # 正文色
    "subtext": "#647878",       # PDF灰蓝 - 辅助文字 (100,120,120)
    "border": "#DCDCDC",        # PDF边框 - 浅灰 (220,220,220)
    "red": "#C0392B",           # 关闭/危险操作
    "warning": "#E67E22",       # 警告色
    "shadow": QColor(0, 120, 60, 30),  # 基于PDF主色的阴影
}

# ── LLM API 配置 ──────────────────────────────────────────
# 配置文件放在可写目录（打包后为 exe 同级），避免写进只读的临时解压目录
CONFIG_PATH = str(APP_DIR / "config.json")
DEFAULT_CONFIG = {
    "llm_provider": "deepseek",
    "deepseek_api_key": "",
    "deepseek_model": "deepseek-chat",
    "dashscope_api_key": "",
    "qwen_model": "qwen-vl-max",
    "enable_llm": False,
    "llm_max_tokens": 2000,
    # 高德地图（可选）：Web服务 Key 用于地址解析；JS API Key 用于导出的网页版地图
    "amap_web_key": "",
    "amap_js_key": "",
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(saved)
            log.info("已加载配置 %s（LLM 开关: %s）", CONFIG_PATH, cfg.get("enable_llm"))
            return cfg
        except Exception as e:
            log.warning("读取配置失败，改用默认配置: %s", e)
    else:
        log.info("未发现 config.json，使用默认配置")
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        log.info("配置已保存到 %s", CONFIG_PATH)
    except OSError as e:
        log.error("保存配置失败: %s", e)

CONFIG = load_config()

def get_api_key():
    if CONFIG["llm_provider"] == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY") or CONFIG["deepseek_api_key"]
    else:
        return os.environ.get("DASHSCOPE_API_KEY") or CONFIG["dashscope_api_key"]

def get_model_name():
    return CONFIG["deepseek_model"] if CONFIG["llm_provider"] == "deepseek" else CONFIG["qwen_model"]

class LLMAnalysisThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, cls_name, image=None, history=None):
        super().__init__()
        self.cls_name = cls_name
        self.image = image  # OpenCV BGR numpy array, 可选
        self.history = history  # 对话历史 [{"role":"user"/"assistant","content":...}], 可选

    def _build_prompt(self):
        return (
            f"你是水稻病虫害防治专家。检测到【{self.cls_name}】。\n"
            f"1. 确认诊断：该病虫害的典型症状\n"
            f"2. 严重程度判断\n"
            f"3. 3条最有效的防治建议\n"
            f"4. 推荐1-2种药剂及用量\n"
            f"请用通俗中文详细回答。"
        )

    def _image_to_base64(self, img_bgr):
        """OpenCV BGR 图像 → 压缩后 base64 JPEG（最长边不超过 768px）"""
        try:
            import cv2
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
        system_msg = {"role": "system", "content": "你是水稻病虫害防治专家，请用通俗中文回答用户问题。"}
        provider = CONFIG["llm_provider"]
        msgs = [system_msg]

        # 追加对话历史
        if self.history:
            msgs.extend(self.history)

        # 构建当前用户消息
        if provider == "qwen" and self.image is not None:
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

    def run(self):
        if not CONFIG["enable_llm"] or not get_api_key():
            self.error.emit("AI 分析未启用，请在设置中配置 API Key")
            return
        try:
            provider = CONFIG["llm_provider"]
            if provider == "deepseek":
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"},
                    json={
                        "model": get_model_name(),
                        "messages": self._build_messages(),
                        "max_tokens": CONFIG["llm_max_tokens"],
                        "temperature": 0.3
                    },
                    timeout=60
                )
            else:
                resp = requests.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    headers={"Authorization": f"Bearer {get_api_key()}", "Content-Type": "application/json"},
                    json={
                        "model": get_model_name(),
                        "messages": self._build_messages(),
                        "max_tokens": CONFIG["llm_max_tokens"],
                        "temperature": 0.3
                    },
                    timeout=60
                )
            if resp.status_code == 200:
                data = resp.json()
                self.finished.emit(data["choices"][0]["message"]["content"])
            else:
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text[:300]
                self.error.emit(f"API 错误 [{resp.status_code}]: {detail}")
        except requests.exceptions.Timeout:
            self.error.emit("API 请求超时，请检查网络")
        except Exception as e:
            self.error.emit(f"异常: {str(e)}")


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


def center_window(window, w, h):
    """将窗口置于屏幕中央"""
    screen = QApplication.primaryScreen().availableGeometry()
    x = (screen.width() - w) // 2
    y = (screen.height() - h) // 2
    window.setGeometry(x, y, w, h)


def make_rounded(window, radius=32, title=""):
    """将窗口设置为无边框抗锯齿圆角，带完整标题栏控件：
       左上: [←后退] [前进→]    中间: 标题    右上: [─最小化] [⛶全屏/窗口] [✕关闭]
    """
    window.setWindowFlags(Qt.FramelessWindowHint)
    window.setAttribute(Qt.WA_TranslucentBackground)
    window._drag_pos = None

    old_central = window.centralWidget()
    old_central.setParent(None)
    old_central.setStyleSheet("background: transparent;")

    outer = QWidget()
    outer.setStyleSheet("background: transparent;")
    outer_layout = QVBoxLayout(outer)
    outer_layout.setContentsMargins(6, 6, 6, 6)
    outer_layout.setSpacing(0)

    inner = QWidget()
    inner.setObjectName("roundedBg")
    inner.setStyleSheet(f"""
        #roundedBg {{
            background-color: {NAU['bg']};
            border-radius: {radius}px;
        }}
    """)
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setSpacing(0)

    # ── 标题栏 ──
    title_bar = QWidget()
    title_bar.setFixedHeight(44)
    title_bar.setStyleSheet("background: transparent; border: none;")
    tb_layout = QHBoxLayout(title_bar)
    tb_layout.setContentsMargins(14, 6, 10, 6)
    tb_layout.setSpacing(4)

    # ── 左上：后退 / 前进 ──
    btn_style = f"""
        QPushButton {{
            background-color: transparent; color: {NAU['subtext']};
            border: 1px solid {NAU['border']}; border-radius: 14px;
            font-size: 14px; padding: 2px 8px;
        }}
        QPushButton:hover {{ background-color: {NAU['border']}; color: {NAU['dark']}; }}
        QPushButton:disabled {{ color: {NAU['border']}; border-color: transparent; }}
    """

    window._btn_back = QPushButton("←")
    window._btn_back.setFont(QFont("Microsoft YaHei", 14))
    window._btn_back.setFixedSize(42, 32)
    window._btn_back.setCursor(Qt.PointingHandCursor)
    window._btn_back.setStyleSheet(btn_style)
    window._btn_back.setEnabled(nav_can_back())
    window._btn_back.clicked.connect(lambda: nav_go_back(window))
    tb_layout.addWidget(window._btn_back)

    window._btn_fwd = QPushButton("→")
    window._btn_fwd.setFont(QFont("Microsoft YaHei", 14))
    window._btn_fwd.setFixedSize(42, 32)
    window._btn_fwd.setCursor(Qt.PointingHandCursor)
    window._btn_fwd.setStyleSheet(btn_style)
    window._btn_fwd.setEnabled(nav_can_forward())
    window._btn_fwd.clicked.connect(lambda: nav_go_forward(window))
    tb_layout.addWidget(window._btn_fwd)

    tb_layout.addStretch()

    # ── 中间：标题 ──
    title_lbl = QLabel(title)
    title_lbl.setFont(QFont("Microsoft YaHei", 11))
    title_lbl.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
    tb_layout.addWidget(title_lbl)

    tb_layout.addStretch()

    # ── 右上：最小化 / 全屏切换 / 关闭 ──
    right_btn_style = f"""
        QPushButton {{
            background-color: transparent; color: {NAU['subtext']};
            border: none; border-radius: 8px;
        }}
        QPushButton:hover {{ background-color: {NAU['border']}; }}
    """

    # 最小化
    btn_min = QPushButton("─")
    btn_min.setFont(QFont("Microsoft YaHei", 14))
    btn_min.setFixedSize(36, 30)
    btn_min.setCursor(Qt.PointingHandCursor)
    btn_min.setStyleSheet(right_btn_style)
    btn_min.clicked.connect(window.showMinimized)
    tb_layout.addWidget(btn_min)

    # 全屏/窗口切换
    window._btn_fs = QPushButton("⛶")
    window._btn_fs.setFont(QFont("Microsoft YaHei", 12))
    window._btn_fs.setFixedSize(36, 30)
    window._btn_fs.setCursor(Qt.PointingHandCursor)
    window._btn_fs.setStyleSheet(right_btn_style)
    window._btn_fs.clicked.connect(lambda: QTimer.singleShot(0, lambda: _toggle_fullscreen(window)))
    tb_layout.addWidget(window._btn_fs)

    # 关闭
    btn_close = QPushButton("✕")
    btn_close.setFont(QFont("Microsoft YaHei", 14))
    btn_close.setFixedSize(36, 30)
    btn_close.setCursor(Qt.PointingHandCursor)
    btn_close.setStyleSheet(f"""
        QPushButton {{
            background-color: transparent; color: {NAU['subtext']};
            border: none; border-radius: 8px;
        }}
        QPushButton:hover {{ background-color: {NAU['red']}; color: white; }}
    """)
    btn_close.clicked.connect(QApplication.quit)
    tb_layout.addWidget(btn_close)

    inner_layout.addWidget(title_bar)
    inner_layout.addWidget(old_central)
    outer_layout.addWidget(inner)
    window.setCentralWidget(outer)

    # 全屏按钮初始文字
    window._btn_fs.setText("🗗" if window.isFullScreen() else "🗖")

    # 窗口拖动
    def _start_drag(event):
        if event.button() == Qt.LeftButton:
            window._drag_pos = event.globalPos() - window.frameGeometry().topLeft()

    def _do_drag(event):
        if event.buttons() == Qt.LeftButton and window._drag_pos is not None:
            window.move(event.globalPos() - window._drag_pos)

    def _end_drag(event):
        window._drag_pos = None

    outer.mousePressEvent = lambda e: _start_drag(e)
    outer.mouseMoveEvent = lambda e: _do_drag(e)
    outer.mouseReleaseEvent = lambda e: _end_drag(e)

    return window


def _toggle_fullscreen(window):
    if window.isFullScreen():
        window.showNormal()
        window._btn_fs.setText("🗖")
    else:
        window.showFullScreen()
        window._btn_fs.setText("🗗")


def card_style():
    """iOS 风格卡片样式"""
    return f"""
        background-color: {NAU['card']};
        border-radius: 40px;
        border: 1px solid {NAU['border']};
    """


def make_shadow(blur=12, offset_y=3):
    """创建 iOS 风格阴影效果"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, offset_y)
    shadow.setColor(NAU["shadow"])
    return shadow


_FONT_CANDIDATES = [
    FONT_PATH,
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def _load_font(size):
    """按候选顺序加载中文字体，全部失败时退回 PIL 默认字体"""
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
    """
    annotated = img_bgr.copy()
    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(FONT_SIZE)

    for item in boxes:
        x1, y1, x2, y2 = map(int, item["box"])
        name = item["name"]
        conf = item["conf"]
        text = f"{name} {conf*100:.1f}%"

        # 用 PIL 画矩形框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        # 画文本背景和文本
        tb = draw.textbbox((x1, max(0, y1 - FONT_SIZE)), text, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(255, 0, 0))
        draw.text((x1, max(0, y1 - FONT_SIZE)), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── 带按键动画的 iOS 风格按钮 ────────────────────────────────
class AnimatedButton(QPushButton):
    """点击时有下沉/弹起阴影动画，模拟真实按键手感"""

    def __init__(self, text="", icon="", color=NAU["primary"], parent=None):
        display = f"{icon}  {text}" if icon else text
        super().__init__(display, parent)
        self._color = color
        self._hover_color = QColor(color).darker(110).name()
        self._press_color = QColor(color).darker(125).name()
        self._anims = []

        # 阴影
        self._shadow = QGraphicsDropShadowEffect()
        self._shadow.setBlurRadius(12)
        self._shadow.setOffset(0, 3)
        self._shadow.setColor(NAU["shadow"])
        self.setGraphicsEffect(self._shadow)

        self.setFont(QFont("Microsoft YaHei", 18))
        self.setMinimumHeight(56)
        self.setCursor(Qt.PointingHandCursor)
        self._refresh_style(self._color)

    def _refresh_style(self, bg):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border-radius: 28px;
                padding: 16px 32px;
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._hover_color};
            }}
            QPushButton:pressed {{
                background-color: {self._press_color};
            }}
        """)

    def mousePressEvent(self, event):
        for a in self._anims:
            a.stop()
        self._anims.clear()

        a1 = QVariantAnimation()
        a1.setDuration(80)
        a1.setStartValue(12.0)
        a1.setEndValue(4.0)
        a1.setEasingCurve(QEasingCurve.OutCubic)
        a1.valueChanged.connect(lambda v: self._shadow.setBlurRadius(v))

        a2 = QVariantAnimation()
        a2.setDuration(80)
        a2.setStartValue(3.0)
        a2.setEndValue(1.0)
        a2.setEasingCurve(QEasingCurve.OutCubic)
        a2.valueChanged.connect(lambda v: self._shadow.setOffset(0, v))

        self._anims.extend([a1, a2])
        a1.start()
        a2.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        for a in self._anims:
            a.stop()
        self._anims.clear()

        a1 = QVariantAnimation()
        a1.setDuration(200)
        a1.setStartValue(4.0)
        a1.setEndValue(12.0)
        a1.setEasingCurve(QEasingCurve.OutBack)
        a1.valueChanged.connect(lambda v: self._shadow.setBlurRadius(v))

        a2 = QVariantAnimation()
        a2.setDuration(200)
        a2.setStartValue(1.0)
        a2.setEndValue(3.0)
        a2.setEasingCurve(QEasingCurve.OutBack)
        a2.valueChanged.connect(lambda v: self._shadow.setOffset(0, v))

        self._anims.extend([a1, a2])
        a1.start()
        a2.start()
        super().mouseReleaseEvent(event)


# ── 后台推理线程 ──────────────────────────────────────────────
class InferenceThread(QThread):
    finished = pyqtSignal(list)

    def __init__(self, model, image):
        super().__init__()
        self.model = model
        self.image = image

    def run(self):
        results = self.model(self.image, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
        result_list = []
        for box in results[0].boxes:
            result_list.append({
                "name": CLASS_NAMES[int(box.cls.item())],
                "conf": float(box.conf.item()),
                "box": box.xyxy[0].tolist()
            })
        self.finished.emit(result_list)


# ── 导航辅助 ──────────────────────────────────────────────────
def _navigate_to(current_window, new_window):
    """保存引用并切换窗口"""
    current_window._next = new_window
    new_window.show()
    current_window.close()


# ── 全局导航历史（支持前进/后退） ──────────────────────────────
NAV_HISTORY = []
NAV_POS = -1


def nav_record(window_class, *args):
    """记录导航到历史，截断前进历史"""
    global NAV_POS
    NAV_HISTORY[:] = NAV_HISTORY[:NAV_POS + 1]
    NAV_HISTORY.append((window_class, args))
    NAV_POS = len(NAV_HISTORY) - 1


def nav_can_back():
    return NAV_POS > 0


def nav_can_forward():
    return NAV_POS < len(NAV_HISTORY) - 1


def nav_go_back(current_window):
    global NAV_POS
    if nav_can_back():
        NAV_POS -= 1
        cls, args = NAV_HISTORY[NAV_POS]
        QTimer.singleShot(0, lambda: _navigate_to(current_window, cls(*args)))


def nav_go_forward(current_window):
    global NAV_POS
    if nav_can_forward():
        NAV_POS += 1
        cls, args = NAV_HISTORY[NAV_POS]
        QTimer.singleShot(0, lambda: _navigate_to(current_window, cls(*args)))


def nav_clear():
    """清除导航历史（返回首页时调用）"""
    global NAV_POS
    NAV_HISTORY.clear()
    NAV_POS = -1


# ── 首页 ──────────────────────────────────────────────────────
class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NAU水稻病虫害识别程序")
        self.setAcceptDrops(False)
        self.model, self.device = get_model()
        self._build()
        make_rounded(self, 28, "南京农业大学 · 水稻病虫害智能诊断系统")
        center_window(self, 1100, 760)

    def _build(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {NAU['bg']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(48, 36, 48, 36)
        root.setSpacing(0)

        # ── LOGO 品牌头图 ──
        logo_card = QWidget()
        logo_card.setStyleSheet(card_style())
        logo_card.setGraphicsEffect(make_shadow(16, 4))
        logo_layout = QVBoxLayout(logo_card)
        logo_layout.setContentsMargins(24, 20, 24, 20)
        logo_layout.setSpacing(10)

        # 校徽 LOGO 图片
        logo_path = res_path("resources/nau_logo.png")
        if os.path.exists(logo_path):
            logo_pix = QPixmap(logo_path).scaledToWidth(500, Qt.SmoothTransformation)
            logo_img = QLabel()
            logo_img.setPixmap(logo_pix)
            logo_img.setAlignment(Qt.AlignCenter)
            logo_img.setStyleSheet("background: transparent; border: none;")
            logo_layout.addWidget(logo_img)

        title = QLabel("水稻病虫害智能识别系统")
        title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")

        subtitle = QLabel("基于 YOLOv8n 深度学习模型 · 精准识别 6 类水稻病虫害")
        subtitle.setFont(QFont("Microsoft YaHei", 14))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")

        logo_layout.addWidget(title)
        logo_layout.addSpacing(4)
        logo_layout.addWidget(subtitle)
        root.addWidget(logo_card)

        root.addSpacing(28)

        # ── 选择提示 ──
        tip = QLabel("请选择识别方式")
        tip.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
        root.addWidget(tip)
        root.addSpacing(20)

        # ── 三个功能按钮（卡片式） ──
        btn_card = QWidget()
        btn_card.setStyleSheet(card_style())
        btn_card.setGraphicsEffect(make_shadow(12, 3))
        btn_layout = QVBoxLayout(btn_card)
        btn_layout.setContentsMargins(24, 20, 24, 20)
        btn_layout.setSpacing(14)

        btn_image = AnimatedButton("拍照上传", "📷", NAU["primary"])
        btn_camera = AnimatedButton("打开摄像头实时识别", "🎥", NAU["primary"])
        btn_map = AnimatedButton("地块地图 · 溯源记录", "🗺", NAU["primary"])
        btn_exit = AnimatedButton("退出识别", "🚪", NAU["red"])

        btn_layout.addWidget(btn_image)
        btn_layout.addWidget(btn_camera)
        btn_layout.addWidget(btn_map)
        btn_layout.addWidget(btn_exit)
        root.addWidget(btn_card)

        root.addSpacing(24)

        # ── 底部状态栏（模型 + 校训） ──
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)

        btn_settings = QPushButton("⚙")
        btn_settings.setFont(QFont("Microsoft YaHei", 16))
        btn_settings.setFixedSize(44, 44)
        btn_settings.setCursor(Qt.PointingHandCursor)
        btn_settings.setToolTip("API 设置")
        btn_settings.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {NAU['subtext']};
                border: 1px solid {NAU['border']}; border-radius: 22px;
            }}
            QPushButton:hover {{ background: {NAU['border']}; color: {NAU['dark']}; }}
        """)
        btn_settings.clicked.connect(self._open_api_settings)
        bottom_row.addWidget(btn_settings)
        bottom_row.addStretch()

        device_text = "GPU (CUDA)" if self.device != "cpu" else "CPU"
        status = QLabel(f"⚡ 模型已就绪  |  运行设备：{device_text}  |  识别类别：{len(CLASS_NAMES)} 种")
        status.setFont(QFont("Microsoft YaHei", 12))
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
        bottom_row.addWidget(status)
        bottom_row.addStretch()
        bottom_row.addWidget(QLabel(""))
        root.addLayout(bottom_row)

        motto = QLabel("诚朴勤仁")
        motto.setFont(QFont("Microsoft YaHei", 13))
        motto.setAlignment(Qt.AlignCenter)
        motto.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none; padding-top: 6px;")
        root.addWidget(motto)

        btn_image.clicked.connect(self.go_image_recog)
        btn_camera.clicked.connect(self.go_camera_recog)
        btn_map.clicked.connect(self.go_map)
        btn_exit.clicked.connect(self.close)

    def _open_api_settings(self):
        nav_record(ApiSettingsWindow)
        _navigate_to(self, ApiSettingsWindow())

    def go_image_recog(self):
        nav_record(ImageRecogWindow, self.model, self.device)
        _navigate_to(self, ImageRecogWindow(self.model, self.device))

    def go_camera_recog(self):
        nav_record(CameraRecogWindow, self.model, self.device)
        _navigate_to(self, CameraRecogWindow(self.model, self.device))

    def go_map(self):
        """打开地块地图：照片的时间戳与经纬度在这里汇成一张田间档案图"""
        if not GEO_AVAILABLE:
            QMessageBox.warning(self, "模块缺失",
                                "未找到 rice_geo.py，地块地图功能不可用。\n"
                                "请确认它与主程序放在同一目录。")
            return
        nav_record(rice_geo.GeoMapWindow, self.model, self.device)
        _navigate_to(self, rice_geo.GeoMapWindow(self.model, self.device))


# ── 图片识别页 ────────────────────────────────────────────────
class ImageRecogWindow(QMainWindow):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.current_image = None
        self.current_path = None             # 原始照片路径（用于读取 EXIF 时间/GPS）
        self._inference_running = False      # 防止重复推理
        self.setWindowTitle("📷  拍照上传识别")
        self.setAcceptDrops(True)
        self._build()
        make_rounded(self, 28, "📷 图片识别")
        center_window(self, 1100, 760)

    def _build(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {NAU['bg']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(48, 36, 48, 36)
        root.setSpacing(0)

        # ── 标题 ──
        title = QLabel("📷  图片识别")
        title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
        root.addWidget(title)
        root.addSpacing(25)

        # ── 拖拽区域 ──
        self.drop_label = QLabel("拖拽图片到此处\n或点击下方按钮选择")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setMinimumHeight(360)
        self.drop_label.setFont(QFont("Microsoft YaHei", 16))
        self.drop_label.setStyleSheet(f"""
            QLabel {{
                border: 3px dashed {NAU['primary']};
                border-radius: 40px;
                background-color: {NAU['primary_light']};
                color: {NAU['subtext']};
                font-size: 18px;
            }}
        """)
        root.addWidget(self.drop_label)
        root.addSpacing(20)

        # ── 操作按钮行 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)

        self.btn_select = AnimatedButton("从电脑选择", "📁", NAU["primary"])
        self.btn_back = AnimatedButton("返回首页", "🏠", "#8B9A8B")

        btn_row.addStretch()
        btn_row.addWidget(self.btn_select)
        btn_row.addWidget(self.btn_back)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addSpacing(15)

        # ── 状态提示 ──
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Microsoft YaHei", 12))
        self.status_label.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
        root.addWidget(self.status_label)

        self.btn_select.clicked.connect(self.select_image)
        self.btn_back.clicked.connect(self.go_home)

    # ── 拖拽事件 ──
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if fp.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self._process(fp)
                break

    def select_image(self):
        if self._inference_running:
            return  # 上一次推理尚未完成，忽略重复点击
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择要识别的图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if fp:
            self._process(fp)

    def _process(self, file_path):
        if self._inference_running:
            return
        self.current_image = cv2.imread(file_path)
        if self.current_image is None:
            QMessageBox.warning(self, "错误", "无法读取图片文件！")
            return
        self.current_path = file_path        # 记录来源，便于把时间/GPS 一起存档

        # 在拖拽框预览
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.drop_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.drop_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.status_label.setText("⏳ 模型正在识别，请稍后...")
        self.btn_select.setEnabled(False)     # 推理中禁用按钮
        self._inference_running = True
        self.thread = InferenceThread(self.model, self.current_image)
        self.thread.finished.connect(self._show_result)
        self.thread.start()

    def _show_result(self, result_list):
        self._inference_running = False
        self.btn_select.setEnabled(True)
        src = getattr(self, "current_path", None)
        nav_record(ResultWindow, self.model, self.device, result_list, self.current_image, "image", src)
        _navigate_to(self, ResultWindow(self.model, self.device, result_list, self.current_image, "image", src))

    def go_home(self):
        nav_clear()
        _navigate_to(self, HomeWindow())


# ── 摄像头实时识别页 ──────────────────────────────────────────
class CameraRecogWindow(QMainWindow):
    INFERENCE_INTERVAL = 30  # 每 30 帧推理一次（约 1 秒一次）

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.frame_count = 0
        self._inference_running = False
        self._last_results = []          # 缓存最近一次真实推理结果

        self.setWindowTitle("🎥  摄像头实时识别")
        self._build()
        make_rounded(self, 28, "🎥 摄像头实时识别")
        center_window(self, 1100, 760)
        self._open_camera()

    def _build(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {NAU['bg']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(48, 36, 48, 36)
        root.setSpacing(0)

        title = QLabel("🎥  摄像头实时识别")
        title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
        root.addWidget(title)
        root.addSpacing(20)

        # 视频画面
        self.video_label = QLabel("正在打开摄像头...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(460)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #000;
                color: white;
                font-size: 18px;
                border-radius: 38px;
            }}
        """)
        root.addWidget(self.video_label)
        root.addSpacing(20)

        # 状态
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Microsoft YaHei", 12))
        self.status_label.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
        root.addWidget(self.status_label)
        root.addSpacing(15)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_back = AnimatedButton("停止识别，返回首页", "🏠", NAU["red"])
        btn_row.addWidget(self.btn_back)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.btn_back.clicked.connect(self.go_home)

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头！")
            self.go_home()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer.start(33)  # ~30fps
        self.status_label.setText("🔴 摄像头已开启 · 模型实时识别中...")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()
        self.frame_count += 1

        # ── 只绘制最近一次真实推理的结果（不再画假框） ──
        display_frame = frame.copy()
        if self._last_results:
            display_frame = annotate_image(display_frame, self._last_results)
        else:
            # 还未产生推理结果时，叠加一个"扫描中"的提示
            cv2.putText(display_frame, "Scanning...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 100), 2)

        # 显示
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # ── 每 N 帧推理一次 ──
        if self.frame_count % self.INFERENCE_INTERVAL == 0 and not self._inference_running:
            self._inference_running = True
            self.thread = InferenceThread(self.model, self.current_frame)
            self.thread.finished.connect(self._on_inference)
            self.thread.start()

    def _on_inference(self, result_list):
        self._inference_running = False
        if result_list:
            self._last_results = result_list
            self.timer.stop()
            if self.cap:
                self.cap.release()
            nav_record(ResultWindow, self.model, self.device, result_list, self.current_frame, "camera")
            _navigate_to(self, ResultWindow(self.model, self.device, result_list, self.current_frame, "camera"))
        else:
            # 本轮未检测到，清空缓存（避免过期框残留）
            self._last_results = []

    def go_home(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        nav_clear()
        _navigate_to(self, HomeWindow())

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ── Markdown → 格式化 HTML ──────────────────────────────────
def md_to_html(text):
    """把 AI 返回的 Markdown 转成 HTML"""
    try:
        return markdown.markdown(text, extensions=['extra', 'nl2br'])
    except Exception as e:
        log.warning("Markdown 渲染失败，退化为纯文本: %s", e)
        return text.replace("\n", "<br>")


# ── 结果展示页（全屏）─────────────────────────────────────────
class ResultWindow(QMainWindow):
    def __init__(self, model, device, result_list, orig_img, mode, source_path=None):
        super().__init__()
        self.model = model
        self.device = device
        self.result_list = result_list
        self.orig_img = orig_img
        self.mode = mode
        self.source_path = source_path      # 原始照片路径（摄像头模式为 None）
        self.setWindowTitle("识别结果 & 治理方案")
        self._build()
        make_rounded(self, 28, "识别结果 & 治理方案")
        self.showFullScreen()

    def _section_title(self, text, size=18):
        lbl = QLabel(text)
        lbl.setFont(QFont("Microsoft YaHei", size, QFont.Bold))
        lbl.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
        return lbl

    def _bullet(self, text, size=18):
        lbl = QLabel(f"  ● {text}")
        lbl.setFont(QFont("Microsoft YaHei", size))
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {NAU['text']}; background: transparent; border: none;")
        return lbl

    def _build(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {NAU['bg']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(36, 24, 36, 24)
        root.setSpacing(0)

        # ── 处理最佳识别结果 ──
        if self.result_list:
            best = max(self.result_list, key=lambda x: x["conf"])
            self.cls_name = best["name"]
            self.conf = best["conf"]
            self.result_img = annotate_image(self.orig_img, self.result_list)
        else:
            self.cls_name = None
            self.conf = 0.0
            self.result_img = self.orig_img

        # ── 三栏主内容 ──
        body = QHBoxLayout()
        body.setSpacing(16)

        # ══════════ 左栏：识别结果 + 典型症状 ══════════
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        left_card = QWidget()
        left_card.setStyleSheet(card_style())
        left_card.setGraphicsEffect(make_shadow(8, 2))
        left = QVBoxLayout(left_card)
        left.setContentsMargins(16, 14, 16, 14)
        left.setSpacing(5)

        left.addWidget(self._section_title("识别结果"))

        if self.cls_name:
            cls_lbl = QLabel(f"病虫害：{self.cls_name}")
            cls_lbl.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
            cls_lbl.setStyleSheet(f"color: {NAU['red']}; background: transparent; border: none;")
            left.addWidget(cls_lbl)

            conf_lbl = QLabel(f"置信度：{self.conf*100:.2f}%")
            conf_lbl.setFont(QFont("Microsoft YaHei", 18))
            conf_lbl.setStyleSheet(f"color: {NAU['dark']}; background: transparent; border: none;")
            left.addWidget(conf_lbl)

            conf_bar = QProgressBar()
            conf_bar.setRange(0, 100)
            conf_bar.setValue(int(self.conf * 100))
            conf_bar.setTextVisible(True)
            conf_bar.setFormat(f"{self.conf*100:.1f}%")
            conf_bar.setFixedHeight(22)
            conf_bar.setFont(QFont("Microsoft YaHei", 16))
            conf_bar.setStyleSheet(f"""
                QProgressBar {{ border:none; border-radius:8px; background:{NAU['border']}; text-align:center; color:white; font-weight:bold; }}
                QProgressBar::chunk {{ border-radius:8px; background:{NAU['primary']}; }}
            """)
            left.addWidget(conf_bar)
        else:
            no_lbl = QLabel("未检测到符合阈值的水稻病虫害")
            no_lbl.setFont(QFont("Microsoft YaHei", 18))
            no_lbl.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
            left.addWidget(no_lbl)

        # 标注图片
        rgb = cv2.cvtColor(self.result_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        img_disp = QLabel()
        img_disp.setPixmap(QPixmap.fromImage(qt_img).scaledToWidth(340, Qt.SmoothTransformation))
        img_disp.setAlignment(Qt.AlignCenter)
        img_disp.setStyleSheet("border-radius: 12px; background: transparent;")
        left.addWidget(img_disp)

        # 各类别置信度
        left.addWidget(self._section_title("各类别置信度"))
        for cls in CLASS_NAMES:
            cls_conf = 0.0
            for r in self.result_list:
                if r["name"] == cls:
                    cls_conf = r["conf"]
                    break
            active = cls == self.cls_name
            row = QHBoxLayout()
            row.setSpacing(4)
            nl = QLabel(cls)
            nl.setFont(QFont("Microsoft YaHei", 16))
            nl.setFixedWidth(70)
            c = NAU["primary"] if active else NAU["border"]
            nl.setStyleSheet(f"color: {c}; background: transparent; font-weight: {'bold' if active else 'normal'};")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(cls_conf * 100))
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setStyleSheet(f"""
                QProgressBar {{ border:none; border-radius:3px; background:{NAU['border']}; }}
                QProgressBar::chunk {{ border-radius:3px; background:{c}; }}
            """)
            row.addWidget(nl)
            row.addWidget(bar)
            left.addLayout(row)

        # ── 典型症状 ──
        left.addSpacing(4)
        left.addWidget(self._section_title("典型症状"))
        if self.cls_name and self.cls_name in TREATMENT_DB:
            pest_img_path = TREATMENT_DB[self.cls_name]["pest_img"]
            pest_card = make_image_card(pest_img_path, f"【{self.cls_name}】", 180, 150)
            left.addWidget(pest_card)
            symptom_label = QLabel(TREATMENT_DB[self.cls_name]["symptom"])
            symptom_label.setFont(QFont("Microsoft YaHei", 18))
            symptom_label.setWordWrap(True)
            symptom_label.setStyleSheet(f"color: {NAU['text']}; background: transparent; border: none; padding: 4px;")
            left.addWidget(symptom_label)
        else:
            placeholder = QLabel("识别未成功")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setFont(QFont("Microsoft YaHei", 18))
            placeholder.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
            left.addWidget(placeholder)

        left.addStretch()
        left_scroll.setWidget(left_card)
        body.addWidget(left_scroll, 3)

        # ══════════ 中栏：治理方案 ══════════
        mid_scroll = QScrollArea()
        mid_scroll.setWidgetResizable(True)
        mid_scroll.setFrameShape(QFrame.NoFrame)
        mid_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        mid_card = QWidget()
        mid_card.setStyleSheet(card_style())
        mid_card.setGraphicsEffect(make_shadow(8, 2))
        mid = QVBoxLayout(mid_card)
        mid.setContentsMargins(18, 16, 18, 16)
        mid.setSpacing(5)

        mid.addWidget(self._section_title("治理方案"))

        if self.cls_name and self.cls_name in TREATMENT_DB:
            tr = TREATMENT_DB[self.cls_name]

            mid.addWidget(self._section_title("农业防治"))
            for m in tr["agri"]:
                mid.addWidget(self._bullet(m))

            if tr.get("bio"):
                mid.addSpacing(4)
                mid.addWidget(self._section_title("生物防治"))
                for m in tr["bio"]:
                    mid.addWidget(self._bullet(m))

            mid.addSpacing(4)
            mid.addWidget(self._section_title("防治指标"))
            threshold_lbl = QLabel(f"  ● {tr['threshold']}")
            threshold_lbl.setFont(QFont("Microsoft YaHei", 18))
            threshold_lbl.setWordWrap(True)
            threshold_lbl.setStyleSheet(f"color: {NAU['red']}; background: transparent; border: none;")
            mid.addWidget(threshold_lbl)

            mid.addSpacing(6)
            mid.addWidget(self._section_title("推荐药剂"))

            drug_grid = QGridLayout()
            drug_grid.setSpacing(8)
            for i, drug in enumerate(tr["drugs"]):
                label_parts = [drug["name"]]
                if drug.get("spec"): label_parts.append(drug["spec"])
                if drug.get("dose"): label_parts.append(drug["dose"])
                card = make_image_card(drug.get("img", ""), "\n".join(label_parts), 140, 110)
                drug_grid.addWidget(card, i // 2, i % 2)
            mid.addLayout(drug_grid)

            for drug in tr["drugs"]:
                drug_text = QLabel(f"● {drug['name']} | {drug.get('spec', '')} | {drug.get('dose', '')}")
                drug_text.setFont(QFont("Microsoft YaHei", 18))
                drug_text.setWordWrap(True)
                drug_text.setStyleSheet(f"color: {NAU['text']}; background: transparent; border: none;")
                mid.addWidget(drug_text)
        else:
            no_treat = QLabel("识别未成功，暂无治理方案推荐")
            no_treat.setFont(QFont("Microsoft YaHei", 18))
            no_treat.setWordWrap(True)
            no_treat.setStyleSheet(f"color: {NAU['subtext']}; background: transparent; border: none;")
            mid.addWidget(no_treat)

        mid.addStretch()
        mid_scroll.setWidget(mid_card)
        body.addWidget(mid_scroll, 5)

        # ══════════ 右栏：AI 智能分析 ══════════
        right_card = QWidget()
        right_card.setStyleSheet(card_style())
        right_card.setGraphicsEffect(make_shadow(8, 2))
        right = QVBoxLayout(right_card)
        right.setContentsMargins(18, 18, 18, 18)
        right.setSpacing(12)

        # 头部
        ai_header = QHBoxLayout()
        ai_header.addWidget(self._section_title("AI 智能分析", 15))
        provider_name = {"deepseek": "DeepSeek", "qwen": "Qwen-VL"}.get(CONFIG["llm_provider"], "AI")
        badge = QLabel(f" {provider_name} ")
        badge.setStyleSheet(f"background: {NAU['primary_light']}; color: {NAU['primary']}; "
                          "padding: 3px 10px; border-radius: 8px; font-size: 11px; font-weight: bold;")
        ai_header.addStretch()

        # 字体调节按钮
        self._ai_font_size = 18
        btn_font_minus = QPushButton("A-")
        btn_font_minus.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        btn_font_minus.setFixedSize(32, 26)
        btn_font_minus.setCursor(Qt.PointingHandCursor)
        btn_font_minus.setToolTip("缩小字体")
        btn_font_minus.setStyleSheet(f"""
            QPushButton {{ background: {NAU['bg']}; color: {NAU['subtext']};
            border: 1px solid {NAU['border']}; border-radius: 6px; }}
            QPushButton:hover {{ border-color: {NAU['primary']}; color: {NAU['primary']}; }}
        """)
        btn_font_minus.clicked.connect(lambda: self._change_ai_font_size(-2))

        btn_font_plus = QPushButton("A+")
        btn_font_plus.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        btn_font_plus.setFixedSize(32, 26)
        btn_font_plus.setCursor(Qt.PointingHandCursor)
        btn_font_plus.setToolTip("放大字体")
        btn_font_plus.setStyleSheet(f"""
            QPushButton {{ background: {NAU['bg']}; color: {NAU['subtext']};
            border: 1px solid {NAU['border']}; border-radius: 6px; }}
            QPushButton:hover {{ border-color: {NAU['primary']}; color: {NAU['primary']}; }}
        """)
        btn_font_plus.clicked.connect(lambda: self._change_ai_font_size(2))

        ai_header.addWidget(btn_font_minus)
        ai_header.addWidget(btn_font_plus)
        ai_header.addWidget(badge)
        right.addLayout(ai_header)

        # AI 对话显示区
        self.ai_display = QTextEdit()
        self.ai_display.setReadOnly(True)
        self.ai_display.setFont(QFont("Microsoft YaHei", 18))
        self.ai_display.setStyleSheet(f"""
            QTextEdit {{
                background: {NAU['bg']}; border: 1px solid {NAU['border']};
                border-radius: 10px; padding: 14px; color: {NAU['text']};
                line-height: 1.7;
            }}
            QScrollBar:vertical {{ width: 5px; background: transparent; }}
            QScrollBar::handle:vertical {{ background: {NAU['border']}; border-radius: 2px; }}
        """)
        self.ai_display.setPlaceholderText("上传图片后 # AI 将自动分析...")
        self.ai_display.setMinimumHeight(300)
        right.addWidget(self.ai_display, 1)

        # 追问输入区
        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("输入问题追问 AI...")
        self.ai_input.setFont(QFont("Microsoft YaHei", 18))
        self.ai_input.setStyleSheet(f"""
            QLineEdit {{
                background: {NAU['bg']}; border: 1px solid {NAU['border']};
                border-radius: 8px; padding: 10px 14px; color: {NAU['text']};
                font-size: 18px;
            }}
            QLineEdit:focus {{ border-color: {NAU['primary']}; }}
        """)
        self.ai_input.setFixedHeight(44)
        self.ai_input.returnPressed.connect(self._send_ai_message)

        self.ai_img_btn = QPushButton("🖼")
        self.ai_img_btn.setFont(QFont("Microsoft YaHei", 18))
        self.ai_img_btn.setFixedSize(44, 44)
        self.ai_img_btn.setCursor(Qt.PointingHandCursor)
        self.ai_img_btn.setToolTip("上传图片追问")
        self.ai_img_btn.setStyleSheet(f"""
            QPushButton {{ background: {NAU['bg']}; color: {NAU['subtext']};
            border: 1px solid {NAU['border']}; border-radius: 8px; }}
            QPushButton:hover {{ border-color: {NAU['primary']}; color: {NAU['primary']}; }}
        """)
        self.ai_img_btn.clicked.connect(self._pick_ai_image)
        self._ai_extra_img = None  # 追问时附加的图片

        btn_send = QPushButton("发送")
        btn_send.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        btn_send.setFixedSize(72, 44)
        btn_send.setCursor(Qt.PointingHandCursor)
        btn_send.setStyleSheet(f"""
            QPushButton {{ background: {NAU['primary']}; color: white; border: none; border-radius: 8px; }}
            QPushButton:hover {{ background: {NAU['primary_hover']}; }}
        """)
        btn_send.clicked.connect(self._send_ai_message)
        input_row.addWidget(self.ai_input)
        input_row.addWidget(self.ai_img_btn)
        input_row.addWidget(btn_send)
        right.addLayout(input_row)

        body.addWidget(right_card, 8)

        root.addLayout(body)
        root.addSpacing(12)

        # ── 底部按钮 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)
        btn_row.addStretch()
        self.btn_next = AnimatedButton("识别下一张", "▶️", NAU["primary"])
        self.btn_map = AnimatedButton("记录到地图", "📍", NAU["primary"])
        self.btn_stop = AnimatedButton("返回首页", "🏠", "#8B9A8B")
        btn_row.addWidget(self.btn_next)
        if GEO_AVAILABLE:
            btn_row.addWidget(self.btn_map)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.btn_next.clicked.connect(self._next_recog)
        if GEO_AVAILABLE:
            self.btn_map.clicked.connect(self._save_to_map)
        self.btn_stop.clicked.connect(self._go_home)

        # 自动启动 AI 分析
        if self.cls_name:
            self._start_ai_analysis()

    def _append_html(self, html):
        """追加 HTML 到对话区并滚动到底部"""
        cursor = self.ai_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(html)
        self.ai_display.ensureCursorVisible()

    def _img_tag(self, img_bgr, max_w=200):
        """OpenCV BGR → HTML <img> base64"""
        try:
            import cv2
            h, w = img_bgr.shape[:2]
            if max(h, w) > max_w:
                scale = max_w / max(h, w)
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            return f"<img src='data:image/jpeg;base64,{b64}' style='max-width:{max_w}px; border-radius:8px; margin-top:6px;'>"
        except Exception as e:
            log.warning("聊天图片编码失败: %s", e)
            return ""

    def _rebuild_chat(self):
        """根据聊天记录重建整个对话显示"""
        try:
            self._do_rebuild_chat()
        except Exception as e:
            self.ai_display.setHtml(
                f"<p style='color:{NAU['warning']}; font-size:23px;'>"
                f"显示异常: {e}</p>")
    def _do_rebuild_chat(self):
        self.ai_display.clear()
        fs = self._ai_font_size
        parts = []

        for entry in self._chat_history:
            role = entry[0]
            content = entry[1]
            extra_img = entry[2] if len(entry) > 2 else None

            if role == "ai_initial":
                html = self._style_md(content, fs)
                parts.append(html)
            elif role == "user":
                img_tag = self._img_tag(extra_img) if extra_img is not None else ""
                parts.append(
                    f"<div style='margin:16px 0 8px 40px; padding:10px 16px; "
                    f"background:{NAU['primary_light']}; border-radius:12px;'>"
                    f"<div style='font-size:{fs}px; color:{NAU['primary']}; font-weight:bold; margin-bottom:4px;'>"
                    f"◀ 你</div>"
                    f"<div style='font-size:{fs}px; color:{NAU['text']}; line-height:1.6;'>{content}</div>"
                    f"{img_tag}"
                    f"</div>")
            elif role == "ai":
                html = self._style_md(content, fs)
                parts.append(
                    f"<div style='margin:8px 40px 16px 0; padding:12px 18px; "
                    f"background:#FAFAFA; border-radius:12px; "
                    f"border-left:3px solid {NAU['accent_blue']};'>"
                    f"<div style='font-size:{fs}px; color:{NAU['accent_blue']}; font-weight:bold; margin-bottom:6px;'>"
                    f"▶ AI</div>"
                    f"{html}"
                    f"</div>")
            elif role == "error":
                parts.append(
                    f"<div style='margin:8px 0; padding:10px 16px; "
                    f"background:#FFF5F5; border-radius:10px; border-left:3px solid {NAU['warning']};'>"
                    f"<span style='color:{NAU['warning']}; font-size:{fs}px;'>⚠ {content}</span>"
                    f"</div>")
            elif role == "system":
                parts.append(
                    f"<div style='padding:4px 0;'>"
                    f"<span style='color:{NAU['subtext']}; font-size:{fs}px;'>{content}</span>"
                    f"</div>")

        if not parts:
            parts.append(
                f"<div style='text-align:center; padding:40px 0;'>"
                f"<div style='font-size:40px; margin-bottom:12px;'>🌾</div>"
                f"<p style='color:{NAU['subtext']}; font-size:{fs}px;'>"
                f"上传图片后 AI 将自动分析...</p></div>")

        self.ai_display.setHtml("".join(parts))
        self.ai_display.verticalScrollBar().setValue(
            self.ai_display.verticalScrollBar().maximum())

    def _style_md(self, text, fs):
        """将 markdown 文本转为带内联样式的 HTML，适配 QTextEdit"""
        html = md_to_html(text)
        html = self._add_ai_icons(html)
        body_s = fs + 4
        h1_s = fs - 2
        h2_s = fs - 3
        h3_s = fs - 4
        html = html.replace('<h1>', f'<h1 style="font-size:{h1_s}px; color:{NAU["primary"]}; border-left:4px solid {NAU["primary"]}; padding-left:12px; margin:18px 0 10px 0;">')
        html = html.replace('<h2>', f'<h2 style="font-size:{h2_s}px; color:{NAU["dark"]}; border-left:3px solid {NAU["accent_gold"]}; padding-left:10px; margin:14px 0 8px 0;">')
        html = html.replace('<h3>', f'<h3 style="font-size:{h3_s}px; color:{NAU["dark"]}; margin:12px 0 6px 0;">')
        html = html.replace('<p>', f'<p style="font-size:{body_s}px; margin:6px 0 12px 0; line-height:1.8;">')
        html = html.replace('<li>', f'<li style="font-size:{body_s}px; margin:3px 0; padding:2px 0;">')
        html = html.replace('<ul>', f'<ul style="margin:4px 0 10px 0; padding-left:18px;">')
        html = html.replace('<ol>', f'<ol style="margin:4px 0 10px 0; padding-left:18px;">')
        html = html.replace('<strong>', f'<strong style="color:{NAU["dark"]};">')
        html = html.replace('<hr>', f'<hr style="border:none; border-top:1px solid {NAU["border"]}; margin:16px 0;">')
        return html

    def _add_ai_icons(self, html):
        """给 AI 回答的关键段落添加图标，增强可读性"""
        # 防治建议 / 治理方案 → 📋
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:防治|治理|方案|措施|建议|方法).*?)(</h[123]>)',
            r'\1📋 \2\3', html, flags=re.IGNORECASE)
        # 药剂 / 药物 / 用药 → 💊
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:药剂|药物|用药|农药|推荐药).*?)(</h[123]>)',
            r'\1💊 \2\3', html, flags=re.IGNORECASE)
        # 诊断 / 症状 / 判断 / 识别 → 🔍
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:诊断|症状|判断|识别|表现|特征).*?)(</h[123]>)',
            r'\1🔍 \2\3', html, flags=re.IGNORECASE)
        # 农业 / 农艺 / 栽培 / 耕作 → 🌿
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:农业|农艺|栽培|耕作|种植|田间).*?)(</h[123]>)',
            r'\1🌿 \2\3', html, flags=re.IGNORECASE)
        # 注意 / 警告 / 严重 / 风险 → ⚠
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:注意|警告|严重|风险|危害).*?)(</h[123]>)',
            r'\1⚠ \2\3', html, flags=re.IGNORECASE)
        # 生物防治 / 天敌 → 🐝
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:生物|天敌|寄生).*?)(</h[123]>)',
            r'\1🐝 \2\3', html, flags=re.IGNORECASE)
        # 综合 / 总结 → 📌
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:综合|总结|汇总|概述).*?)(</h[123]>)',
            r'\1📌 \2\3', html, flags=re.IGNORECASE)
        # 气候 / 天气 / 环境 → 🌤
        html = re.sub(
            r'(<h[123][^>]*>)(.*?(?:气候|天气|环境|湿度|温度).*?)(</h[123]>)',
            r'\1🌤 \2\3', html, flags=re.IGNORECASE)
        return html

    def _change_ai_font_size(self, delta):
        """调整 AI 回答字体大小"""
        self._ai_font_size = max(8, min(32, self._ai_font_size + delta))
        self._rebuild_chat()

    def _start_ai_analysis(self):
        self._chat_history = []
        self._api_history = []  # LLM 对话上下文
        self._chat_history.append(("system",
            f"⏳ 正在调用 AI 分析 <b>{self.cls_name}</b> ..."))
        self._rebuild_chat()
        self.llm_thread = LLMAnalysisThread(self.cls_name, self.result_img)
        self.llm_thread.finished.connect(self._on_ai_result)
        self.llm_thread.error.connect(self._on_ai_error)
        self.llm_thread.start()

    def _on_ai_result(self, text):
        try:
            self._chat_history = [("ai_initial", text)]
            self._api_history = [{"role": "assistant", "content": text}]
            self._rebuild_chat()
        except Exception as e:
            self._chat_history = [("error", f"显示结果失败: {e}")]
            self._rebuild_chat()

    def _on_ai_error(self, msg):
        self._chat_history = [("error", msg)]
        self._rebuild_chat()
        QMessageBox.warning(self, "AI 调用失败", msg)

    def _pick_ai_image(self):
        fp, _ = QFileDialog.getOpenFileName(self, "选择追问图片", "",
                                            "Images (*.jpg *.jpeg *.png *.bmp)")
        if fp:
            img = cv2.imread(fp)
            if img is not None:
                self._ai_extra_img = img
                self.ai_img_btn.setStyleSheet(f"""
                    QPushButton {{ background: {NAU['primary_light']}; color: {NAU['primary']};
                    border: 2px solid {NAU['primary']}; border-radius: 8px; }}
                """)
                self.ai_img_btn.setToolTip("已选择图片，点击重新选择")

    def _send_ai_message(self):
        msg = self.ai_input.text().strip()
        extra_img = self._ai_extra_img
        self._ai_extra_img = None
        self.ai_img_btn.setStyleSheet(f"""
            QPushButton {{ background: {NAU['bg']}; color: {NAU['subtext']};
            border: 1px solid {NAU['border']}; border-radius: 8px; }}
            QPushButton:hover {{ border-color: {NAU['primary']}; color: {NAU['primary']}; }}
        """)
        self.ai_img_btn.setToolTip("上传图片追问")
        if not msg:
            return
        self.ai_input.clear()
        self._chat_history.append(("user", msg, extra_img))
        self._chat_history.append(("system", "⏳ AI 思考中..."))
        self._rebuild_chat()
        self._pending_user_msg = msg  # 等 AI 回复后再加入上下文
        self.llm_thread2 = LLMAnalysisThread(self.cls_name, extra_img, list(self._api_history))
        self.llm_thread2._build_prompt = lambda: msg
        self.llm_thread2.finished.connect(self._on_ai_followup)
        self.llm_thread2.error.connect(self._on_ai_followup_error)
        self.llm_thread2.start()

    def _on_ai_followup(self, text):
        try:
            self._chat_history[-1] = ("ai", text)
            if self._pending_user_msg:
                self._api_history.append({"role": "user", "content": self._pending_user_msg})
                self._pending_user_msg = None
            self._api_history.append({"role": "assistant", "content": text})
            self._rebuild_chat()
        except Exception as e:
            self._chat_history[-1] = ("error", f"显示失败: {e}")
            self._rebuild_chat()

    def _on_ai_followup_error(self, msg):
        self._chat_history[-1] = ("error", msg)
        self._rebuild_chat()
        QMessageBox.warning(self, "AI 调用失败", msg)

    def _save_to_map(self):
        """把这次识别结果+时间+GPS 存进地块地图档案，并跳到地图页"""
        if not GEO_AVAILABLE:
            return
        try:
            pending = {
                "image": self.orig_img,
                "annotated": self.result_img,
                "results": self.result_list,
                "cls": self.cls_name or "",
                "conf": self.conf or 0.0,
                "source_path": self.source_path,
            }
            nav_record(rice_geo.GeoMapWindow, self.model, self.device)
            _navigate_to(self, rice_geo.GeoMapWindow(self.model, self.device, pending))
        except Exception as e:
            QMessageBox.warning(self, "保存失败", "%s" % e)

    def _next_recog(self):
        if self.mode == "image":
            nav_record(ImageRecogWindow, self.model, self.device)
            _navigate_to(self, ImageRecogWindow(self.model, self.device))
        else:
            nav_record(CameraRecogWindow, self.model, self.device)
            _navigate_to(self, CameraRecogWindow(self.model, self.device))

    def _go_back(self):
        nav_go_back(self)

    def _go_home(self):
        nav_clear()
        _navigate_to(self, HomeWindow())


# ── API 设置页（全页面）───────────────────────────────────────
class ApiSettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("API 设置")
        self._build()
        make_rounded(self, 28, "API 设置")
        center_window(self, 1100, 760)

    def _build(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {NAU['bg']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(80, 48, 80, 48)
        root.setSpacing(0)

        title = QLabel("API 配置")
        title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title.setStyleSheet(f"color: {NAU['dark']}; background: transparent;")
        root.addWidget(title)

        desc = QLabel("配置 AI 大模型接入，用于智能分析病虫害")
        desc.setFont(QFont("Microsoft YaHei", 13))
        desc.setStyleSheet(f"color: {NAU['subtext']}; background: transparent;")
        root.addWidget(desc)
        root.addSpacing(24)

        # 设置卡片
        card = QWidget()
        card.setStyleSheet(card_style())
        card.setGraphicsEffect(make_shadow(10, 3))
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 24)
        cl.setSpacing(14)

        # 服务商切换
        lbl_provider = QLabel("服务商")
        lbl_provider.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        lbl_provider.setStyleSheet(f"color: {NAU['text']}; background: transparent;")
        cl.addWidget(lbl_provider)

        provider_row = QHBoxLayout()
        provider_row.setSpacing(10)
        self.btn_ds = QPushButton("DeepSeek")
        self.btn_qw = QPushButton("千问 (阿里云)")
        self._provider_btns = {"deepseek": self.btn_ds, "qwen": self.btn_qw}

        base_style = f"""
            QPushButton {{
                background: {NAU['bg']}; color: {NAU['text']};
                border: 1px solid {NAU['border']}; border-radius: 8px; padding: 6px 20px;
            }}
            QPushButton:hover {{ border-color: {NAU['primary']}; color: {NAU['primary']}; }}
        """
        active_style = f"""
            QPushButton {{
                background: {NAU['primary']}; color: white;
                border: 1px solid {NAU['primary']}; border-radius: 8px; padding: 6px 20px;
            }}
            QPushButton:hover {{ background: {NAU['primary_hover']}; }}
        """
        for btn in (self.btn_ds, self.btn_qw):
            btn.setFixedHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFont(QFont("Microsoft YaHei", 11))
            btn.setStyleSheet(base_style)
        self.btn_ds.clicked.connect(lambda: self._switch_provider("deepseek"))
        self.btn_qw.clicked.connect(lambda: self._switch_provider("qwen"))
        provider_row.addWidget(self.btn_ds)
        provider_row.addWidget(self.btn_qw)
        provider_row.addStretch()
        cl.addLayout(provider_row)

        # API Key
        lbl_key = QLabel("API Key")
        lbl_key.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        lbl_key.setStyleSheet(f"color: {NAU['text']}; background: transparent;")
        cl.addWidget(lbl_key)

        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("sk-...")
        self.key_input.setFont(QFont("Microsoft YaHei", 11))
        self.key_input.setFixedHeight(38)
        self.key_input.setStyleSheet(f"""
            QLineEdit {{ background: {NAU['bg']}; border: 1px solid {NAU['border']};
            border-radius: 8px; padding: 8px 14px; color: {NAU['text']}; }}
            QLineEdit:focus {{ border-color: {NAU['primary']}; }}
        """)
        cl.addWidget(self.key_input)

        # 模型选择
        lbl_model = QLabel("模型")
        lbl_model.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        lbl_model.setStyleSheet(f"color: {NAU['text']}; background: transparent;")
        cl.addWidget(lbl_model)

        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Microsoft YaHei", 11))
        self.model_combo.setFixedHeight(38)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{ background: {NAU['bg']}; border: 1px solid {NAU['border']};
            border-radius: 8px; padding: 8px 14px; color: {NAU['text']}; }}
            QComboBox:hover {{ border-color: {NAU['primary']}; }}
            QComboBox::drop-down {{ border: none; width: 28px; }}
            QComboBox QAbstractItemView {{ background: {NAU['card']}; border: 1px solid {NAU['border']};
            border-radius: 6px; padding: 4px; outline: none; }}
            QComboBox QAbstractItemView::item {{ padding: 6px 12px; border-radius: 4px; }}
            QComboBox QAbstractItemView::item:hover {{ background: {NAU['primary_light']}; color: {NAU['primary']}; }}
        """)
        cl.addWidget(self.model_combo)

        # 启用开关
        self.enable_check = QCheckBox("启用云端 AI 深度分析")
        self.enable_check.setFont(QFont("Microsoft YaHei", 12))
        self.enable_check.setStyleSheet(f"""
            QCheckBox {{ color: {NAU['text']}; spacing: 8px; background: transparent; }}
            QCheckBox::indicator {{ width: 20px; height: 20px; border-radius: 4px;
            border: 2px solid {NAU['border']}; background: {NAU['bg']}; }}
            QCheckBox::indicator:checked {{ background: {NAU['primary']}; border-color: {NAU['primary']}; }}
        """)
        cl.addWidget(self.enable_check)
        cl.addSpacing(8)

        # ── 高德地图（可选）──
        lbl_amap = QLabel("高德地图 Key（可选）")
        lbl_amap.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        lbl_amap.setStyleSheet(f"color: {NAU['text']}; background: transparent;")
        cl.addWidget(lbl_amap)

        hint_amap = QLabel("用于「地址解析」与导出的网页版地图。到 lbs.amap.com 免费申请：\n"
                           "Web服务 → 填到上一行；Web端(JS API) → 填到下一行。不填也能用离线地图底图。")
        hint_amap.setFont(QFont("Microsoft YaHei", 10))
        hint_amap.setWordWrap(True)
        hint_amap.setStyleSheet(f"color: {NAU['subtext']}; background: transparent;")
        cl.addWidget(hint_amap)

        self.amap_key_input = QLineEdit()
        self.amap_key_input.setPlaceholderText("高德 Web服务 Key（逆地理编码/静态图）")
        self.amap_key_input.setFont(QFont("Microsoft YaHei", 11))
        self.amap_key_input.setFixedHeight(38)
        self.amap_key_input.setStyleSheet(f"""
            QLineEdit {{ background: {NAU['bg']}; border: 1px solid {NAU['border']};
            border-radius: 8px; padding: 8px 14px; color: {NAU['text']}; }}
            QLineEdit:focus {{ border-color: {NAU['primary']}; }}
        """)
        cl.addWidget(self.amap_key_input)

        self.amap_js_input = QLineEdit()
        self.amap_js_input.setPlaceholderText("高德 Web端(JS API) Key（网页版地图）")
        self.amap_js_input.setFont(QFont("Microsoft YaHei", 11))
        self.amap_js_input.setFixedHeight(38)
        self.amap_js_input.setStyleSheet(self.amap_key_input.styleSheet())
        cl.addWidget(self.amap_js_input)
        cl.addSpacing(8)

        # 保存按钮
        btn_save = QPushButton("保存设置")
        btn_save.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        btn_save.setFixedSize(140, 42)
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setStyleSheet(f"""
            QPushButton {{ background: {NAU['primary']}; color: white; border: none; border-radius: 10px; }}
            QPushButton:hover {{ background: {NAU['primary_hover']}; }}
        """)
        btn_save.clicked.connect(self._save)
        cl.addWidget(btn_save, alignment=Qt.AlignLeft)

        root.addWidget(card)
        root.addStretch()

        self._load_values()

    def _switch_provider(self, provider):
        CONFIG["llm_provider"] = provider
        base = f"""
            QPushButton {{
                background: {NAU['bg']}; color: {NAU['text']};
                border: 1px solid {NAU['border']}; border-radius: 8px; padding: 6px 20px;
            }}
            QPushButton:hover {{ border-color: {NAU['primary']}; color: {NAU['primary']}; }}
        """
        active = f"""
            QPushButton {{
                background: {NAU['primary']}; color: white;
                border: 1px solid {NAU['primary']}; border-radius: 8px; padding: 6px 20px;
            }}
            QPushButton:hover {{ background: {NAU['primary_hover']}; }}
        """
        for key, btn in self._provider_btns.items():
            btn.setStyleSheet(active if key == provider else base)
        self._update_model_list()
        self._update_key_field()
        self._refresh_nav_buttons()

    def _refresh_nav_buttons(self):
        """确保前进后退按钮状态与当前导航历史一致"""
        if hasattr(self, '_btn_back'):
            self._btn_back.setEnabled(nav_can_back())
        if hasattr(self, '_btn_fwd'):
            self._btn_fwd.setEnabled(nav_can_forward())

    def _update_model_list(self):
        self.model_combo.clear()
        if CONFIG["llm_provider"] == "deepseek":
            self.model_combo.addItems(["deepseek-chat", "deepseek-reasoner"])
        else:
            self.model_combo.addItems(["qwen-vl-plus", "qwen-vl-max", "qwen-plus", "qwen-max"])

    def _update_key_field(self):
        if CONFIG["llm_provider"] == "deepseek":
            self.key_input.setText(CONFIG["deepseek_api_key"])
            self.key_input.setPlaceholderText("sk-...  (DeepSeek API Key)")
        else:
            self.key_input.setText(CONFIG["dashscope_api_key"])
            self.key_input.setPlaceholderText("sk-...  (DashScope API Key)")

    def _load_values(self):
        self._switch_provider(CONFIG["llm_provider"])
        current_model = CONFIG.get(
            "deepseek_model" if CONFIG["llm_provider"] == "deepseek" else "qwen_model", "")
        idx = self.model_combo.findText(current_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.enable_check.setChecked(CONFIG["enable_llm"])
        self.amap_key_input.setText(CONFIG.get("amap_web_key", ""))
        self.amap_js_input.setText(CONFIG.get("amap_js_key", ""))

    def _save(self):
        provider = CONFIG["llm_provider"]
        key = self.key_input.text().strip()
        model = self.model_combo.currentText()
        if provider == "deepseek":
            CONFIG["deepseek_api_key"] = key
            CONFIG["deepseek_model"] = model
        else:
            CONFIG["dashscope_api_key"] = key
            CONFIG["qwen_model"] = model
        CONFIG["enable_llm"] = self.enable_check.isChecked()
        CONFIG["amap_web_key"] = self.amap_key_input.text().strip()
        CONFIG["amap_js_key"] = self.amap_js_input.text().strip()
        save_config(CONFIG)
        QMessageBox.information(self, "成功", "API 设置已保存！")
        self._refresh_nav_buttons()


# ── 把主程序的窗口装饰与首页工厂注入地图模块，保证界面风格统一 ──
if GEO_AVAILABLE:
    rice_geo.install_chrome(center_fn=center_window, round_fn=make_rounded,
                            home_factory=lambda: HomeWindow())


# ── 主入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 全局异常捕获，防止闪退
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        log.error("未捕获异常: %s", exc_value, exc_info=(exc_type, exc_value, exc_tb))
        QMessageBox.critical(None, "程序异常",
            f"{exc_type.__name__}: {exc_value}\n\n{traceback.format_exc()[-500:]}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    log.info("程序启动 | 资源目录: %s | 数据目录: %s", BASE_DIR, APP_DIR)
    if not os.path.exists(MODEL_PATH):
        log.error("模型文件缺失: %s", MODEL_PATH)

    os.makedirs(APP_DIR / "resources", exist_ok=True)
    init_treatment_dirs()
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    else:
        log.warning("图标文件缺失，使用系统默认图标: %s", ICON_PATH)
    window = HomeWindow()
    nav_record(HomeWindow)
    window.show()
    sys.exit(app.exec_())
