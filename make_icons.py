# -*- coding: utf-8 -*-
"""把 resources/icons/_src/*.svg 渲染成界面用的 PNG 图标。

图标来自 Lucide（https://lucide.dev，ISC 许可，见 resources/icons/NOTICE.txt）。
SVG 里用的是 stroke="currentColor"，这里按色调替换后渲染成 PNG：

    resources/icons/green/   绿色 —— AI 面板、状态栏等浅色背景
    resources/icons/white/   白色 —— 深色按钮上（南农绿/红按钮）
    resources/icons/gray/    灰色 —— 标题栏的小按钮、浅色图标按钮

换颜色/尺寸改下面的 TONES / SIZE 再跑一次即可：

    python make_icons.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QByteArray, Qt                     # noqa: E402
from PyQt5.QtGui import QColor, QImage, QPainter            # noqa: E402
from PyQt5.QtSvg import QSvgRenderer                        # noqa: E402
from PyQt5.QtWidgets import QApplication                    # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "resources", "icons", "_src")
DST_ROOT = os.path.join(ROOT, "resources", "icons")

TONES = {
    "green": "#0F7B4A",     # 主色：和标题/主按钮同色
    "white": "#FFFFFF",     # 深色实心按钮
    "gray": "#7C8783",      # 标题栏 / 次要图标
    "red": "#C0392B",       # 危险操作（描边按钮）
}
SIZE = 64                   # 渲染得比显示大，缩放下来更锐利（显示时约 16-30px）


def render(svg_text, color, size=SIZE):
    data = svg_text.replace('stroke="currentColor"', 'stroke="%s"' % color)
    renderer = QSvgRenderer(QByteArray(data.encode("utf-8")))
    if not renderer.isValid():
        return None
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    renderer.render(painter)
    painter.end()
    return img


def main():
    app = QApplication.instance() or QApplication([])       # noqa: F841
    if not os.path.isdir(SRC_DIR):
        print("找不到源目录:", SRC_DIR)
        return 1
    svgs = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith(".svg"))
    if not svgs:
        print("源目录里没有 svg")
        return 1
    total = 0
    for tone, color in TONES.items():
        out_dir = os.path.join(DST_ROOT, tone)
        os.makedirs(out_dir, exist_ok=True)
        for fn in svgs:
            text = open(os.path.join(SRC_DIR, fn), encoding="utf-8").read()
            img = render(text, color)
            if img is None:
                print("跳过（SVG 解析失败）:", fn)
                continue
            img.save(os.path.join(out_dir, fn[:-4] + ".png"))
            total += 1
        print("  %-6s %s  (-> resources/icons/%s/)" % (tone, color, tone))
    print("共生成 %d 个图标（%d 个源文件 x %d 种色调）" % (total, len(svgs), len(TONES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
