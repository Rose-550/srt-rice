# -*- coding: utf-8 -*-
"""程序化生成的水稻植株 + 软件三维渲染器（零依赖，只用 QPainter）。

为什么自己写而不用 Three.js / PyVista：
  · 零新依赖，不用再折腾 pip（这台机器上已经踩过好几次依赖坑）
  · 全部自己实现，答辩时讲得清楚，不是"调了个库"
  · 科普展示不需要照片级真实感，懂结构就行

实现要点：
  1. 建模：把茎、叶、穗都表示成一串四边形（billboard 条带），够用且简单
  2. 变换：绕 Y 轴（水平旋转）+ 绕 X 轴（俯仰）转，再透视投影
  3. 绘制：画家算法（按面片平均深度从远到近画），颜色按朝向做假光照
"""
import math

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QColor, QPolygonF, QPen, QBrush
from PyQt5.QtWidgets import QWidget


# ── 向量工具 ──────────────────────────────────────────────
class V3(object):
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return V3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return V3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k):
        return V3(self.x * k, self.y * k, self.z * k)

    def cross(self, o):
        return V3(self.y * o.z - self.z * o.y,
                  self.z * o.x - self.x * o.z,
                  self.x * o.y - self.y * o.x)

    def norm(self):
        n = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z) or 1.0
        return V3(self.x / n, self.y / n, self.z / n)


def lerp(a, b, t):
    return a + (b - a) * t


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return V3(u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
              u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
              u * u * u * p0.z + 3 * u * u * t * p1.z + 3 * u * t * t * p2.z + t * t * t * p3.z)


# ── 建模 ─────────────────────────────────────────────────
# 部位名 -> 基色（受病害影响时会被高亮覆盖）
PART_COLOR = {
    "stem":    (150, 178, 96),
    "leaf":    (108, 168, 82),
    "leaf_tip": (186, 176, 88),
    "panicle": (216, 190, 112),
    "grain":   (236, 214, 146),
    "sheath":  (172, 184, 116),
}


class Plant(object):
    """一株水稻。faces 里每项是 (四个 V3 顶点, 部位名, 明暗系数)"""

    def __init__(self, seed=7):
        self.faces = []
        self.parts = []          # 去重后的部位名列表
        import random
        rnd = random.Random(seed)
        self._build(rnd)

    # -- 条带：把一串中心点扫成宽度渐变的四边形 --
    def _strip(self, pts, w0, w1, part, shade=1.0, axis=None):
        n = len(pts)
        for i in range(n - 1):
            t0, t1 = i / float(n - 1), (i + 1) / float(n - 1)
            a, b = pts[i], pts[i + 1]
            d = (b - a)
            # 用给定轴或世界 Z 轴求侧向
            ax = axis or V3(0, 0, 1)
            side = d.cross(ax)
            if abs(side.x) + abs(side.y) + abs(side.z) < 1e-6:
                side = V3(1, 0, 0)
            side = side.norm()
            wa, wb = lerp(w0, w1, t0) * 0.5, lerp(w0, w1, t1) * 0.5
            self.faces.append(([a + side * wa, b + side * wb,
                                b - side * wb, a - side * wa], part, shade))

    def _build(self, rnd):
        H = 1.0                        # 株高
        n_tiller = 4
        for ti in range(n_tiller):
            ang = ti * (2 * math.pi / n_tiller) + rnd.uniform(-0.25, 0.25)
            lean = 0.10 + rnd.uniform(0, 0.10)
            tilt = rnd.uniform(0.05, 0.20)
            h = H * rnd.uniform(0.82, 1.0)
            dx, dz = math.cos(ang) * lean, math.sin(ang) * lean
            # 茎：略微外倾
            stem = []
            for k in range(9):
                t = k / 8.0
                stem.append(V3(dx * t * t, h * t, dz * t * t))
            self._strip(stem, 0.030, 0.018, "stem", 0.92)
            # 叶鞘
            self._strip(stem[:3], 0.048, 0.034, "sheath", 0.86)
            # 叶：从茎的几个高度抽出，向外下垂
            for li in range(3):
                lv = 0.30 + li * 0.22
                base = stem[min(8, int(lv * 8))]
                la = ang + (li - 1.0) * 1.05 + rnd.uniform(-0.18, 0.18)
                L = 0.50 + rnd.uniform(-0.06, 0.10)
                p0 = base
                p1 = base + V3(math.cos(la) * L * 0.40, 0.16, math.sin(la) * L * 0.40)
                p2 = base + V3(math.cos(la) * L * 0.82, 0.10, math.sin(la) * L * 0.82)
                p3 = base + V3(math.cos(la) * L * 1.10, -0.14, math.sin(la) * L * 1.10)
                pts = [bezier(p0, p1, p2, p3, k / 9.0) for k in range(10)]
                self._strip(pts[:6], 0.042, 0.050, "leaf", 1.0)
                self._strip(pts[:6], 0.012, 0.014, "leaf", 0.86,
                            axis=V3(0, 0, 1))
                self._strip(pts[5:], 0.050, 0.004, "leaf_tip", 1.06)
            # 穗：只在部分分蘖上
            if ti % 2 == 0:
                top = stem[-1]
                pa = ang + rnd.uniform(-0.3, 0.3)
                q0 = top
                q1 = top + V3(0, 0.16, 0)
                q2 = top + V3(math.cos(pa) * 0.16, 0.22, math.sin(pa) * 0.16)
                q3 = top + V3(math.cos(pa) * 0.30, 0.06, math.sin(pa) * 0.30)
                pts = [bezier(q0, q1, q2, q3, k / 8.0) for k in range(9)]
                self._strip(pts, 0.016, 0.009, "panicle", 0.98)
                # 谷粒
                for k in range(1, 9):
                    t = k / 9.0
                    c = bezier(q0, q1, q2, q3, t)
                    for s in (-1, 1):
                        o = V3(c.x + s * 0.022, c.y, c.z + s * 0.022)
                        g = 0.012
                        self.faces.append(([o + V3(-g, -g, 0), o + V3(g, -g, 0),
                                            o + V3(g, g, 0), o + V3(-g, g, 0)],
                                           "grain", 1.12 + 0.03 * s))
        # 记录部位
        seen = []
        for _v, part, _s in self.faces:
            if part not in seen:
                seen.append(part)
        self.parts = seen


# ── 渲染 ─────────────────────────────────────────────────
class Rice3DView(QWidget):
    """可旋转/缩放的水稻三维视图。

    鼠标左键拖动旋转，滚轮缩放；支持自动缓慢旋转（演示用），
    以及按部位高亮（用来说明病害发生在植株的哪个位置）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plant = Plant()
        self.yaw = -0.55
        self.pitch = 0.30
        self.zoom = 1.0
        self.highlight = {}
        self.auto_spin = True
        self.show_ground = True
        self._drag = None
        self._pulse = 0.0
        self.setMinimumSize(320, 380)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self._timer = None
        self._start_timer()

    # -- 对外接口 --
    def set_highlight(self, parts, color=(214, 69, 65)):
        """高亮指定部位（病害发生的位置）"""
        self.highlight = {p: QColor(*color) for p in parts}
        self.update()

    def clear_highlight(self):
        self.highlight = {}
        self.update()

    # -- 自动旋转 --
    def _start_timer(self):
        from PyQt5.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def _tick(self):
        if self.auto_spin and self._drag is None:
            self.yaw += 0.007
        self._pulse = (self._pulse + 0.06) % (2 * math.pi)
        self.update()

    # -- 交互 --
    def mousePressEvent(self, e):
        self._drag = e.pos()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            d = e.pos() - self._drag
            self.yaw += d.x() * 0.012
            self.pitch = max(-0.9, min(1.15, self.pitch + d.y() * 0.008))
            self._drag = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, e):
        self.auto_spin = not self.auto_spin

    def wheelEvent(self, e):
        step = 1.0 + (e.angleDelta().y() / 1200.0)
        self.zoom = max(0.45, min(2.6, self.zoom * step))
        self.update()

    # -- 投影 --
    def _project(self, v, w, h):
        # 植株高度是 1.0，得按视口高度留够余量，否则顶部会被切掉
        scale = min(w * 0.60, h * 0.66) * self.zoom
        cx, cy = w * 0.5, h * 0.86
        cyaw, syaw = math.cos(self.yaw), math.sin(self.yaw)
        x = v.x * cyaw + v.z * syaw
        z = -v.x * syaw + v.z * cyaw
        y = v.y
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        y2 = y * cp - z * sp
        z2 = y * sp + z * cp
        d = 3.0
        f = d / (d + z2)
        return QPointF(cx + x * scale * f, cy - y2 * scale * f), z2

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()

        # 背景：淡淡的径向渐变，让植株"浮"起来
        from PyQt5.QtGui import QRadialGradient
        g = QRadialGradient(w * 0.5, h * 0.45, max(w, h) * 0.75)
        g.setColorAt(0.0, QColor(255, 255, 255))
        g.setColorAt(1.0, QColor(243, 246, 240))
        p.fillRect(self.rect(), QBrush(g))

        # 地面阴影
        if self.show_ground:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 16))
            gw, gh = w * 0.30 * self.zoom, h * 0.035 * self.zoom
            p.drawEllipse(QPointF(w * 0.5, h * 0.86 + h * 0.02 * self.zoom),
                          gw, gh)

        # 逐面片投影 + 深度排序（画家算法：远的先画）
        draw = []
        for verts, part, shade in self.plant.faces:
            pts = []
            zsum = 0.0
            for v in verts:
                q, z = self._project(v, w, h)
                pts.append(q)
                zsum += z
            # 面法线做假光照：正对光源的亮，背光暗
            a, b, c = verts[0], verts[1], verts[2]
            n = (b - a).cross(c - a).norm()
            light = 0.62 + 0.38 * abs(n.z * 0.55 + n.y * 0.45 + n.x * 0.25)
            draw.append((zsum / 4.0, pts, part, shade * light, b - a))
        draw.sort(key=lambda t: -t[0])

        pulse = 0.5 + 0.5 * math.sin(self._pulse)
        for _z, pts, part, k, _d in draw:
            base = PART_COLOR.get(part, (120, 150, 90))
            if part in self.highlight:
                hc = self.highlight[part]
                mix = 0.34 + 0.30 * pulse
                col = QColor(int(lerp(base[0], hc.red(), mix)),
                             int(lerp(base[1], hc.green(), mix)),
                             int(lerp(base[2], hc.blue(), mix)))
            else:
                col = QColor(int(base[0] * k), int(base[1] * k), int(base[2] * k))
            col = QColor(max(0, min(255, col.red())), max(0, min(255, col.green())),
                         max(0, min(255, col.blue())))
            p.setPen(QPen(col.darker(118), 0.6))
            p.setBrush(col)
            p.drawPolygon(QPolygonF(pts))
        p.end()


# 六种病虫害 → 受影响部位（科普讲解用）
DISEASE_PARTS = {
    "稻纵卷叶螟": (["leaf", "leaf_tip"], (224, 122, 62), "幼虫吐丝卷叶取食叶肉，叶片形成白色条斑"),
    "稻水象甲":   (["leaf", "stem"], (86, 122, 196), "成虫啃食叶肉、幼虫蛀食根部，植株矮小"),
    "稻叶蝉":     (["stem", "sheath"], (186, 96, 176), "刺吸汁液传播病毒，茎鞘出现褐色斑"),
    "白叶枯病":   (["leaf", "leaf_tip"], (214, 69, 65), "叶缘先现水渍状黄斑，沿叶脉扩展成灰白枯条"),
    "胡麻斑病":   (["leaf"], (150, 108, 52), "叶片生出褐色椭圆病斑，中央灰白、边缘深褐"),
    "稻瘟病":     (["leaf", "panicle", "grain"], (196, 60, 60), "叶片现梭形病斑，穗颈受害导致白穗"),
}

