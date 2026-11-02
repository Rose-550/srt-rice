# -*- coding: utf-8 -*-
"""把 docs/design/mockups/ 下的 HTML 视觉稿渲染成 PNG，并生成 A/B 对比板。

依赖：
  · Node.js + npx playwright（用系统 Edge/Chrome 渲染，不下载浏览器）
  · Pillow（rice-gpu 环境已带）

用法（任意环境，只要 npx 和 PIL 可用）：
    python scripts/render_ui_mockups.py

产物（覆盖写）：
    docs/design/mockups/结果页_方案A_主从两栏.png
    docs/design/mockups/结果页_方案B_控制塔.png
    docs/design/mockups/结果页_方案对比.png
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOCK = ROOT / "docs" / "design" / "mockups"
PORT = 8766
VIEWPORT = "2760,1800"          # HTML 内以 zoom:2 按 1380×900 设计

SHOTS = [
    ("首页_方案_仪表盘.html", "首页_方案_仪表盘.png", "首页 · 仪表盘方案"),
    ("自动巡检_方案_监控台.html", "自动巡检_方案_监控台.png", "自动巡检 · 监控台方案"),
    ("教学与考核_方案_答题工作台.html", "教学与考核_方案_答题工作台.png", "教学与考核 · 答题工作台"),
    ("图片识别_方案_画布信息栏.html", "图片识别_方案_画布信息栏.png", "图片识别 · 画布+信息栏"),
    ("报告记录_方案_记录中心.html", "报告记录_方案_历史会话.png", "报告记录 · 历史会话"),
    ("报告记录_方案_记录中心.html#reports", "报告记录_方案_报告文件.png", "报告记录 · 报告文件"),
    ("地块地图_方案_地图工作台.html", "地块地图_方案_地图工作台.png", "地块地图 · 地图工作台"),
    ("病害科普_方案_3D图鉴.html", "病害科普_方案_3D图鉴.png", "病害科普 · 3D 图鉴"),
    ("防治库_方案_数据维护台.html", "防治库_方案_数据维护台.png", "防治库 · 数据维护台"),
    ("设置_方案_服务与模型.html", "设置_方案_服务与模型.png", "设置 · 服务与模型"),
    ("结果页_方案A_主从两栏.html", "结果页_方案A_治理方案态.png", "方案A · 治理方案（默认）"),
    ("结果页_方案A_主从两栏.html#ai", "结果页_方案A_AI问答态.png", "方案A · AI 问答（占满整列）"),
    ("结果页_方案B_控制塔.html", "结果页_方案B_控制塔.png", "方案B · 控制塔"),
]

# 对比板只收结果页的三个状态；首页单独出图
COMPARE = [s for s in SHOTS if "结果页" in s[0]]


def _url(name):
    """中文名 percent-encode；#fragment 原样保留，用于切换标签页状态"""
    if "#" in name:
        path, frag = name.split("#", 1)
        return "http://127.0.0.1:%d/docs/design/mockups/%s#%s" % (
            PORT, urllib.parse.quote(path), frag)
    return "http://127.0.0.1:%d/docs/design/mockups/%s" % (PORT, urllib.parse.quote(name))


def render(server):
    for i, (src, dst, _) in enumerate(SHOTS):
        url = _url(src)
        # npx 在 Windows 上是 .cmd，需经 shell；输出先落到纯 ASCII 临时路径，避免中文路径问题
        tmp = Path(tempfile.gettempdir()) / ("ui_mock_%d.png" % i)
        cmd = ('npx --yes playwright screenshot --channel msedge '
               '--viewport-size %s --wait-for-timeout 1000 "%s" "%s"'
               % (VIEWPORT, url, tmp))
        print("[render]", src)
        subprocess.run(cmd, check=True, shell=(os.name == "nt"))
        shutil.move(str(tmp), str(MOCK / dst))


def make_compare():
    from PIL import Image, ImageDraw, ImageFont
    imgs = [(Image.open(MOCK / dst), label) for _, dst, label in COMPARE]
    scale = 0.33
    tile_w = int(imgs[0][0].width * scale)
    tile_h = int(imgs[0][0].height * scale)
    gap, top = 30, 84
    W = tile_w * len(imgs) + gap * (len(imgs) + 1)
    H = top + tile_h + gap
    board = Image.new("RGB", (W, H), "#F7F6F1")
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    x = gap
    for img, label in imgs:
        board.paste(img.resize((tile_w, tile_h), Image.LANCZOS), (x, top))
        draw.text((x, 32), label, font=font, fill="#12301F")
        x += tile_w + gap
    out = MOCK / "结果页_方案对比.png"
    board.save(out)
    print("[compare]", out, board.size)


def make_page_compare(before, after_dst, out_name, before_label, after_label):
    """任一页面：当前素材截图 vs 新版视觉稿，并排对比"""
    from PIL import Image, ImageDraw, ImageFont
    after = MOCK / after_dst
    if not before.exists() or not after.exists():
        print("[compare] 跳过：缺少截图", before)
        return
    H = 880
    tiles = []
    for path, label in ((before, before_label), (after, after_label)):
        im = Image.open(path)
        w = int(im.width * H / im.height)
        tiles.append((im.resize((w, H), Image.LANCZOS), label, w))
    gap, top = 36, 80
    W = sum(t[2] for t in tiles) + gap * (len(tiles) + 1)
    board = Image.new("RGB", (W, top + H + gap), "#F7F6F1")
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    x = gap
    for img, label, w in tiles:
        board.paste(img, (x, top))
        draw.text((x, 30), label, font=font, fill="#12301F")
        x += w + gap
    out = MOCK / out_name
    board.save(out)
    print("[compare]", out, board.size)


def make_home_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "01_电脑端_首页.png",
        "首页_方案_仪表盘.png", "首页_改版对比.png", "现状（v3.10）", "改版方案 · 仪表盘")


def make_patrol_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "04_电脑端_自动巡检台账.png",
        "自动巡检_方案_监控台.png", "自动巡检_改版对比.png", "现状（v3.10）", "改版方案 · 监控台")


def make_teach_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "07_电脑端_教学与考核.png",
        "教学与考核_方案_答题工作台.png", "教学与考核_改版对比.png", "现状（v3.10）", "改版方案 · 答题工作台")


def make_imgrec_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "02_电脑端_图片识别.png",
        "图片识别_方案_画布信息栏.png", "图片识别_改版对比.png", "现状（v3.10）", "改版方案 · 画布+信息栏")


def make_report_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "08_电脑端_报告记录.png",
        "报告记录_方案_历史会话.png", "报告记录_改版对比.png", "现状（v3.10）", "改版方案 · 记录中心")


def make_map_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "05_电脑端_地块地图.png",
        "地块地图_方案_地图工作台.png", "地块地图_改版对比.png", "现状（v3.10）", "改版方案 · 地图工作台")


def make_science_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "09_电脑端_病害科普3D.png",
        "病害科普_方案_3D图鉴.png", "病害科普_改版对比.png", "现状（v3.10）", "改版方案 · 3D 图鉴")


def make_treat_compare():
    make_page_compare(
        ROOT / "AI+教育大赛" / "素材截图" / "06_电脑端_防治库维护.png",
        "防治库_方案_数据维护台.png", "防治库_改版对比.png", "现状（v3.10）", "改版方案 · 数据维护台")


def main():
    if not MOCK.is_dir():
        sys.exit("找不到目录：%s" % MOCK)
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT),
                               "--bind", "127.0.0.1"], cwd=str(ROOT),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        urllib.request.urlopen("http://127.0.0.1:%d/" % PORT, timeout=5)
        render(server)
        make_compare()
        make_home_compare()
        make_patrol_compare()
        make_teach_compare()
        make_imgrec_compare()
        make_report_compare()
        make_map_compare()
        make_science_compare()
        make_treat_compare()
    finally:
        server.terminate()
        server.wait(timeout=5)
    print("完成。")


if __name__ == "__main__":
    main()
