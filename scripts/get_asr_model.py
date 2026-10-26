# -*- coding: utf-8 -*-
"""一键准备语音输入：自检并修复环境 + 装依赖 + 下载 Vosk 中文模型。

用法（在本项目目录下，用普通终端运行，不要经过任何沙箱）：
    D:\\anoconda3\\envs\\rice-gpu\\python.exe scripts\\get_asr_model.py
"""
import glob
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 的上一级是项目根
MODEL_DIR = os.path.join(BASE, "resources", "asr")
MODEL_NAME = "vosk-model-small-cn-0.22"


def broken_certifi():
    """找出安装元数据损坏的 certifi（pip 被中途打断会留下半个 dist-info）。"""
    try:
        import certifi
    except ImportError:
        return None
    sp = os.path.dirname(os.path.dirname(os.path.abspath(certifi.__file__)))
    for d in glob.glob(os.path.join(sp, 'certifi-*.dist-info')):
        if not os.path.exists(os.path.join(d, 'METADATA')):
            return d
    return None


def repair_certifi():
    """删掉坏目录再重装。pip 读不到 METADATA 就会拒绝安装任何包。"""
    d = broken_certifi()
    if not d:
        return True
    print("[0/3] 检测到 certifi 安装信息损坏，正在修复…")
    print("       ", d)
    shutil.rmtree(d, ignore_errors=True)
    r = subprocess.call([sys.executable, "-m", "pip", "install",
                         "--force-reinstall", "--no-deps", "certifi"])
    if r == 0:
        print("       修复完成")
        return True
    print("       修复失败，请手动运行：")
    print("       %s -m pip install --force-reinstall --no-deps certifi" % sys.executable)
    return False


def ensure_packages():
    missing = []
    for mod, pkg in (("vosk", "vosk"), ("sounddevice", "sounddevice")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        print("[1/3] 依赖已就绪")
        return True
    print("[1/3] 安装依赖:", ", ".join(missing))
    r = subprocess.call([sys.executable, "-m", "pip", "install"] + missing)
    return r == 0


def ensure_model():
    target = os.path.join(MODEL_DIR, MODEL_NAME)
    if os.path.isdir(target):
        print("[2/3] 模型已存在")
        return True
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[2/3] 下载中文模型（42MB，请稍候）…")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=900).read()
        print("       下载完成 %.1f MB，解压中…" % (len(data) / 1024 / 1024))
        zipfile.ZipFile(io.BytesIO(data)).extractall(MODEL_DIR)
        print("       解压完成")
        return True
    except Exception as e:
        print("       下载失败:", e)
        print("       请手动下载 %s" % MODEL_URL)
        print("       解压到 %s" % MODEL_DIR)
        return False


if __name__ == "__main__":
    ok0 = repair_certifi()
    ok1 = ensure_packages() if ok0 else False
    ok2 = ensure_model()          # 模型下载不依赖 pip，单独进行
    print()
    print("[3/3] 检查结果")
    if ok0 and ok1 and ok2:
        print("  全部就绪。运行 rice_dp1_fluent.py，AI 面板的「语音」按钮即可使用。")
    else:
        print("  有步骤未完成，请看上面的提示。")
        if not ok1:
            print("  依赖没装上 → 语音输入不可用（朗读不受影响，它是系统自带的）")