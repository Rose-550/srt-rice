# -*- coding: utf-8 -*-
"""一键准备本地语音合成（ChatTTS）。

用法：
    D:\\anoconda3\\envs\\rice-gpu\\python.exe scripts\\get_tts_model.py

两个关键点：
1. 装依赖和下载模型必须分开两个进程（同进程 sys.modules 里的旧模块换不掉）
2. 走 HuggingFace 镜像 hf-mirror.com：ChatTTS 自带的 modelscope 源有 bug
   （modelscope 包没装时会 UnboundLocalError: download_path）
"""
import importlib
import os
import subprocess
import sys
import time

MIRROR = "https://hf-mirror.com"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 的上一级是项目根


def pip(*args):
    r = subprocess.call([sys.executable, '-m', 'pip', 'install'] + list(args))
    importlib.invalidate_caches()
    return r == 0


def stage1():
    ok = True
    try:
        import ChatTTS  # noqa
        print('[1/2] ChatTTS 已就绪')
    except Exception:
        print('[1/2] 安装 ChatTTS…')
        ok = pip('ChatTTS')
    try:
        import transformers
        ver = transformers.__version__
        if int(ver.split('.')[0]) < 5:
            print('[2/2] transformers %s 兼容' % ver)
        else:
            print('[2/2] transformers %s 太新，降级到 4.x…' % ver)
            ok = pip('transformers>=4.41,<5') and ok
    except ImportError:
        print('[2/2] 安装 transformers 4.x…')
        ok = pip('transformers>=4.41,<5') and ok
    return ok


def stage2():
    # 镜像必须在 import huggingface_hub 之前设好
    os.environ['HF_ENDPOINT'] = MIRROR
    from transformers.utils import logging as hf_log
    hf_log.set_verbosity_error()
    try:
        import ChatTTS
        print('[1/2] ChatTTS 导入成功')
    except Exception as e:
        print('[1/2] 导入失败:', type(e).__name__, str(e)[:120])
        return False

    # 优先用本地已有缓存；否则从镜像下
    local = os.path.join(BASE, 'resources', 'tts', 'chattts')
    chat = ChatTTS.Chat()
    loaded = False
    if os.path.isdir(local) and os.listdir(local):
        print('[2/2] 用本地模型:', local)
        try:
            loaded = chat.load(source='local', custom_path=local, compile=False)
        except Exception as e:
            print('      本地加载失败:', str(e)[:90])
    if not loaded:
        print('[2/2] 从镜像 %s 下载模型（约 1GB，请耐心等）…' % MIRROR)
        t0 = time.time()
        try:
            loaded = chat.load(source='huggingface', compile=False)
            print('      就绪，用时 %.0f 秒' % (time.time() - t0))
        except Exception as e:
            print('      下载失败:', type(e).__name__, str(e)[:150])
            return False
    if not loaded:
        print('      加载返回 False')
        return False
    print('      试合成一句…')
    try:
        import torch
        import torchaudio
        torch.manual_seed(42)
        t1 = time.time()
        wavs = chat.infer(['白叶枯病建议使用噻唑锌防治，每亩用百分之二十悬浮剂一百到一百二十五毫升。'])
        out = os.path.join(BASE, '_chattts_test.wav')
        torchaudio.save(out, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
        print('      合成 %.1f 秒 -> %s' % (time.time() - t1, os.path.basename(out)))
        os.startfile(out)
        return True
    except Exception as e:
        print('      合成失败:', type(e).__name__, str(e)[:150])
        return False


if __name__ == '__main__':
    if '--stage2' in sys.argv:
        ok = stage2()
    else:
        ok = stage1()
        if ok:
            print()
            print('依赖就绪，换新进程继续（旧模块还在内存里，必须换）…')
            print('-' * 56)
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), '--stage2'])
            ok = (rc == 0)
            print('-' * 56)
    print()
    if ok:
        print('全部就绪。启动 rice_dp1_fluent.py，点「朗读」就是本地语音。')
    else:
        print('未完成，看上面提示。')
    sys.exit(0 if ok else 1)