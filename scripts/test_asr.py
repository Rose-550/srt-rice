# -*- coding: utf-8 -*-
"""验证语音识别能不能用。在项目目录下运行：
    D:\\anoconda3\\envs\\rice-gpu\\python.exe scripts\\test_asr.py
"""
import os
import sys
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 的上一级是项目根
SRC = os.path.join(ROOT, 'resources', 'asr', 'vosk-model-small-cn-0.22')

print('1. 模型目录:', SRC)
print('   存在:', os.path.isdir(SRC))
print('   路径含中文:', not all(ord(c) < 128 for c in SRC))
print()
print('2. 找一个纯英文目录放模型…')
TARGET = None
for base in (r'C:\Users\Public\rice_dp1\asr',
             r'C:\ProgramData\rice_dp1\asr',
             r'D:\rice_dp1\asr'):
    dest = os.path.join(base, 'vosk-model-small-cn-0.22')
    try:
        if not os.path.isdir(dest):
            os.makedirs(base, exist_ok=True)
            t0 = time.time()
            shutil.copytree(SRC, dest)
            print('   已复制到', dest, '(%.1f 秒)' % (time.time() - t0))
        else:
            print('   已存在', dest)
        TARGET = dest
        break
    except Exception as e:
        print('   ', base, '不可用:', type(e).__name__)
print()
if not TARGET:
    print('!!! 没有可写的英文目录，请手动把 resources\\asr 复制到 C:\\Users\\Public\\ 下')
    sys.exit(1)
print('3. 用 Vosk 加载模型…')
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)
try:
    t0 = time.time()
    model = Model(TARGET)
    rec = KaldiRecognizer(model, 16000)
    print('   加载成功！耗时 %.1f 秒' % (time.time() - t0))
    print()
    print('结论：语音识别可用。启动 rice_dp1_fluent.py 即可。')
except Exception as e:
    print('   加载失败:', type(e).__name__, e)
    sys.exit(1)