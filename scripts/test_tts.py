# -*- coding: utf-8 -*-
"""试听语音：Edge 在线 -> Piper 本地 -> 系统 SAPI。
运行：  D:\\anoconda3\\envs\\rice-gpu\\python.exe scripts\\test_tts.py
"""
import inspect
import os
import sys
import tempfile
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 的上一级是项目根
MODEL = os.path.join(ROOT, 'resources', 'tts', 'zh_CN-huayan-medium.onnx')
TEXT = '白叶枯病建议使用噻唑锌防治，每亩用百分之二十悬浮剂100到125毫升，兑水30公斤喷雾。'
D = tempfile.gettempdir()


def piper_synth(voice, text, wav_path):
    rate = getattr(getattr(voice, 'config', None), 'sample_rate', 22050)

    def _write(how):
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(rate))
            if how == 'wav':
                voice.synthesize_wav(text, wf)
            elif how == 'two':
                voice.synthesize(text, wf)
            else:
                for chunk in voice.synthesize(text):
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
    for how in order:
        try:
            frames = _write(how)
        except Exception as e:
            print('        方式 %s 报错: %s' % (how, str(e)[:60]))
            continue
        print('        方式 %s -> %d 帧' % (how, frames))
        if frames > 0:
            return frames, rate, how
    return 0, rate, None


def try_edge():
    print('[1/3] Edge 在线语音')
    try:
        import edge_tts_lite
        out = os.path.join(D, 'test_edge.mp3')
        t0 = time.time()
        n = edge_tts_lite.synthesize(TEXT, 'zh-CN-XiaoxiaoNeural', out, rate='-8%')
        print('      OK  %.1f KB  用时 %.1f 秒' % (n / 1024, time.time() - t0))
        return out
    except Exception as e:
        print('      不可用:', str(e)[:70])
        return None


def try_piper():
    print('[2/3] Piper 本地语音（离线）')
    if not os.path.exists(MODEL):
        print('      模型不存在:', MODEL)
        return None
    try:
        import piper
        from piper import PiperVoice
        print('      piper-tts 版本:', getattr(piper, '__version__', '?'))
    except ImportError:
        print('      没装: %s -m pip install piper-tts' % sys.executable)
        return None
    try:
        out = os.path.join(D, 'test_piper.wav')
        t0 = time.time()
        voice = PiperVoice.load(MODEL)
        load_t = time.time() - t0
        t0 = time.time()
        frames, rate, how = piper_synth(voice, TEXT, out)
        syn_t = time.time() - t0
        if frames <= 0:
            print('      三种方式都没产出音频')
            return None
        print('      OK  %d 帧 / %.1f 秒音频  @%dHz  加载%.1fs 合成%.1fs'
              % (frames, frames / float(rate), rate, load_t, syn_t))
        return out
    except Exception as e:
        print('      失败:', type(e).__name__, str(e)[:100])
        return None


if __name__ == '__main__':
    print('测试文本:', TEXT[:22] + '…')
    print()
    e = try_edge()
    p = try_piper()
    print('[3/3] 系统 SAPI（兜底）')
    try:
        import win32com.client
        sapi = win32com.client.Dispatch('SAPI.SpVoice')
        print('      可用（音质一般）')
    except Exception as ex:
        sapi = None
        print('      不可用:', str(ex)[:50])
    print()
    print('=== 试听（优先放 Edge 晓晓）===')
    played = e or p
    if played:
        print('播放:', os.path.basename(played))
        os.startfile(played)
        if e and p:
            print('（Piper 的版本在 %s，想对比可以手动打开）' % os.path.basename(p))
    elif sapi:
        sapi.Speak(TEXT)
    print()
    print('小结: Edge %s / Piper %s / SAPI %s' % ('OK' if e else 'X', 'OK' if p else 'X', 'OK' if sapi else 'X'))