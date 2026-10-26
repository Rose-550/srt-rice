# -*- coding: utf-8 -*-
"""最小 Edge 语音合成客户端（自写，不用 edge-tts 包）。

为什么不用 edge-tts：它在 websockets 上用 headers= 传参，会顶掉库的默认头，
导致 Sec-WebSocket-Version 重复，握手必 403（实测定位）。

协议：连上 → 发配置 → 发 SSML → 收二进制音频帧 → Path:turn.end 结束。
长文本会按标点自动分段，逐段合成再拼接，避免单次请求超时。
"""
import asyncio
import hashlib
import json
import re
import time
import uuid

TOKEN = "6A5AA1D4EAFF4E9FB37E23D68491D6F4"
EDGE_VERSION = "1-132.0.2957.140"
URL = "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1"
HEADERS = {
    "Origin": "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold",
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                   " (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0"),
}
FORMAT = "audio-24khz-48kbitrate-mono-mp3"
CHUNK_LIMIT = 280          # 单次合成的字数上限


def sec_ms_gec():
    """微软要求的鉴权摘要，随时间变化（每 5 分钟一档）"""
    ticks = int(time.time() + 11644473600) * 10_000_000
    ticks -= ticks % 3_000_000_000
    return hashlib.sha256((str(ticks) + TOKEN).encode("ascii")).hexdigest().upper()


def split_text(text, limit=CHUNK_LIMIT):
    """按句末标点切分，保证每段不超过 limit 字"""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；!?;])", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > limit and cur:
            chunks.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        chunks.append(cur)
    return chunks


def _ts():
    return time.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)",
                         time.gmtime())


def _escape(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _synth_bytes(text, voice, rate, volume, pitch):
    """合成一段，返回 mp3 字节"""
    import websockets
    qs = ("?TrustedClientToken=" + TOKEN + "&ConnectionId=" + uuid.uuid4().hex +
          "&Sec-MS-GEC=" + sec_ms_gec() + "&Sec-MS-GEC-Version=" + EDGE_VERSION)
    async with websockets.connect(URL + qs, additional_headers=HEADERS,
                                  open_timeout=30, close_timeout=10,
                                  max_size=None) as ws:
        cfg = json.dumps({"context": {"synthesis": {"audio": {
            "metadataoptions": {"sentenceBoundaryEnabled": "false",
                                "wordBoundaryEnabled": "false"},
            "outputFormat": FORMAT}}}})
        await ws.send("X-Timestamp:" + _ts() +
                      "\r\nContent-Type:application/json; charset=utf-8" +
                      "\r\nPath:speech.config\r\n\r\n" + cfg)
        ssml = ("<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
                "xml:lang='zh-CN'><voice name='" + voice + "'><prosody pitch='" + pitch +
                "' rate='" + rate + "' volume='" + volume + "'>" + _escape(text) +
                "</prosody></voice></speak>")
        await ws.send("X-RequestId:" + uuid.uuid4().hex +
                      "\r\nContent-Type:application/ssml+xml" +
                      "\r\nX-Timestamp:" + _ts() + "Z\r\nPath:ssml\r\n\r\n" + ssml)
        audio = bytearray()
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=90)
            if isinstance(msg, bytes):
                if len(msg) < 2:
                    continue
                hlen = int.from_bytes(msg[:2], "big")
                head = msg[2:2 + hlen].decode("utf-8", "ignore")
                if "Path:audio" in head:
                    audio.extend(msg[2 + hlen:])
            elif "Path:turn.end" in msg:
                break
        return bytes(audio)


def synthesize(text, voice="zh-CN-XiaoxiaoNeural", out_path=None,
               rate="-8%", volume="+0%", pitch="+0Hz", retry=2):
    """合成整段文本到 out_path，返回字节数。

    长文本自动分段，逐段合成后拼接；单段失败会重试，中途失败则保留已合成部分。
    """
    if out_path is None:
        import tempfile
        out_path = tempfile.gettempdir() + "/edge_lite.mp3"
    chunks = split_text(text)
    if not chunks:
        return 0
    audio = bytearray()
    for i, c in enumerate(chunks):
        for attempt in range(retry):
            try:
                audio.extend(asyncio.run(_synth_bytes(c, voice, rate, volume, pitch)))
                break
            except Exception:
                if attempt == retry - 1:
                    if i == 0:
                        raise
                    # 后面某段失败：保留已合成的，别让整条朗读白费
                    with open(out_path, "wb") as f:
                        f.write(bytes(audio))
                    return len(audio)
                time.sleep(0.8)
    with open(out_path, "wb") as f:
        f.write(bytes(audio))
    return len(audio)
