# -*- coding: utf-8 -*-
"""中转服务部署自检：从公网地址完整走一遍「建记录 → 传图 → 补传识别结果 → 取图」

用法：
    python check_relay.py https://your-domain.com [口令]

它会验证：
  1) /api/ping 通不通（域名/证书/反代是否配好）
  2) 三步接口是否都能用
  3) 取回的图片与上传的**字节是否完全一致**（证明不转码、EXIF/GPS 不丢）
  4) 识别结果（label/score）是否真的存进去了 —— 这一步最容易漏
"""
import io
import sys
import json
import hashlib
import urllib.request

try:
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational as R
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def make_photo():
    """造一张带 GPS 的小图；没有 Pillow 就退回一段最小 JPEG 字节"""
    if not HAVE_PIL:
        return (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                b"\xff\xd9")
    buf = io.BytesIO()
    im = Image.new("RGB", (240, 180), (200, 215, 190))
    ex = Image.Exif()
    ex[0x010F] = "check_relay"
    ex[0x0110] = "self-check"
    ex[0x8825] = {1: "N", 2: (R(32, 1), R(0, 1), R(38, 1)),
                  3: "E", 4: (R(118, 1), R(37, 1), R(48, 1))}
    im.save(buf, format="JPEG", exif=ex, quality=88)
    return buf.getvalue()


def call(url, data=None, token="", raw=False, method=None, headers=None, req_raw=False):
    """raw=响应当二进制收（下图）；req_raw=请求体是二进制（传图）。别混：传图的响应仍是 JSON。"""
    # 中转服务认的是 X-Auth-Token（或 ?token=）；写错头名会 401，且只有设了口令才暴露
    hdr = {"X-Auth-Token": token} if token else {}
    hdr.update(headers or {})
    body = None
    if data is not None:
        body = data if isinstance(data, (bytes, bytearray)) else json.dumps(data).encode("utf-8")
        hdr.setdefault("Content-Type", "application/octet-stream" if req_raw
                       else "application/json; charset=utf-8")
    req = urllib.request.Request(url, data=body, headers=hdr,
                                 method=method or ("POST" if body is not None else "GET"))
    with urllib.request.urlopen(req, timeout=30) as resp:
        blob = resp.read()
        return resp.status, (blob if raw else json.loads(blob.decode("utf-8")))


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    base = argv[0].rstrip("/")
    token = argv[1] if len(argv) > 1 else ""
    ok = fail = 0

    def report(name, passed, extra=""):
        nonlocal ok, fail
        if passed:
            ok += 1
            print("  ✅ %s %s" % (name, extra))
        else:
            fail += 1
            print("  ❌ %s %s" % (name, extra))

    print("中转服务自检：%s" % base)
    try:
        st, body = call(base + "/api/ping")
        report("健康检查 /api/ping", st == 200 and body.get("ok"), json.dumps(body, ensure_ascii=False)[:80])
    except Exception as e:
        report("健康检查 /api/ping", False, "连不上：%s" % e)
        print("\n常见原因：域名没解析、证书没装、Nginx 没转发、阿里云安全组没放行 443")
        return 1

    photo = make_photo()
    sha = hashlib.sha256(photo).hexdigest()
    try:
        st, rec = call(base + "/api/records", {"device": "check_relay", "lat": 32.0106,
                                              "lng": 118.63, "note": "部署自检"},
                       token=token)
        rid = rec.get("id")
        report("建记录 /api/records", st == 200 and rid, rid or json.dumps(rec, ensure_ascii=False)[:80])
        if not rid:
            return 1
        st, up = call(base + "/api/upload/" + rid, photo, token=token, req_raw=True,
                      headers={"X-Filename": "check.jpg"})
        report("传图片 /api/upload", st == 200 and up.get("ok"), "%d 字节" % len(photo))
        st, fin = call(base + "/api/finish/" + rid,
                       {"label": "稻瘟病", "score": 0.88, "boxes": 2, "model": "check_relay"},
                       token=token)
        report("补传识别结果 /api/finish", st == 200 and fin.get("ok"))
        st, got = call(base + "/api/image/" + rid, token=token, raw=True)
        same = hashlib.sha256(got).hexdigest() == sha
        report("取回的图片与上传字节一致（不转码）", same, "%d 字节" % len(got))
        st, listing = call(base + "/api/records?since=0&limit=50", token=token)
        rows = listing.get("records") or []
        mine = [r for r in rows if r.get("id") == rid]
        report("识别结果确实存进去了（label/score）",
               bool(mine) and mine[0].get("label") == "稻瘟病"
               and abs(float(mine[0].get("score") or 0) - 0.88) < 1e-6,
               json.dumps(mine[0], ensure_ascii=False)[:110] if mine else "查不到该记录")
    except Exception as e:
        report("三步接口", False, "异常：%s" % e)

    print("\n结果：通过 %d 项，失败 %d 项" % (ok, fail))
    if fail == 0:
        print("部署成功。接下来在程序里：「自动巡检 → 云端同步」填 %s 与口令，点一下就通。" % base)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
