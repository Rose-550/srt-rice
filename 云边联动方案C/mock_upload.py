# -*- coding: utf-8 -*-
"""模拟小程序上传（用于在没有真机/小程序的情况下验证全链路）

它走的是和小程序完全一样的三步接口：
    1) POST /api/records            创建记录，拿到 record_id
    2) POST /api/upload/<id>        上传图片二进制
    3) POST /api/finish/<id>        补传备注等元数据（可选）

用法：
    python mock_upload.py                     # 上传 3 张，模拟南京周边点位
    python mock_upload.py --count 5
    python mock_upload.py --server http://127.0.0.1:9000
    python mock_upload.py --images D:\some\dir

注意：这里用 urllib 手工拼 multipart 之外的最简形式（直接 POST 二进制），
与小程序 wx.uploadFile 的行为一致——wx.uploadFile 也是把文件作为请求体上传。
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
# 默认从项目里找几张现成样本图
DEFAULT_IMAGE_DIRS = [
    os.path.join(ROOT, "..", "手机端可跑版", "samples"),
    os.path.join(ROOT, "..", "data", "raw", "JPEGImages"),
]

# 模拟的识别结论候选（与程序里的六类一致；不传结论就测不到端云一致性）
MOCK_LABELS = ["稻瘟病", "白叶枯病", "胡麻斑病", "稻纵卷叶螟", "稻水象甲", "稻叶蝉"]

# 南京农业大学周边 + 江苏稻区的几个大致点位（模拟田间取点）
BASE_POINTS = [
    (32.0405, 118.8335, "江苏省南京市玄武区·孝陵卫"),
    (31.9870, 118.8760, "江苏省南京市江宁区·秣陵街道"),
    (32.1150, 118.7900, "江苏省南京市浦口区·永宁街道"),
]


def pick_images(count, image_dir=None):
    dirs = [image_dir] if image_dir else DEFAULT_IMAGE_DIRS
    files = []
    for d in dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if os.path.splitext(n)[1].lower() in (".jpg", ".jpeg", ".png"):
                files.append(os.path.join(d, n))
        if len(files) >= count:
            break
    if not files:
        print("找不到可用图片。请用 --images 指定一个含 jpg/png 的目录。")
        sys.exit(1)
    random.shuffle(files)
    return files[:count]


def post_json(url, obj, token=""):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("X-Auth-Token", token)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post_file(url, path, token=""):
    with open(path, "rb") as fh:
        blob = fh.read()
    req = urllib.request.Request(url, data=blob, method="POST", headers={
        "Content-Type": "image/jpeg",
        "X-Filename": os.path.basename(path),
    })
    if token:
        req.add_header("X-Auth-Token", token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("RELAY_BASE", "http://127.0.0.1:9000"))
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--images", default=None)
    ap.add_argument("--token", default=os.environ.get("RELAY_TOKEN", ""))
    # 默认就带识别结论：不带的话，「手机结论有没有汇总到电脑端」这条路径永远测不到
    ap.add_argument("--label", default=None,
                    help="模拟的识别结论（默认随机取一类）")
    ap.add_argument("--no-label", action="store_true",
                    help="只传图不传结论（模拟未接入结论的旧版小程序）")
    args = ap.parse_args()

    base = args.server.rstrip("/")
    files = pick_images(args.count, args.images)
    print("目标服务器: %s" % base)
    print("使用图片  : %d 张\n" % len(files))

    ok = 0
    for i, fp in enumerate(files, 1):
        lat0, lng0, addr = BASE_POINTS[(i - 1) % len(BASE_POINTS)]
        lat = round(lat0 + random.uniform(-0.01, 0.01), 6)   # 加一点抖动，模拟真实取点
        lng = round(lng0 + random.uniform(-0.01, 0.01), 6)
        try:
            created = post_json(base + "/api/records", {
                "device": "mock-miniprogram",
                "lat": lat, "lng": lng,
                "accuracy": random.choice([8, 12, 20, 35]),
                "address": addr,
                "note": "模拟上传 #%d" % i,
            }, args.token)
            rid = created["id"]
            up = post_file(base + "/api/upload/" + rid, fp, args.token)
            # 识别结果必须走 finish —— create 的白名单里没有 label/score
            patch = {"note": "模拟上传 #%d（含图片）" % i}
            label = ""
            if not args.no_label:
                label = args.label or random.choice(MOCK_LABELS)
                patch.update({"label": label,
                              "score": round(random.uniform(0.72, 0.96), 3),
                              "boxes": random.randint(1, 3),
                              "model": "yolov8n-onnx"})
            post_json(base + "/api/finish/" + rid, patch, args.token)
            print("  [%d/%d] OK  id=%s  %.6f,%.6f  %s  %s  (%d bytes)%s" % (
                i, len(files), rid, lat, lng, os.path.basename(fp), addr, up.get("bytes", 0),
                ("  结论=" + label) if label else "  （不传结论）"))
            ok += 1
        except urllib.error.HTTPError as e:
            print("  [%d/%d] 失败 HTTP %s: %s" % (i, len(files), e.code, e.read().decode("utf-8", "ignore")[:200]))
        except Exception as e:
            print("  [%d/%d] 失败 %s: %s" % (i, len(files), type(e).__name__, e))
        time.sleep(0.2)

    print("\n完成：成功 %d / %d" % (ok, len(files)))


if __name__ == "__main__":
    main()
