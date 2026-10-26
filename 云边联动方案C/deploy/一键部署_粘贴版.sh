#!/usr/bin/env bash
# 稻视界 · 中转服务「一键粘贴版」
# 用法：在服务器终端（控制台「远程连接」的网页终端即可）粘贴本文件全部内容并回车。
# 可选：bash 本脚本 域名 口令   （不传就用下面的默认值）
set -e
DOMAIN="$1"
if [ -z "$DOMAIN" ]; then DOMAIN="rice-srt.cn"; fi
TOKEN="$2"
if [ -z "$TOKEN" ]; then TOKEN="rice-srt-2026-1014"; fi
APP=/opt/rice-relay
echo "==> 域名 $DOMAIN / 口令 $TOKEN"

echo "==> 建目录与用户"
id -u rice >/dev/null 2>&1 || useradd --system --home $APP --shell /sbin/nologin rice
mkdir -p $APP/data

echo "==> 写入服务代码"
cat > $APP/relay_server.py <<'__RELAY_PY__'
# -*- coding: utf-8 -*-
"""稻视界 · 云边联动（方案 C）中转服务端

只用 Python 标准库，零依赖。本地测试和丢到云服务器上跑的是同一份代码。

接口：
    GET  /api/ping                      健康检查
    POST /api/records                   创建记录，返回 record_id 与上传地址（小程序调用）
    POST /api/upload/<record_id>        上传图片二进制（小程序调用）
    POST /api/finish/<record_id>        可选：补传/更新元数据（小程序调用）
    GET  /api/records?since=&limit=     增量拉取记录（电脑端调用）
    GET  /api/image/<record_id>         下载图片（电脑端调用）

存储：
    data/records.jsonl      记录索引，一行一条，方便增量拉取
    data/images/<id>.jpg    原始图片
    data/meta/<id>.json     单条元数据

启动：
    python relay_server.py            # 本地，端口 9000
    python relay_server.py 8080       # 指定端口

部署到云服务器时：
    1) 用 systemd / supervisor / nohup 常驻
    2) 前面挂 Nginx 并配好 HTTPS（小程序正式版要求 https 域名）
    3) 设置环境变量 RELAY_TOKEN 作为电脑端拉取口令
"""
import json
import os
import re
import ssl
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(DATA, "images")
META_DIR = os.path.join(DATA, "meta")
INDEX = os.path.join(DATA, "records.jsonl")

MAX_UPLOAD = 20 * 1024 * 1024          # 单张图片上限 20MB
LOCK = threading.Lock()                # 索引写入串行化
RELAY_TOKEN = os.environ.get("RELAY_TOKEN", "")   # 为空则不校验（仅建议本地测试时为空）

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


def ensure_dirs():
    for d in (DATA, IMG_DIR, META_DIR):
        os.makedirs(d, exist_ok=True)


def load_index():
    """读全部记录（按时间升序）。数据量小时足够；上万条再换 sqlite。"""
    if not os.path.exists(INDEX):
        return []
    out = []
    with open(INDEX, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # 跳过写坏的行，不让一条脏数据卡死整个接口
    out.sort(key=lambda r: r.get("created_at", 0))
    return out


def append_index(rec):
    # 目录不存在就建：全新部署（data/ 还没有）时，第一条记录会直接写失败——
    # 之前没做这步，手机端第一次上传就报 500。
    os.makedirs(os.path.dirname(INDEX) or ".", exist_ok=True)
    with LOCK:
        with open(INDEX, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())     # 断电也不丢记录


def save_meta(rec):
    os.makedirs(META_DIR, exist_ok=True)
    with open(os.path.join(META_DIR, rec["id"] + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)


def find_record(rid):
    for r in load_index():
        if r["id"] == rid:
            return r
    return None


def safe_ext(name, ctype):
    ext = os.path.splitext(name or "")[1].lower()
    if ext in ALLOWED_EXT:
        return ext
    m = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/bmp": ".bmp"}
    return m.get((ctype or "").split(";")[0].strip().lower(), ".jpg")


class Handler(BaseHTTPRequestHandler):
    server_version = "RiceRelay/1.0"

    # ---------- 工具 ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"ok": False, "error": msg}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def _json_body(self):
        raw = self._body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError("请求体不是合法 JSON: %s" % e)

    def _authed(self):
        """电脑端拉取接口的口令校验；未设置 RELAY_TOKEN 时放行。"""
        if not RELAY_TOKEN:
            return True
        tok = self.headers.get("X-Auth-Token") or parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return tok == RELAY_TOKEN

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s - %s\n" % (time.strftime("%H:%M:%S"), self.address_string(), fmt % args))

    # ---------- 路由 ----------
    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)

        if path in ("/api/ping", "/ping"):
            return self._json({"ok": True, "service": "rice-relay",
                               "time": time.time(), "records": len(load_index())})

        if path == "/api/records":
            if not self._authed():
                return self._err(401, "口令不正确（请检查电脑端 RELAY_TOKEN）")
            try:
                since = float(qs.get("since", ["0"])[0])
            except ValueError:
                return self._err(400, "since 必须是数字时间戳")
            try:
                limit = max(1, min(500, int(qs.get("limit", ["200"])[0])))
            except ValueError:
                return self._err(400, "limit 必须是整数")
            # 索引是追加式的（创建/上传/补传各写一行），这里必须**按 id 合并成最新状态**，
            # 否则电脑端会先拿到"还没补传识别结果"的那一行，label 是空的。
            merged = {}
            for r in load_index():
                rid = r.get("id")
                if rid:
                    merged.setdefault(rid, {}).update(r)
            rows = [r for r in merged.values() if r.get("created_at", 0) > since]
            rows.sort(key=lambda r: r.get("created_at", 0))
            rows = rows[:limit]
            # 图片下载地址返回相对路径，由电脑端拼自己配置的 base URL
            for r in rows:
                r["image_url"] = "/api/image/" + r["id"]
            return self._json({"ok": True, "count": len(rows), "records": rows})

        m = re.fullmatch(r"/api/image/([A-Za-z0-9_-]+)", path)
        if m:
            if not self._authed():
                return self._err(401, "口令不正确")
            rec = find_record(m.group(1))
            if not rec:
                return self._err(404, "记录不存在")
            fp = os.path.join(IMG_DIR, rec["id"] + rec.get("ext", ".jpg"))
            if not os.path.exists(fp):
                return self._err(404, "图片文件不存在")
            with open(fp, "rb") as fh:
                blob = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Content-Disposition",
                             'attachment; filename="%s"' % os.path.basename(fp))
            self.end_headers()
            self.wfile.write(blob)
            return

        return self._err(404, "未知接口")

    def do_POST(self):
        u = urlparse(self.path)
        path = u.path

        # 1) 小程序创建记录
        if path == "/api/records":
            try:
                meta = self._json_body()
            except ValueError as e:
                return self._err(400, str(e))
            rid = uuid.uuid4().hex[:16]
            now = time.time()
            rec = {
                "id": rid,
                "created_at": now,
                "created_at_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "device": str(meta.get("device", ""))[:64],
                "lat": meta.get("lat"),
                "lng": meta.get("lng"),
                "accuracy": meta.get("accuracy"),
                "address": str(meta.get("address", ""))[:200],
                "note": str(meta.get("note", ""))[:500],
                "source": "miniprogram",
                "uploaded": False,
            }
            append_index(rec)
            save_meta(rec)
            base = os.environ.get("RELAY_PUBLIC_BASE", "").rstrip("/")
            return self._json({
                "ok": True,
                "id": rid,
                "upload_url": base + "/api/upload/" + rid,
                "finish_url": base + "/api/finish/" + rid,
            })

        # 2) 小程序上传图片
        m = re.fullmatch(r"/api/upload/([A-Za-z0-9_-]+)", path)
        if m:
            rid = m.group(1)
            rec = find_record(rid)
            if not rec:
                return self._err(404, "记录不存在，请先调用 POST /api/records")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self._err(400, "没有收到图片数据")
            if n > MAX_UPLOAD:
                return self._err(413, "图片超过 %d MB 上限" % (MAX_UPLOAD // 1024 // 1024))
            blob = self._body()
            ext = safe_ext(self.headers.get("X-Filename", ""), self.headers.get("Content-Type"))
            os.makedirs(IMG_DIR, exist_ok=True)      # 目录被清理过也能自愈，否则会静默丢图
            dst = os.path.join(IMG_DIR, rid + ext)
            with open(dst, "wb") as fh:
                fh.write(blob)
            if os.path.getsize(dst) != len(blob):
                return self._err(500, "图片写入不完整，请重试")
            rec["uploaded"] = True
            rec["bytes"] = len(blob)
            rec["ext"] = ext
            rec["file"] = rid + ext
            save_meta(rec)
            append_index(rec)          # 追加一条更新态；电脑端按 id 去重取最新
            return self._json({"ok": True, "id": rid, "bytes": len(blob), "ext": ext})

        # 3) 可选：补传元数据
        m = re.fullmatch(r"/api/finish/([A-Za-z0-9_-]+)", path)
        if m:
            rid = m.group(1)
            rec = find_record(rid)
            if not rec:
                return self._err(404, "记录不存在")
            try:
                patch = self._json_body()
            except ValueError as e:
                return self._err(400, str(e))
            # 除了位置与备注，也接受**小程序端的识别结果**——
            # 电脑端要拿它跟自己的检测结论做「端云一致性」对比，
            # 所以这里的白名单必须包含 label/score 等字段（否则手机端白算一遍）。
            for k in ("lat", "lng", "accuracy", "address", "note", "device",
                      "label", "score", "boxes", "model", "elapsed_ms", "raw"):
                if k in patch:
                    rec[k] = patch[k]
            rec["finished_at"] = time.time()
            save_meta(rec)
            append_index(rec)
            return self._json({"ok": True, "id": rid})

        return self._err(404, "未知接口")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    ensure_dirs()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)   # 注意绑 0.0.0.0，不是 127.0.0.1
    n = len(load_index())
    print("稻视界 · 云边联动中转服务已启动")
    print("  端口      : %d" % port)
    print("  数据目录  : %s" % DATA)
    print("  已有记录  : %d" % n)
    print("  拉取口令  : %s" % ("已启用（RELAY_TOKEN）" if RELAY_TOKEN else "未启用（仅本地测试可用）"))
    print("  健康检查  : http://127.0.0.1:%d/api/ping" % port)
    print("  按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()

__RELAY_PY__

echo "==> 写入自检脚本"
cat > $APP/check_relay.py <<'__CHECK_PY__'
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

__CHECK_PY__

echo "==> 写入配置与服务单元"
cat > $APP/relay.env <<EOF
RELAY_TOKEN=$TOKEN
RELAY_PUBLIC_BASE=http://$DOMAIN
EOF
chmod 600 $APP/relay.env
cat > /etc/systemd/system/rice-relay.service <<'__SVC__'
# 稻视界 · 中转服务 systemd 单元
# 安装：见 deploy/README.md（install.sh 会自动放好并启用）
[Unit]
Description=Rice Relay (mini-program -> PC bridge)
After=network.target

[Service]
Type=simple
User=rice
Group=rice
WorkingDirectory=/opt/rice-relay
# 口令等敏感配置放这里，权限 600，不进代码库
EnvironmentFile=/opt/rice-relay/relay.env
ExecStart=/usr/bin/python3 /opt/rice-relay/relay_server.py 9000
Restart=always
RestartSec=3
# 让服务只能写自己的数据目录
ReadWritePaths=/opt/rice-relay/data
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target

__SVC__
chown -R rice:rice $APP
systemctl daemon-reload
systemctl enable --now rice-relay
sleep 2

echo "==> 本机自检"
curl -s http://127.0.0.1:9000/api/ping || echo "（没起来：journalctl -u rice-relay -n 30）"
echo
echo "=== 下一步：装 Nginx + 申请证书 ==="
echo "apt-get update && apt-get install -y nginx certbot python3-certbot-nginx"
echo "然后把这行整段粘贴（会自动写好反代配置）："
echo "  printf '%s\\n' 'server {' '  listen 80;' '  server_name $DOMAIN;' '  client_max_body_size 30m;' '  location /api/ {' '    proxy_pass http://127.0.0.1:9000;' '    proxy_set_header Host \$host;' '    proxy_read_timeout 120s;' '    proxy_request_buffering off;' '  }' '}' > /etc/nginx/conf.d/rice-relay.conf"
echo "  nginx -t && systemctl reload nginx"
echo "  certbot --nginx -d $DOMAIN -m admin@$DOMAIN --agree-tos --redirect"
echo "=== 全部完成后的自检 ==="
echo "  python3 /opt/rice-relay/check_relay.py https://$DOMAIN $TOKEN"
