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

# ── 手机传图页（公网版）────────────────────────────────────────────
# 手机在**任何网络**（4G/校园网/外地）打开 https://你的域名/relay/ 就能传原图，
# 不需要和小程序、也不需要和电脑在同一网络。页内请求全部用**相对路径**，
# 这样 nginx 把 /relay/ 前缀剥掉后依然能正确命中 /api/*。
RELAY_UPLOAD_PAGE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>稻视界 · 田间传图</title><style>
body{font:16px/1.7 -apple-system,"Microsoft YaHei",sans-serif;margin:0;padding:18px;
     background:#f4f7f5;color:#22312b}
h2{margin:0 0 4px;font-size:20px}.tip{color:#5f6b66;font-size:14px;margin-bottom:14px}
.box{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 10px rgba(0,0,0,.07)}
.btn{display:block;padding:15px;border-radius:10px;background:#2e7d5b;color:#fff;
     font-weight:600;text-align:center;margin-bottom:10px}
.btn.ghost{background:#e8efeb;color:#22312b}
.picker{position:absolute;left:-9999px;width:1px;height:1px;opacity:0}
#bar{height:8px;border-radius:4px;background:#e8efeb;overflow:hidden;margin-top:10px;display:none}
#bar i{display:block;height:100%;width:0;background:#2e7d5b;transition:width .15s}
#log{font-size:13px;color:#5f6b66;margin-top:10px;white-space:pre-wrap}
li{font-size:13px;padding:6px 0;border-bottom:1px solid #eef2ef}
ul{list-style:none;padding:0;margin:0}
</style></head><body>
<h2>稻视界 · 田间传图</h2>
<div class="tip">手机在任何网络都能传（4G 也行）。原图直传不压缩，电脑端会自动取走并识别。<br>
<b style="color:#c62828">⚠ 只有在拍摄现场才点「我此刻就在这块地里」</b>——它记的是你<b>现在</b>的位置。
回家后再传老照片请留空，回电脑上手动落点才准。</div>
<div class="box">
  <label class="btn" for="fc">📷 拍照</label>
  <label class="btn ghost" for="f">🖼 从相册选</label>
  <input id="f" class="picker" type="file" accept="image/*" multiple>
  <input id="fc" class="picker" type="file" accept="image/*" capture="environment">
  <div class="row" style="margin-top:8px">
    <button id="loc" type="button" style="flex:1;padding:11px;border-radius:9px;border:0;
            background:#e8efeb;color:#22312b;font-size:15px">📍 我此刻就在这块地里</button>
    <span id="locout" style="font-size:12px;color:#5f6b66;align-self:center">未取</span>
  </div>
  <div id="bar"><i></i></div>
  <div id="log">选好照片会自动上传。</div>
</div>
<div class="box"><div style="font-weight:600;margin-bottom:6px">已传 <span id="n">0</span> 张</div>
<ul id="list"></ul></div>
<script>
var $=function(i){return document.getElementById(i);};
var items=[];
function render(){
  $('n').textContent=items.length;
  $('list').innerHTML=items.map(function(it){
    return '<li>'+it.name+' '+(it.ok?'✅ 已送达电脑端待取':'❌ '+it.err)+'</li>';}).join('');
}
var curLoc = null;   // 安卓相册选择器会剥掉 EXIF 定位，所以传图时现取一次
function askLoc(){
  var o=$('locout');
  if(!navigator.geolocation){o.textContent='不支持定位';return;}
  o.textContent='定位中…';
  navigator.geolocation.getCurrentPosition(function(p){
    curLoc={lat:p.coords.latitude,lng:p.coords.longitude,acc:p.coords.accuracy};
    o.textContent='✓ 已取（±'+Math.round(p.coords.accuracy||0)+'m）';
  },function(e){
    o.textContent='✗ '+(e.message||'取不到，请允许定位权限');
  },{enableHighAccuracy:true,timeout:12000,maximumAge:0});
}
$('loc').onclick=askLoc;
function up(file,onp){
  return new Promise(function(res,rej){
    // ① 建记录（相对路径 → /relay/api/records）
    fetch('api/records',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({device:(navigator.userAgent||'').slice(0,60),note:'网页直传',
        lat:curLoc?curLoc.lat:null,lng:curLoc?curLoc.lng:null,
        accuracy:curLoc?curLoc.acc:null})})
    .then(function(r){return r.json();}).then(function(j){
      if(!j||!j.ok){rej(new Error('建记录失败'));return;}
      // ② 传图（原始字节）
      var x=new XMLHttpRequest();
      x.open('POST','api/upload/'+j.id);
      x.setRequestHeader('Content-Type','application/octet-stream');
      x.setRequestHeader('X-Filename',file.name||'photo.jpg');
      x.upload.onprogress=function(e){if(e.lengthComputable&&onp)onp(e.loaded/e.total);};
      x.onload=function(){try{res(JSON.parse(x.responseText));}catch(e){rej(new Error('响应异常'));}};
      x.onerror=function(){rej(new Error('网络中断'));};
      x.send(file);
    }).catch(rej);
  });
}
async function handle(files){
  if(!files||!files.length)return;
  $('bar').style.display='block';
  for(var i=0;i<files.length;i++){
    var f=files[i],it={name:f.name,ok:false,err:''};items.push(it);render();
    $('log').textContent='正在上传 '+(i+1)+'/'+files.length+'：'+f.name;
    try{
      var r=await up(f,function(pp){$('bar').firstElementChild.style.width=
        Math.round(((i+pp)/files.length)*100)+'%';});
      it.ok=!!(r&&r.ok); if(!it.ok)it.err=(r&&r.error)||'失败';
    }catch(e){it.err=e.message;}
    render();
  }
  $('bar').firstElementChild.style.width='100%';
  $('log').textContent='完成。电脑端在程序里点「自动巡检 → 云端同步」就能取走。';
  setTimeout(function(){$('bar').style.display='none';$('bar').firstElementChild.style.width='0';},1200);
  $('f').value='';$('fc').value='';
}
$('f').onchange=function(){handle(this.files);};
$('fc').onchange=function(){handle(this.files);};
</script></body></html>"""

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

        if path in ("/", "/up", "/index.html"):
            body = RELAY_UPLOAD_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

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
