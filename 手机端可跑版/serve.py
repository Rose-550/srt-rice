# -*- coding: utf-8 -*-
"""手机端演示服务器。

用法：
    python serve.py            # 默认端口 8765
    python serve.py 9000

启动后终端会打印两个地址：
    http://127.0.0.1:8765/          本机自测
    http://192.168.x.x:8765/        手机访问（需与本机同一 WiFi）

设置 COOP/COEP 是为了让浏览器进入「跨源隔离」状态，
onnxruntime-web 才能使用多线程 WASM（否则自动退化为单线程，速度较慢但可用）。
"""
import http.server
import socket
import socketserver
import sys
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".json": "application/json",
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # 开发期禁缓存：否则改完代码浏览器仍用旧的 mjs/模块缓存
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_POST(self):
        """自检用：把浏览器端渲染好的画布回传到磁盘（仅用于开发验证）。"""
        if self.path.split("?")[0] != "/__save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        with open(os.path.join(ROOT, "annotated-overlay.png"), "wb") as fh:
            fh.write(data)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("223.5.5.5", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    os.chdir(ROOT)
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print("稻视界 · 手机端演示服务器已启动")
        print("  本机自测 : http://127.0.0.1:%d/" % PORT)
        print("  手机访问 : http://%s:%d/   （手机连同一 WiFi）" % (lan_ip(), PORT))
        print("  按 Ctrl+C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")
