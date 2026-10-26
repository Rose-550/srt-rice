# -*- coding: utf-8 -*-
"""单独启动「手机传图」接收服务（复用程序里的同一份实现）

用法：
    python checks\phone_upload_server.py [端口]

启动后手机连同一 WiFi，浏览器打开打印出来的地址，从相册选原图上传。
上传的图片原字节落进收件箱（inbox），有 GPS 的会自动落点。
按 Ctrl+C 或结束进程即可停止。
"""
import os
import sys
import time
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 只要服务，不需要窗口
os.environ.setdefault("RICE_NO_WEBENGINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rice_dp1_fluent as m                     # noqa: E402
from PyQt5.QtWidgets import QApplication        # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else m.PHONE_UPLOAD_PORT

app = QApplication(sys.argv)
inbox = m.WATCH_DIR
srv = m.PhoneUploadThread(inbox, port=PORT)


def on_saved(path, gps):
    name = os.path.basename(path)
    if gps.get("has_gps"):
        print("[收到] %s  ✅ 含 GPS %.6f, %.6f" % (name, gps.get("lon") or 0, gps.get("lat") or 0),
              flush=True)
    else:
        print("[收到] %s  （无 GPS，稍后在地图上点一下落位）" % name, flush=True)

    # 收到就立刻检测：独立起服务也能直接出结果，手机页面几秒后就能看到
    # （不必再开着主程序；主程序开着时它自己也会巡检，两条路互不冲突）
    def work():
        t0 = time.time()
        try:
            rec = m.process_image_file(path, do_cascade=True)
            print("[识别] %s → %s（%.0f%%）· 严重度 %s · 用时 %.1f 秒"
                  % (name, rec.get("cls") or "未检出", (rec.get("conf") or 0) * 100,
                     rec.get("severity") or "-", time.time() - t0), flush=True)
        except Exception as e:
            print("[识别] %s 失败：%s" % (name, e), flush=True)

    threading.Thread(target=work, daemon=True).start()


srv.saved.connect(on_saved)
srv.start()
time.sleep(1.0)          # 等端口真的绑上再打印
print("=" * 56, flush=True)
print("稻视界 · 手机传图服务已启动", flush=True)
print("  手机访问（连同一 WiFi）: http://%s:%d/" % (m.lan_ip(), PORT), flush=True)
print("  本机自测              : http://127.0.0.1:%d/" % PORT, flush=True)
print("  图片落盘目录          : %s" % inbox, flush=True)
print("  停止：Ctrl+C 或关闭本窗口", flush=True)
print("=" * 56, flush=True)
sys.exit(app.exec_())