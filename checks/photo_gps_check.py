# -*- coding: utf-8 -*-
"""检查照片有没有 GPS / 拍摄时间（直接复用程序里的 EXIF 读取逻辑）

用法：
    python checks\photo_gps_check.py <照片或文件夹> [更多路径...]

输出每张照片的：拍摄时间 / 是否有 GPS / 经纬度 / 设备，最后给出汇总与建议。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("RICE_NO_WEBENGINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rice_geo as geo   # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}


def collect(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in EXTS]
        elif p.suffix.lower() in EXTS:
            out.append(p)
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    files = collect(argv)
    if not files:
        print("没找到图片文件（支持 %s）" % "、".join(sorted(EXTS)))
        return 1
    have_gps, no_gps, err = [], [], []
    for f in files:
        meta = geo.read_photo_meta(str(f))
        taken = meta.get("taken_at")
        taken_s = taken.strftime("%Y-%m-%d %H:%M:%S") if taken else "—"
        if meta.get("has_gps"):
            have_gps.append(f)
            print("✅ %-38s %s  GPS %.6f, %.6f  %s"
                  % (f.name[:38], taken_s, meta["lon"], meta["lat"],
                     (meta.get("device") or "")[:18]))
        else:
            no_gps.append(f)
            src = meta.get("source")
            why = ("无 EXIF 时间（用的是文件时间）" if src == "filetime"
                   else "有 EXIF 但里面没有 GPS")
            print("❌ %-38s %s  %s  %s"
                  % (f.name[:38], taken_s, why, (meta.get("device") or "")[:18]))
    n = len(files)
    print("\n" + "=" * 66)
    print("共 %d 张：有 GPS %d 张（%.0f%%）· 无 GPS %d 张"
          % (n, len(have_gps), len(have_gps) * 100.0 / n, len(no_gps)))
    if no_gps:
        print("\n没有 GPS 的照片怎么办：")
        print("  1) 程序里不会丢——导入时会进「待落点」，在地图上点一下实际位置即可；")
        print("  2) 想批量补：用带轨迹记录的设备（手机 GPS 日志/GPX）按拍摄时间批量写入 EXIF；")
        print("  3) 现场拍照建议用系统相机并打开定位，别用微信内置相机；")
        print("  4) 传图时选「原图」——微信/QQ 压缩会剥掉 EXIF 位置信息。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
