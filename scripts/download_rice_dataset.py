# -*- coding: utf-8 -*-
"""
水稻叶部病害数据集 下载 + 体检脚本
================================================================
目标数据集：Kaggle  yusufmurtaza01/rice-leaf-diseases
            （就是 HyperAI 51498 的上游源站）
许可证    ：CC0 Public Domain（公有领域，无使用限制）
体积      ：约 494 MB
内容      ：8,665 张图 / 28,765 个框 / 9 类，YOLO txt 格式

为什么选它（跟我们模型的对应关系）
----------------------------------------------------------------
它刚好覆盖我们最弱的三类【病】，且完全不含我们最强的三类【虫】：
    白叶枯病  <-  Rice__BacterialLeafBlight   (我们的 R = 0.7404，最弱)
    稻瘟病    <-  Rice__LeafBlast             (mAP50-95 0.4033)
    褐斑病    <-  Rice__BrownSpot / NarrowBrownLeafSpot (mAP50-95 0.4005，最差)
属于"精准补短板"，不是简单堆图片数量。

用法
----------------------------------------------------------------
方式一（推荐，自动下载）：
    1. 去 https://www.kaggle.com/settings 页面 -> API -> Create New Token
       会下载一个 kaggle.json 文件
    2. 把它放到   C:\\Users\\罗浔\\.kaggle\\kaggle.json
    3. 运行：  D:\\anoconda3\\envs\\rice-gpu\\python.exe scripts\download_rice_dataset.py

方式二（手动下载，不想配 token）：
    1. 浏览器打开 https://www.kaggle.com/datasets/yusufmurtaza01/rice-leaf-diseases
    2. 登录后点 Download，得到一个 .zip
    3. 运行：  ...python.exe scripts\download_rice_dataset.py --zip "刚下载的zip路径"

只体检、不下载（数据已经在本地了）：
    ...python.exe scripts\download_rice_dataset.py --skip-download
================================================================
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------- 配置
OWNER, SLUG = "yusufmurtaza01", "rice-leaf-diseases"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{OWNER}/{SLUG}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "_download" / f"{SLUG}.zip"
EXTRACT_DIR = PROJECT_ROOT / "data" / "external" / SLUG
DEFAULT_CRED = Path.home() / ".kaggle" / "kaggle.json"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 官方页面上公布的每类 (图片数, 框数)。9 类的图片数相加 = 8665，说明一张图只属一类
EXPECTED = {
    "Rice__BacterialLeafBlight": (2000, 2353),   # 白叶枯病   <== 我们要的
    "Rice__BrownSpot":           (2000, 15257),  # 褐斑病     <== 我们要的
    "Rice__LeafBlast":           (449,  763),    # 稻瘟病     <== 我们要的
    "Rice__Healthy":             (433,  821),
    "Rice__Hispa":               (431,  933),
    "Rice__LeafScald":           (456,  564),
    "Rice__LeafSmut":            (2000, 6386),
    "Rice__NarrowBrownLeafSpot": (443,  737),
    "Rice__NeckBlast":           (453,  951),
}
EXPECTED_IMAGES = 8665
EXPECTED_BOXES = 28765

# 我们模型现在的 6 类（取自 dataset.yaml）
LOCAL_CLASSES = ["稻纵卷叶螟", "稻水象甲", "稻叶蝉", "白叶枯病", "胡麻斑病", "稻瘟病"]

# 外部数据集 -> 我们类别 的对应关系
#   「胡麻斑病」的中文正名是「稻胡麻斑病 / 胡麻叶枯病」，英文就是 rice brown spot，
#   病原为稻平脐蠕孢 Bipolaris oryzae（异名 Cochliobolus miyabeanus），
#   跟外部的 Rice__BrownSpot 是同一个病，不是两个病 —— 已核实，可以合并。
MAP_SUGGEST = {
    "Rice__BacterialLeafBlight": "白叶枯病       同一种病，可直接并入",
    "Rice__LeafBlast":           "稻瘟病         同一种病，可直接并入",
    "Rice__BrownSpot":           "胡麻斑病       同一种病（brown spot），可直接并入",
    "Rice__NarrowBrownLeafSpot": "窄褐斑病       我们 6 类里没有，别硬塞",
    "Rice__NeckBlast":           "穗颈瘟           稻瘟病发生在穗颈部位，可考虑并入「稻瘟病」",
}


def hr(char="─", n=66):
    print(char * n)


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", "replace").decode("ascii", "replace"))


# ---------------------------------------------------------------- 凭据
def load_credential(explicit: str = None):
    path = Path(explicit) if explicit else DEFAULT_CRED
    if not path.exists():
        return None, path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        safe_print(f"[!] 读不了 {path}：{exc}")
        return None, path
    user = data.get("username")
    key = data.get("key")
    if not user or not key:
        safe_print(f"[!] {path} 里缺 username 或 key")
        return None, path
    return (user, key), path


# ---------------------------------------------------------------- 下载
def download(zip_path: Path, cred) -> bool:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and zip_path.stat().st_size > 1024 * 1024:
        safe_print(f"[i] 压缩包已经在了，跳过下载：{zip_path}")
        safe_print(f"    大小 {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True

    user, key = cred
    req = Request(DOWNLOAD_URL)
    req.add_header("Authorization", "Basic " + _b64(f"{user}:{key}"))
    req.add_header("User-Agent", "rice-dataset-fetch/1.0")

    safe_print(f"[*] 开始下载 {OWNER}/{SLUG} ...")
    safe_print(f"    保存到 {zip_path}")
    t0 = time.time()
    try:
        with urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            safe_print(f"    服务器报告大小：{total / 1024 / 1024:.1f} MB" if total else "    服务器未报告大小")
            got, last = 0, 0.0
            with open(zip_path, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last > 3:
                        last = now
                        pct = f" ({got * 100 / total:.1f}%)" if total else ""
                        safe_print(f"    ...{got / 1024 / 1024:7.1f} MB{pct}")
    except HTTPError as exc:
        safe_print(f"[X] 下载被服务器拒绝：HTTP {exc.code} {exc.reason}")
        if exc.code in (401, 403):
            safe_print("    原因：kaggle.json 里的 username/key 不对，或者文件没放对位置。")
        elif exc.code == 404:
            safe_print("    原因：没有带有效凭据。Kaggle 对匿名请求统一回 404。")
        if zip_path.exists():
            zip_path.unlink()
        return False
    except URLError as exc:
        safe_print(f"[X] 连不上 Kaggle：{exc.reason}")
        return False
    except Exception as exc:
        safe_print(f"[X] 下载中断：{exc}")
        return False

    dt = time.time() - t0
    size = zip_path.stat().st_size / 1024 / 1024
    safe_print(f"[√] 下载完成：{size:.1f} MB，用了 {dt:.0f} 秒")
    return True


def _b64(s: str) -> str:
    import base64
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------- 解压
def extract(zip_path: Path, out_dir: Path) -> bool:
    if not zip_path.exists():
        safe_print(f"[X] 找不到压缩包：{zip_path}")
        return False
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        safe_print(f"[i] 已经解压过了，跳过：{out_dir}")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_print(f"[*] 解压到 {out_dir} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            safe_print(f"    压缩包内 {len(names)} 个条目")
            zf.extractall(out_dir)
    except Exception as exc:
        safe_print(f"[X] 解压失败：{exc}")
        return False
    marker.write_text("ok", encoding="utf-8")
    safe_print("[√] 解压完成")
    return True


# ---------------------------------------------------------------- 定位 YOLO 根目录
def find_yolo_root(base: Path):
    if not base.exists():
        return None
    candidates = []
    for p in [base] + [d for d in base.rglob("*") if d.is_dir()]:
        if (p / "images").is_dir() and (p / "labels").is_dir():
            candidates.append(p)
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))


def read_names(yolo_root: Path):
    for cand in list(yolo_root.glob("*.yaml")) + list(yolo_root.glob("*.yml")):
        try:
            txt = cand.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        m = re.search(r"names\s*:\s*(\[.*?\]|\n(?:\s*-\s*.+\n?)+)", txt, re.S)
        if not m:
            continue
        block = m.group(1)
        if block.strip().startswith("["):
            names = re.findall(r"['\"]([^'\"]+)['\"]", block)
        else:
            names = re.findall(r"-\s*(.+)", block)
            names = [n.strip().strip("'\"") for n in names]
        if names:
            return cand, names
    return None, []


# ---------------------------------------------------------------- 体检
def inspect(yolo_root: Path):
    safe_print("")
    hr("═")
    safe_print("体检报告")
    hr("═")

    yaml_file, ext_names = read_names(yolo_root)
    if yaml_file:
        safe_print(f"dataset 配置：{yaml_file}")
        safe_print(f"类别数 nc = {len(ext_names)}")
        for i, n in enumerate(ext_names):
            safe_print(f"    {i}: {n}")
    else:
        safe_print("[i] 没找到 dataset.yaml，类别名只能从文件名前缀猜")

    grand_img = grand_lab = grand_box = grand_empty = 0
    per_class_img = defaultdict(int)
    per_class_box = defaultdict(int)
    bad_lines = []
    out_of_range = 0
    no_label = 0
    splits_found = []

    for split in ("train", "val", "test", "valid", "validation"):
        img_dir = yolo_root / "images" / split
        lab_dir = yolo_root / "labels" / split
        if not img_dir.is_dir():
            continue
        splits_found.append(split)
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
        n_box = n_empty = n_nolabel = 0
        n_oor = 0
        for img in images:
            lab = (lab_dir / (img.stem + ".txt")) if lab_dir.is_dir() else None
            if lab is None or not lab.exists():
                n_nolabel += 1
                continue
            try:
                content = lab.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not content:
                n_empty += 1
                continue
            for ln_no, line in enumerate(content.splitlines(), 1):
                parts = line.split()
                if len(parts) < 5:
                    bad_lines.append(f"{lab.name}:{ln_no} {line[:50]}")
                    continue
                try:
                    cid = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:5]]
                except ValueError:
                    bad_lines.append(f"{lab.name}:{ln_no} {line[:50]}")
                    continue
                if any(v < 0 or v > 1 for v in vals):
                    n_oor += 1
                n_box += 1
                per_class_box[cid] += 1
                per_class_img[cid] += 1

        safe_print("")
        safe_print(f"  [{split}]  图片 {len(images)} 张 | 标注框 {n_box} 个 | "
                   f"空标注 {n_empty} | 无标注文件 {n_nolabel}")
        grand_img += len(images)
        grand_box += n_box
        grand_empty += n_empty
        grand_lab += len(images) - n_nolabel
        no_label += n_nolabel
        out_of_range += n_oor

    safe_print("")
    hr()
    safe_print(f"总计：{grand_img} 张图 / {grand_box} 个框（官方公布 {EXPECTED_IMAGES} / {EXPECTED_BOXES}）")
    safe_print(f"     有标注文件的图 {grand_lab} 张 | 空标注文件 {grand_empty} 个 | 完全没有标注 {no_label} 张")
    if out_of_range:
        safe_print(f"     [!] 有 {out_of_range} 个框的坐标超出 0~1 范围")
    if bad_lines:
        safe_print(f"     [!] 有 {len(bad_lines)} 行格式不对，前 5 行：")
        for b in bad_lines[:5]:
            safe_print(f"         {b}")

    safe_print("")
    hr()
    safe_print("按类别的框数统计")
    hr()
    idx2name = {i: n for i, n in enumerate(ext_names)}
    for cid in sorted(set(list(per_class_box.keys()) + list(idx2name.keys()))):
        name = idx2name.get(cid, f"<未命名类 {cid}>")
        safe_print(f"  类 {cid:>2}  {name:<28} 框 {per_class_box.get(cid, 0):>7}")
    total_by_name = defaultdict(int)
    for cid, c in per_class_box.items():
        total_by_name[idx2name.get(cid, f"<{cid}>")] += c

    safe_print("")
    hr("═")
    safe_print("跟我们现有 6 类的对得上的部分")
    hr("═")
    safe_print("我们现在的类别：" + " / ".join(LOCAL_CLASSES))
    safe_print("")
    for ext_name, boxes in sorted(total_by_name.items(), key=lambda kv: -kv[1]):
        hit = MAP_SUGGEST.get(ext_name)
        origin = re.sub(r"^Rice__", "", ext_name)
        tag = f"  ->  {hit}" if hit else "  (我们不需要)"
        safe_print(f"  {origin:<26} 框 {boxes:>7}{tag}")
    safe_print("")
    safe_print("提醒：这份数据里【一个虫类都没有】。它补的是「病」，不是「虫」。")


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="下载并体检水稻叶部病害数据集")
    ap.add_argument("--zip", help="已手动下载好的 zip 路径，给了这个就不联网下载")
    ap.add_argument("--cred", help="kaggle.json 路径（默认 %USERPROFILE%\\.kaggle\\kaggle.json）")
    ap.add_argument("--skip-download", action="store_true", help="跳过下载，只体检本地已有数据")
    ap.add_argument("--dest", help=f"解压目录（默认 {EXTRACT_DIR}）")
    args = ap.parse_args()

    dest = Path(args.dest) if args.dest else EXTRACT_DIR

    safe_print("")
    hr("═")
    safe_print("水稻叶部病害数据集  获取 + 体检")
    safe_print(f"目标：Kaggle {OWNER}/{SLUG}   (CC0 公有领域, 约 494 MB, 8665张/28765框/9类)")
    hr("═")
    safe_print(f"Python   : {sys.version.split()[0]}")
    safe_print(f"项目目录 : {PROJECT_ROOT}")
    try:
        import shutil
        free = shutil.disk_usage(str(PROJECT_ROOT)).free / 1024 ** 3
        safe_print(f"磁盘剩余 : {free:.1f} GB （够，需要约 1.2 GB 装压缩包+解压）")
    except Exception:
        pass

    if args.skip_download:
        zip_path = None
    elif args.zip:
        zip_path = Path(args.zip)
        safe_print(f"\n[i] 使用你手动下载的包：{zip_path}")
        if not zip_path.exists():
            safe_print("[X] 这个路径不存在，检查一下")
            return 2
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.resolve() != ZIP_PATH.resolve():
            import shutil
            safe_print(f"    复制到 {ZIP_PATH}")
            shutil.copy2(zip_path, ZIP_PATH)
        zip_path = ZIP_PATH
    else:
        safe_print("")
        hr()
        cred, cred_path = load_credential(args.cred)
        if cred is None:
            safe_print(f"[!] 没找到可用凭据：{cred_path}")
            safe_print("")
            safe_print("    这不是报错，是还没配。两条路选一条：")
            safe_print("    路 A：去 https://www.kaggle.com/settings  ->  API  ->  Create New Token")
            safe_print(f"          把下载到的 kaggle.json 放到  {DEFAULT_CRED.parent}")
            safe_print("          然后重新运行本脚本，就全自动了。")
            safe_print("    路 B：浏览器手动下载 zip，再运行")
            safe_print("          python scripts\download_rice_dataset.py --zip \"zip的路径\"")
            safe_print("")
            safe_print("    （Kaggle 账号免费，注册只需要邮箱。CC0 数据集没有附加限制。）")
            return 1
        safe_print(f"[√] 读到凭据：用户 {cred[0]}（来自 {cred_path}）")
        if not download(ZIP_PATH, cred):
            safe_print("\n[X] 自动下载失败，可以改用手动下载 + --zip 方式。")
            return 3
        zip_path = ZIP_PATH

    if zip_path is not None:
        if not extract(zip_path, dest):
            return 4

    yolo_root = find_yolo_root(dest)
    if yolo_root is None:
        safe_print(f"\n[X] 在 {dest} 里找不到 YOLO 结构（应该有 images/ 和 labels/ 两个文件夹）")
        safe_print("    如果是手动下载的包，可能目录结构不一样，把实际解压出来的内容发我看看。")
        return 5
    safe_print(f"\n[√] 找到 YOLO 根目录：{yolo_root}")
    inspect(yolo_root)

    safe_print("")
    hr("═")
    safe_print("下一步（等你确认数据没问题再动手）")
    hr("═")
    safe_print("1) 先用这份数据做「只推理不训练」，测出跨域掉多少分（我们验收缺的第 5 项）")
    safe_print("2) 再把 白叶枯病/稻瘟病 的标注合并进训练集，重训一版对比")
    safe_print("3) 合并前务必确认：外部的 BrownSpot 和我们的「胡麻斑病」是不是同一种病")
    return 0


if __name__ == "__main__":
    sys.exit(main())
