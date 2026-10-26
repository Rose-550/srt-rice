# -*- coding: utf-8 -*-
"""
修正 YOLO 权重文件内嵌的类别名（names）顺序。

背景：
    本项目 best.pt 内部记录的 names 顺序与真实训练的类别 id 映射不一致
    （经 val 集 300 张图片文件名交叉验证，程序 CLASS_NAMES 的顺序才是对的）。
    直接用 YOLO("best.pt") 加载时，输出的中文类别名会张冠李戴。

本工具：
    1. 类别顺序的唯一权威来源是 dataset.yaml 的 names 字段，不硬编码；
    2. 只改 names 元数据，不触碰任何权重张量（保存前后做指纹校验）；
    3. 默认另存为新文件，绝不覆盖原文件。

用法：
    python fix_pt_names.py best.pt
    python fix_pt_names.py best.pt -o best_fixed.pt
    python fix_pt_names.py runs/detect/train/weights/best.pt --inplace
"""
import argparse
import hashlib
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
import torch

# 类别顺序取自当前训练集；旧根目录 dataset.yaml 已归档到 archive/legacy_dataset/
DEFAULT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_fieldlike", "dataset.yaml")


def load_class_order(yaml_path):
    """从 dataset.yaml 读取 names 列表（不依赖 pyyaml，兼容 names: [...] 与 names: 换行列表两种写法）"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("names:"):
            continue
        body = s[len("names:"):].strip()
        if body.startswith("["):
            items = body.strip("[]").split(",")
            return [x.strip().strip("'\"") for x in items if x.strip()]
        # 换行列表：names:
        #   - a
        #   - b
        out = []
        for nxt in lines[i + 1:]:
            t = nxt.rstrip()
            if not t.strip():
                continue
            if not t.lstrip().startswith("-"):
                break
            out.append(t.lstrip().lstrip("-").strip().strip("'\""))
        return out
    raise ValueError("dataset.yaml 中未找到 names 字段: %s" % yaml_path)


def fingerprint(model):
    """对全部权重张量做指纹，用于证明只改了元数据、没动参数"""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        t = sd[k]
        try:
            arr = t.detach().cpu().float().numpy()
        except Exception:
            arr = t.detach().cpu().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def fix_one(src, dst, order):
    print("=" * 74)
    print("源文件 :", src)
    print("目标   :", dst)
    print("体积   : %.2f MB" % (os.path.getsize(src) / 1024 / 1024))

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("不是 Ultralytics 检查点格式：%s" % src)

    model = ckpt["model"]
    old = dict(model.names) if getattr(model, "names", None) else {}
    n_param_before = sum(p.numel() for p in model.parameters())
    fp_before = fingerprint(model)

    print("\n修改前 names :", old)
    if len(old) not in (0, len(order)):
        print("  [警告] 原有类别数 %d 与 dataset.yaml 的 %d 不一致，请人工确认" % (len(old), len(order)))

    correct = {i: n for i, n in enumerate(order)}
    model.names = correct
    # yaml 里若也存了 names，一并改掉，避免不同加载路径取到旧值
    if isinstance(getattr(model, "yaml", None), dict) and model.yaml.get("names") is not None:
        model.yaml["names"] = list(order)
    ckpt["model"] = model

    print("修改后 names :", correct)
    print("目标顺序来源 : dataset.yaml")

    fp_after = fingerprint(ckpt["model"])
    n_param_after = sum(p.numel() for p in ckpt["model"].parameters())
    print("\n参数量 %d -> %d" % (n_param_before, n_param_after))
    print("权重指纹一致 :", fp_before == fp_after)
    if (fp_before != fp_after) or (n_param_before != n_param_after):
        raise RuntimeError("权重张量发生变化，已中止保存！")

    torch.save(ckpt, dst)
    print("已保存 :", dst, "(%.2f MB)" % (os.path.getsize(dst) / 1024 / 1024))

    # 回读校验
    re_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
    back = dict(re_ckpt["model"].names)
    print("回读校验 :", back)
    assert back == correct, "回读 names 不一致！"
    assert fingerprint(re_ckpt["model"]) == fp_before, "回读权重指纹不一致！"
    print("校验通过：names 已修正，权重张量未改动")
    return True


def main():
    ap = argparse.ArgumentParser(description="修正 YOLO 权重内嵌的类别名顺序")
    ap.add_argument("weights", help="待修正的 .pt 权重路径")
    ap.add_argument("-o", "--out", default=None, help="输出路径；默认在原文件旁生成 <名字>_names_fixed.pt")
    ap.add_argument("--yaml", default=DEFAULT_YAML, help="类别顺序来源（默认 dataset.yaml）")
    ap.add_argument("--inplace", action="store_true", help="原地覆盖（会先备份为 .bak）")
    args = ap.parse_args()

    src = os.path.abspath(args.weights)
    if not os.path.exists(src):
        print("[错误] 文件不存在:", src)
        return 1
    if not os.path.exists(args.yaml):
        print("[错误] 类别顺序来源不存在:", args.yaml)
        return 1

    order = load_class_order(args.yaml)
    print("dataset.yaml 类别顺序 :", {i: n for i, n in enumerate(order)})
    if len(order) < 2:
        print("[错误] 解析出的类别顺序异常:", order)
        return 1

    if args.inplace:
        bak = src + ".bak"
        shutil.copy2(src, bak)
        print("已备份原文件 ->", bak)
        dst = src
    else:
        if args.out:
            dst = os.path.abspath(args.out)
        else:
            stem, ext = os.path.splitext(src)
            dst = stem + "_names_fixed" + ext
        if os.path.abspath(dst) == src:
            print("[错误] 输出与输入相同，请加 --inplace 或指定 -o")
            return 1

    fix_one(src, dst, order)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
