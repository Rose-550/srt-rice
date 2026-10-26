"""Probe: what does DetectionValidator.get_stats() actually return in 8.4.12?"""
import os

import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

ROOT = os.path.join(os.environ["USERPROFILE"], "Desktop", "rice")
MODEL = os.path.join(ROOT, "runs", "detect", "yolov8n_clean", "weights", "best.pt")
DATA = os.path.join(ROOT, "dataset_fieldlike", "dataset.yaml")
OUT = os.path.join(ROOT, "checks", "pred_lowconf_audit")


class Probe(DetectionValidator):
    def get_stats(self):
        s = super().get_stats()
        self.captured = s
        print("\n>>> get_stats returned a", type(s).__name__, flush=True)
        if isinstance(s, dict):
            for k, v in s.items():
                try:
                    print("    %-14s %-16s shape=%s dtype=%s"
                          % (k, type(v).__name__, getattr(v, "shape", None),
                             getattr(v, "dtype", None)), flush=True)
                except Exception as e:
                    print("    %-14s <err %s>" % (k, e), flush=True)
        print(">>> validator.stats during get_stats:", flush=True)
        st = getattr(self, "stats", None)
        if isinstance(st, dict):
            for k, v in st.items():
                print("    %-14s %-16s len=%s" % (k, type(v).__name__, len(v)), flush=True)
        return s


model = YOLO(MODEL)
vargs = {**model.overrides, "data": DATA, "split": "val", "imgsz": 640,
         "batch": 16, "device": 0, "conf": 0.001, "iou": 0.7,
         "rect": True, "mode": "val",
         "plots": False, "verbose": False, "workers": 0,
         "project": OUT, "name": "val_probe", "exist_ok": True}
v = Probe(args=vargs, _callbacks=model.callbacks)
v(model=model.model)

print("\n>>> after __call__:", flush=True)
print("    validator.stats is", type(getattr(v, "stats", None)).__name__, flush=True)
print("    captured keys:", sorted(getattr(v, "captured", {}).keys()), flush=True)
print("    attrs with 'stat':",
      [a for a in dir(v) if "stat" in a.lower()], flush=True)
try:
    cm = v.confusion_matrix
    print("    confusion_matrix ok, shape", np.asarray(cm.matrix).shape, flush=True)
except Exception as e:
    print("    confusion_matrix err:", e, flush=True)
