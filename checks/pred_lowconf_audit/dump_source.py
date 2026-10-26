"""Dump the source of the validator methods that handle the raw stats."""
import inspect

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.engine.validator import BaseValidator

for cls in (DetectionValidator, BaseValidator):
    for name in ("gather_stats", "get_stats", "update_metrics", "_process_batch", "init_metrics"):
        fn = getattr(cls, name, None)
        if fn is None:
            continue
        print("=" * 70)
        print("%s.%s" % (cls.__name__, name))
        print("=" * 70)
        try:
            print(inspect.getsource(fn))
        except Exception as e:
            print("<no source: %s>" % e)
        print()
