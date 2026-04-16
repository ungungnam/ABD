import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))

ABD_SRC = _this_dir
PAPA_SRC = os.path.abspath(os.path.join(_this_dir, '../../PaPA/src'))
PAPA_ROOT = os.path.abspath(os.path.join(_this_dir, '../../PaPA'))

# ABD_SRC must end up at position 0 (highest priority).
# PaPA shares module names (e.g. utils.camera_utils, dataset.dataset_recorder) —
# ABD's versions must shadow PaPA's.  We always remove-then-reinsert so that even
# if a caller already added one of these paths, the final order is guaranteed:
#   [ABD_SRC, PAPA_SRC, PAPA_ROOT, ...]
for p in [PAPA_ROOT, PAPA_SRC, ABD_SRC]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
