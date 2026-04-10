import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))

ABD_SRC = _this_dir
PAPA_SRC = os.path.abspath(os.path.join(_this_dir, '../../PaPA/src'))
PAPA_ROOT = os.path.abspath(os.path.join(_this_dir, '../../PaPA'))

# ABD_SRC must be inserted last so it ends up at position 0 (highest priority).
# PaPA shares some module names (e.g. dataset.dataset_recorder) — ABD's version
# must shadow PaPA's to avoid importing the wrong class.
for p in [PAPA_ROOT, PAPA_SRC, ABD_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)
