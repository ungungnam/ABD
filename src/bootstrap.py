import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))

ABD_SRC = _this_dir
PAPA_SRC = os.path.abspath(os.path.join(_this_dir, '../../PaPA/src'))
PAPA_ROOT = os.path.abspath(os.path.join(_this_dir, '../../PaPA'))

for p in [ABD_SRC, PAPA_SRC, PAPA_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)
