"""ABD Data Collection — Hydra entry point."""

import sys
import os

# Bootstrap: add ABD src/ and PaPA src/ to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401, E402

import hydra
from omegaconf import DictConfig

from runner.collection_runner import CollectionRunner


def _restore_terminal():
    """Restore terminal to cooked mode.

    VS Code debugpy (and any prior run that crashed inside tty.setraw) can
    leave the terminal with OPOST/ICRNL/ICANON disabled, causing log lines
    to appear indented and input() to hang on Enter.
    """
    import termios
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        attrs[0] |= termios.ICRNL
        attrs[1] |= termios.OPOST
        attrs[3] |= termios.ECHO | termios.ICANON | termios.ISIG
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except Exception:
        pass


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    _restore_terminal()
    runner = CollectionRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
