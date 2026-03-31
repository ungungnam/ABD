"""ABD Data Collection — Hydra entry point."""

import sys
import os

# Bootstrap: add ABD src/ and PaPA src/ to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import bootstrap  # noqa: F401, E402

import hydra
from omegaconf import DictConfig

from runner.collection_runner import CollectionRunner


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    runner = CollectionRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
