import os
import sys
import select
import termios
import tty

import numpy as np
from omegaconf import OmegaConf

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DatasetRecorder:
    def __init__(self, config):
        self.config = config
        self.features = OmegaConf.to_container(self.config.features, resolve=True)

        for k in self.features.keys():
            if k in ['action', 'observation.state']:
                self.features[k]['shape'] = tuple(self.features[k]['shape'])

        self.lerobot_dataset = self._get_lerobot_dataset()

    def _get_lerobot_dataset(self):
        if os.path.exists(self.config.root):
            return LeRobotDataset(
                repo_id=self.config.root,
            )
        else:
            return LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                features=self.features,
                root=self.config.root,
            )

    def _wait_true_false_key(
        self,
        prompt: str,
        true_keys = (" ", "\n", "\r", "y", "Y"),      # Space 또는 Enter
        false_keys = ("\x7f", "\x08","\b", "n", "N", "q", "Q"),        # Backspace(보통 DEL=\x7f)
    ) -> bool:
        sys.stdout.write(prompt)
        sys.stdout.flush()

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # 1글자 즉시 입력
            while True:
                r, _, _ = select.select([sys.stdin], [], [])
                if not r:
                    continue
                ch = sys.stdin.read(1)

                if ch in true_keys:
                    print()
                    return True
                if ch in false_keys:
                    print()
                    return False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def is_success(self) -> bool:
        return self._wait_true_false_key(
            prompt="(Space/Enter)=저장  /  Backspace=저장안함 > "
        )

    def is_collecting_dataset(self) -> bool:
        return self._wait_true_false_key(
            prompt="(Space/Enter)=수집  /  Backspace=중단 > "
        )

    def finalize(self):
        self.lerobot_dataset.finalize()

    def save_episode(self):
        self.lerobot_dataset.save_episode()

    def clear_episode_buffer(self):
        self.lerobot_dataset.clear_episode_buffer()

    def add_frame(self, frame):
        self.lerobot_dataset.add_frame(frame)