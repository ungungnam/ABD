"""Terminal-based human reset interface.

Prompts the human operator to physically reset the workspace,
then waits for confirmation before continuing.
Uses the same tty approach as PaPA's DatasetRecorder.
"""

import sys
import time
import select
import termios
import tty
from typing import Tuple

from task.task_family import TaskDefinition


class HumanResetInterface:
    """Terminal-based human reset prompt."""

    def request_reset(self, task: TaskDefinition) -> Tuple[bool, float]:
        """Block until human confirms the workspace has been reset.

        Returns:
            (should_continue, confirm_timestamp):
                should_continue: True if reset confirmed, False if aborted.
                confirm_timestamp: time.time() when the key was pressed.
        """
        print()
        print("=" * 60)
        print("  HUMAN RESET REQUESTED")
        print(f"  Task: {task.name}")
        print(f"  Please reset the workspace to the canonical state.")
        print(f"  Press SPACE/ENTER when done, or Q to abort.")
        print("=" * 60)

        return self._wait_true_false_key(
            prompt="  Reset complete? (Space/Enter=continue, Q=abort) > ",
        )

    @staticmethod
    def _wait_true_false_key(
        prompt: str,
        true_keys=(" ", "\n", "\r", "y", "Y"),
        false_keys=("q", "Q"),
    ) -> Tuple[bool, float]:
        """Wait for single keypress.

        Returns:
            (result, timestamp): result is True (continue) or False (abort),
            timestamp is time.time() when the key was pressed.
        """
        sys.stdout.write(prompt)
        sys.stdout.flush()

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                r, _, _ = select.select([sys.stdin], [], [])
                if not r:
                    continue
                ch = sys.stdin.read(1)
                confirm_time = time.time()
                if ch in true_keys:
                    print()
                    return True, confirm_time
                if ch in false_keys:
                    print()
                    return False, confirm_time
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
