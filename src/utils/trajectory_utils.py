import numpy as np
import matplotlib.pyplot as plt

from typing import List

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def smoothen_trajectory(trajectory: List[np.ndarray], sg_window=9, sg_polyorder=3) -> List[np.ndarray]:
    """
    Smooth a trajectory given as List[SE(3)] (each element is a 4x4 transform).

    Strategy:
        - translation: Savitzky–Golay directly on (x,y,z)
        - rotation: convert to relative rotvecs -> cumulative rotvec -> SG -> reconstruct
        (more stable than smoothing absolute rotations)

    Args:
        trajectory: list of (4,4) np.ndarray

    Returns:
        smoothed_trajectory: list of (4,4) np.ndarray (same length)
    """
    if trajectory is None or len(trajectory) <= 2:
        return trajectory

    # ---------- sanitize window/polyorder ----------
    n = len(trajectory)

    window = min(sg_window, n if (n % 2 == 1) else (n - 1))
    if window % 2 == 0:
        window -= 1
    polyorder = min(sg_polyorder, window - 1)

    # ---------- stack ----------
    T = np.stack(trajectory, axis=0).astype(np.float64)  # (N,4,4)
    t = T[:, :3, 3]                                      # (N,3)
    Rm = T[:, :3, :3]                                    # (N,3,3)

    # ---------- smooth translation ----------
    t_s = savgol_filter(t, window_length=window, polyorder=polyorder, axis=0)

    # ---------- smooth rotation (relative-cumulative rotvec trick) ----------
    R_seq = R.from_matrix(Rm)

    rel_rotvec = np.zeros((n, 3), dtype=np.float64)
    for i in range(1, n):
        R_rel = R_seq[i - 1].inv() * R_seq[i]
        rel_rotvec[i] = R_rel.as_rotvec()

    cum = np.cumsum(rel_rotvec, axis=0)  # approx log(R0^T Ri)
    cum_s = savgol_filter(cum, window_length=window, polyorder=polyorder, axis=0)

    R0 = R_seq[0]
    R_s = [(R0 * R.from_rotvec(cum_s[i])).as_matrix() for i in range(n)]

    # ---------- recompose ----------
    smoothed = []
    for i in range(n):
        Ti = np.eye(4, dtype=np.float64)
        Ti[:3, :3] = R_s[i]
        Ti[:3, 3] = t_s[i]
        smoothed.append(Ti)

    # Preserve start and end poses exactly — SG boundary effect drifts them
    smoothed[0] = trajectory[0].copy()
    smoothed[-1] = trajectory[-1].copy()

    return smoothed


def visualize_trajectory(
    trajectory: List[np.ndarray],
    show_rotation: bool = True,
    frame_stride: int = 5,
    axis_len: float = 0.05,
    elev: float = 25,
    azim: float = -60,
):
    """
    Visualize a List[SE(3)] trajectory in 3D.

    Args:
        trajectory: list of (4,4) np.ndarray
        show_rotation: if True, draw orientation frames (x,y,z axes) along the path
        frame_stride: draw a frame every N waypoints (to avoid clutter)
        axis_len: length of the drawn axes (in same unit as translation)
        elev, azim: initial view angles for matplotlib 3D camera
    """
    if trajectory is None or len(trajectory) == 0:
        print("[visualize_trajectory] Empty trajectory.")
        return

    T = np.stack(trajectory, axis=0).astype(np.float64)  # (N,4,4)
    pts = T[:, :3, 3]                                    # (N,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Path
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", linewidth=2)
    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=60, marker="o")
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=60, marker="X")

    # Optional: draw rotation frames
    if show_rotation:
        idxs = range(0, len(trajectory), max(1, int(frame_stride)))
        for i in idxs:
            R = T[i, :3, :3]
            p = pts[i]

            # axis directions in world
            x_dir = R[:, 0]
            y_dir = R[:, 1]
            z_dir = R[:, 2]

            # draw 3 arrows (no explicit colors requested)
            ax.quiver(p[0], p[1], p[2], x_dir[0], x_dir[1], x_dir[2], length=axis_len, normalize=True)
            ax.quiver(p[0], p[1], p[2], y_dir[0], y_dir[1], y_dir[2], length=axis_len, normalize=True)
            ax.quiver(p[0], p[1], p[2], z_dir[0], z_dir[1], z_dir[2], length=axis_len, normalize=True)

    # Pretty axes (equal aspect)
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    span = max(span, 1e-9)

    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_zlim(cz - span / 2, cz + span / 2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Trajectory (N={len(trajectory)})  |  show_rotation={show_rotation}")

    plt.show()