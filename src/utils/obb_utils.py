import numpy as np

from utils.transform_utils import normalize, rot_about_axis

# ----------------------------
# OBB fitting (PCA-based)
# ----------------------------
def fit_obb_pca(points, margin=0.0, min_points=10):
    """
    Fit an oriented bounding box (OBB) using PCA.

    Args:
        points: (N,3) np.ndarray
        margin: float, expand extents by this amount on each side (same unit as points)
        min_points: minimum points required

    Returns:
        obb: dict or None
          {
            "center": (3,),            # world center
            "R": (3,3),                # columns are OBB axes in world (right-handed)
            "extents": (3,),           # full lengths along each axis (L0,L1,L2)
            "mins_local": (3,),        # min coords in OBB local frame
            "maxs_local": (3,),        # max coords in OBB local frame
          }
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if points.shape[0] < min_points:
        return None

    # PCA
    mean = points.mean(axis=0)
    X = points - mean
    C = (X.T @ X) / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]                 # columns = principal axes

    # enforce right-handed
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    # project to local
    P_local = (R.T @ (points - mean).T).T
    mins = P_local.min(axis=0) - margin
    maxs = P_local.max(axis=0) + margin
    q1 = np.percentile(P_local, 25, axis=0)
    q3 = np.percentile(P_local, 75, axis=0)
    extents = (maxs - mins)

    # center in world: mean + R @ center_local
    # center_local = 0.5 * (mins + maxs)
    center_local = 0.5 * (q1+q3)
    center = mean + R @ center_local

    return {
        "center": center,
        "R": R,
        "extents": extents,
        "mins_local": mins,
        "maxs_local": maxs,
    }


# ----------------------------
# Grasp generation from OBB
# ----------------------------
def _make_grasp_T(center, approach, closing):
    """
    Build grasp frame:
      z = approach
      x = closing
      y = z x x
    """
    z = normalize(approach)
    x = normalize(closing)

    # make x orthogonal to z
    x = x - z * np.dot(x, z)
    x = normalize(x)

    y = normalize(np.cross(z, x))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = center
    return T

def generate_grasps_from_obb(
    obb,
    rotation,                    # (3,3) REQUIRED: forced grasp rotation in world
    yaw_samples=1,               # forced rotation이면 기본 1이 자연스러움
    yaw_axis=[0,0,1],                # "x"|"y"|"z" or np.ndarray(3,)
    pregrasp_offset=0.1,
    grasp_depth=0.0,             # object center에서 approach 방향으로 추가 이동(원하면)
    approach_axis="z",           # approach 축을 forced rotation의 어느 축으로 둘지
):
    """
    Use OBB only for position (object center), and FORCE the grasp rotation.

    Args:
        obb: dict from fit_obb_pca, uses only obb["center"]
        rotation: (3,3) forced orientation in world frame
        yaw_samples: if >1, sample yaw around yaw_axis of the forced rotation
        yaw_axis: which axis to yaw around ("x","y","z") or explicit (3,) vector in world
        pregrasp_offset: backoff along -approach
        grasp_depth: shift the grasp position along +approach
        approach_axis: which axis of forced rotation is "approach" ("x","y","z")

    Returns:
        grasps: list of dicts with forced rotation
    """
    if obb is None:
        return []

    # 1) position from OBB (object position)
    c = np.asarray(obb["center"], dtype=np.float64).reshape(3)

    # 2) forced rotation
    Rf = np.asarray(rotation, dtype=np.float64)
    assert Rf.shape == (3, 3), f"rotation must be (3,3), got {Rf.shape}"
    if np.linalg.det(Rf) < 0:
        # keep right-handed
        Rf = Rf.copy()
        Rf[:, 2] *= -1.0

    axis_idx = {"x": 0, "y": 1, "z": 2}

    # 3) define approach direction from forced rotation axis
    assert approach_axis in axis_idx, "approach_axis must be one of 'x','y','z'"
    approach_w = Rf[:, axis_idx[approach_axis]]
    approach_w = normalize(approach_w)

    # 4) grasp position shift (optional)
    grasp_center = c + grasp_depth * approach_w

    # 5) yaw axis (optional sampling)
    if isinstance(yaw_axis, str):
        assert yaw_axis in axis_idx, "yaw_axis must be one of 'x','y','z' or a (3,) vector"
        yaw_axis_w = Rf[:, axis_idx[yaw_axis]]
    else:
        yaw_axis_w = np.asarray(yaw_axis, dtype=np.float64).reshape(3)
    yaw_axis_w = normalize(yaw_axis_w)

    grasps = []
    for k in range(max(int(yaw_samples), 1)):
        yaw = 0.0 if yaw_samples <= 1 else (2 * np.pi * k / yaw_samples)
        R_yaw = rot_about_axis(yaw_axis_w, yaw)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_yaw @ Rf          # ✅ orientation 완전 강제
        T[:3, 3]  = grasp_center        # ✅ position은 object center 기반

        pre_T = T.copy()
        pre_T[:3, 3] = T[:3, 3] - pregrasp_offset * T[:3, 2]  # -approach (T의 z축 기준)

        grasps.append({
            "T_wg": T,
            "pre_T_wg": pre_T,
            "meta": {
                "forced_rotation": True,
                "yaw_idx": k,
                "yaw_samples": yaw_samples,
                "approach_axis": approach_axis,
                "yaw_axis": yaw_axis if isinstance(yaw_axis, str) else "custom",
            }
        })

    return grasps