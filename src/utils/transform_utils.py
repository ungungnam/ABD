import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rot_mat(q, quat='wxyz'):
    if quat == 'wxyz':
        w, x, y, z = q
    elif quat == 'xyzw':
        x, y, z, w = q
    else:
        raise Exception("Unknown quat")

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], dtype=np.float64)
    return R

def rt_to_transform_matrix(R, t):
    """
    Build a 4x4 homogeneous transform from rotation and translation.

    Args:
        R: (3,3) rotation matrix
        t: (3,) or (3,1) translation vector

    Returns:
        T: (4,4) homogeneous transform matrix
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    assert R.shape == (3, 3), f"R must be (3,3), got {R.shape}"

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rpy_to_rot_mat(rpy, degrees=False):
    rpy = np.asarray(rpy, dtype=np.float64)
    if degrees:
        rpy = np.deg2rad(rpy)
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def pose6d_to_transform_matrix(pose6d, degrees=False):
    pose6d = np.asarray(pose6d, dtype=np.float64).reshape(-1)
    t = pose6d[:3]
    rpy = pose6d[3:]
    R = rpy_to_rot_mat(rpy, degrees=degrees)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def pq_to_transform_matrix(p, q, quat='wxyz'):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot_mat(q, quat)
    T[:3, 3] = p
    return T

def transform_matrix_to_pq(T, quat='wxyz'):
    T = np.asarray(T, dtype=np.float64)
    assert T.shape == (4, 4), f"T must be (4,4), got {T.shape}"

    p = T[:3, 3].copy()
    R = T[:3, :3]
    q = rot_mat_to_quat(R, quat=quat)
    return p, q

def rot_mat_to_quat(R, quat='wxyz'):
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3), f"R must be (3,3), got {R.shape}"

    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)

    # Optional: enforce a consistent sign (w >= 0)
    if q_wxyz[0] < 0:
        q_wxyz *= -1.0

    if quat == 'wxyz':
        return q_wxyz
    elif quat == 'xyzw':
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    else:
        raise ValueError("quat must be 'wxyz' or 'xyzw'")

def rot_mat_to_rpy(R, degrees=False, eps=1e-12):
    """
    Inverse of rpy_to_rot_mat where R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Returns rpy = [roll, pitch, yaw].
    """
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3), f"R must be (3,3), got {R.shape}"

    # pitch = asin(-R[2,0]) for ZYX
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0 + eps, 1.0 - eps)
    pitch = np.arcsin(sy)

    # Handle gimbal lock when cos(pitch) ~ 0
    cy = np.cos(pitch)
    if np.abs(cy) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch ~= +/- pi/2
        # yaw and roll are coupled; set roll=0 and compute yaw from other terms
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    rpy = np.array([roll, pitch, yaw], dtype=np.float64)
    if degrees:
        rpy = np.rad2deg(rpy)
    return rpy

def transform_matrix_to_pose6d(T, degrees=False):
    """
    Convert 4x4 transform to pose6d = [x, y, z, roll, pitch, yaw].
    roll/pitch/yaw follow the same convention as rpy_to_rot_mat:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    T = np.asarray(T, dtype=np.float64)
    assert T.shape == (4, 4), f"T must be (4,4), got {T.shape}"

    t = T[:3, 3].copy()
    R = T[:3, :3]
    rpy = rot_mat_to_rpy(R, degrees=degrees)

    pose6d = np.concatenate([t, rpy], axis=0).astype(np.float64)
    return pose6d

def transform_points(T, pts, multi_points=True):
    if pts.ndim == 1:
        pts = pts[None, :]  # (1,3)
        multi_points=False
    
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)

    pts_out_h = (T @ pts_h.T).T  # (N,4)
    pts_out = pts_out_h[:, :3]

    return pts_out if multi_points else pts_out[0]

def transform_rays(T, rays, multi_rays=True, normalize_out=False):
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim == 1:
        rays = rays[None, :]
        multi_rays = False

    R = np.asarray(T[:3, :3], dtype=np.float64)
    rays_out = (R @ rays.T).T  # (N,3)

    if normalize_out:
        norms = np.linalg.norm(rays_out, axis=1, keepdims=True)
        rays_out = rays_out / (norms + 1e-12)

    return rays_out if multi_rays else rays_out[0]

def inverse_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + eps)

def rot_about_axis(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)

def rotation_geodesic(R1, R2, eps=1e-12):
    R = R1.T @ R2
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return np.arccos(cos_theta)

def lerp(a, b, t):
    return (1.0 - t) * a + t * b

def SE3_mean(T_list, weights=None):
    R_list, t_list = [], []
    for T in T_list:
        R_list.append(T[:3,:3])
        t_list.append(T[:3,3])

    R_stack = R.from_matrix(R_list)
    R_mean = R_stack.mean(weights).as_matrix()
    t_mean = np.mean(np.stack(t_list, axis=0), axis=0)

    T_mean = rt_to_transform_matrix(R_mean, t_mean)
    return T_mean