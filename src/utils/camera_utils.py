import numpy as np
import cv2
from pupil_apriltags import Detector

from utils.transform_utils import inverse_transform, rt_to_transform_matrix, SE3_mean


_APRILTAG_DETECTOR = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

def pts_to_pixels(
    pts,
    K,
    axis_forward="z"
):
    """
    Project camera-frame 3D points to image pixels.

    Args:
        pts: (N,3) camera-frame points
        K: (3,3) intrinsic matrix
        axis_forward: "z" (standard pinhole) or "x" (Coppelia/Isaac-style)

    Returns:
        uv: (N,2) pixel coordinates (float)
        valid: (N,) bool mask (points in front of camera)
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if axis_forward == "z":
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        valid = z > 0

        u = fx * (x / (z + 1e-12)) + cx
        v = fy * (y / (z + 1e-12)) + cy

    elif axis_forward == "x":
        # 네가 앞에서 쓰던 CoppeliaSim 스타일
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        valid = x > 0

        u = fx * (-y / (x + 1e-12)) + cx
        v = fy * (-z / (x + 1e-12)) + cy

    else:
        raise ValueError("axis_forward must be 'z' or 'x'")

    uv = np.stack([u, v], axis=1)
    return uv, valid

def estimate_T_ec_from_image_and_fk_list(
    image_and_fk_list,
    K,
    dist=None,
    chessboard_size=(7, 6),
    square_size=0.025,
    reproj_thresh_px=0.8,
    method=cv2.CALIB_HAND_EYE_TSAI,
):
    """
    image_and_fk_list: [{"rgb": (H,W,3) RGB, "end_pose": ...}, ...]
    end_pose -> 반드시 4x4 T_we (base->ee)로 변환 가능해야 함.
              이미 end_pose가 4x4라면 그대로 사용.
              아니면 아래 'to_T_we' 부분을 네 로봇 형식에 맞게 바꿔야 함.
    Returns:
      T_ec (4,4), used_count, stats(dict)
    """
    if dist is None:
        dist = np.zeros((5,), dtype=np.float64)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    used = 0
    all_err = []

    for item in image_and_fk_list:
        rgb = item["rgb"]
        end_pose = item["end_pose"]

        # --- 1) end_pose -> T_we (base->ee) ---
        # case A: already 4x4
        T_we = np.asarray(end_pose, dtype=np.float64)
        if T_we.shape != (4,4):
            raise ValueError("end_pose must be a 4x4 transform (T_we). "
                             "Convert your robot end_pose format to T_we here.")

        # OpenCV wants gripper->base, so invert base->gripper
        T_ew = inverse_transform(T_we)

        # --- 2) RGB -> T_chc (checkerboard->camera) ---
        T_chc, err = estimate_T_chc_from_checkerboard(
            rgb=rgb,
            K=K,
            chessboard_size=chessboard_size,
            square_size=square_size,
            dist=dist,
            refine=True,
        )
        if T_chc is None:
            continue

        if err is None or err > reproj_thresh_px:
            continue

        all_err.append(err)

        # collect
        R_gripper2base.append(T_ew[:3, :3])
        t_gripper2base.append(T_ew[:3, 3].reshape(3, 1))

        R_target2cam.append(T_chc[:3, :3])
        t_target2cam.append(T_chc[:3, 3].reshape(3, 1))

        used += 1

    if used < 6:
        raise RuntimeError(f"Not enough valid samples: {used}. "
                           "Need more poses with successful checkerboard detection.")

    R_ce, t_ce = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=method,
    )

    T_ce = np.eye(4, dtype=np.float64)
    T_ce[:3, :3] = R_ce
    T_ce[:3, 3]  = t_ce.reshape(3)
    T_ec = inverse_transform(T_ce)

    stats = {
        "used": used,
        "mean_reproj_err_px": float(np.mean(all_err)) if all_err else None,
        "max_reproj_err_px": float(np.max(all_err)) if all_err else None,
    }
    return T_ec, used, stats

def estimate_T_chc_from_checkerboard(
    rgb,                 # (H,W,3) RGB
    K,                   # (3,3)
    chessboard_size=(7, 6),  # (cols, rows) inner corners
    square_size=0.025,       # meters
    dist=None,               # (5,) or (8,) etc.
    refine=True
):
    """
    Returns:
      T_chc: (4,4) checkerboard -> camera
      reproj_err_px: float
    """
    if dist is None:
        dist = np.zeros((5,), dtype=np.float64)

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)

    cols, rows = chessboard_size
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

    if not found:
        return None, None

    if refine:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    # checkerboard 3D points in checkerboard frame (ch)
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    T_chc = rt_to_transform_matrix(R, tvec.reshape(3))

    # reprojection error
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    obs = corners.reshape(-1, 2)
    reproj_err_px = float(np.mean(np.linalg.norm(proj - obs, axis=1)))

    return T_chc, reproj_err_px

def estimate_T_ec_from_image_and_fk_list_apriltag(
    image_and_fk_list,
    K,
    tag_size=0.04,               # meters
    tag_id=None,                 # 특정 tag만 쓰면 지정
    min_decision_margin=30.0,    # 낮으면 outlier 가능성 ↑ (상황에 맞게 조절)
    method=cv2.CALIB_HAND_EYE_TSAI,
):
    """
    image_and_fk_list: [{"rgb": RGB(undistorted), "end_pose": (4,4) T_we}, ...]
    Returns: T_ec (4,4), used, stats
    """
    R_gripper2base, t_gripper2base = [], []
    R_target2cam,   t_target2cam   = [], []

    used = 0
    margins = []

    for item in image_and_fk_list:
        rgb = item["rgb"]
        T_we = np.asarray(item["end_pose"], dtype=np.float64)
        if T_we.shape != (4, 4):
            raise ValueError("end_pose must be 4x4 T_we (base->ee).")

        # AprilTag pose: tag->camera
        T_ct, margin = estimate_T_ct_from_apriltag(
            rgb=rgb, K=K, tag_size=tag_size, tag_id=tag_id
        )
        if T_ct is None:
            continue
        if margin is not None and margin < float(min_decision_margin):
            continue

        margins.append(float(margin) if margin is not None else 0.0)

        R_gripper2base.append(T_we[:3, :3])
        t_gripper2base.append(T_we[:3, 3].reshape(3, 1))

        R_target2cam.append(T_ct[:3, :3])
        t_target2cam.append(T_ct[:3, 3].reshape(3, 1))

        used += 1

    if used < 6:
        raise RuntimeError(f"Not enough valid samples: {used}. Need >= 6 with valid tag detection.")

    R_ec, t_ec = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam,   t_target2cam,
        method=method,
    )

    T_ec = rt_to_transform_matrix(R_ec, t_ec)

    stats = {
        "used": used,
        "mean_decision_margin": float(np.mean(margins)) if margins else None,
        "min_decision_margin": float(np.min(margins)) if margins else None,
    }
    return T_ec, used, stats

def estimate_T_ct_from_apriltag(
    rgb,                    # (H,W,3) RGB (undistorted 권장)
    K,                      # (3,3)
    board_T_bt={
        0:np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ]),
        1:np.array([
            [1,0,0,0.15],
            [0,1,0,-0.13],
            [0,0,1,0],
            [0,0,0,1],
        ]),
        2:np.array([
            [1,0,0,0],
            [0,1,0,-0.13],
            [0,0,1,0],
            [0,0,0,1],
        ]),
        3:np.array([
            [1,0,0,0.15],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ]),
    },
    tag_size=0.1,          # meters (실측!)
    min_tags=2,  # 한 프레임에서 최소 몇 개 태그가 필요?
    decision_margin_min=10.0,  # 너무 낮은 검출 제거 (환경 따라 10~60 조절)
    outlier_trans_thresh=0.03,  # (m) 후보들 사이 translation outlier 컷 (예: 3cm)
):
    """
    Returns:
      T_tagc: (4,4) tag -> camera
      quality: float (작을수록/클수록? 라이브러리마다 다름 → 여기선 pose_err 기반)
    """
    # pupil-apriltags Detector 생성 시 families가 고정되므로, family 바꾸려면 detector를 다시 만들어야 함.
    # 여기서는 tag36h11 기준으로만 둠.

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    dets = _APRILTAG_DETECTOR.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=float(tag_size),
    )

    if not dets:
        return None, None

    T_cb_candidates = []
    weights = []
    for d in dets:
        tid = int(d.tag_id)
        if tid not in board_T_bt:
            continue

        dm = float(getattr(d, "decision_margin", 0.0))
        if dm < decision_margin_min:
            continue

        # tag -> camera (네 기존 코드와 동일하게 사용)
        R = np.asarray(d.pose_R, dtype=np.float64)
        t = np.asarray(d.pose_t, dtype=np.float64).reshape(3)
        T_ct= rt_to_transform_matrix(R, t)     # tag -> camera
        T_bt = board_T_bt[tid]                 # board -> tag
        T_cb = T_ct @ inverse_transform(T_bt)

        T_cb_candidates.append(T_cb)
        weights.append(dm)

    if len(T_cb_candidates) < min_tags:
        return None, None

    # ---------- (옵션) 후보들 간 translation outlier 제거 ----------
    # 빠르고 단순한 컷: 후보들의 translation 중앙값 기준으로 먼 것 제거
    ts = np.array([T[:3, 3] for T in T_cb_candidates])
    med = np.median(ts, axis=0)
    keep = [i for i, t in enumerate(ts) if np.linalg.norm(t - med) <= outlier_trans_thresh]

    if len(keep) < min_tags:
        return None, None

    T_cb_kept = [T_cb_candidates[i] for i in keep]
    w_kept = [weights[i] for i in keep]

    # ---------- SE(3) 평균 ----------
    # SE3_mean이 가중치를 지원하면 weights=w_kept로 넘기고,
    # 지원하지 않으면 그냥 평균.
    T_cb_mean = SE3_mean(T_cb_kept, weights=w_kept)

    quality = float(np.mean(w_kept)) if w_kept else 0.0
    return T_cb_mean, quality