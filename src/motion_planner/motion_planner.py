import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from utils.camera_utils import pts_to_pixels, detect_single_tag_world_pose, detect_tag_world_pose_stereo

log = logging.getLogger(__name__)
from utils.obb_utils import fit_obb_pca, generate_grasps_from_obb
from utils.transform_utils import inverse_transform, normalize, pq_to_transform_matrix, rotation_geodesic, transform_points, transform_rays
from utils.utils import map_vlm_action_to_vector
from utils.trajectory_utils import smoothen_trajectory, visualize_trajectory


class MotionPlanner():
    def __init__(self, config, env, robot, cameras):
        self.config = config
        self.env = env
        self.robot = robot
        self.cameras = cameras
        # tag_id -> T: pre-populated with hardcoded defaults, overwritten by forward steps
        #   tag 6 (pink):   [x, y] = [0.3476801,  0.15095109]
        #   tag 5 (purple): [x, y] = [0.3476801, -0.05095109]
        self._tag_position_cache: dict = self._make_default_tag_cache()
        # Last detected drawer tag pose and yaw delta from pre-execution reference
        self.last_drawer_tag_pose: Optional[np.ndarray] = None
        self.last_drawer_rotation_deg: Optional[float] = None
        self._drawer_ref_yaw: Optional[float] = None  # atan2 yaw at init pose before execution
        # Last detected world XYZ from pick_place perception (None until first plan).
        self.last_pick_position: Optional[Tuple[float, float, float]] = None
        self.last_place_position: Optional[Tuple[float, float, float]] = None

    @classmethod
    def _make_default_tag_cache(cls) -> dict:
        """Pre-populate cache with hardcoded original positions for each cup tag."""
        defaults = {
            6: [0.3476801,  0.15095109],   # pink cup
            5: [0.3476801, -0.05095109],   # purple cup
        }
        cache = {}
        for tag_id, (x, y) in defaults.items():
            T = np.eye(4, dtype=np.float64)
            T[0, 3] = x
            T[1, 3] = y
            T[2, 3] = cls._TAG_Z
            cache[tag_id] = T
        return cache

    # ------------------------------------------------------------------ #
    # Stack-cups geometry constants
    # ------------------------------------------------------------------ #
    _CUP_HEIGHT       = 0.09    # 9 cm
    _TABLE_Z          = -0.063  # table surface Z in robot base frame (measured)
    _TAG_Z            = _TABLE_Z + _CUP_HEIGHT  # = 0.027 m — fixed tag height
    _GRASP_BELOW_TAG  = 0.025   # gripper TIP offset below tag
    _GRASP_BELOW_TAG_unstack  = 0.020 
    _PICK_TAG_XY_OFFSET  = [ 0.0,  0.0]   # pick correction: no offset
    _PLACE_TAG_XY_OFFSET = [ 0.0,  0.0]   # place correction: no offset
    _STACK_OFFSET     = _CUP_HEIGHT - _GRASP_BELOW_TAG  # exact height above place-tag
    _STACK_DROP_EXTRA = 0.05 # extra Z so cup drops naturally onto blue cup
    _GRASP_TILT_Y_DEG = -30   # gripper tilt around Y axis for pick/place (degrees)

    # ---- Drawer constants ----
    _HANDLE_TAG_OFFSET  = np.array([0.05, 0.035, 0.03])  # handle pos in AprilTag frame (x,y,z)
    _DRAWER_PULL_DIST   = 0.110    # open/close distance along world X (m)
    _DRAWER_APPROACH    = 0.08    # pre-pick offset in +X from handle (m)

    _DRAWER_TILT_Y_DEG = +45  # gripper tilt around Y axis for drawer grasp (degrees)

    @classmethod
    def _drawer_R(cls) -> np.ndarray:
        """Gripper rotation for drawer handle grasp.

        Base: tip points in +X world (closing direction of drawer).
        Then tilt _DRAWER_TILT_Y_DEG around Y for a more stable approach angle.
        """
        base_R = np.array([[ 0., 0., 1.],
                            [ 0., 1., 0.],
                            [-1., 0., 0.]], dtype=np.float64)
        a = np.deg2rad(cls._DRAWER_TILT_Y_DEG)
        Ry = np.array([[ np.cos(a), 0., np.sin(a)],
                       [        0., 1.,         0.],
                       [-np.sin(a), 0., np.cos(a)]], dtype=np.float64)
        return Ry @ base_R

    @classmethod
    def _grasp_R(cls) -> np.ndarray:
        """Forced gripper rotation with Y-axis tilt applied."""
        base_R = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], dtype=np.float64)
        a = np.deg2rad(cls._GRASP_TILT_Y_DEG)
        Ry = np.array([[ np.cos(a), 0., np.sin(a)],
                       [        0., 1.,         0.],
                       [-np.sin(a), 0., np.cos(a)]], dtype=np.float64)
        return Ry @ base_R

    def _finalize(self, trajectory, events, key_poses):
        smoothened = smoothen_trajectory(trajectory)
        partial_trajectory, partial_events = self.get_partial_trajectory(
            smoothened, events, proportion=1.0)
        return partial_trajectory, partial_events, key_poses

    def plan_pick_place(self, pick_perception, place_perception, vlm_action=None, place_offset=None, place_bowl_center_xy=False):
        pick_grasp_pose  = self._get_best_grasp_pose_from_perception(pick_perception,  hover_offset=0.05)
        place_grasp_pose = self._get_best_grasp_pose_from_perception(place_perception, hover_offset=0.05, use_median_xy=place_bowl_center_xy)
        # Cache last detected world XYZ for downstream observers (collection_runner
        # injects these into the policy prompt as object positions). Keep the
        # previous value when a fresh detection is unavailable, so policies still
        # see the most recent known position.
        if pick_grasp_pose is not None:
            self.last_pick_position = tuple(pick_grasp_pose["T_wg"][:3, 3].tolist())
        if place_grasp_pose is not None:
            self.last_place_position = tuple(place_grasp_pose["T_wg"][:3, 3].tolist())
        if place_grasp_pose is not None:
            # Force place orientation to base _grasp_R() so the held object
            # always ends up along world X (closing axis = world Y) regardless
            # of the place container's PCA — keeps banana laid front-back.
            base_R = self._grasp_R()
            place_grasp_pose['T_wg'][:3, :3] = base_R
            place_grasp_pose['pre_T_wg'][:3, :3] = base_R
            # Always drop from a fixed height = TABLE_Z + 3 cm, regardless of
            # the perceived container Z, so the object falls cleanly.
            drop_z = self._TABLE_Z + 0.03
            place_grasp_pose['T_wg'][2, 3] = drop_z
            place_grasp_pose['pre_T_wg'][2, 3] = drop_z + 0.05
        if place_grasp_pose is not None and place_offset is not None:
            offset = np.array(place_offset, dtype=np.float64)
            place_grasp_pose['T_wg'][:3, 3]     += offset
            place_grasp_pose['pre_T_wg'][:3, 3] += offset
            log.info(f"[MotionPlanner] place_offset applied: {offset}")
        if pick_grasp_pose and place_grasp_pose:
            trajectory, events, key_poses = self.generate_trajectory(
                vlm_action=vlm_action,
                pick_grasp_pose=pick_grasp_pose,
                place_grasp_pose=place_grasp_pose,
                post_lift=True,
                post_lift_height=0.05,
                descent_max_step=0.010,
                lift_max_step=0.010,
                max_step=0.03,
                close_trans_thresh=0.005,
                ext_len=3,
            )
        else:
            trajectory, events, key_poses = self.generate_trajectory(vlm_action=vlm_action)
        return self._finalize(trajectory, events, key_poses)

    def plan_stack_cups(self, pick_tag_id, place_tag_id, place_xy_offset, stack_step, vlm_action=None):
        pick_grasp_pose, place_grasp_pose, pick_detected, place_detected = \
            self._get_stack_grasp_poses(
                pick_tag_id, place_tag_id, place_xy_offset, stack_step=stack_step)
        if pick_grasp_pose and place_grasp_pose:
            trajectory, events, key_poses = self.generate_trajectory(
                vlm_action=vlm_action,
                pick_grasp_pose=pick_grasp_pose,
                place_grasp_pose=place_grasp_pose,
                descent_max_step=0.004,
                lift_max_step=0.008,
            )
            return self._finalize(trajectory, events, key_poses), pick_detected, place_detected
        else:
            # AprilTag not detected — signal failure cleanly (no VLM fallback)
            return (None, [], []), pick_detected, place_detected

    def plan_open_drawer(self, pick_tag_id, stack_step):
        action = "open" if stack_step == "forward" else "close"
        pick_grasp_pose, place_grasp_pose = self._get_drawer_grasp_poses(pick_tag_id, action=action)
        if pick_grasp_pose and place_grasp_pose:
            trajectory, events, key_poses = self.generate_drawer_trajectory(
                pick_grasp_pose, place_grasp_pose, action=action)
        else:
            trajectory, events, key_poses = [], [], []
        return self._finalize(trajectory, events, key_poses)

    def get_reference_object_points(self, perception) -> np.ndarray:
        used_cameras = list(perception.keys())

        best_score = 0.0
        best_pts = None

        for i in range(len(used_cameras)):
            for j in range(len(used_cameras)):
                if i <= j:
                    continue
                camera_1 = used_cameras[i]
                camera_2 = used_cameras[j]

                mask_1 = np.array(perception[camera_1]['mask']).reshape(
                    (self.cameras[camera_1].height, self.cameras[camera_1].width))
                mask_2 = np.array(perception[camera_2]['mask']).reshape(
                    (self.cameras[camera_2].height, self.cameras[camera_2].width))

                score, pts = self._get_score(
                    cameras=[camera_1, camera_2],
                    masks=[mask_1, mask_2]
                )

                if score > best_score:
                    best_score = score
                    best_pts = pts

        return best_pts

    def get_cone(self, camera, mask, boundary_only=True):
        H, W = mask.shape

        K = camera.get_intrinsic_matrix()

        if boundary_only:
            # simple 4-neighborhood boundary: mask True but has any False neighbor
            m = mask
            # pad to avoid border issues
            mp = np.pad(m, ((1, 1), (1, 1)), mode="constant",
                        constant_values=False)
            up = mp[0:H,     1:W+1]
            down = mp[2:H+2,   1:W+1]
            left = mp[1:H+1,   0:W]
            right = mp[1:H+1,   2:W+2]
            boundary = m & (~up | ~down | ~left | ~right)
            v, u = np.where(boundary)
        else:
            v, u = np.where(mask)
        pixels_uv = np.stack([u, v], axis=1)

        rays_c = self.get_rays(pixels=pixels_uv, K=K, axis_forward='z')

        T_wc = camera.get_extrinsic_matrix()
        p_wc = T_wc[:3,3]
        rays_w = transform_rays(T_wc, rays_c)

        return {
            'origin_w': p_wc,
            'rays_w': rays_w,
            'pixels_uv': pixels_uv
        }

    def current_pose(self):
        return self.env.current_pose()

    def get_rays(self, pixels, K, axis_forward="x", normalize=True):
        """
        Build camera-frame rays from pixel coordinates.

        Args:
            pixels: (N,2) array-like of (u, v) pixel coords.
                    u: x-axis (cols), v: y-axis (rows)
            K: (3,3) intrinsic matrix
            axis_forward: "z" for standard pinhole (Z forward, OpenCV-like)
                          "x" for Coppelia/Isaac-style you used earlier (X forward)
            normalize: whether to normalize rays to unit vectors

        Returns:
            rays: (N,3) float array, camera-frame ray directions
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim == 1:
            pixels = pixels[None, :]  # (1,2)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        u = pixels[:, 0]
        v = pixels[:, 1]

        # Standard pinhole: ray = [(u-cx)/fx, (v-cy)/fy, 1]
        x_n = (u - cx) / (fx + 1e-12)
        y_n = (v - cy) / (fy + 1e-12)

        if axis_forward.lower() == "z":
            rays = np.stack([x_n, y_n, np.ones_like(x_n)], axis=1)  # (N,3)

        elif axis_forward.lower() == "x":
            # Your earlier convention:
            # x = 1, y = -(u-cx)/fx, z = -(v-cy)/fy
            rays = np.stack([np.ones_like(x_n), -x_n, -y_n], axis=1)  # (N,3)

        else:
            raise ValueError("axis_forward must be 'z' or 'x'")

        if normalize:
            rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12)

        return rays

    def sample_cone_intersection(
        self,
        cone,
        camera,
        mask,
        depth_range=(0.1, 1.0),
        n_depth=300,
        pick='frontmost'
    ):
        H, W = mask.shape

        origin_w = cone["origin_w"]
        rays_w = cone["rays_w"]
        
        T_wc = camera.get_extrinsic_matrix()
        T_cw = inverse_transform(T_wc)

        depths = np.linspace(depth_range[0], depth_range[1], n_depth).astype(np.float64)
        pts_keep_list = []

        for ray_w in rays_w:
            pt_w = origin_w[None, :] + depths[:, None] * ray_w[None, :]
            pt_c = transform_points(T_cw, pt_w)

            pixels_c, valid = pts_to_pixels(pts=pt_c, K=camera.get_intrinsic_matrix())
            
            u = np.round(pixels_c[:, 0]).astype(int)
            v = np.round(pixels_c[:, 1]).astype(int)
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

            ok = valid & in_img
            idx = np.where(ok)[0]

            if not np.any(ok):
                continue

            inside = np.zeros_like(ok, dtype=bool)
            inside[idx] = mask[v[idx], u[idx]] 

            if not np.any(inside):
                continue

            if pick == "frontmost":
                first = np.argmax(inside)      # first True along depth samples
                pts_keep_list.append(pt_w[first])
            elif pick == 'all':
                pts_keep_list.append(pt_w[inside])
            elif pick =='median':
                idxs = np.where(inside)[0]
                mid = idxs[len(idxs) // 2]
                pts_keep_list.append(pt_w[mid])
            elif pick == 'IQR':
                idxs = np.where(inside)[0]
                d_inside = depths[idxs]
                q1, q3 = np.percentile(d_inside, [25, 75])
                keep = (depths >= q1) & (depths <= q3) & inside
                pts_keep_list.append(pt_w[keep])

        if len(pts_keep_list) == 0:
            return np.empty((0, 3), dtype=np.float64)
        pts_keep = np.vstack([p.reshape(1, 3) if p.ndim == 1 else p for p in pts_keep_list])
        
        return pts_keep

    def sample_best_grasp(self, grasp_poses, w_trans=1.0, w_rot=0.25):
        """
        grasp_poses: list of dict
        - each dict must contain key "T_wg" : (4,4)
        Return:
        best_idx: int
        best_T: (4,4)
        best_score: float
        """
        if len(grasp_poses) == 0:
            return None

        T_cur = self.current_pose()          # (4,4)
        R_cur = T_cur[:3, :3]
        t_cur = T_cur[:3, 3]

        best_idx = -1
        best_score = np.inf

        for i, g in enumerate(grasp_poses):
            T_g = g["T_wg"]
            R_g = T_g[:3, :3]
            t_g = T_g[:3, 3]

            d_trans = np.linalg.norm(t_g - t_cur)
            d_rot = rotation_geodesic(R_cur, R_g)

            score = w_trans * d_trans + w_rot * d_rot

            if score < best_score:
                best_score = score
                best_idx = i

        return grasp_poses[best_idx]

    def _make_vlm_subgoal(self, vlm_action, step=0.1, keep_rotation=True):
        """
        Make a short-horizon subgoal from current pose moving along v0.
        """
        T_cur = self.current_pose()
        t = T_cur[:3, 3]

        vlm_action_vector = map_vlm_action_to_vector(vlm_action)
        vlm_action_vector_g = self._transform_to_gripper_frame(vlm_action_vector)
        vlm_action_vector_w = transform_rays(T_cur, vlm_action_vector_g)
        v = normalize(vlm_action_vector_w)
        t_sub = t + step * v

        T_sub = T_cur.copy()
        T_sub[:3, 3] = t_sub

        if not keep_rotation:
            # Optionally: keep current anyway (default), or you could align yaw, etc.
            pass

        return T_sub

    def _interpolate_poses_linear(self, T_a, T_b, max_step=0.05, min_n=2, max_n=60):
        """
        max_step: 한 waypoint 간 최대 이동 거리 (m 단위면 0.01~0.03 정도가 흔함)
        """
        if T_a is None or T_b is None:
            return []

        ta = T_a[:3, 3]
        tb = T_b[:3, 3]
        dist = float(np.linalg.norm(tb - ta))

        # 거리 기반으로 점 개수 자동 결정
        n = int(np.ceil(dist / max_step)) + 1
        n = max(min_n, min(max_n, n))

        Ra = T_a[:3, :3]
        Rb = T_b[:3, :3]

        poses = []
        for i in range(n):
            t = i / max(n - 1, 1)

            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = (1.0 - t) * ta + t * tb

            # rotation: cheap schedule (필요하면 slerp로 교체 가능)
            T[:3, :3] = Ra if t < 0.5 else Rb
            poses.append(T)

        return poses

    def generate_trajectory(
            self,
            vlm_action,
            pick_grasp_pose = None,
            place_grasp_pose = None,
            pick_event: str = "CLOSE",
            place_event: str = "OPEN",
            step: float = 0.06,
            close_trans_thresh: float = 0.03,
            close_rot_thresh_deg: float = 10.0,
            ext_len=3,
            post_lift: bool = True,
            post_lift_height: float = 0.05,
            descent_max_step: float = 0.05,
            lift_max_step: float = 0.05,
            max_step: float = 0.05,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Return:
          traj_T: List[4x4] end-effector poses in world (SE(3))
          events: List of event dicts, e.g. {"at": idx, "cmd": "CLOSE"}
                 - 'at' is the waypoint index where you *intend* to trigger.
                 - execution code should still gate by pose error.
          key_poses: list of key SE(3) poses (empty when post_lift=False)
        """
        def _is_T_close(T1: np.ndarray, T2: np.ndarray) -> bool:
            d_trans = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
            d_rot = rotation_geodesic(T1[:3, :3], T2[:3, :3])  # rad
            return (d_trans < close_trans_thresh) and (d_rot < np.deg2rad(close_rot_thresh_deg))

        def _append(traj: List[np.ndarray], seg: List[np.ndarray]) -> None:
            if seg:
                traj.extend(seg)

        T_cur = self.current_pose()
        traj_T: List[np.ndarray] = []
        events: List[Dict[str, Any]] = []
        key_poses: List[np.ndarray] = []

        # ---- Case 1: no pick & no place -> VLM-only ----
        if pick_grasp_pose is None and place_grasp_pose is None:
            T_vlm = self._make_vlm_subgoal(vlm_action, step=step, keep_rotation=True)
            traj_T = self._interpolate_poses_linear(T_cur, T_vlm)
            traj_T = self._prune_duplicates(traj_T)
            return traj_T, [], []

        # ---- Case 2: require BOTH pick and place ----
        else:
            pre_pick = pick_grasp_pose["pre_T_wg"]
            T_pick = pick_grasp_pose["T_wg"]
            pre_place = place_grasp_pose["pre_T_wg"]
            T_place = place_grasp_pose["T_wg"]

            if post_lift:
                post_pick = T_pick.copy()
                post_pick[:3, 3] = T_pick[:3, 3] + np.array([0.0, 0.0, post_lift_height])
                post_place = T_place.copy()
                post_place[:3, 3] = T_place[:3, 3] + np.array([0.0, 0.0, post_lift_height])
                key_poses = [pre_pick, T_pick, pre_place, T_place]

            # 1) cur -> vlm(pick)
            if vlm_action is not None:
                T_vlm = self._make_vlm_subgoal(vlm_action, step=step, keep_rotation=True)
            else:
                T_vlm = T_cur
            _append(traj_T, self._interpolate_poses_linear(T_cur, T_vlm, max_step=max_step))
            traj_T = self._prune_duplicates(traj_T)

            # 2) -> pre_pick -> pick
            if not _is_T_close(T_vlm, pre_pick):
                seg = self._interpolate_poses_linear(T_vlm, pre_pick, max_step=max_step)
                _append(traj_T, self._prune_duplicates(seg))
            else:
                log.warning("[Traj] pre_pick SKIPPED (T_vlm ≈ pre_pick, d=%.4fm)",
                            np.linalg.norm(T_vlm[:3, 3] - pre_pick[:3, 3]))

            T_at_pre_pick = traj_T[-1] if traj_T else T_vlm
            if not _is_T_close(T_at_pre_pick, T_pick):
                seg = self._interpolate_poses_linear(T_at_pre_pick, T_pick, max_step=descent_max_step)
                _append(traj_T, self._prune_duplicates(seg))
            else:
                log.warning("[Traj] pick descent SKIPPED (T_at_pre_pick ≈ T_pick, d=%.4fm)",
                            np.linalg.norm(T_at_pre_pick[:3, 3] - T_pick[:3, 3]))

            # 3) dwell + CLOSE
            _append(traj_T, [traj_T[-1]]*ext_len)
            events.append({"at": len(traj_T) - 1, "cmd": pick_event})

            if post_lift:
                seg = self._interpolate_poses_linear(traj_T[-1], post_pick, max_step=lift_max_step)
                _append(traj_T, self._prune_duplicates(seg))

            _append(traj_T, self._interpolate_poses_linear(traj_T[-1], T_cur, max_step=max_step))

            # 4) -> pre_place -> place
            T_start = traj_T[-1]
            if not _is_T_close(T_start, pre_place):
                seg = self._interpolate_poses_linear(T_start, pre_place, max_step=max_step)
                _append(traj_T, self._prune_duplicates(seg))
            else:
                log.warning("[Traj] pre_place SKIPPED (T_start ≈ pre_place, d=%.4fm)",
                            np.linalg.norm(T_start[:3, 3] - pre_place[:3, 3]))

            T_at_pre_place = traj_T[-1] if traj_T else T_start
            if not _is_T_close(T_at_pre_place, T_place):
                seg = self._interpolate_poses_linear(T_at_pre_place, T_place, max_step=descent_max_step)
                _append(traj_T, self._prune_duplicates(seg))
            else:
                log.warning("[Traj] place descent SKIPPED (T_at_pre_place ≈ T_place, d=%.4fm)",
                            np.linalg.norm(T_at_pre_place[:3, 3] - T_place[:3, 3]))

            # 5) dwell + OPEN
            _append(traj_T, [traj_T[-1]]*ext_len)
            events.append({"at": len(traj_T) - 1, "cmd": place_event})

            if post_lift:
                seg = self._interpolate_poses_linear(traj_T[-1], post_place, max_step=lift_max_step)
                _append(traj_T, self._prune_duplicates(seg))

            _append(traj_T, self._interpolate_poses_linear(traj_T[-1], T_cur, max_step=max_step))

            return traj_T, events, key_poses

    def _prune_duplicates(self, traj_T: List[np.ndarray], eps: float = 1e-6) -> List[np.ndarray]:
        if len(traj_T) <= 1:
            return traj_T
        pruned = [traj_T[0]]
        for T in traj_T[1:]:
            if np.linalg.norm(T[:3, 3] - pruned[-1][:3, 3]) > eps:
                pruned.append(T)
        return pruned

    def get_partial_trajectory(
        self,
        trajectory,
        events,
        proportion=0.5,
        min_steps=10,
    ):
        """
        MPC-style partial execution.
        Returns:
        partial_traj: List[SE(3)]
        partial_events: List[event dict]
        """
        if trajectory is None or len(trajectory) == 0:
            return [], []

        n = len(trajectory)
        k = int(np.ceil(proportion * n))

        # ensure at least min_steps if possible
        k = max(min_steps, k)
        k = min(n, k)

        partial_traj = trajectory[:k]

        # 🔑 keep only events that fall inside [0, k-1]
        partial_events = []
        for e in events:
            if e["at"] <= k:
                partial_events.append(e.copy())

        return partial_traj, partial_events

    def _transform_to_gripper_frame(self, v):
        R = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        return R@v

    def _get_score(self, cameras, masks):
        camera_1, camera_2 = cameras
        mask_1, mask_2 = masks

        cone_1 = self.get_cone(
            camera=self.cameras[camera_1],
            mask=mask_1
        )
        cone_2 = self.get_cone(
            camera=self.cameras[camera_2],
            mask=mask_2
        )

        points_12 = self.sample_cone_intersection(
            cone=cone_1,
            camera=self.cameras[camera_2],
            mask=mask_2,
            pick='IQR'
        )
        points_21 = self.sample_cone_intersection(
            cone=cone_2,
            camera=self.cameras[camera_1],
            mask=mask_1,
            pick='IQR'
        )

        score_12 = len(points_12) / len(cone_1['rays_w'])
        score_21 = len(points_21) / len(cone_2['rays_w'])

        score = np.mean((score_12, score_21))
        pts = np.vstack((points_12, points_21))

        return score, pts

    def _make_grasp_pose_from_tag_world(self, T_wt, z_offset, hover_offset=0.05, xy_offset=None):
        """Build a grasp pose dict from an already-resolved tag world transform.

        Args:
            T_wt:         4×4 tag→world transform
            z_offset:     vertical shift in world Z (negative=below tag, positive=above)
            hover_offset: pre_T_wg is this far above T_wg
            xy_offset:    [dx, dy] lateral shift in world XY

        Returns:
            {"T_wg": ..., "pre_T_wg": ...}
        """
        forced_R = self._grasp_R()

        offset = np.array([0.0, 0.0, z_offset])
        if xy_offset is not None:
            offset[0] += xy_offset[0]
            offset[1] += xy_offset[1]

        T_wg = np.eye(4, dtype=np.float64)
        T_wg[:3, :3] = forced_R
        T_wg[:3, 3]  = T_wt[:3, 3] + offset

        pre_T_wg = T_wg.copy()
        pre_T_wg[:3, 3] = T_wg[:3, 3] + np.array([0.0, 0.0, hover_offset])

        return {"T_wg": T_wg, "pre_T_wg": pre_T_wg}

    # Per-step pose offsets: (pick_hover, place_z, place_hover)
    # reverse place_z = -_GRASP_BELOW_TAG: gripper releases cup at same height as table-level pick
    _STACK_STEP_OFFSETS = {
        "forward_1": (0.07, 0.015, 0.10),
        "forward_2": (0.07, 0.015, 0.11),
        "reverse_1": (0.05, -0.015, 0.10),
        "reverse_2": (0.05, -0.015, 0.10),
    }

    @staticmethod
    def _average_tag_poses(transforms: List[np.ndarray]) -> np.ndarray:
        """Return a tag world transform whose translation is the equal-weight average
        of all input transforms.  Rotation is taken from the first entry (it is
        overridden by _grasp_R() / _drawer_R() downstream anyway).
        """
        avg_t = np.mean([T[:3, 3] for T in transforms], axis=0)
        result = transforms[0].copy()
        result[:3, 3] = avg_t
        return result

    def _get_stack_grasp_poses(self, pick_tag_id, place_tag_id, place_xy_offset,
                               stack_step=None):
        """Detect AprilTags and return (pick_pose, place_pose, pick_detected, place_detected).

        Stack (forward_1 / forward_2):
          Priority 1 — cameras that see BOTH tags simultaneously (best relative-XY
                        accuracy).  If multiple such cameras exist, their positions
                        are averaged with equal weights.
          Priority 2 — if no single camera sees both tags, use cameras that see
                        each tag individually (pick cameras for pick pose, place
                        cameras for place pose), averaged per tag.
          Priority 3 — stereo triangulation fallback when no camera sees a tag.

        Unstack (reverse_1 / reverse_2):
          Pick cup is on top → detect its tag from any camera (first found).
          Place position = cached original position of pick cup (no tag detection).

        Returns:
            (pick_grasp_pose, place_grasp_pose, pick_detected, place_detected)
            pick_detected / place_detected: bool — whether the tag was actually seen.
            On failure, returns (None, None, False, False).
        """
        pick_hover, place_z, place_hover = self._STACK_STEP_OFFSETS.get(
            stack_step, (0.07, 0.01, 0.10)
        )

        T_pick_wt = None
        T_place_wt = None

        if place_xy_offset:
            # Unstack: pick cup is on top — average across all cameras that see it.
            solo_picks: List[np.ndarray] = []
            solo_pick_cams: List[str] = []
            for name, camera in self.cameras.items():
                tp = detect_single_tag_world_pose(camera, pick_tag_id)
                if tp is not None:
                    solo_picks.append(tp)
                    solo_pick_cams.append(name)
            if solo_picks:
                T_pick_wt = self._average_tag_poses(solo_picks)
                log.info(
                    f"[StackCups] Pick tag detected by {len(solo_picks)} camera(s) "
                    f"{solo_pick_cams} (reverse) — averaged pose"
                )
            else:
                T_pick_wt = detect_tag_world_pose_stereo(self.cameras, pick_tag_id)

            pick_detected = T_pick_wt is not None
            # Place uses cache — not a live tag detection.
            place_detected = None  # N/A for reverse steps

        else:
            # Stack: scan all cameras and categorise detections.
            both_picks: List[np.ndarray] = []   # T_pick  from cameras seeing BOTH tags
            both_places: List[np.ndarray] = []  # T_place from cameras seeing BOTH tags
            both_cams: List[str] = []
            solo_picks: List[np.ndarray] = []   # T_pick  from cameras seeing pick only
            solo_places: List[np.ndarray] = []  # T_place from cameras seeing place only
            solo_pick_cams: List[str] = []
            solo_place_cams: List[str] = []

            for name, camera in self.cameras.items():
                tp  = detect_single_tag_world_pose(camera, pick_tag_id)
                tpl = detect_single_tag_world_pose(camera, place_tag_id)
                if tp is not None and tpl is not None:
                    both_picks.append(tp)
                    both_places.append(tpl)
                    both_cams.append(name)
                else:
                    if tp  is not None:
                        solo_picks.append(tp)
                        solo_pick_cams.append(name)
                    if tpl is not None:
                        solo_places.append(tpl)
                        solo_place_cams.append(name)

            if both_picks:
                # Priority 1: average across all cameras seeing both tags
                T_pick_wt  = self._average_tag_poses(both_picks)
                T_place_wt = self._average_tag_poses(both_places)
                log.info(
                    f"[StackCups] Both tags visible in {len(both_picks)} camera(s) "
                    f"{both_cams} — averaged pick/place pose"
                )
            else:
                # Priority 2: use per-tag cameras independently
                if solo_picks:
                    T_pick_wt = self._average_tag_poses(solo_picks)
                    log.info(
                        f"[StackCups] Pick tag detected by {len(solo_picks)} camera(s) "
                        f"{solo_pick_cams} (solo)"
                    )
                else:
                    T_pick_wt = detect_tag_world_pose_stereo(self.cameras, pick_tag_id)

                if solo_places:
                    T_place_wt = self._average_tag_poses(solo_places)
                    log.info(
                        f"[StackCups] Place tag detected by {len(solo_places)} camera(s) "
                        f"{solo_place_cams} (solo)"
                    )
                else:
                    T_place_wt = detect_tag_world_pose_stereo(self.cameras, place_tag_id)

            pick_detected = T_pick_wt is not None
            place_detected = T_place_wt is not None

            # Cache pick cup's original position keyed by tag_id (used for unstack place).
            if T_pick_wt is not None:
                self._tag_position_cache[pick_tag_id] = T_pick_wt
                log.info(
                    f"[StackCups] cache[{pick_tag_id}] <- "
                    f"{T_pick_wt[:3,3].round(3).tolist()} (forward step={stack_step})"
                )

        if T_pick_wt is None:
            return None, None, False, False

        pick_pose = self._make_grasp_pose_from_tag_world(
            T_pick_wt, z_offset=-self._GRASP_BELOW_TAG, hover_offset=pick_hover,
            xy_offset=self._PICK_TAG_XY_OFFSET)

        if place_xy_offset:
            # Unstack: use cached position (defaults pre-populated in __init__,
            # overwritten by forward steps when tag is detected).
            cached_T = self._tag_position_cache.get(pick_tag_id)
            if cached_T is None:
                log.warning(
                    f"[StackCups] No cached position for pick tag {pick_tag_id} "
                    f"— cannot compute unstack place pose."
                )
                return None, None, False, False
            place_pose = self._make_grasp_pose_from_tag_world(
                cached_T, z_offset=place_z, hover_offset=place_hover,
                xy_offset=self._PLACE_TAG_XY_OFFSET)
            log.info(
                f"[StackCups] reverse step={stack_step} tag={pick_tag_id} | "
                f"pick_T_wt={T_pick_wt[:3,3].round(3).tolist()} | "
                f"cached_T={cached_T[:3,3].round(3).tolist()} | "
                f"place T_wg={place_pose['T_wg'][:3,3].round(3).tolist()} | "
                f"place pre_T_wg={place_pose['pre_T_wg'][:3,3].round(3).tolist()}"
            )
        else:
            # Stack: place on top of the place tag's current position.
            if T_place_wt is None:
                return None, None, pick_detected, False
            place_pose = self._make_grasp_pose_from_tag_world(
                T_place_wt, z_offset=place_z, hover_offset=place_hover,
                xy_offset=self._PLACE_TAG_XY_OFFSET)

        return pick_pose, place_pose, pick_detected, place_detected

    def _get_drawer_grasp_poses(self, tag_id: int, action: str = "open"):
        """Compute pick and place poses for the drawer task via AprilTag detection.

        Args:
            tag_id: AprilTag ID attached to the drawer (tag 7).
            action: "open"  → pick at handle (closed), place at handle - pull_dist in X
                    "close" → pick at handle (open),   place at handle + pull_dist in X

        Handle position = T_wt @ _HANDLE_TAG_OFFSET (offset in tag frame).
        Gripper tip points in +X world (drawer closing direction).
        Pre-pick hovers approach_dist in +X from handle so the robot comes
        from the +X side and slides in.
        """
        detections = []
        for camera in self.cameras.values():
            T = detect_single_tag_world_pose(camera, tag_id, tag_size=0.06)
            if T is not None:
                detections.append(T)
        if detections:
            T_wt = self._average_tag_poses(detections)
            normals = [T[:3, 2] for T in detections]
            avg_normal = np.mean(normals, axis=0) / np.linalg.norm(np.mean(normals, axis=0))
        else:
            T_wt = detect_tag_world_pose_stereo(self.cameras, tag_id, tag_size=0.06)
            avg_normal = T_wt[:3, 2] if T_wt is not None else None

        if T_wt is None or avg_normal is None:
            log.warning(f"[MotionPlanner] AprilTag {tag_id} not found for drawer task.")
            # Keep the last successfully-detected pose so downstream observers
            # (e.g. policy prompt) can still reference it as the most recent
            # known position.
            return None, None

        log.info(f"[Drawer] Tag detected by {len(detections)} camera(s) for trajectory planning")

        self.last_drawer_tag_pose = T_wt

        # Handle position in world frame
        p_handle = T_wt[:3, :3] @ self._HANDLE_TAG_OFFSET + T_wt[:3, 3]

        R = self._drawer_R()

        T_pick = np.eye(4, dtype=np.float64)
        T_pick[:3, :3] = R
        T_place = np.eye(4, dtype=np.float64)
        T_place[:3, :3] = R

        if action == "open":
            # Pick: grasp closed handle, shift 1 cm toward robot (-X), 2 cm lower Z
            T_pick[:3, 3] = p_handle + np.array([-0.01, 0.0, -0.02])
            # Place: pull -X by a random distance in [6 cm, 9 cm], 2 cm lower Z
            pull_dist = np.random.uniform(0.07, 0.11)
            T_place[:3, 3] = p_handle + np.array([-pull_dist, 0.0, -0.02])
        else:
            # Pick: open handle position (closed handle - pull_dist), -1cm X buffer, -4cm Z
            T_pick[:3, 3] = p_handle + np.array([-self._DRAWER_PULL_DIST, 0.0, -0.04])
            # Place: tag-derived closed handle X, Y/Z same as pick
            T_place[:3, 3] = np.array([p_handle[0], p_handle[1], p_handle[2] - 0.04])

        pre_T_pick = T_pick.copy()
        pre_T_pick[:3, 3] = T_pick[:3, 3] + np.array([-0.05, 0.0, 0.0])

        pick_pose  = {"T_wg": T_pick,  "pre_T_wg": pre_T_pick}
        place_pose = {"T_wg": T_place, "pre_T_wg": T_place.copy()}

        return pick_pose, place_pose

    def reset_drawer_reference(self):
        """Clear yaw reference so the next measure_drawer_rotation() call captures a new one."""
        self._drawer_ref_yaw = None
        self.last_drawer_rotation_deg = None

    def measure_drawer_rotation(self, tag_id: int) -> Optional[float]:
        """Measure how far the drawer has rotated since the last reset_drawer_reference() call.

        Uses atan2(R[1,0], R[0,0]) from the tag's world rotation matrix, which is
        robust for both vertical and horizontal tags (unlike Z-axis dot-product which
        breaks when the tag Z-axis is nearly vertical).

        First call after reset_drawer_reference(): captures the reference yaw, returns 0.0.
        Subsequent calls: return the signed delta from the reference in degrees.
        Returns None if the tag cannot be detected.
        """
        detections = []
        for camera in self.cameras.values():
            T = detect_single_tag_world_pose(camera, tag_id, tag_size=0.06)
            if T is not None:
                detections.append(T)
        if not detections:
            T_wt = detect_tag_world_pose_stereo(self.cameras, tag_id, tag_size=0.06)
            if T_wt is None:
                return None
            detections = [T_wt]

        # Average atan2 yaw across detections (handles per-camera noise independently)
        yaws = [np.degrees(np.arctan2(float(T[1, 0]), float(T[0, 0]))) for T in detections]
        # Circular mean to handle wraparound
        avg_yaw = float(np.degrees(np.arctan2(
            np.mean(np.sin(np.radians(yaws))),
            np.mean(np.cos(np.radians(yaws))),
        )))

        log.info(
            f"[Drawer] measure_rotation: {len(detections)} cam(s) | "
            f"yaws={[round(y, 1) for y in yaws]} → avg={avg_yaw:.1f} deg"
        )

        if self._drawer_ref_yaw is None:
            self._drawer_ref_yaw = avg_yaw
            self.last_drawer_rotation_deg = 0.0
            log.info(f"[Drawer] reference captured: {avg_yaw:.1f} deg → delta=0.0")
        else:
            raw_delta = avg_yaw - self._drawer_ref_yaw
            # Wrap to [-180, 180]
            delta = float(np.degrees(np.arctan2(
                np.sin(np.radians(raw_delta)), np.cos(np.radians(raw_delta))
            )))
            self.last_drawer_rotation_deg = delta
            log.info(
                f"[Drawer] ref={self._drawer_ref_yaw:.1f} cur={avg_yaw:.1f} → delta={delta:.1f} deg"
            )

        return self.last_drawer_rotation_deg

    @staticmethod
    def compute_drawer_yaw_deg(T_wt: np.ndarray) -> float:
        """Compute how many degrees the drawer is rotated left/right from front-facing.

        Assumes the tag's Z axis is the drawer face normal, and that the face
        normal points toward the robot (approximately -X world direction) when
        the drawer is perfectly front-facing.

        Returns:
            Signed angle in degrees.  Positive = rotated left (+Y side toward robot),
            negative = rotated right (-Y side toward robot), from the robot's perspective.
        """
        face_normal = T_wt[:3, 2]          # tag Z axis in world frame
        n_xy = face_normal[:2].copy()
        norm = float(np.linalg.norm(n_xy))
        if norm < 1e-6:
            return 0.0
        n_xy /= norm
        # Expected when front-facing: face normal ≈ [+1, 0] (away from robot, into drawer)
        expected = np.array([1.0, 0.0])
        cos_a = float(np.clip(np.dot(n_xy, expected), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_a)))
        # Sign: +Y component → rotated left; -Y → rotated right
        sign = float(np.sign(n_xy[1])) if abs(n_xy[1]) > 1e-6 else 1.0
        return sign * angle

    def generate_drawer_trajectory(
            self,
            pick_grasp_pose,
            place_grasp_pose,
            action: str = "open",
            pick_event: str = "CLOSE",
            place_event: str = "OPEN",
            ext_len: int = 3,
    ):
        """Trajectory for drawer open/close.

        open:  cur → pre_pick → pick → [CLOSE] → place → [OPEN] → post_place → cur
        close: cur → pre_pick → pick (contact, no grip) → push to place → post_place → cur
        """
        def _append(traj, seg):
            if seg:
                traj.extend(seg)

        T_cur = self.current_pose()
        traj_T = []
        events = []

        pre_pick = pick_grasp_pose["pre_T_wg"]
        T_pick   = pick_grasp_pose["T_wg"]
        T_place  = place_grasp_pose["T_wg"]

        # post_place: place 위치에서 X축 -3cm 후퇴
        post_place = T_place.copy()
        post_place[:3, 3] = T_place[:3, 3] + np.array([-0.05, 0.0, 0.0])

        key_poses = [pre_pick, T_pick, T_place, post_place]

        # 1) cur → pre_pick
        _append(traj_T, self._interpolate_poses_linear(T_cur, pre_pick))
        traj_T = self._prune_duplicates(traj_T)

        # 2) pre_pick → pick (slow horizontal)
        seg = self._interpolate_poses_linear(traj_T[-1], T_pick, max_step=0.004)
        _append(traj_T, self._prune_duplicates(seg))

        if action == "open":
            # 3) dwell + CLOSE
            _append(traj_T, [traj_T[-1]] * ext_len)
            events.append({"at": len(traj_T) - 1, "cmd": pick_event})

        # 4) push/pull to place (slow linear along X)
        seg = self._interpolate_poses_linear(traj_T[-1], T_place, max_step=0.005)
        _append(traj_T, self._prune_duplicates(seg))

        if action == "open":
            # 5) dwell + OPEN
            _append(traj_T, [traj_T[-1]] * ext_len)
            events.append({"at": len(traj_T) - 1, "cmd": place_event})

        # 6) post_place (-3cm X) → cur (zero position)
        seg = self._interpolate_poses_linear(traj_T[-1], post_place, max_step=0.005)
        _append(traj_T, self._prune_duplicates(seg))
        _append(traj_T, self._interpolate_poses_linear(traj_T[-1], T_cur))

        return traj_T, events, key_poses

    def _get_grasp_pose_from_apriltag(self, tag_id, z_offset, hover_offset=0.10, xy_offset=None):
        """Compute grasp pose from a single AprilTag (convenience wrapper)."""
        T_wt = None
        for camera in self.cameras.values():
            T_wt = detect_single_tag_world_pose(camera, tag_id)
            if T_wt is not None:
                break
        if T_wt is None:
            return None
        return self._make_grasp_pose_from_tag_world(T_wt, z_offset, hover_offset, xy_offset)

    def _get_best_grasp_pose_from_perception(self, perception, hover_offset=0.10, use_median_xy=False):
        if perception['responses_result_is_valid']:
            object_points = self.get_reference_object_points(perception['responses_result'])
        else:
            object_points = None

        if object_points is not None:
            z = object_points[:, 2]
            z_mean, z_std = float(z.mean()), float(z.std())
            object_points = object_points[np.abs(z - z_mean) < 1.5 * z_std]
            z = object_points[:, 2]
            log.info(
                f"[Perception Z] n={len(z)} min={z.min():.4f} p10={np.percentile(z,10):.4f} "
                f"p25={np.percentile(z,25):.4f} mean={z.mean():.4f} "
                f"p75={np.percentile(z,75):.4f} max={z.max():.4f}"
            )
            obb = fit_obb_pca(object_points)
            grasp_poses = generate_grasps_from_obb(obb, rotation=self.current_pose()[:3, :3])
            best_grasp_pose = self.sample_best_grasp(grasp_poses)
            forced_R = self._adapt_R_to_long_axis(obb)
            best_grasp_pose['T_wg'][:3, :3] = forced_R
            if use_median_xy:
                best_grasp_pose['T_wg'][0, 3], best_grasp_pose['T_wg'][1, 3] = \
                    self._bowl_center_xy(object_points)
                # Pan interior ≈ TABLE_Z; use fixed depth instead of percentile
                # (1.5σ z-filter discards low-Z interior points, biasing percentile to rim)
                best_grasp_pose['T_wg'][2, 3] = self._TABLE_Z + 0.03
            else:
                best_grasp_pose['T_wg'][2, 3] = float(np.percentile(object_points[:, 2], 30))
            best_grasp_pose['pre_T_wg'][:3, :3] = forced_R
            best_grasp_pose['pre_T_wg'][:3, 3] = (
                best_grasp_pose['T_wg'][:3, 3] + np.array([0.0, 0.0, hover_offset])
            )

        else:
            best_grasp_pose = None

        return best_grasp_pose

    @classmethod
    def _adapt_R_to_long_axis(cls, obb) -> np.ndarray:
        """Rotate gripper orientation around world-Z so the closing axis is
        PERPENDICULAR to the object's longest horizontal axis (jaws grip across
        the object's length).

        Uses _grasp_R() (Y-tilt included) as base so IK behaviour matches
        other tasks (stack_cups, drawer).  Only applied when the object is
        elongated (long/mid axis ratio >= 1.5) and the long axis has a
        meaningful horizontal component.
        For roughly circular objects (pan, plate) the base rotation is returned unchanged.
        """
        base_R = cls._grasp_R()
        if obb is None:
            log.info("[MotionPlanner] Long-axis grasp SKIPPED: obb is None")
            return base_R

        extents = obb["extents"]
        long_axis = obb["R"][:, 0]
        lx, ly, lz = float(long_axis[0]), float(long_axis[1]), float(long_axis[2])
        aspect = extents[0] / extents[1] if extents[1] > 1e-6 else float("inf")
        log.info(
            f"[MotionPlanner] OBB extents=({extents[0]:.3f}, {extents[1]:.3f}, "
            f"{extents[2]:.3f}) aspect={aspect:.2f} long_axis=({lx:+.2f}, "
            f"{ly:+.2f}, {lz:+.2f}) hypot_xy={np.hypot(lx, ly):.2f}"
        )

        if extents[1] < 1e-6 or aspect < 1.5:
            log.info(
                f"[MotionPlanner] Long-axis grasp SKIPPED: aspect {aspect:.2f} "
                f"< 1.5 (not elongated enough)"
            )
            return base_R

        if np.hypot(lx, ly) < 0.15:
            log.info(
                f"[MotionPlanner] Long-axis grasp SKIPPED: long axis nearly "
                f"vertical (hypot_xy={np.hypot(lx, ly):.2f} < 0.15)"
            )
            return base_R

        # Normalize to [-pi/2, pi/2] (long axis is bipolar)
        theta = np.arctan2(ly, lx)
        if theta > np.pi / 2:
            theta -= np.pi
        elif theta < -np.pi / 2:
            theta += np.pi

        # Rotate base_R around world-Z by delta = theta. Because base closing
        # axis (gripper Y) is world +Y, the resulting Y column = (-sin θ, cos θ, 0)
        # which is perpendicular to the long axis (cos θ, sin θ, 0). Jaws then
        # close ACROSS the object's length. delta in [-π/2, π/2] guarantees
        # shortest wrist path (≤90°).
        delta = theta

        c, s = np.cos(delta), np.sin(delta)
        Rz = np.array([[ c, -s, 0.],
                       [ s,  c, 0.],
                       [0., 0., 1.]], dtype=np.float64)
        adapted_R = Rz @ base_R

        log.info(
            f"[MotionPlanner] Long-axis grasp: theta={np.degrees(theta):.1f}° "
            f"delta_yaw={np.degrees(delta):.1f}° "
            f"aspect={extents[0]/extents[1]:.2f}"
        )
        return adapted_R

    def _bowl_center_xy(self, object_points: np.ndarray):
        """Return (cx, cy) of the pan bowl by finding the far-X edge and subtracting the bowl radius.

        Handle is toward the robot (-X), bowl is away from the robot (+X).
        Bowl center X ≈ max(X) - bowl_radius.
        Bowl center Y ≈ median(Y) (bowl is symmetric along Y).
        """
        _BOWL_RADIUS = 0.085  # 21 cm diameter / 2, shifted +3 cm, pulled -1 cm toward robot

        xy = object_points[:, :2]
        x_far = float(np.percentile(xy[:, 0], 95))  # far bowl rim (robust max)
        cx = x_far - _BOWL_RADIUS
        cy = float(np.median(xy[:, 1]))
        log.info(
            f"[BowlCenter] x_far={x_far:.4f} bowl_r={_BOWL_RADIUS:.3f} "
            f"cx={cx:.4f} cy={cy:.4f}"
        )
        return cx, cy

