import time
import numpy as np
import cv2
import pyrealsense2 as rs
from utils.camera_utils import estimate_T_ct_from_apriltag
from utils.transform_utils import SE3_mean

class RealSenseCamera:
    def __init__(
        self,
        serial_number,
        width=640,
        height=480,
        fps=30,
        use_depth=False,
        calibration_tries=10
    ):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.use_depth = use_depth
        self._calibration_tries = calibration_tries

        self.pipeline = None
        self.config = None
        self.profile = None
        self.intrinsics = None
        self.K = None
        self.T_wc = None

        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial_number)

        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        self.profile = self.pipeline.start(self.config)

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = (
            color_stream.as_video_stream_profile().get_intrinsics()
        )

        self.K = np.array(
            [
                [self.intrinsics.fx, 0, self.intrinsics.ppx],
                [0, self.intrinsics.fy, self.intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.dist = np.array(self.intrinsics.coeffs)

        # warm-up
        for _ in range(10):
            self.pipeline.wait_for_frames()

        self._initialized = True

    def get_rgb(self):
        self._lazy_init()

        color_frame = np.asanyarray(self.pipeline.wait_for_frames().get_color_frame().get_data())
        color_image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        return color_image 
    
    def get_intrinsic_matrix(self):
        self._lazy_init()
        return self.K

    def get_T_ct(self):
        T_ct_list = []
        for _ in range(self._calibration_tries):
            rgb = self.get_rgb()
            rgb = cv2.undistort(rgb, self.K, self.dist)
            T_ct, _ = estimate_T_ct_from_apriltag(
                rgb=rgb, K=self.K
            )
            T_ct_list.append(T_ct)
            time.sleep(0.1)
        T_ct_mean = SE3_mean(T_ct_list)
        return T_ct_mean


class WristRealSenseCamera(RealSenseCamera):
    def __init__(
        self,
        serial_number,
        mounted_on=None,
        width=640,
        height=480,
        fps=30,
        use_depth=False,
    ):
        """
        T_ec: (4,4) end-effector -> camera (hand-eye calibration result)
        fk_fn: function(robot_state) -> T_we
        """
        super().__init__(
            serial_number,
            width=width,
            height=height,
            fps=fps,
            use_depth=use_depth,
        )

        self.T_ec = np.array([
            [0.0, 0.947, 0.320, -0.075],
            [-1.0, 0.0, 0.0, -0.014],
            [0.0, -0.320, 0.947, 0.038],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.robot = mounted_on

    def get_extrinsic_matrix(self) -> np.ndarray:
        T_we = self.robot.get_end_pose(gripper_depth=False)
        T_wc = T_we @ self.T_ec
        return T_wc

    def get_T_wt(self):
        T_wc = self.get_extrinsic_matrix()
        T_ct = self.get_T_ct()

        return T_wc @ T_ct

class FixedRealSenseCamera(RealSenseCamera):
    def __init__(
        self,
        serial_number,
        width=640,
        height=480,
        fps=30,
        use_depth=False,
    ):
        super().__init__(
            serial_number,
            width=width,
            height=height,
            fps=fps,
            use_depth=use_depth,
        )

    def get_extrinsic_matrix(self) -> np.ndarray:
        return self.T_wc

    def set_extrinsic_matrix(self, T):
        setattr(self, 'T_wc', T)