import time
import numpy as np

from piper_sdk import *

from utils.transform_utils import pose6d_to_transform_matrix

class Piper:
    def __init__(self, config):
        self.config = config
        self.piper = C_PiperInterface_V2()
        self.fk = self._fk(C_PiperForwardKinematics())
        self._init_pose = [
            50000, 0, 350000, 0, 120000, 0, 0
        ]
        self.T_eg = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.136],
            [0, 0, 0, 1],
        ])
        # When input_frame == "tip", step() interprets the first six entries of
        # the incoming pose as the gripper-tip pose (in world frame) and
        # converts back to EE before sending to the arm. Default "ee" preserves
        # existing behavior.
        self._input_frame = getattr(config, "input_frame", "ee")
        self._gripper_effort = 1000  # default effort (pick_place); stack_cups uses 300

    def _lazy_init(self, set_to_zero=True):
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        
        if set_to_zero:
            self._control_end_pose(self._init_pose)
            self._open_gripper()

    def reset(self):
        joints = [0,0,0,0,0,0,0]
        self._control_joint(
            joints
        )

    def step(self, pose_6d, gripper):
        pose_6d = list(pose_6d)
        if self._input_frame == "tip":
            pose_6d = self._tip_to_ee(pose_6d)
        end_pose = [int(x) for x in pose_6d] + [int(gripper)]
        self._control_end_pose(
            end_pose
        )
        time.sleep(0.2)

    def _tip_to_ee(self, pose_6d_tip):
        """Convert a tip-frame pose (raw Piper units) to EE-frame pose.

        Orientation is preserved (T_eg is a pure +Z translation in the EE
        frame); only translation shifts by -R_tip @ [0, 0, 0.136].
        """
        x, y, z, rx, ry, rz = [float(v) for v in pose_6d_tip[:6]]
        pose_si = [x * 1e-6, y * 1e-6, z * 1e-6,
                   rx * 1e-3, ry * 1e-3, rz * 1e-3]
        T_wg = pose6d_to_transform_matrix(pose_si, degrees=True)
        offset_world = T_wg[:3, :3] @ np.array([0.0, 0.0, self.T_eg[2, 3]])
        p_ee = T_wg[:3, 3] - offset_world
        return [p_ee[0] * 1e6, p_ee[1] * 1e6, p_ee[2] * 1e6, rx, ry, rz]

    def go_to_init_pose(self):
        self._control_end_pose(self._init_pose)
        self._open_gripper()

    def get_joints(self):
        joints_deg = self.piper.GetArmJointMsgs().joint_state
        joints_rad = [
            np.deg2rad(joints_deg.joint_1 / 1000),
            np.deg2rad(joints_deg.joint_2 / 1000),
            np.deg2rad(joints_deg.joint_3 / 1000),
            np.deg2rad(joints_deg.joint_4 / 1000),
            np.deg2rad(joints_deg.joint_5 / 1000),
            np.deg2rad(joints_deg.joint_6 / 1000),
        ]
        return joints_rad

    def get_end_pose(self, gripper_depth=True):
        raw_end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        end_pose = [
            raw_end_pose.X_axis / 1000000,
            raw_end_pose.Y_axis / 1000000,
            raw_end_pose.Z_axis / 1000000,
            np.deg2rad(raw_end_pose.RX_axis / 1000),
            np.deg2rad(raw_end_pose.RY_axis / 1000),
            np.deg2rad(raw_end_pose.RZ_axis / 1000)
        ]
        T_we = pose6d_to_transform_matrix(end_pose)

        if gripper_depth:
            T = T_we @ self.T_eg
        else:
            T = T_we
        return T

    def get_gripper(self):
        gripper = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        gripper = np.clip(gripper,min=0,max=70000)
        return gripper

    def _fk(self, fk_interface):
        def fk_wrapper(joints):
            fk_list = fk_interface.CalFK(joints)
            pose_6d = np.asarray(fk_list[-1])

            pose_6d[:3] /= 1000

            T = pose6d_to_transform_matrix(pose_6d, degrees=True)
            return T
        return fk_wrapper

    def _close_gripper(self):
        self.piper.GripperCtrl(42000, 300, 0x01, 0)

    def _open_gripper(self):
        self.piper.GripperCtrl(70000, 300, 0x01, 0)

    def _control_joint(self, joints):
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(joints[0], joints[1], joints[2],joints[3],joints[4],joints[5])
        self.piper.GripperCtrl(joints[6], 300, 0x01, 0)
        time.sleep(0.005)

    def _control_end_pose(self, end_pose):
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        self.piper.EndPoseCtrl(end_pose[0], end_pose[1], end_pose[2], end_pose[3], end_pose[4], end_pose[5])
        self.piper.GripperCtrl(end_pose[6], self._gripper_effort, 0x01, 0)
        time.sleep(0.01)
