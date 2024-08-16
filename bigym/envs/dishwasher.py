"""Dishwasher interaction tasks."""
from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH, HandSide
from bigym.envs.props.dishwasher import Dishwasher
from bigym.utils.physics_utils import get_joint_position


SUBTASK_MAGNITUDE = 1.0

class _DishwasherEnv(BiGymEnv, ABC):
    """Base dishwasher environment."""

    RESET_ROBOT_POS = np.array([0, -0.8, 0])

    _PRESET_PATH = PRESETS_PATH / "dishwasher.yaml"
    _TOLERANCE = 0.05

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]

    def _check_grasped(self, side, obj):
        return self.robot.is_gripper_holding_object(obj, side)

    def _grasp_reward(self, side, obj):
        gripper_pos = self.robot._grippers[side].pinch_position
        # obj_joint = get_joint_position(obj.joint, True)
        obj_pos = obj.body.get_position()
        return np.exp(-np.linalg.norm(gripper_pos - obj_pos))

    def _joint_reward(self, obj, val=1.0):
        obj_joint = get_joint_position(obj.joint, True)
        return np.exp(-np.linalg.norm(obj_joint - val))


class DishwasherOpen(_DishwasherEnv):
    """Open the dishwasher door and pull out all trays."""

    def _initialize_env(self):
        super()._initialize_env()
        self._current_subtask = 0

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=0, bottom_tray=0, middle_tray=0)
        self._current_subtask = 0

    def _reward(self):
        dishwasher_joints = self.dishwasher.get_state()
        door_pulled, tray_bottom_pulled, tray_middle_pulled = np.isclose(dishwasher_joints, 1, atol=self._TOLERANCE)
        grasp_door_reward, grasp_tray_bottom_reward, grasp_tray_middle_reward = (
            self._grasp_reward(HandSide.LEFT, self.dishwasher.door),
            self._grasp_reward(HandSide.LEFT, self.dishwasher.tray_bottom),
            self._grasp_reward(HandSide.LEFT, self.dishwasher.tray_middle),
        )

        if self._success():
            if self._current_subtask == 5:
                self._current_subtask += 1
            _reward = 0.0
        elif tray_bottom_pulled and door_pulled:
            if self._current_subtask == 3:
                self._current_subtask += 1
            tray_middle_grasped = self._check_grasped(HandSide.LEFT, self.dishwasher.tray_middle)
            if not tray_middle_grasped and self._current_subtask == 4:
                _reward = grasp_tray_middle_reward
            else:
                if self._current_subtask == 4:
                    self._current_subtask += 1
                _reward = self._joint_reward(self.dishwasher.tray_middle)
        elif door_pulled:
            if self._current_subtask == 1:
                self._current_subtask += 1
            tray_bottom_grasped = self._check_grasped(HandSide.LEFT, self.dishwasher.tray_bottom)
            if not tray_bottom_grasped and self._current_subtask == 2:
                _reward = grasp_tray_bottom_reward
            else:
                if self._current_subtask == 2:
                    self._current_subtask += 1
                _reward = self._joint_reward(self.dishwasher.tray_bottom)
        else:
            door_grasped = self._check_grasped(HandSide.LEFT, self.dishwasher.door)
            if not door_grasped and self._current_subtask == 0:
                _reward = grasp_door_reward
            else:
                if self._current_subtask == 0:
                    self._current_subtask += 1
                _reward = self._joint_reward(self.dishwasher.door)

        reward = SUBTASK_MAGNITUDE * self._current_subtask + _reward
        return reward


class DishwasherClose(_DishwasherEnv):
    """Push back all trays and close the door of the dishwasher."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=1)


class DishwasherCloseTrays(DishwasherClose):
    """Push the dishwasher’s trays back with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 0, atol=self._TOLERANCE)


class DishwasherOpenTrays(DishwasherClose):
    """Pull out the dishwasher’s trays with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=0, middle_tray=0)
