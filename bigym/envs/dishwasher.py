"""Dishwasher interaction tasks."""
from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH, HandSide
from bigym.envs.props.dishwasher import Dishwasher


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
        return np.exp(-np.linalg.norm(self.robot.get_hand_pos(side) - obj.body.get_position()))

    def _position_reward(self, obj, val):
        return np.exp(-np.linalg.norm(obj.body.get_position() - val))


class DishwasherOpen(_DishwasherEnv):
    """Open the dishwasher door and pull out all trays."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=0, bottom_tray=0, middle_tray=0)

    def _reward(self):
        dishwasher_joints = self.dishwasher.get_state()
        door_pulled, tray_bottom_pulled, tray_middle_pulled = np.isclose(dishwasher_joints, 1, atol=self._TOLERANCE)
        grasp_door_reward, grasp_tray_bottom_reward, grasp_tray_middle_reward = (
            self._grasp_reward(HandSide.LEFT, self.dishwasher.door),
            self._grasp_reward(HandSide.LEFT, self.dishwasher.tray_bottom),
            self._grasp_reward(HandSide.LEFT, self.dishwasher.tray_middle),
        )

        if tray_middle_pulled:
            reward = 6.0
        elif tray_bottom_pulled:
            tray_middle_grasped = self._check_grasped(HandSide.LEFT, self.dishwasher.tray_middle)
            if not tray_middle_grasped:
                reward = 4.0 + grasp_tray_middle_reward
            else:
                reward = 5.0 + self._position_reward(self.dishwasher.tray_middle, 1.0)
        elif door_pulled:
            tray_bottom_grasped = self._check_grasped(HandSide.LEFT, self.dishwasher.tray_bottom)
            if not tray_bottom_grasped:
                reward = 2.0 + grasp_tray_bottom_reward
            else:
                reward = 3.0 + self._position_reward(self.dishwasher.tray_bottom, 1.0)
        else:
            door_grasped = self._check_grasped(HandSide.LEFT, self.dishwasher.door)
            if not door_grasped:
                reward = grasp_door_reward
            else:
                reward = 1.0 + self._position_reward(self.dishwasher.door, 1.0)

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
