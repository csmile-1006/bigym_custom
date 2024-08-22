"""Set of plate moving tasks."""
from abc import ABC

import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH, HandSide
from bigym.envs.props.holders import DishDrainer
from bigym.envs.props.kitchenware import Plate
from bigym.envs.props.tables import Table
from bigym.utils.physics_utils import distance

from dm_control.utils import rewards


RACK_BOUNDS = np.array([0.05, 0.05, 0])
RACK_POSITION_LEFT = np.array([0.7, 0.3, 0.95])
RACK_POSITION_RIGHT = np.array([0.7, -0.3, 0.95])

PLATE_OFFSET_POS = np.array([0, 0.01, 0])
PLATE_OFFSET_ROT = Quaternion(axis=[1, 0, 0], degrees=-5).elements


class _MovePlatesEnv(BiGymEnv, ABC):
    """Base plates environment."""

    _PRESET_PATH = PRESETS_PATH / "move_plates.yaml"

    _SUCCESSFUL_DIST = 0.05
    _SUCCESS_ROT = np.deg2rad(20)

    _PLATES_COUNT = 1

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]
        self.rack_start = self._preset.get_props(DishDrainer)[0]
        self.rack_target = self._preset.get_props(DishDrainer)[1]
        self.plates = [Plate(self._mojo) for _ in range(self._PLATES_COUNT)]

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for plate in self.plates:
            if np.all(
                [
                    distance(plate.body, site) > self._SUCCESSFUL_DIST
                    for site in self.rack_target.sites
                ]
            ):
                return False
            plate_up = Quaternion(plate.body.get_quaternion()).rotate(up)
            angle = np.arccos(np.clip(np.dot(plate_up, right), -1.0, 1.0))
            if angle > self._SUCCESS_ROT:
                return False
            if not plate.is_colliding(self.rack_target):
                return False
            if plate.is_colliding(self.table):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(plate, side):
                    return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for plate in self.plates:
            if plate.is_colliding(self.floor):
                return True
        return False

    def _on_reset(self):
        offset = np.random.uniform(-RACK_BOUNDS, RACK_BOUNDS)
        self.rack_start.body.set_position(RACK_POSITION_LEFT + offset)
        offset = np.random.uniform(-RACK_BOUNDS, RACK_BOUNDS)
        self.rack_target.body.set_position(RACK_POSITION_RIGHT + offset)

        sites = np.array(self.rack_start.sites)
        sites = np.random.choice(sites, size=len(self.plates), replace=False)

        for site, plate in zip(sites, self.plates):
            plate.body.set_position(site.get_position() + PLATE_OFFSET_POS, True)
            quat = Quaternion(site.get_quaternion())
            quat *= PLATE_OFFSET_ROT
            plate.body.set_quaternion(quat.elements, True)

        self._prev_qvel = np.zeros_like(self._robot.qvel)
        self._prev_qpos = np.zeros_like(self._robot.qpos)
        self._original_zpos = [plate.body.get_position()[2] for plate in self.plates]

    def _reward(self):
        reward = 0.0
        reward_info = {}
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])

        for idx, plate in enumerate(self.plates):
            plate_up = Quaternion(plate.body.get_quaternion()).rotate(up)
            # Subtask 5: Grasp the Target Dish
            # Reward for grasping the plate
            grasp_reward = (
                1.0
                if self.robot.is_gripper_holding_object(plate, HandSide.LEFT)
                else 0.0
            )
            reward += grasp_reward
            reward_info["grasp_reward"] = grasp_reward

            if not grasp_reward:
                source_rack_dist = np.linalg.norm(
                    self.robot.pelvis.get_position()
                    - self.rack_start.body.get_position()
                )
                source_rack_reward = 0.5 * np.exp(-source_rack_dist)
                reward += source_rack_reward
                reward_info["source_rack_reward"] = source_rack_reward

                # Subtask 4 & 9: Position Hand for Grasping/Placement
                # Reward for gripper being close to the plate
                gripper_pos = self.robot.grippers[HandSide.LEFT].pinch_position
                gripper_plate_dist = np.linalg.norm(
                    gripper_pos - plate.body.get_position()
                )
                gripper_plate_reward = np.exp(-gripper_plate_dist)
                reward += gripper_plate_reward
                reward_info["gripper_plate_reward"] = gripper_plate_reward

            else:
                plate_z_pos = plate.body.get_position()[2]
                lift_reward = np.linalg.norm(self._original_zpos[idx] - plate_z_pos)
                reward += lift_reward
                reward_info["lift_reward"] = lift_reward

                target_rack_dist = np.linalg.norm(
                    self.robot.pelvis.get_position()
                    - self.rack_target.body.get_position()
                )
                approach_rack_reward = 0.5 * np.exp(-target_rack_dist)
                reward += approach_rack_reward
                reward_info["approach_rack_reward"] = approach_rack_reward

                # Subtask 10: Place the Target Dish
                # Reward for placing the plate on the target rack
                place_reward = 0.0
                for site in self.rack_target.sites:
                    site_dist = np.linalg.norm(
                        plate.body.get_position() - site.get_position()
                    )
                    place_reward = max(place_reward, np.exp(-site_dist))
                angle = np.arccos(np.clip(np.dot(plate_up, right), -1.0, 1.0))
                angle_reward = -1 + rewards.tolerance(
                    angle, bounds=(0, self._SUCCESS_ROT), margin=0.3
                )
                reward += place_reward + angle_reward
                reward_info["place_reward"] = place_reward
                reward_info["angle_reward"] = angle_reward

        # Humanoid Movement Criteria
        # Penalize for large changes in velocity (smoothness and efficiency)
        velocity_change = np.linalg.norm(self._robot.qvel - self._prev_qvel)
        smoothness_reward = -0.004 * velocity_change
        reward += smoothness_reward
        reward_info["smoothness_reward"] = smoothness_reward
        self._prev_qvel = self._robot.qvel.copy()

        # Penalize termination by failure or truncation
        fail_penalty = -250.0 if self.fail else 0.0
        not_healthy_penalty = -250.0 if self.truncate else 0.0
        reward += fail_penalty + not_healthy_penalty
        reward_info["fail_penalty"] = fail_penalty
        reward_info["not_healthy_penalty"] = not_healthy_penalty

        success_reward = 10.0 if self.success else 0.0
        reward += success_reward * self._PLATES_COUNT
        reward_info["success_reward"] = success_reward
        # Add other penalties or rewards for humanoid movement criteria as needed
        return reward

    # def _reward(self):
    #     SUBTASK_SOLVE_SCALE = 3.0
    #     if self._success():
    #         return 15.0 * self._PLATES_COUNT + self._regularization_reward()
    #     else:
    #         up = np.array([0, 0, 1])
    #         right = np.array([0, -1, 0])

    #         reward = 0.0
    #         for plate in self.plates:
    #             plate_grasped = self.robot.is_gripper_holding_object(
    #                 plate, HandSide.LEFT
    #             )
    #             plate_up = Quaternion(plate.body.get_quaternion()).rotate(up)
    #             angle = np.arccos(np.clip(np.dot(plate_up, right), -1.0, 1.0))
    #             if not plate_grasped:
    #                 plate_grasp_reward = 0.0
    #                 # for side in self.robot.grippers:
    #                 gripper_pos = self.robot.grippers[HandSide.LEFT].pinch_position
    #                 obj_pos = plate.body.get_position()
    #                 plate_grasp_reward += SUBTASK_SOLVE_SCALE * np.exp(
    #                     -np.linalg.norm(gripper_pos - obj_pos)
    #                 )
    #                 # plate_grasp_reward /= len(self.robot.grippers)
    #                 reward += plate_grasp_reward
    #             elif plate.is_colliding(self.table):
    #                 reward += -100.0
    #             else:
    #                 plate_grasp_reward = SUBTASK_SOLVE_SCALE
    #                 plate_place_reward = np.max(
    #                     [
    #                         SUBTASK_SOLVE_SCALE * np.exp(-distance(plate.body, site))
    #                         for site in self.rack_target.sites
    #                     ]
    #                 )
    #                 plate_angle_reward = SUBTASK_SOLVE_SCALE * rewards.tolerance(
    #                     angle, bounds=(0, self._SUCCESS_ROT), margin=0.3
    #                 )
    #                 reward += (
    #                     plate_grasp_reward + plate_place_reward + plate_angle_reward
    #                 )

    #         return reward + self._regularization_reward()


class MovePlate(_MovePlatesEnv):
    """Move one plate from one rack to another."""

    def _get_task_privileged_obs_space(self):
        return {
            "rack_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "plate_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
        }

    def _get_task_privileged_obs(self):
        return {
            "rack_pose": np.array(self.rack_target.get_pose(), np.float32).flatten(),
            "plate_pose": np.array(self.plates[0].get_pose(), np.float32).flatten(),
        }

class MoveTwoPlates(_MovePlatesEnv):
    """Move two plates from one rack to another."""

    _PLATES_COUNT = 2

    def _get_task_privileged_obs_space(self):
        return {}

    def _get_task_privileged_obs(self):
        return {}
