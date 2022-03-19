import gym
import mujoco_py
import numpy as np
import torch

# from a2c_ppo_acktr.model import PrimitivePolicy
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer
from gym import spaces
from gym.spaces.box import Box
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from robosuite.wrappers.gym_wrapper import GymWrapper

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers.dm_backend_wrappers import DMControlBackendRobosuiteEnv
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.torch.model_based.dreamer.conv_networks import CNNMLP


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        gym.Wrapper.__init__(self, env)
        self._duration = duration
        self._elapsed_steps = 0
        self._max_episode_steps = duration
        self._step = None

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        self._step += 1
        self._elapsed_steps += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        self._elapsed_steps = 0
        return self.env.reset()


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount):
        gym.Wrapper.__init__(self, env)
        self._amount = amount

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self.env.step(
                action,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class NormalizeActions(gym.Wrapper):
    def __init__(self, env, unused=None):
        gym.Wrapper.__init__(self, env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        obs, reward, done, info = self.env.step(
            original,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class ImageUnFlattenWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = Box(
            0, 255, (3, self.env.imwidth, self.env.imheight), dtype=np.uint8
        )

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")

    def reset(self):
        obs = self.env.reset()
        return obs.reshape(-1, self.env.imwidth, self.env.imheight)

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        obs, reward, done, info = self.env.step(
            action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        return (
            obs.reshape(-1, self.env.imwidth, self.env.imheight),
            reward,
            done,
            info,
        )


class MetaworldWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        reward_type="dense",
        imwidth=84,
        imheight=84,
        reward_scale=1.0,
        use_image_obs=False,
    ):
        super().__init__(env)
        self.reward_type = reward_type
        self.env.imwdith = imwidth
        self.env.imheight = imheight
        self.imwidth = imwidth
        self.imheight = imheight
        self.observation_space = Box(
            0, 255, (3 * self.imwidth * self.imheight,), dtype=np.uint8
        )
        self.image_shape = (3, self.imwidth, self.imheight)
        self.reward_scale = reward_scale
        self.use_image_obs = use_image_obs

    def _get_image(self):
        if hasattr(self.env, "_use_dm_backend"):
            img = self.env.render(
                mode="rgb_array", imwidth=self.imwidth, imheight=self.imheight
            )
        else:
            img = self.env.sim.render(
                imwidth=self.imwidth,
                imheight=self.imheight,
            )

        img = img.transpose(2, 0, 1).flatten()
        return img

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")

    def reset(self):
        obs = super().reset()
        if self.use_image_obs:
            if self.env.set_primitive_goals:
                return np.concatenate((self._get_image(), self.env.primitive_goal))
            else:
                return self._get_image()
        else:
            return obs

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        obs, reward, done, info = self.env.step(
            action,
        )
        self.unset_render_every_step()
        if self.use_image_obs:
            if self.env.set_primitive_goals:
                obs = np.concatenate((self._get_image(), self.env.primitive_goal))
            else:
                obs = self._get_image()
        reward = self.reward_scale * reward
        new_info = {}
        for key, value in info.items():
            if value is not None:
                new_info[key] = value
        if self.reward_type == "sparse":
            reward = info["success"]
        return obs, reward, done, new_info


class SawyerXYZEnvMetaworldPrimitives(SawyerXYZEnv):
    def reset_camera(self, camera_settings):
        if camera_settings is None:
            camera_settings = {}
        self.renderer = DMRenderer(self.sim, camera_settings=camera_settings)

    def reset_action_space(
        self,
        control_mode="end_effector",
        action_scale=1 / 100,
        camera_settings=None,
        collect_primitives_info=False,
        render_intermediate_obs_to_info=False,
        num_low_level_actions_per_primitive=10,
        goto_pose_iterations=300,
        open_gripper_iterations=200,
        close_gripper_iterations=300,
        pos_ctrl_action_scale=0.05,
        primitive_model_kwargs=None,
        primitive_model_path=None,
        low_level_reward_type="none",
        set_primitive_goals=False,
        primitive_to_model=None,
        relabel_high_level_actions=False,
        split_top_grasp=False,
    ):
        self.split_top_grasp = split_top_grasp
        self.relabel_high_level_actions = relabel_high_level_actions
        self.primitive_to_model = primitive_to_model
        self.set_primitive_goals = set_primitive_goals
        self.goto_pose_iterations = goto_pose_iterations
        self.open_gripper_iterations = open_gripper_iterations
        self.close_gripper_iterations = close_gripper_iterations
        self.pos_ctrl_action_scale = pos_ctrl_action_scale
        self.reset_camera(camera_settings)
        self.action_scale = action_scale
        self.render_intermediate_obs_to_info = render_intermediate_obs_to_info
        self.num_low_level_actions_per_primitive = num_low_level_actions_per_primitive

        # primitives
        # self.primitive_idx_to_name = {
        #     0: "move_delta_ee",
        #     1: "top_x_y_grasp",
        #     2: "lift",
        #     3: "drop",
        #     4: "move_left",
        #     5: "move_right",
        #     6: "move_forward",
        #     7: "move_backward",
        #     8: "open_gripper",
        #     9: "close_gripper",
        # }
        # self.primitive_idx_to_num_low_level_steps = {
        #     0: goto_pose_iterations,
        #     1: goto_pose_iterations * 3
        #     + open_gripper_iterations
        #     + close_gripper_iterations,
        #     2: goto_pose_iterations,
        #     3: goto_pose_iterations,
        #     4: goto_pose_iterations,
        #     5: goto_pose_iterations,
        #     6: goto_pose_iterations,
        #     7: goto_pose_iterations,
        #     8: open_gripper_iterations,
        #     9: close_gripper_iterations,
        # }
        # self.primitive_name_to_func = dict(
        #     move_delta_ee=self.move_delta_ee,
        #     top_x_y_grasp=self.top_x_y_grasp,
        #     lift=self.lift,
        #     drop=self.drop,
        #     move_left=self.move_left,
        #     move_right=self.move_right,
        #     move_forward=self.move_forward,
        #     move_backward=self.move_backward,
        #     open_gripper=self.open_gripper,
        #     close_gripper=self.close_gripper,
        # )
        # self.primitive_name_to_action_idx = dict(
        #     move_delta_ee=[0, 1, 2],
        #     top_x_y_grasp=[3, 4, 5, 6],
        #     lift=7,
        #     drop=8,
        #     move_left=9,
        #     move_right=10,
        #     move_forward=11,
        #     move_backward=12,
        #     open_gripper=13,
        #     close_gripper=14,
        # )
        # self.max_arg_len = 15
        # self.primitive_idx_to_name = {
        #     0: "move_delta_ee",
        #     1: "top_x_y_grasp",
        #     2: "lift",
        #     3: "drop",
        #     4: "move_left",
        #     5: "move_right",
        #     6: "move_forward",
        #     7: "move_backward",
        #     8: "open_gripper",
        #     9: "close_gripper",
        # }
        self.primitive_idx_to_name = {
            0: "move_delta_ee",
            # 1: "top_x_y_grasp",
            1: "move_along_x",
            2: "move_along_y",
            3: "move_along_z",
            4: "move_gripper",
        }
        self.primitive_idx_to_num_low_level_steps = {
            0: goto_pose_iterations,
            # 1: goto_pose_iterations * 4,
            1: goto_pose_iterations,
            2: goto_pose_iterations,
            3: goto_pose_iterations,
            4: goto_pose_iterations,
        }
        self.primitive_name_to_func = dict(
            move_delta_ee=self.move_delta_ee,
            # top_x_y_grasp=self.top_x_y_grasp,
            move_along_x=self.move_along_x,
            move_along_y=self.move_along_y,
            move_along_z=self.move_along_z,
            move_gripper=self.move_gripper,
        )
        self.primitive_name_to_action_idx = dict(
            move_delta_ee=[0, 1, 2],
            # top_x_y_grasp=[3, 4, 5, 6],
            move_along_x=3,
            move_along_y=4,
            move_along_z=5,
            move_gripper=6,
        )
        self.max_arg_len = 7
        self.num_primitives = len(self.primitive_name_to_func)
        self.control_mode = control_mode
        if self.control_mode == "primitives":
            action_space_low = -1 * np.ones(self.max_arg_len)
            action_space_high = np.ones(self.max_arg_len)
            self.reset_mocap2body_xpos(self.sim)
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, action_space_low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    action_space_high,
                )
            )
            self.action_space = Box(act_lower, act_upper, dtype=np.float32)

        self.unset_render_every_step()
        self.collect_primitives_info = collect_primitives_info
        if primitive_model_kwargs is not None:
            primitive_model_kwargs["state_encoder_kwargs"]["input_size"] = (
                self.action_space.low.shape[0] + 5
            )
            # self.primitive_model = PrimitivePolicy(
            #     (self.observation_space.low.size + self.action_space.low.size,),
            #     (5,),
            #     base_kwargs=primitive_model_kwargs,
            # )
            self.primitive_model = CNNMLP(**primitive_model_kwargs)
            self.primitive_model_path = primitive_model_path
        self.use_primitive_model = False
        self.low_level_reward_type = low_level_reward_type
        self.set_render_every_step()
        if self.set_primitive_goals:
            self.start_ee_pos = self.get_endeff_pos()
            if (
                self.primitive_to_model == "close_gripper"
                or self.primitive_to_model == "open_gripper"
            ):
                self.primitive_goal_space = Box(
                    np.concatenate((self.start_ee_pos, np.array([0]))),
                    np.concatenate((self.start_ee_pos, np.array([1]))),
                )
            elif self.primitive_to_model == "top_x_y_grasp":
                self.primitive_goal_space = Box(
                    np.concatenate((self.mocap_low, np.array([0]))),
                    np.concatenate((self.mocap_high, np.array([-1]))),
                )
            elif self.primitive_to_model == "move_left":
                self.primitive_goal_space = Box(
                    np.concatenate(
                        (
                            self.mocap_low[0:1],
                            self.start_ee_pos[1:3],
                            np.array([1, 0, 1, 0, 0]),
                        )
                    ),
                    np.concatenate((self.start_ee_pos[0:3], np.array([0]))),
                )
            elif self.primitive_to_model == "move_right":
                self.primitive_goal_space = Box(
                    np.concatenate((self.start_ee_pos[0:3], np.array([0]))),
                    np.concatenate(
                        (
                            self.mocap_high[0:1],
                            self.start_ee_pos[1:3],
                            np.array([0]),
                        )
                    ),
                )
            elif self.primitive_to_model == "move_backward":
                self.primitive_goal_space = Box(
                    np.concatenate(
                        (
                            self.start_ee_pos[0:1],
                            self.mocap_low[1:2],
                            self.start_ee_pos[2:3],
                            np.array([0]),
                        )
                    ),
                    np.concatenate((self.start_ee_pos[0:3], np.array([0]))),
                )
            elif self.primitive_to_model == "move_backward":
                self.primitive_goal_space = Box(
                    np.concatenate((self.start_ee_pos[0:3], np.array([0]))),
                    np.concatenate(
                        (
                            self.start_ee_pos[0:1],
                            self.mocap_high[1:2],
                            self.start_ee_pos[2:3],
                            np.array([0]),
                        )
                    ),
                )
            elif self.primitive_to_model == "drop":
                self.primitive_goal_space = Box(
                    np.concatenate(
                        (
                            self.start_ee_pos[0:2],
                            self.mocap_low[2:3],
                            np.array([0]),
                        )
                    ),
                    np.concatenate((self.start_ee_pos[0:3], np.array([0]))),
                )
            elif self.primitive_to_model == "lift":
                self.primitive_goal_space = Box(
                    np.concatenate((self.start_ee_pos[0:3], np.array([0]))),
                    np.concatenate(
                        (
                            self.start_ee_pos[0:2],
                            self.mocap_high[2:3],
                            np.array([0]),
                        )
                    ),
                )
            elif self.primitive_to_model == "move_delta_ee":
                self.primitive_goal_space = Box(
                    np.concatenate((self.mocap_low, np.array([0]))),
                    np.concatenate((self.mocap_high, np.array([0]))),
                )
            self.primitive_goal_dim = self.primitive_goal_space.low.size + 1
            if self.set_primitive_goals:
                self.primitive_goal = self.primitive_goal_space.sample()
                if (
                    self.primitive_to_model == "close_gripper"
                    or self.primitive_to_model == "top_x_y_grasp"
                ):
                    self.primitive_goal = np.concatenate(
                        (
                            self.primitive_goal[:-1],
                            [self.primitive_goal[-1], -self.primitive_goal[-1]],
                        )
                    )
                elif self.primitive_to_model == "open_gripper":
                    self.primitive_goal = np.concatenate(
                        (
                            self.primitive_goal[:-1],
                            [-self.primitive_goal[-1], self.primitive_goal[-1]],
                        )
                    )
                else:
                    self.primitive_goal = np.concatenate(
                        (
                            self.primitive_goal[:-1],
                            [self.primitive_goal[-1], self.primitive_goal[-1]],
                        )
                    )

    def sync_primitive_model_from_path(self, path):
        # TODO: figure out how to get this to be on GPU without blowing up GPU memory usage
        self.primitive_model.load_state_dict(torch.load(path, map_location=ptu.device))
        self.primitive_model = self.primitive_model.to(ptu.device)

    def sync_primitive_model(self):
        self.sync_primitive_model_from_path(self.primitive_model_path)

    def set_use_primitive_model(self):
        self.use_primitive_model = True

    def _reset_hand(self):
        super()._reset_hand()
        self.prev_grasp = 1  # corresponds to open
        self.did_reset = True

    def set_render_every_step(
        self,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        self.render_every_step = render_every_step
        self.render_mode = render_mode
        self.render_im_shape = render_im_shape

    def unset_render_every_step(self):
        self.render_every_step = False

    @_assert_task_is_set
    def step(
        self,
        action,
    ):
        if self.did_reset:
            if self.set_primitive_goals:
                self.primitive_goal = self.primitive_goal_space.sample()
                if (
                    self.primitive_to_model == "close_gripper"
                    or self.primitive_to_model == "top_x_y_grasp"
                ):
                    self.primitive_goal = np.concatenate(
                        (
                            self.primitive_goal[:-1],
                            [self.primitive_goal[-1], -self.primitive_goal[-1]],
                        )
                    )
                elif self.primitive_to_model == "open_gripper":
                    self.primitive_goal = np.concatenate(
                        (
                            self.primitive_goal[:-1],
                            [-self.primitive_goal[-1], self.primitive_goal[-1]],
                        )
                    )
                else:
                    self.primitive_goal = np.concatenate(
                        (
                            self.primitive_goal[:-1],
                            [self.primitive_goal[-1], self.primitive_goal[-1]],
                        )
                    )
            self.did_reset = False
        action = np.clip(action, -1.0, 1.0)
        if self.control_mode in [
            "joint_position",
            "joint_velocity",
            "torque",
            "end_effector",
        ]:
            self.prev_low_level_action = action
            if self.control_mode == "end_effector":
                self.set_xyz_action(action[:3])
                self.do_simulation([action[-1], -action[-1]])
        else:
            self.img_array = []
            self.primitives_info = {}
            self.primitives_info["low_level_action"] = []
            self.primitives_info["low_level_obs"] = []
            self.primitives_info["low_level_float_obs"] = []
            self.primitives_info["low_level_reward"] = []
            self.primitives_info["low_level_terminal"] = []
            self.primitive_step_counter = 0
            self._num_low_level_steps_total = 0
            self.combined_prev_action = np.zeros(3, dtype=np.float32)
            self.act(action)
            self.primitives_info["low_level_terminal"][-1] = 1
            self.primitives_info["high_level_action"] = self.high_level_action
            self.primitives_info[
                "processed_high_level_action"
            ] = self.processed_high_level_action

        self.curr_path_length += 1

        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            return (
                self._last_stable_obs,
                0.0,
                False,
                {  # info
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )

        self._last_stable_obs = self._get_obs()
        reward, info = self.evaluate_state(
            self._last_stable_obs, self.prev_low_level_action
        )
        if self.set_primitive_goals:
            reward = self.compute_low_level_reward(
                np.array([action[-1], -action[-1]]), self.primitive_goal
            )
        if self.control_mode == "primitives":
            if self.collect_primitives_info:
                info.update(self.primitives_info)
            info["relabel_distance"] = self.relabel_distance
            info["num low level steps"] = (
                self._num_low_level_steps_total // self.frame_skip
            )
            info["num low level steps true"] = self._num_low_level_steps_total
            self._num_low_level_steps_total = 0
        return self._last_stable_obs, reward, False, info

    def _get_site_pos(self, siteName):
        return self.data.site_xpos[self.model.site_name2id(siteName)]

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx

    def get_site_xpos(self, name):
        id = self.sim.model.site_name2id(name)
        return self.sim.data.site_xpos[id]

    def get_mocap_pos(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        return self.sim.data.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        body_id = self.sim.model.body_name2id(name)
        mocap_id = self.sim.model.body_mocapid[body_id]
        self.sim.data.mocap_quat[mocap_id] = value

    def ctrl_set_action(self, sim, action):
        sim.data.ctrl[0] = action[0]
        sim.data.ctrl[1] = action[1]

    def mocap_set_action(self, sim, action):
        if sim.model.nmocap > 0:
            action, _ = np.split(action, (sim.model.nmocap * 7,))
            action = action.reshape(sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]
            self.reset_mocap2body_xpos(sim)
            new_mocap_pos = self.data.mocap_pos + pos_delta[None]
            new_mocap_quat = self.data.mocap_quat + quat_delta[None]

            new_mocap_pos[0, :] = np.clip(
                new_mocap_pos[0, :],
                self.mocap_low,
                self.mocap_high,
            )
            self.data.set_mocap_pos("mocap", new_mocap_pos)
            self.data.set_mocap_quat("mocap", new_mocap_quat)

    def _set_action(self, action):
        assert action.shape == (9,)

        action = action.copy()
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7:9]

        pos_ctrl *= self.pos_ctrl_action_scale
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl])
        self.combined_prev_action += pos_ctrl
        if (self.primitive_step_counter + 1) % (
            self.num_low_level_steps // self.num_low_level_actions_per_primitive
        ) == 0:
            if hasattr(self, "primitive_model"):
                self.primitives_info["low_level_action"].append(
                    np.concatenate([self.combined_prev_action, gripper_ctrl])
                    / self.primitive_model.scale
                )
            else:
                self.primitives_info["low_level_action"].append(
                    np.concatenate([self.combined_prev_action, gripper_ctrl])
                )
            self.combined_prev_action = np.zeros(3, dtype=np.float32)

        # Apply action to simulation.
        self.mocap_set_action(self.sim, action)
        self.ctrl_set_action(self.sim, gripper_ctrl)

    def low_level_step(self, action, render_obs=False):
        self.mocap_set_action(self.sim, action[:7])
        self.ctrl_set_action(self.sim, action[7:])
        self.sim.step()
        self._last_stable_obs = self._get_obs()
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        if render_obs:
            obs = (
                self.render(
                    "rgb_array",
                    64,
                    64,
                )
                .transpose(2, 0, 1)
                .flatten()
            )
        else:
            obs = None
        return obs, reward, False, info

    def reset_mocap2body_xpos(self, sim):
        if (
            sim.model.eq_type is None
            or sim.model.eq_obj1id is None
            or sim.model.eq_obj2id is None
        ):
            return
        for eq_type, obj1_id, obj2_id in zip(
            sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
        ):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue

            mocap_id = sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                body_idx = obj2_id
            else:
                mocap_id = sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert mocap_id != -1
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]

    def call_render_every_step(self):
        if self.render_intermediate_obs_to_info:
            if (self.primitive_step_counter + 1) % (
                self.num_low_level_steps // self.num_low_level_actions_per_primitive
            ) == 0:
                obs = (
                    self.render(
                        "rgb_array",
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                    .transpose(2, 0, 1)
                    .flatten()
                )
                self.primitives_info["low_level_obs"].append(obs.astype(np.uint8))
        if self.render_every_step:
            if self.render_mode == "rgb_array":
                self.img_array.append(
                    self.render(
                        self.render_mode,
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                )
            else:
                self.render(
                    self.render_mode,
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )

    def get_gripper_pos(
        self,
    ):
        finger_right, finger_left = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )
        return np.linalg.norm(finger_right - finger_left)

    def get_endeff_quat(self):
        return self.sim.data.get_body_xquat("hand")

    def get_endeff_pose(self):
        ee_pos = self.get_endeff_pos()
        ee_quat = self.get_endeff_quat()
        return np.concatenate((ee_pos, ee_quat))

    def compute_low_level_reward(self, low_level_action_gripper, target):
        current = np.concatenate(
            (self.get_endeff_pos(), low_level_action_gripper)
        )  # 7DOF pose and gripper ctrl
        if self.low_level_reward_type == "argument_achievement":
            reward = -1 * np.linalg.norm(current - target)
        else:
            reward = 0
        return reward

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def unsubsample_and_execute_ll(
        self,
        num_subsample_steps,
        target,
        compute_action,
    ):
        prev_grasp = self.get_gripper_pos()
        self.processed_high_level_action = np.zeros_like(self.high_level_action)
        self.processed_high_level_action[
            : self.num_primitives
        ] = self.high_level_action[: self.num_primitives]
        if self.primitive_name == "move_delta_ee":
            self.processed_high_level_action[
                self.num_primitives
                + np.array(self.primitive_name_to_action_idx[self.primitive_name])
            ] = self.high_level_action[
                self.num_primitives
                + np.array(self.primitive_name_to_action_idx[self.primitive_name])
            ]
        else:
            self.processed_high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = self.high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ]
        for sample_step in range(self.num_low_level_actions_per_primitive):
            self.primitives_info["low_level_float_obs"].append(
                np.concatenate((self.get_endeff_pos(), [self.get_gripper_pos()]))
            )
            observation = (
                self.render(
                    "rgb_array",
                    64,
                    64,
                )
                .transpose(2, 0, 1)
                .flatten()
            )

            action = np.concatenate(
                (
                    self.primitives_info["low_level_float_obs"][-1],
                    self.processed_high_level_action,
                    [1 / (sample_step + 1)],
                )
            )
            inputs = np.concatenate((observation, action))
            low_level_action, _ = self.primitive_model.get_action(
                np.expand_dims(inputs, 0)
            )
            low_level_action *= self.primitive_model.scale
            t = self.get_endeff_pos() + low_level_action[:3]
            for step in range(num_subsample_steps):
                a = (t - self.get_endeff_pos()) / num_subsample_steps
                # a = np.concatenate(
                #     (a, low_level_action[3:])
                # )  # assume rotation should not be subsampled/unsubsampled
                a = np.concatenate((a, np.array([1, 0, 1, 0, *low_level_action[-2:]])))
                if self.render_every_step:
                    if self.render_mode == "rgb_array":
                        self.img_array.append(
                            self.render(
                                self.render_mode,
                                self.render_im_shape[0],
                                self.render_im_shape[1],
                            )
                        )
                    else:
                        self.render(
                            self.render_mode,
                            self.render_im_shape[0],
                            self.render_im_shape[1],
                        )
                self.mocap_set_action(self.sim, a[:7])
                self.ctrl_set_action(self.sim, a[7:])
                self.sim.step()
                self.primitive_step_counter += 1
                self._num_low_level_steps_total += 1
            obs = (
                self.render(
                    "rgb_array",
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )
                .transpose(2, 0, 1)
                .flatten()
            )
            self.primitives_info["low_level_obs"].append(obs.astype(np.uint8))
            self.primitives_info["low_level_action"].append(
                low_level_action / self.primitive_model.scale
            )
            reward = self.compute_low_level_reward(a[7:], target)
            self.primitives_info["low_level_reward"].append(reward)
            self.primitives_info["low_level_terminal"].append(0)
        self.primitives_info["low_level_float_obs"].append(
            np.concatenate((self.get_endeff_pos(), [self.get_gripper_pos()]))
        )
        return low_level_action

    def execute_primitive_model(self, target, compute_action):
        (
            _,
            _,
            primitive_idx,
        ) = self.get_primitive_info_from_high_level_action(self.high_level_action)
        num_subsample_steps = (
            self.primitive_idx_to_num_low_level_steps[primitive_idx]
            // self.num_low_level_actions_per_primitive
        )
        action = self.unsubsample_and_execute_ll(
            num_subsample_steps,
            target,
            compute_action,
        )
        self.prev_low_level_action = action
        return action

    def execute_primitive(self, compute_action, num_iterations, target):
        if self.use_primitive_model:
            action = self.execute_primitive_model(target, compute_action)
        else:
            for _ in range(num_iterations):
                if (self.primitive_step_counter + 1) % (
                    self.num_low_level_steps // self.num_low_level_actions_per_primitive
                ) == 0:
                    self.primitives_info["low_level_float_obs"].append(
                        np.concatenate(
                            (self.get_endeff_pos(), [self.get_gripper_pos()])
                        )
                    )
                action = compute_action()
                self._set_action(action)
                self.sim.step()
                self.call_render_every_step()
                self.primitive_step_counter += 1
                self._num_low_level_steps_total += 1
                if (self.primitive_step_counter + 1) % (
                    self.num_low_level_steps // self.num_low_level_actions_per_primitive
                ) == 0:
                    low_level_reward = self.compute_low_level_reward(action[7:], target)
                    self.primitives_info["low_level_reward"].append(low_level_reward)
                if (self.primitive_step_counter + 1) % (
                    self.num_low_level_steps // self.num_low_level_actions_per_primitive
                ) == 0:
                    self.primitives_info["low_level_terminal"].append(0)
            self.primitives_info["low_level_float_obs"].append(
                np.concatenate((self.get_endeff_pos(), [self.get_gripper_pos()]))
            )
        self.processed_high_level_action = np.zeros_like(self.high_level_action)
        self.processed_high_level_action[
            : self.num_primitives
        ] = self.high_level_action[: self.num_primitives]
        if self.primitive_name == "move_along_x":
            self.high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_endeff_pos()[0] - self.pre_action_pos[0])
            self.processed_high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_endeff_pos()[0] - self.pre_action_pos[0])
        elif self.primitive_name == "move_along_y":
            self.high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_endeff_pos()[1] - self.pre_action_pos[1])
            self.processed_high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_endeff_pos()[1] - self.pre_action_pos[1])
        elif self.primitive_name == "move_along_z":
            self.high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_endeff_pos()[2] - self.pre_action_pos[2])
            self.processed_high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_endeff_pos()[2] - self.pre_action_pos[2])
        elif self.primitive_name == "move_delta_ee":
            self.high_level_action[
                self.num_primitives
                + np.array(self.primitive_name_to_action_idx[self.primitive_name])
            ] = (self.get_endeff_pos() - self.pre_action_pos)
            self.processed_high_level_action[
                self.num_primitives
                + np.array(self.primitive_name_to_action_idx[self.primitive_name])
            ] = (self.get_endeff_pos() - self.pre_action_pos)
        elif self.primitive_name == "top_x_y_grasp":
            self.high_level_action[
                self.num_primitives
                + np.array(self.primitive_name_to_action_idx[self.primitive_name])
            ] = np.concatenate(
                (
                    self.get_endeff_pos() - self.pre_action_pos,
                    [(self.get_gripper_pos() - 0.065) / 0.035],
                )
            )
            self.processed_high_level_action[
                self.num_primitives
                + np.array(self.primitive_name_to_action_idx[self.primitive_name])
            ] = np.concatenate(
                (
                    self.get_endeff_pos() - self.pre_action_pos,
                    [(self.get_gripper_pos() - 0.065) / 0.035],
                )
            )
        elif self.primitive_name == "move_gripper":
            self.high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_gripper_pos() - 0.065) / 0.035
            self.processed_high_level_action[
                self.num_primitives
                + self.primitive_name_to_action_idx[self.primitive_name]
            ] = (self.get_gripper_pos() - 0.065) / 0.035
        self.relabel_distance = np.linalg.norm(self.high_level_action - self.old_hl)
        if not self.relabel_high_level_actions:
            self.high_level_action = self.old_hl
            self.processed_high_level_action = np.zeros_like(self.high_level_action)
            self.processed_high_level_action[
                : self.num_primitives
            ] = self.high_level_action[: self.num_primitives]
            if self.primitive_name == "move_delta_ee":
                self.processed_high_level_action[
                    self.num_primitives
                    + np.array(self.primitive_name_to_action_idx[self.primitive_name])
                ] = self.high_level_action[
                    self.num_primitives
                    + np.array(self.primitive_name_to_action_idx[self.primitive_name])
                ]
            else:
                self.processed_high_level_action[
                    self.num_primitives
                    + self.primitive_name_to_action_idx[self.primitive_name]
                ] = self.high_level_action[
                    self.num_primitives
                    + self.primitive_name_to_action_idx[self.primitive_name]
                ]
        return action

    def move_gripper(self, d, target=None, iterations=None):
        pose = self.get_endeff_pos()
        gripper_target = d * 0.035 + 0.065

        def compute_action():
            delta = pose - self.get_endeff_pos()
            current_pos = self.get_gripper_pos()
            gripper_ctrl = [
                -(gripper_target - current_pos) * 1 / (0.07),
                (gripper_target - current_pos) * 1 / (0.07),
            ]
            action = np.array(
                [delta[0], delta[1], delta[2], 1.0, 0.0, 1.0, 0.0, *gripper_ctrl]
            )
            return action

        if target is None:
            target = np.concatenate((self.get_endeff_pos(), [gripper_target]))
        if iterations is None:
            iterations = self.goto_pose_iterations
        action = self.execute_primitive(compute_action, iterations, target)
        self.prev_low_level_action = action

    def move_along_x(self, x_dist):
        self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0]),
        )

    def move_along_y(self, y_dist):
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0]),
        )

    def move_along_z(self, z_dist):
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, z_dist]),
        )

    def close_gripper(self, d, target=None, iterations=None):
        d = np.abs(d)
        compute_action = lambda: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, d, -d])
        if target is None:
            target = np.concatenate((self.get_endeff_pos(), [d, -d]))
        if iterations is None:
            iterations = self.close_gripper_iterations
        action = self.execute_primitive(compute_action, iterations, target)
        self.prev_low_level_action = action
        self.prev_grasp = -d

    def open_gripper(self, d, target=None, iterations=None):
        d = np.abs(d)
        compute_action = lambda: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -d, d])
        if target is None:
            target = np.concatenate((self.get_endeff_pos(), [-d, d]))
        if iterations is None:
            iterations = self.open_gripper_iterations
        action = self.execute_primitive(compute_action, iterations, target)
        self.prev_low_level_action = action
        self.prev_grasp = d

    def goto_pose(self, pose, target=None, iterations=None):
        prev_grasp = self.get_gripper_pos()

        def compute_action():
            delta = pose - self.get_endeff_pos()
            current_pos = self.get_gripper_pos()
            gripper_ctrl = [
                -(prev_grasp - current_pos) * 1 / (0.07),
                (prev_grasp - current_pos) * 1 / (0.07),
            ]
            action = np.array(
                [delta[0], delta[1], delta[2], 1.0, 0.0, 1.0, 0.0, *gripper_ctrl]
            )
            return action

        if target is None:
            target = np.concatenate((pose, [prev_grasp]))
        if iterations is None:
            iterations = self.goto_pose_iterations
        action = self.execute_primitive(compute_action, iterations, target)
        self.prev_low_level_action = action

    def top_x_y_grasp(self, xyzd):
        x_dist, y_dist, z_dist, d = xyzd
        target = np.concatenate(
            (
                self.get_endeff_pos() + np.array([x_dist, y_dist, z_dist]),
                [d * 0.035 + 0.065],
            )
        )
        if self.use_primitive_model:
            self.execute_primitive_model(target, None)
        else:
            if self.split_top_grasp:
                iterations = self.goto_pose_iterations // 4
                self.num_low_level_steps = self.goto_pose_iterations
                self.goto_pose(
                    self.get_endeff_pos() + np.array([0.0, y_dist, 0]),
                    target=target,
                    iterations=iterations,
                )
                self.goto_pose(
                    self.get_endeff_pos() + np.array([x_dist, 0.0, 0]),
                    target=target,
                    iterations=iterations,
                )
                self.goto_pose(
                    self.get_endeff_pos() + np.array([0.0, 0, z_dist]),
                    target=target,
                    iterations=iterations,
                )
                self.close_gripper(d, target=target, iterations=iterations)
            else:
                self.goto_pose(
                    self.get_endeff_pos() + np.array([0.0, y_dist, 0]),
                    target=target,
                )
                self.goto_pose(
                    self.get_endeff_pos() + np.array([x_dist, 0.0, 0]),
                    target=target,
                )
                self.goto_pose(
                    self.get_endeff_pos() + np.array([0.0, 0, z_dist]),
                    target=target,
                )
                self.move_gripper(d, target=target)

    def move_delta_ee(self, pose):
        self.goto_pose(self.get_endeff_pos() + pose)

    def lift(self, z_dist):
        z_dist = np.abs(z_dist)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, z_dist]),
        )

    def drop(self, z_dist):
        z_dist = np.abs(z_dist)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, 0.0, -z_dist]),
        )

    def move_left(self, x_dist):
        x_dist = np.abs(x_dist)
        self.goto_pose(
            self.get_endeff_pos() + np.array([-x_dist, 0.0, 0.0]),
        )

    def move_right(self, x_dist):
        x_dist = np.abs(x_dist)
        self.goto_pose(
            self.get_endeff_pos() + np.array([x_dist, 0.0, 0.0]),
        )

    def move_forward(self, y_dist):
        y_dist = np.abs(y_dist)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, y_dist, 0.0]),
        )

    def move_backward(self, y_dist):
        y_dist = np.abs(y_dist)
        self.goto_pose(
            self.get_endeff_pos() + np.array([0.0, -y_dist, 0.0]),
        )

    def break_apart_action(self, action):
        broken_a = {}
        for key, value in self.primitive_name_to_action_idx.items():
            broken_a[key] = action[value]
        return broken_a

    def get_primitive_info_from_high_level_action(self, hl):
        primitive_idx, primitive_args = (
            np.argmax(hl[: self.num_primitives]),
            hl[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        return primitive_name, primitive_args, primitive_idx

    def act(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action * self.action_scale
        argmax = np.argmax(action[: self.num_primitives], axis=-1)
        one_hots = np.eye(self.num_primitives)[argmax]
        action = np.concatenate((one_hots, action[self.num_primitives :]), axis=-1)
        self.high_level_action = action
        (
            primitive_name,
            primitive_args,
            primitive_idx,
        ) = self.get_primitive_info_from_high_level_action(action)
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        self.num_low_level_steps = self.primitive_idx_to_num_low_level_steps[
            primitive_idx
        ]
        self.old_hl = self.high_level_action.copy()
        self.pre_action_pos = self.get_endeff_pos()
        self.pre_action_grip = self.get_gripper_pos()
        self.primitive_name = primitive_name
        primitive(
            primitive_action,
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class RobosuiteWrapper(GymWrapper):
    def __init__(
        self,
        env,
        reset_action_space_kwargs,
        keys=None,
    ):
        super().__init__(
            env,
            keys=keys,
        )
        if hasattr(self.env, "reset_action_space"):
            self.env.reset_action_space(**reset_action_space_kwargs)
            if self.control_mode == "primitives":
                self.action_space = self.env.action_space
        self.image_shape = (3, self.imwidth, self.imheight)
        self.imlength = self.imwidth * self.imheight
        self.imlength *= 3
        self.observation_space = spaces.Box(0, 255, (self.imlength,), dtype=np.uint8)

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")

    def reset(self):
        o = super().reset()
        o = self.env.render(
            render_mode="rgb_array", imheight=self.imheight, imwidth=self.imwidth
        )
        o = (
            o.reshape(self.imwidth, self.imheight, 3)[:, :, ::-1]
            .transpose(2, 0, 1)
            .flatten()
        )
        return o

    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        self.env.set_render_every_step(render_every_step, render_mode, render_im_shape)
        obs, reward, done, info = super().step(action)
        o = self.env.render(
            render_mode="rgb_array", imheight=self.imheight, imwidth=self.imwidth
        )
        self.env.unset_render_every_step()
        new_info = {}
        for key, value in info.items():
            if value is not None:
                new_info[key] = value
        o = (
            o.reshape(self.imwidth, self.imheight, 3)[:, :, ::-1]
            .transpose(2, 0, 1)
            .flatten()
        )
        return obs, reward, done, new_info

    def __getattr__(self, name):
        if name != "env":
            return getattr(self.env, name)
        else:
            raise AttributeError("")


class NormalizeBoxEnvFixed(NormalizedBoxEnv):
    def step(
        self,
        action,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        wrapped_step = self._wrapped_env.step(
            scaled_action,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info


class RobosuitePrimitives(DMControlBackendRobosuiteEnv):
    def set_render_every_step(
        self,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(64, 64),
    ):
        self.render_every_step = render_every_step
        self.render_mode = render_mode
        self.render_im_shape = render_im_shape

    def unset_render_every_step(self):
        self.render_every_step = False

    def reset_action_space(
        self,
        control_mode="robosuite",
        action_scale=1,
        max_path_length=200,
        camera_settings=None,
        workspace_low=(),
        workspace_high=(),
        imwidth=64,
        imheight=64,
        go_to_pose_iterations=100,
    ):
        self.imwidth = imwidth
        self.imheight = imheight
        self.workspace_low = np.array(workspace_low)
        self.workspace_high = np.array(workspace_high)
        if camera_settings is None:
            camera_settings = {}
        self.camera_settings = camera_settings
        self.max_path_length = max_path_length
        self.action_scale = action_scale
        self.go_to_pose_iterations = go_to_pose_iterations

        # primitives
        self.primitive_idx_to_name = {
            0: "move_delta_ee",
            1: "top_grasp",
            2: "lift",
            3: "drop",
            4: "move_left",
            5: "move_right",
            6: "move_forward",
            7: "move_backward",
            8: "open_gripper",
            9: "close_gripper",
        }
        self.primitive_name_to_func = dict(
            move_delta_ee=self.move_delta_ee,
            top_grasp=self.top_grasp,
            lift=self.lift,
            drop=self.drop,
            move_left=self.move_left,
            move_right=self.move_right,
            move_forward=self.move_forward,
            move_backward=self.move_backward,
            open_gripper=self.open_gripper,
            close_gripper=self.close_gripper,
        )
        self.primitive_name_to_action_idx = dict(
            move_delta_ee=[0, 1, 2],
            top_grasp=3,
            lift=4,
            drop=5,
            move_left=6,
            move_right=7,
            move_forward=8,
            move_backward=9,
            open_gripper=[],  # doesn't matter
            close_gripper=[],  # doesn't matter
        )
        self.max_arg_len = 10

        self.num_primitives = len(self.primitive_name_to_func)
        self.control_mode = control_mode

        if self.control_mode == "primitives":
            action_space_low = -1 * np.ones(self.max_arg_len)
            action_space_high = np.ones(self.max_arg_len)
            act_lower_primitive = np.zeros(self.num_primitives)
            act_upper_primitive = np.ones(self.num_primitives)
            act_lower = np.concatenate((act_lower_primitive, action_space_low))
            act_upper = np.concatenate(
                (
                    act_upper_primitive,
                    action_space_high,
                )
            )
            self.action_space = Box(act_lower, act_upper, dtype=np.float32)

    def step(self, action):
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        if self.control_mode == "robosuite":
            policy_step = True
            target_pos = action[:3] + self._eef_xpos
            target_pos = np.clip(target_pos, self.workspace_low, self.workspace_high)
            action[:3] = target_pos - self._eef_xpos
            action[3:6] = 0

            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                self._update_observables()
                policy_step = False
                if self.render_every_step:
                    self.render()
            stats = [0, 0]
        else:
            self.img_array = []
            self._num_low_level_steps_total = 0
            stats = self.act(action)
            self._update_observables()

        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)
        if self.control_mode == "primitives":
            reward = float(stats[1] > 0)
            info["success"] = float(stats[1] > 0)
            info["num low level steps"] = self._num_low_level_steps_total // (
                int(self.control_timestep / self.model_timestep)
            )
            info["num low level steps true"] = self._num_low_level_steps_total
            self._num_low_level_steps_total = 0
        else:
            info["success"] = float(self._check_success())

        return self._get_observations(force_update=True), reward, done, info

    def render(self, render_mode="human", imwidth=64, imheight=64):
        if self._use_dm_backend:
            if render_mode == "human":
                self.renderer.render_to_window()
            else:
                img = self.renderer.render_offscreen(
                    imwidth,
                    imheight,
                )
                return img
        else:
            super().render()

    def call_render_every_step(self):
        if self.render_every_step:
            if self.render_mode == "rgb_array":
                self.img_array.append(
                    self.render(
                        self.render_mode,
                        self.render_im_shape[0],
                        self.render_im_shape[1],
                    )
                )
            else:
                self.render(
                    self.render_mode,
                    self.render_im_shape[0],
                    self.render_im_shape[1],
                )

    def close_gripper(self, unused=None):
        total_reward, total_success = 0, 0
        for _ in range(150):
            action = [0, 0, 0, 0, 0, 0, 1]
            self.robots[0].control(action, policy_step=False)
            self.sim.step()
            self.call_render_every_step()
            r = self.reward(action)
            total_reward += r
            total_success += float(self._check_success())
            self._num_low_level_steps_total += 1
        return np.array((total_reward, total_success))

    def open_gripper(self, unused=None):
        total_reward, total_success = 0, 0
        for _ in range(300):
            action = [0, 0, 0, 0, 0, 0, -1]
            self.robots[0].control(action, policy_step=False)
            self.sim.step()
            self.call_render_every_step()
            r = self.reward(action)
            total_reward += r
            total_success += float(self._check_success())
            self._num_low_level_steps_total += 1
        return np.array((total_reward, total_success))

    def goto_pose(self, pose, grasp=False):
        total_reward, total_success = 0, 0
        prev_delta = np.zeros_like(pose)
        pose = np.clip(pose, self.workspace_low, self.workspace_high)
        for _ in range(self.go_to_pose_iterations):
            delta = pose - self._eef_xpos
            if grasp:
                gripper = 1
            else:
                gripper = -1
            action = [*delta, 0, 0, 0, gripper]
            if np.allclose(delta - prev_delta, 1e-4):
                break
            policy_step = True
            prev_delta = delta
            for i in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                policy_step = False
                self.call_render_every_step()
                self.cur_time += self.control_timestep
                r = self.reward(action)
                total_reward += r
                total_success += float(self._check_success())
                self._num_low_level_steps_total += 1
        return np.array((total_reward, total_success))

    def top_grasp(self, z_down):
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, 0, -np.abs(z_down)]), grasp=False
        )
        stats += self.close_gripper()
        return stats

    def move_delta_ee(self, pose):
        stats = self.goto_pose(
            self._eef_xpos + pose,
        )
        return stats

    def lift(self, z_dist):
        z_dist = np.abs(z_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, z_dist]),
        )
        return stats

    def drop(self, z_dist):
        z_dist = np.abs(z_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0.0, 0.0, -z_dist]),
        )
        return stats

    def move_left(self, x_dist):
        x_dist = np.abs(x_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, -x_dist, 0.0]),
        )
        return stats

    def move_right(self, x_dist):
        x_dist = np.abs(x_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([0, x_dist, 0.0]),
        )
        return stats

    def move_forward(self, y_dist):
        y_dist = np.abs(y_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([y_dist, 0, 0.0]),
        )
        return stats

    def move_backward(self, y_dist):
        y_dist = np.abs(y_dist, 0.0)
        stats = self.goto_pose(
            self._eef_xpos + np.array([-y_dist, 0, 0.0]),
        )
        return stats

    def break_apart_action(self, action):
        broken_a = {}
        for key, value in self.primitive_name_to_action_idx.items():
            broken_a[key] = a[value]
        return broken_a

    def act(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action * self.action_scale
        primitive_idx, primitive_args = (
            np.argmax(action[: self.num_primitives]),
            action[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        stats = primitive(primitive_action)
        return stats

    def get_idx_from_primitive_name(self, primitive_name):
        for idx, pn in self.primitive_idx_to_name.items():
            if pn == primitive_name:
                return idx
