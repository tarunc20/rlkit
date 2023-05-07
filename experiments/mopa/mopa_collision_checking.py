import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import sys
from rlkit.mprl.hierarchical_policies import StepBasedSwitchingPolicy
import rlkit.mopa.env
import cv2
from rlkit.mopa.env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.mopa.util.env import joint_convert, mat2quat, quat_mul, rotation_matrix, quat2mat
from rlkit.mopa.util.transform_utils import mat2pose, convert_quat, pose2mat
from rlkit.mopa.config import sawyer
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from rlkit.mopa.config.default_configs import LIFT_CONFIG, LIFT_OBSTACLE_CONFIG, ASSEMBLY_OBSTACLE_CONFIG
from rlkit.mopa.mopa_env import *
import mujoco_py
import collections 
import time

def video_func(algorithm, epoch):
    import copy
    import os
    import pickle

    import numpy as np

    from rlkit.core import logger
    from rlkit.core.batch_rl_algorithm import BatchModularRLAlgorithm
    from rlkit.torch.model_based.dreamer.visualization import make_video

    if epoch % 50 == 0 or epoch == -1 and epoch != 0:
        eval_collector = algorithm.eval_data_collector
        eval_env = algorithm.eval_env
        policy = eval_collector._policy
        max_path_length = algorithm.max_path_length
        num_envs = eval_env.num_envs
        frames = [[] for _ in range(num_envs)]
        success_rate = 0
        policy.reset()
        obs = eval_env.reset()
        images = eval_env.env_method("get_image")
        for i in range(num_envs):
            frames[i].append(images[i])
        for step in range(max_path_length):
            actions, _ = policy.get_action(obs)
            obs, rewards, dones, infos = eval_env.step(copy.deepcopy(actions))
            images = eval_env.env_method("get_image")
            for i in range(num_envs):
                frames[i].append(images[i])
            if all(dones):
                break
        print(
            f"r: {rewards}, is grasped: {eval_env.env_method('check_grasp')}, logged grasp: {infos['grasped']}"
        )
        success_rate = infos["success"]
        logdir = logger.get_snapshot_dir()
        flattened_frames = []
        for i in range(num_envs):
            for frame in frames[i]:
                flattened_frames.append(frame)
        make_video(flattened_frames, logdir, epoch)
        print(f"Saved video for epoch {epoch}")
        print(f"Success rate: {np.mean(success_rate)}")
        # env = algorithm.trainer.env
        # algorithm.trainer.env = None
        # pickle.dump(
        #     algorithm.trainer, open(os.path.join(logdir, f"control_{epoch}.pkl"), "wb")
        # )
        # algorithm.trainer.env = env
        # if isinstance(algorithm, BatchModularRLAlgorithm):
        #     algorithm.planner_trainer.env = None
        #     pickle.dump(
        #         algorithm.planner_trainer,
        #         open(os.path.join(logdir, f"planner_{epoch}.pkl"), "wb"),
        #     )
        #     algorithm.planner_trainer.env = env

def make_env(variant):
    env = gym.make(**LIFT_CONFIG)
    ik_env = gym.make(**LIFT_CONFIG)
    env = MoPAMPEnv(
        "SawyerLift-v0",
        env,
        ik_env,
        config=LIFT_CONFIG,
        plan_to_learned_goals=variant["plan_to_learned_goals"],
        num_ll_actions_per_hl_action=variant["num_ll_actions_per_hl_action"]
    )
    return env

# hardcoded policy no teleport on grasp
# create environment 
import glfw
#LIFT_CONFIG["action_range"] = 1.0
env = gym.make(**LIFT_CONFIG)
ik_env = gym.make(**LIFT_CONFIG)
#glfw.show_window()
# train only lifting policy
env = MoPAMPEnv(
    "SawyerLift-v0",
    env,
    ik_env,
    config=LIFT_CONFIG,
    plan_to_learned_goals=False,
    num_ll_actions_per_hl_action=25,
    teleport_on_grasp=False,
    vertical_displacement=0.05,
)
o = env.reset()
print(f"Initial State")
save_img(env._wrapped_env, f"initial_state.png")
print(env._wrapped_env.sim.data.qpos[env._wrapped_env.ref_gripper_joint_pos_indexes])
print(env._wrapped_env.sim.data.qvel[env._wrapped_env.ref_gripper_joint_pos_indexes])
action = np.zeros(7)
action[:3] = np.array([0., 0., -0.5])
action[-1] = -1.0
# go down 
for _ in range(3):
    o, r, d, i = env.step(action)
print(env._wrapped_env.sim.data.qpos[env._wrapped_env.ref_gripper_joint_pos_indexes])
print(env._wrapped_env.sim.data.qvel[env._wrapped_env.ref_gripper_joint_pos_indexes])
# grip action
assert False
action = np.zeros(7)
action[-1] = 5.0
o, r, d, i = env.step(action)
save_img(env._wrapped_env, f"down.png")
action = np.zeros(7)
action[:3] = np.array([0., 0., 0.5])
action[-1] = 1.0 
for _ in range(10):
    o, r, d, i = env.step(action)
save_img(env._wrapped_env, f"down.png")    
assert False
action = np.zeros(7)
action[:3] = np.array([0., 0., -0.02])
action[-1] = -1.0
for _ in range(4):
    o, r, d, i = env.step(action)
    for _ in range(500):
        env._wrapped_env.render("human")
for _ in range(500):
    env._wrapped_env.render("human")
o, r, d, i = env.step(np.array([0., 0., 0., 0., 0., 0., -1.0]))
for _ in range(500):
    env._wrapped_env.render("human")
# go downwards
action = np.zeros(7)
action[:3] = np.array([0., 0., -0.02])
action[-1] = -1.0
for _ in range(4):
    o, r, d, i = env.step(action)
    for _ in range(500):
        env._wrapped_env.render("human")


# grip 
action = np.zeros(7)
action[-1] = 5.0
for _ in range(500):
    env._wrapped_env.render("human")

# grip and go up
action = np.zeros(7)
action[:3] = np.array([0., 0., 0.40])
action[-1] = 5.0
for _ in range(5):
    o, r, d, i = env.step(action)
    for _ in range(500):
        env._wrapped_env.render("human")
# #env.render()
# #assert False
# # check collisions
# # open gripper action
# action = np.zeros(7)
# action[-1] = -1.0
# o, r, d, i = env.step(action)
# # visualize teleportation 
# save_img(env._wrapped_env, 'start.png')
# # do a grasp action
# # add 3d viewer
# for _ in range(5):
#     action = np.zeros(7)
#     action[:3] = np.array([0., 0., -0.02])
#     action[3] = 1.0
#     o, r, d, i = env.step(action)
# save_img(env._wrapped_env, 'end.png')
# print(f"Done: {d}")

# if __name__ == "__main__":
#     main()