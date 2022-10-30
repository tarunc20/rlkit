from matplotlib import pyplot as plt
import numpy as np
from rlkit.torch.model_based.dreamer.visualization import make_video
import robosuite as suite
import torch
from robosuite.utils.transform_utils import *
from robosuite.wrappers.gym_wrapper import GymWrapper
from tqdm import tqdm

import rlkit.torch.pytorch_util as ptu
from rlkit.mprl.mp_env import (
    RobosuiteEnv,
)

if __name__ == "__main__":
    robosuite_args = dict(
        robots="Panda",
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
        use_object_obs=True,
        env_name="PickPlaceBread",
    )
    # OSC controller spec
    controller_args = dict(
        type="OSC_POSE",
        input_max=1,
        input_min=-1,
        output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        kp=150,
        damping=1,
        impedance_mode="fixed",
        kp_limits=[0, 300],
        damping_limits=[0, 10],
        position_limits=None,
        orientation_limits=None,
        uncouple_pos_ori=True,
        control_delta=True,
        interpolation=None,
        ramp_ratio=0.2,
    )
    robosuite_args["controller_configs"] = controller_args
    env = suite.make(
        **robosuite_args,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_names="frontview",
        camera_heights=1024,
        camera_widths=1024,
    )
    np.random.seed(1)
    env = RobosuiteEnv(GymWrapper(env))
    num_episodes = 1
    total = 0
    ptu.device = torch.device("cuda")
    success_rate = 0
    frames = []
    target_pos = np.array(
        [
            0.2,
            0.15,
            env.sim.data.qpos[18] + 0.1,
        ]
    )
    for s in tqdm(range(num_episodes)):
        o = env.reset()
        rs = []
        for i in range(300):
            a = np.concatenate(
                (
                    env.sim.data.qpos[16:19] + np.array([0, 0, 0.1]) - env._eef_xpos,
                    [0, 0, 0, -1],
                )
            )
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, -0.2], [0, 0, 0, -1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, 0], [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, 0.2], [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(200):
            a = np.concatenate((target_pos - env._eef_xpos, [0, 0, 0, 1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()
        for i in range(50):
            a = np.concatenate(([0, 0, -0.0], [0, 0, 0, -1]))
            o, r, d, info = env.step(a)
            rs.append(r)
            env.render()

        print(env._check_success())
        plt.plot(rs)
        plt.show()
        success_rate += env._check_success()
        make_video(frames, "test", 0)
    print(f"Success Rate: {success_rate/num_episodes}")
