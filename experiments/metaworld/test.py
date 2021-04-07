import time

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv, make_metaworld_env
from rlkit.envs.primitives_wrappers import ImageEnvMetaworld, TimeLimit

if __name__ == "__main__":
    for env_name in [
        "assembly-v2",
        "basketball-v2",
        "bin-picking-v2",
        "box-close-v2",
        "button-press-topdown-v2",
        "button-press-topdown-wall-v2",
        "button-press-v2",
        "button-press-wall-v2",
        "coffee-button-v2",
        "coffee-pull-v2",
        "coffee-push-v2",
        "dial-turn-v2",
        "disassemble-v2",
        "door-close-v2",
        "door-lock-v2",
        "door-open-v2",
        "door-unlock-v2",
        # "hand-insert-v2", #no goal
        # "drawer-close-v2", #no goal
        # "drawer-open-v2", #no goal
        # "faucet-open-v2", #no goal
        # "faucet-close-v2", #no goal
        "hammer-v2",
        "handle-press-side-v2",
        "handle-press-v2",
        "handle-pull-side-v2",
        "handle-pull-v2",
        "lever-pull-v2",
        "peg-insert-side-v2",
        "pick-place-wall-v2",
        "pick-out-of-hole-v2",
        "reach-v2",
        "push-back-v2",
        "push-v2",
        "pick-place-v2",
        "plate-slide-v2",
        "plate-slide-side-v2",
        "plate-slide-back-v2",
        "plate-slide-back-side-v2",
        "peg-unplug-side-v2",
        "soccer-v2",
        "stick-push-v2",
        "stick-pull-v2",
        "push-wall-v2",
        "reach-wall-v2",
        "shelf-place-v2",
        "sweep-into-v2",
        "sweep-v2",
        # "window-open-v2", #no goal
        # "window-close-v2", #no goal
    ]:
        print(env_name)
        # env = ImageEnvMetaworld(
        #     make_metaworld_env(
        #         env_name,
        #         dict(
        #             control_mode="primitives",
        #             use_combined_action_space=True,
        #             action_scale=0.5,
        #             max_path_length=5,
        #         ),
        #         # {}
        #     ),
        #     imwidth=64,
        #     imheight=64,
        # )
        # obs = env.reset()
        num_envs = 20
        env_fns = [
            lambda: TimeLimit(
                ImageEnvMetaworld(
                    make_metaworld_env(env_name, {}), imwidth=64, imheight=64
                ),
                5,
            )
            for _ in range(num_envs)
        ]
        envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="forkserver")
        envs.reset()
        for i in range(10000 // num_envs):
            # a = np.zeros(env.action_space.low.size)
            # primitive = 'angled_x_y_grasp'
            # a[env.get_idx_from_primitive_name(primitive)] = 1
            # a[
            #         env.num_primitives
            #         + np.array(env.primitive_name_to_action_idx[primitive])
            #     ] = np.array([-np.pi / 6, -0.3, 1.4])
            a = [envs.action_space.sample() for _ in range(num_envs)]
            # print(a[-14:])
            # envs.step(
            #     a, render_every_step=False, render_mode="human"
            # )
            print(i)
            envs.step(
                a,
            )
            # env.render()
            if i % 5 == 0:
                # if i % 150 == 0:
                # st = time.time()

                # print((time.time() - st) / (i + 1))
                envs.reset()
        # env.reset()
    # cv2.imwrite("test.png", obs.reshape(3, 64, 64).transpose(1, 2, 0))
    # plt.imshow(obs.reshape(3, 84, 84).transpose(1, 2, 0))
    # plt.show()
    # import gym

    # gym.logger.set_level(40)
    # num_envs = 15
    # performance = {}
    # for num_envs in range(33, 34):
    #     env_fns = [
    #         lambda: TimeLimit(
    #             ImageEnvMetaworld(
    #                 make_metaworld_env("assembly-v2", {}), imwidth=64, imheight=64
    #             ),
    #             150,
    #         )
    #         for _ in range(num_envs)
    #     ]
    #     envs = StableBaselinesVecEnv(env_fns=env_fns, start_method="forkserver")
    #     envs.reset()
    #     d = [False] * num_envs
    #     import time

    #     st = time.time()
    #     for i in range(10000):
    #         o, r, d, _ = envs.step(
    #             envs.action_space.sample().reshape(1, -1).repeat(num_envs, 0)
    #         )
    #         if i % 150 == 0:
    #             envs.reset()
    #     performance[num_envs] = (time.time() - st) / 10000 / num_envs
    # print(performance)
