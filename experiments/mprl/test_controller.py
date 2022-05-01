import cv2
import numpy as np
from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv
from rlkit.mprl.mp_env import MPEnv, set_robot_based_on_ee_pos, update_controller_config
from robosuite.controllers import controller_factory
from robosuite.controllers.controller_factory import load_controller_config
from robosuite.utils.transform_utils import mat2euler, quat2mat
from robosuite.wrappers.gym_wrapper import GymWrapper
import robosuite as suite

if __name__ == "__main__":
    environment_kwargs = {
        "control_freq": 20,
        "controller": "OSC_POSE",
        "env_name": "PickPlaceBread",
        "hard_reset": False,
        "ignore_done": True,
        "reward_scale": 1.0,
        "robots": "Panda",
    }
    controller = environment_kwargs.pop("controller")
    controller_config = load_controller_config(default_controller=controller)
    env = suite.make(
        **environment_kwargs,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_object_obs=True,
        use_camera_obs=False,
        reward_shaping=True,
        controller_configs=controller_config,
    )
    env = MPEnv(NormalizedBoxEnv(GymWrapper(env)))
    # env.reset()
    # ik_controller_config = {
    #         "type": "IK_POSE",
    #         "ik_pos_limit": 0.02,
    #         "ik_ori_limit": 0.05,
    #         "interpolation": None,
    #         "ramp_ratio": 0.2,
    #         "converge_steps": 100,
    #     }
    # update_controller_config(env, ik_controller_config)
    # ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
    # ik_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)
    # qpos, qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
    num_steps = 10
    total = 0
    for s in range(num_steps):
        env.reset()
        ik_controller_config = {
                "type": "IK_POSE",
                "ik_pos_limit": 0.02,
                "ik_ori_limit": 0.05,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "converge_steps": 100,
            }
        update_controller_config(env, ik_controller_config)
        ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
        ik_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)
        qpos, qvel = env.sim.data.qpos.copy(), env.sim.data.qvel.copy()
        # env.sim.data.qpos[:] = qpos
        # env.sim.data.qvel[:] = qvel
        # env.sim.forward()
        # curr_pos = env._eef_xpos.copy()
        # action = np.random.uniform(-.25, .25, size=3)
        # target_pos = curr_pos + action[:3]
        target_z_pos = env.sim.data.body_xpos[env.obj_body_id[env.obj_to_use]][-1] + 0.03
        pose = np.array(
                [
                    0.2,
                    0.15,
                    target_z_pos,
                ]
            )
        target_pos = pose
        ori = env._eef_xquat.copy()
        for i in range(0):
            env.step(env.action_space.sample())
        # total += np.linalg.norm(env._eef_xpos - target_pos)
        error = set_robot_based_on_ee_pos(
            env,
            target_pos,
            ori,
            ik_ctrl,
            qpos,
            qvel,
            None,
            is_grasped=False,
        )
        cv2.imwrite(f'test_{s}.png', env.get_image())
        total += error
    print(f"Avg Distance to target: {total/num_steps})")
