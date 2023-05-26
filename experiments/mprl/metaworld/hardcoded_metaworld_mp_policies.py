import time

import numpy as np
import mujoco_py
from matplotlib import pyplot as plt
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import (
    euler2mat,
    mat2pose,
    mat2quat,
    pose2mat,
    quat2mat,
    quat_conjugate,
    quat_multiply,
)
from PIL import Image
from rlkit.mprl.experiment import make_env
from rlkit.mprl.mp_env_metaworld import (get_object_pos, set_robot_based_on_ee_pos, 
                                        check_object_grasp, get_object_string, 
                                        gripper_contact, check_robot_string,
                                        get_object_pose, set_object_pose, mp_to_point) 
from rlkit.mprl.inverse_kinematics import qpos_from_site_pose
from rlkit.mprl import module
import cv2 

def save_img(env, filename):
    import matplotlib.pyplot as plt 
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig(filename)
    plt.close()

def hardcoded_assembly_policy():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="assembly-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=1024,
                imheight=1024,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    env.name = "assembly-v2"
    o = env.reset()
    save_img(env, "start.png") 
    env.mp_bounds_low = (-3, -3, -3)
    env.mp_bounds_high = (3, 3, 3)
    mp_to_point(
        env,
        None, 
        None,
        env._get_pos_objects() + np.array([0.0, -0.0, 0.05]),
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        ignore_object_collision=False,
    )
    save_img(env, "start.png") 

if __name__ == "__main__":
    hardcoded_assembly_policy()