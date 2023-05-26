import time
import mujoco_py
import numpy as np
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

from rlkit.mprl.experiment import make_env
from rlkit.mprl.mp_env_metaworld import (get_object_pos, set_robot_based_on_ee_pos, 
                                        check_object_grasp, get_object_string, 
                                        gripper_contact, check_robot_string,
                                        get_object_pose, set_object_pose,
                                        body_check_grasp) 
from rlkit.mprl.inverse_kinematics import qpos_from_site_pose
from rlkit.mprl import module
import cv2 

def save_img(env, filename):
    frame = env.render("rgb_array", camera_name="corner2", resolution=(1024, 1024))
    plt.imshow(frame)
    plt.savefig(filename)
    plt.close()
    return 

def main():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="peg-insert-side-v2",
            env_kwargs=dict(
                reward_type="dense",
                usage_kwargs=dict(
                    use_dm_backend=False,
                    use_raw_action_wrappers=False,
                    use_image_obs=False,
                    max_path_length=500,
                    unflatten_images=False,
                ),
                imwidth=480,
                imheight=480,
                action_space_kwargs=dict(
                    control_mode="end_effector",
                    action_scale=1 / 100,
                ),
            ),
        ),
        mprl=True,
        mp_env_kwargs=dict()
    )
    env = make_env(variant)
    env.reset()
    while not body_check_grasp(env):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1])))
    for _ in range(125):
        o, r, d, i = env.step(np.concatenate((np.array([-0.5, 0.0, 0.]), [1])))
    print(f"Info: {i}")
    save_img(env._wrapped_env, "start.png")
    # # test hardcoded policy 
    # while not body_check_grasp(env):
    #     o, r, d, i = env.step(np.concatenate((np.zeros(3), [1])))
    # for _ in range(50):
    #     o, r, d, i = env.step(np.concatenate((np.array([0., 0.5, 0.]), [1])))
    # save_img(env._wrapped_env, "start.png")

if __name__ == "__main__":
    main()