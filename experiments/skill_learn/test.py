import cv2
import numpy as np

from rlkit.envs.primitives_make_env import make_env

if __name__ == "__main__":
    env_suite = "metaworld"
    env_name = "assembly-v2"
    env_kwargs = dict(
        use_image_obs=True,
        imwidth=64,
        imheight=64,
        reward_type="sparse",
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            unflatten_images=False,
            max_path_length=5,
        ),
        action_space_kwargs=dict(
            control_mode="primitives",
            action_scale=1,
            camera_settings={
                "distance": 0.38227044687537043,
                "lookat": [0.21052547, 0.32329237, 0.587819],
                "azimuth": 141.328125,
                "elevation": -53.203125160653144,
            },
        ),
    )
    env = make_env(
        env_suite,
        env_name,
        env_kwargs,
    )
    o = env.reset()
    cv2.imwrite("test0.png", o.reshape(3, 64, 64).transpose(1, 2, 0))

    a = np.zeros_like(env.action_space.sample())
    a[5] = 1
    a[env.num_primitives + env.primitive_name_to_action_idx["move_gripper"]] = -1
    o, r, d, i = env.step(a)
    cv2.imwrite("test1.png", o.reshape(3, 64, 64).transpose(1, 2, 0))

    a = np.zeros_like(env.action_space.sample())
    a[5] = 1
    a[env.num_primitives + env.primitive_name_to_action_idx["move_gripper"]] = 1
    o, r, d, i = env.step(a)
    cv2.imwrite("test2.png", o.reshape(3, 64, 64).transpose(1, 2, 0))
