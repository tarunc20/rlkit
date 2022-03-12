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
    env.reset()
    a = env.action_space.sample()
    env.step(a)
