"""
Test script for camera transforms. This test will read the ground-truth 
object state in the Lift environment, transform it into a pixel location
in the camera frame, then transform it back to the world frame, and assert
that the values are close.
"""
import random

import numpy as np

import robosuite
import robosuite.utils.camera_utils as CU
from robosuite.controllers import load_controller_config


def test_camera_transforms():
    # set seeds
    random.seed(0)
    np.random.seed(0)

    camera_name = "agentview"
    camera_height = 480
    camera_width = 640
    env = robosuite.make(
        "PickPlaceCereal",
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_depths=[True],
        camera_heights=[camera_height],
        camera_widths=[camera_width],
        reward_shaping=True,
        control_freq=20,
    )
    obs_dict = env.reset()
    sim = env.sim

    # ground-truth object position
    obj_pos = obs_dict["object-state"][:3]

    # camera frame
    image = obs_dict["{}_image".format(camera_name)][::-1]

    # unnormalized depth map
    depth_map = obs_dict["{}_depth".format(camera_name)][::-1]
    segmentation_map = CU.get_camera_segmentation(camera_name=camera_name, camera_width=camera_width, camera_height=camera_height, sim=sim)
    # visualize segmentation map
    object_string = 'Cereal'
    geom_ids = np.unique(segmentation_map[:, :, 1])
    object_id = None
    for geom_id in geom_ids:
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name is None or geom_name.startswith('Visual'):
            continue
        if object_string in sim.model.geom_id2name(geom_id):
            object_id = geom_id
            break
    cube_mask = segmentation_map[:, :, 1] == object_id
    depth_map = CU.get_real_depth_map(sim=env.sim, depth_map=depth_map)

    # get camera matrices
    world_to_camera = CU.get_camera_transform_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
    )
    camera_to_world = np.linalg.inv(world_to_camera)

    obj_pixels = np.argwhere(cube_mask)
    # transform from camera pixel back to world position
    # can we do this batched somehow...
    obj_poses = []
    for obj_pixel in obj_pixels:
        estimated_obj_pos = CU.transform_from_pixels_to_world(
            pixels=obj_pixel,
            depth_map=depth_map,
            camera_to_world_transform=camera_to_world,
        )
        obj_poses.append(estimated_obj_pos)
    estimated_obj_pos = np.mean(obj_poses, axis=0)

    z_err = np.abs(obj_pos[2] - estimated_obj_pos[2])

    print("obj pos: {}".format(obj_pos))
    print("estimated obj pos: {}".format(estimated_obj_pos))
    print("z err: {}".format(z_err))
    print("err: {}".format(np.linalg.norm(obj_pos - estimated_obj_pos)))

    env.close()


if __name__ == "__main__":

    test_camera_transforms()
