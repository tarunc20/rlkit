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
                                        get_object_pose, set_object_pose) 
from rlkit.mprl.inverse_kinematics import qpos_from_site_pose
from rlkit.mprl import module
import cv2 

"""
Things to figure out 
- what are gripper indices
- what are object indices 
- 
"""

def get_camera_segmentation(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera segmentation matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        im (np.array): 2-channel segmented image where the first contains the
            geom types and the second contains the geom IDs
    """
    return sim.render(camera_name=camera_name, height=camera_height, width=camera_width, segmentation=True)[::-1]

def save_img(env, filename, resolution, camera_name):
    frame = env.render("rgb_array", camera_name=camera_name, resolution=resolution)
    # plt.imshow(frame)
    # plt.savefig(filename)
    # plt.close()
    img = Image.fromarray(frame, "RGB")
    img.show()
    img.save(filename)
    return 

# def get_object_pose(env):
#     pass 

def check_string(string, other_string):
    if string is None:
        return False
    return other_string in string 

def hammer_check_grasp(env):
    object_gripper_contact = False
    d = env.sim.data
    obj_string = get_object_string(env)
    left_gripper_contact = False
    right_gripper_contact = False
    object_in_contact_with_env = False
    pad_names = ["leftpad", "rightpad"]
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        body1 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom1]
        )
        body2 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom2]
        )
        if body1 == "leftpad" and body2 == "hammer" or body2 in "leftpad"and body1 == "hammer":
            left_gripper_contact = True 
        if body1 == "rightpad" and body2 == "hammer" or body2 in "rightpad"and body1 == "hammer":
            right_gripper_contact = True 
    return left_gripper_contact and right_gripper_contact 

def peg_insert_check_grasp(env):
    object_gripper_contact = False
    d = env.sim.data
    obj_string = get_object_string(env)
    left_gripper_contact = False
    right_gripper_contact = False
    object_in_contact_with_env = False
    pad_names = ["leftpad", "rightpad"]
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        body1 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom1]
        )
        body2 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom2]
        )
        if body1 == "leftpad" and body2 == "peg" or body2 in "leftpad"and body1 == "peg":
            left_gripper_contact = True 
        if body1 == "rightpad" and body2 == "peg" or body2 in "rightpad"and body1 == "peg":
            right_gripper_contact = True 
    return left_gripper_contact and right_gripper_contact 

def body_check_grasp(env):
    # get correct object name 
    if env.name == "hammer-v2":
        obj_name = "hammer"
    elif env.name == "peg-insert-side-v2":
        obj_name = "peg"
    elif env.name == "sweep-v2":
        obj_name = "obj"
    else:
        raise NotImplementedError
    object_gripper_contact = False
    d = env.sim.data
    obj_string = get_object_string(env)
    left_gripper_contact = False
    right_gripper_contact = False
    object_in_contact_with_env = False
    pad_names = ["leftpad", "rightpad"]
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        body1 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom1]
        )
        body2 = env.sim.model.body_id2name(
            env.sim.model.geom_bodyid[d.contact[coni].geom2]
        )
        try: # check to see if there needs to be any change 
            if body1 == "leftpad" and body2 == obj_name or body2 in "leftpad"and body1 == obj_name:
                left_gripper_contact = True 
            if body1 == "rightpad" and body2 == obj_name or body2 in "rightpad"and body1 == obj_name:
                right_gripper_contact = True 
        except:
            continue 
        # also check geom grasp
    return left_gripper_contact and right_gripper_contact 

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
    env.reset()
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects() + np.array([0.0, -0.0, 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    # close gripper step
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1.])))
    # go up step
    peg_position = env.sim.data.body_xpos[env.sim.model.body_name2id("peg")] + np.array([0.13, -0.047, 0.10])
    delta_ac = peg_position - env._eef_xpos 
    for _ in range(100): # used to be 150
        o, r, d, i = env.step(np.concatenate((delta_ac, [1])))
    save_img(env, "metaworld_assembly_wide.png", (960, 540), "corner")
    for _ in range(30):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [-1.0])))
    print(f"Info: {i}")
    #save_img(env, "drop.png") 

def hardcoded_hammer_policy():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="hammer-v2",
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
        mprl=False,
    )
    env = make_env(variant)
    env.name = 'hammer-v2'
    env.reset()
    # teleport to hammer 
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects()[:3] + np.array([0.09, -0.02, 0.05]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    # gripper action 
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1])))
    print(f"Checking grasp {hammer_check_grasp(env)}")
    #save_img(env, "hammer_grasp_state.png")
    # print gripper qpos and qvel
    print(f'odl qpos vel: {env.sim.data.qpos[7:9]}')
    # teleport again
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects()[3:] + np.array([-0.05, -0.20, 0.05]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        True,
    )
    save_img(env, "metaworld_hammer_wide.png", (960, 540), "corner3")
    for _ in range(50):
        o, r, d, i = env.step(np.concatenate((np.array([0., 0.5, 0.]), [1])))
    print(f"Info: {i}")
    #save_img(env, "hammer_raised_state.png")

def hardcoded_disassemble_policy():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="disassemble-v2",
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
        mprl=False,
    )
    env = make_env(variant)
    env.name = "disassemble-v2"
    env.reset()
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects() + np.array([0.0, -0.0, 0.0]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    # close gripper step
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1.])))
    # go up step
    for _ in range(25): # was originally 50
        o, r, d, i = env.step(np.concatenate((np.array([0., 0.,0.5]), [1.])))
    save_img(env, "metaworld_disassemble_wide.png", (960, 540), "corner")
    print(f"Info: {i}")

def hardcoded_stick_pull_policy():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="stick-pull-v2",
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
        mprl=False,
    )
    env = make_env(variant)
    env.name = "stick-pull-v2"
    env.reset()
    # teleport down 
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects()[:3] + np.array([-0.02, -0.0, 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    # grasp
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1.])))
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects()[3:] + np.array([-0.06, -0.02, 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        True,
    )
    #save_img(env, "start.png")
    # move forward 
    for _ in range(25): # used to be 75
        o, r, d, i = env.step(np.concatenate((
            np.array([0.5, -0.3, 0.]), [1]
        )))
    print(f"Info: {i}")
    save_img(env, "metaworld_stick_pull_wide.png", (960, 540), "corner2")

def hardcoded_peg_insert_policy():
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
                    action_scale=1 / 100, # used to be 1 / 100
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    env.name = "peg-insert-side-v2"
    env.reset()
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects() + np.array([0.08, -0.0, 0.03]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1])))
    set_robot_based_on_ee_pos(
        env, 
        env.sim.data.get_site_xpos("hole") + np.array([0.25, 0., 0.035]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        True,
    )
    save_img(env, "metaworld_peg_insert_square.png", (1024, 1024), "corner2")
    for _ in range(75):
        o, r, d, i = env.step(np.concatenate((
            np.array([-1.0, 0., -0.01]),
            [1]
        )))
    print(f"Info: {i}")

def hardcoded_sweep_policy():
    np.random.seed(0)
    variant = dict(
        env_suite="metaworld",
        expl_environment_kwargs=dict(
            env_name="sweep-v2",
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
                    action_scale=1 / 100, # used to be 1 / 100
                ),
            ),
        ),
        mprl=False,
    )
    env = make_env(variant)
    env.name = "sweep-v2"
    env.reset()
    for _ in range(100):
        o, r, d, i = env.step(np.concatenate((
            env._get_pos_objects() - env._eef_xpos, 
            [0]
        )))
    for _ in range(25):
        o, r, d, i = env.step(np.concatenate((np.zeros(3), [1])))
    print(body_check_grasp(env) or check_object_grasp(env))
    save_img(env, "start.png")

def hardcoded_door_open_policy():
    return 

def check_segmentation():
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
    env.reset()
    seg = np.flipud(get_camera_segmentation(env.sim, "corner", 540, 960)[:, :, -1])
    plt.imshow(seg)
    plt.savefig("test.png")

def color_agent(env, agent_geom_ids):
    pass

def change_color_tests():
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
    env.reset()
    print(env.sim.model.body_names)
    # set visualized indicator
    # reset visualized indicator 
    # first test - try to set robot to cyan
    ALL_ROBOT_BODIES = [
        'right_l1', 'right_l2', 'right_l3', 'right_l4', 'right_arm_itb',
        'right_l5', 'right_hand_camera', 'right_wrist', 'right_l6', 'right_hand', 'hand', 'rightclaw', 
        'rightpad', 'leftclaw', 'leftpad', 'right_l4_2', 'right_l2_2', 'right_l1_2',
    ]
    body_ids = [env.sim.model.body_name2id(body) for body in ALL_ROBOT_BODIES]
    geom_ids = []
    for geom_id, body_id in enumerate(env.sim.model.geom_bodyid):
        if body_id in body_ids:
            geom_ids.append(geom_id)
    # set color 
    for idx in geom_ids:
        color = env.sim.model.geom_rgba[idx]
        color = np.array([0.1, 0.3, 0.7, 1.0])
        env.sim.model.geom_rgba[idx] = color
    env.sim.forward()
    frame = env.render("rgb_array", camera_name="corner", resolution=(960, 540))
    save_img(env, "color.png", resolution=(960, 540), camera_name="corner")

    # set robot pos 
    set_robot_based_on_ee_pos(
        env, 
        env._get_pos_objects() + np.array([0.0, -0.0, 0.02]),
        env._eef_xquat,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(),
        False,
    )
    act_img = env.render("rgb_array", camera_name="corner", resolution=(960, 540))
    # get segmentation 
    seg = np.flipud(get_camera_segmentation(env.sim, "corner", 540, 960)[:, :, -1])
    # mask out image 
    seg = seg * np.isin(seg, geom_ids)
    plt.imshow(seg)
    plt.savefig("segmentation.png")
    plt.close()

    new_frame = env.render("rgb_array", camera_name="corner", resolution=(960, 540))
    # print(seg.shape)
    new_frame *= np.expand_dims(np.isin(seg, geom_ids).astype(np.uint8), axis=-1)
    print(new_frame.shape)
    plt.imshow(new_frame)
    plt.savefig("new_frame.png")
    # do alpha blending
    import cv2
    dest = cv2.addWeighted(frame, 0.75, act_img, 0.25, 0.0)
    plt.close()
    plt.imshow(dest)
    plt.savefig("img.png")
    #cv2.imwrite('img.png', dest)
    




if __name__ == "__main__":
    change_color_tests()
    #check_segmentation()
    #hardcoded_peg_insert_policy()