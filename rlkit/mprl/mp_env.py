import cv2
import numpy as np
from robosuite.controllers import controller_factory
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import quat2mat
from rlkit.core import logger
from rlkit.envs.proxy_env import ProxyEnv
import matplotlib
try:
    # graph-tool and py-OMPL have some minor issues coexisting with each other.  Both modules
    # define conversions to C++ STL containers (i.e. std::vector), and the module that is imported
    # first will have its conversions used.  Order doesn't seem to matter on Linux,
    # but on Apple, graph_tool will not be imported properly if OMPL comes first.
    import graph_tool.all as gt

    graphtool = True
except ImportError:
    print("Failed to import graph-tool.  PlannerData will not be analyzed or plotted")
    graphtool = False

try:
    from ompl import base as ob
    from ompl import util as ou
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import util as ou
    from ompl import geometric as og

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def set_robot_based_on_ee_pos(
    env, pos, quat, ctrl, qpos, qvel, ee_to_object_translation, is_grasped=False
):
    env.sim.data.qpos[:] = qpos
    env.sim.data.qvel[:] = qvel
    env.sim.forward()

    ctrl.sync_state()
    desired_rot = quat2mat(quat)
    cur_rot = quat2mat(env._eef_xquat)
    rot_diff = desired_rot @ np.linalg.inv(cur_rot)
    joint_pos = ctrl.joint_positions_for_eef_command(pos - env._eef_xpos, rot_diff)
    env.robots[0].set_robot_joint_positions(joint_pos)
    if is_grasped:
        # TODO: we should also match relative orientation of object wrt ee
        if env.name.endswith("Lift"):
            env.sim.data.qpos[9:12] = env._eef_xpos + ee_to_object_translation
        elif env.name.endswith("PickPlaceBread"):
            env.sim.data.qpos[16:19] = env._eef_xpos + ee_to_object_translation
        env.sim.forward()
    return np.linalg.norm(env._eef_xpos - pos)


def check_robot_string(string):
    return string.startswith("robot") or string.startswith("gripper")


def check_robot_collision(env, ignore_object_collision):
    if env.name.endswith("Lift"):
        obj_string = "cube"
    elif env.name.endswith("PickPlaceBread"):
        obj_string = "Bread"
    d = env.sim.data
    for coni in range(d.ncon):
        con1 = env.sim.model.geom_id2name(d.contact[coni].geom1)
        con2 = env.sim.model.geom_id2name(d.contact[coni].geom2)
        if con1 is not None and con2 is not None:
            if check_robot_string(con1) ^ check_robot_string(con2):
                if (
                    con1.startswith(obj_string)
                    or con2.startswith(obj_string)
                    and ignore_object_collision
                ):
                    continue
                return True
            if ignore_object_collision:
                if con1.startswith(obj_string) or con2.startswith(obj_string):
                    # if the robot and the object collide, then we can ignore the collision
                    # if we are supposed to be "ignoring object collisions" then we assume the
                    # robot is "joined" to the object. so if the object collides with any non-robot
                    # object, then we should call that a collision
                    return True
    return False


def backtracking_search_from_goal(
    env,
    ik_ctrl,
    ignore_object_collision,
    start_pos,
    goal_pos,
    ori,
    qpos,
    qvel,
    movement_fraction=0.001,
    max_iters=1000,
    ee_to_object_translation=None,
    is_grasped=False,
):
    # only search over the xyz position, orientation should be the same as commanded
    curr_pos = goal_pos.copy()
    set_robot_based_on_ee_pos(
        env, curr_pos, ori, ik_ctrl, qpos, qvel, ee_to_object_translation, is_grasped
    )
    collision = check_robot_collision(env, ignore_object_collision)
    iters = 0
    while collision and iters < max_iters:
        curr_pos = curr_pos - movement_fraction * (goal_pos - start_pos)
        set_robot_based_on_ee_pos(
            env,
            curr_pos,
            ori,
            ik_ctrl,
            qpos,
            qvel,
            ee_to_object_translation,
            is_grasped,
        )
        collision = check_robot_collision(env, ignore_object_collision)
        iters += 1
    if collision:
        return start_pos  # assumption is this is always valid!
    else:
        return curr_pos


def update_controller_config(env, controller_config):
    controller_config["robot_name"] = env.robots[0].name
    controller_config["sim"] = env.robots[0].sim
    controller_config["eef_name"] = env.robots[0].gripper.important_sites["grip_site"]
    controller_config["eef_rot_offset"] = env.robots[0].eef_rot_offset
    controller_config["joint_indexes"] = {
        "joints": env.robots[0].joint_indexes,
        "qpos": env.robots[0]._ref_joint_pos_indexes,
        "qvel": env.robots[0]._ref_joint_vel_indexes,
    }
    controller_config["actuator_range"] = env.robots[0].torque_limits
    controller_config["policy_freq"] = env.robots[0].control_freq
    controller_config["ndim"] = len(env.robots[0].robot_joints)


def apply_controller(controller, action, robot, policy_step):
    gripper_action = None
    if robot.has_gripper:
        gripper_action = action[
            controller.control_dim :
        ]  # all indexes past controller dimension indexes
        arm_action = action[: controller.control_dim]
    else:
        arm_action = action

    # Update the controller goal if this is a new policy step
    if policy_step:
        controller.set_goal(arm_action)

    # Now run the controller for a step
    torques = controller.run_controller()

    # Clip the torques
    low, high = robot.torque_limits
    torques = np.clip(torques, low, high)

    # Get gripper action, if applicable
    if robot.has_gripper:
        robot.grip_action(gripper=robot.gripper, gripper_action=gripper_action)

    # Apply joint torque control
    robot.sim.data.ctrl[robot._ref_joint_actuator_indexes] = torques


def useGraphTool(pd):
    # Extract the graphml representation of the planner data
    graphml = pd.printGraphML()
    f = open("graph.graphml", "w")
    f.write(graphml)
    f.close()

    # Load the graphml data using graph-tool
    graph = gt.load_graph("graph.graphml", fmt="xml")
    edgeweights = graph.edge_properties["weight"]

    # Write some interesting statistics
    avgdeg, stddevdeg = gt.vertex_average(graph, "total")
    avgwt, stddevwt = gt.edge_average(graph, edgeweights)

    print("---- PLANNER DATA STATISTICS ----")
    print(
        str(graph.num_vertices()) + " vertices and " + str(graph.num_edges()) + " edges"
    )
    print(
        "Average vertex degree (in+out) = "
        + str(avgdeg)
        + "  St. Dev = "
        + str(stddevdeg)
    )
    print("Average edge weight = " + str(avgwt) + "  St. Dev = " + str(stddevwt))

    _, hist = gt.label_components(graph)
    print("Strongly connected components: " + str(len(hist)))

    # Make the graph undirected (for weak components, and a simpler drawing)
    graph.set_directed(False)
    _, hist = gt.label_components(graph)
    print("Weakly connected components: " + str(len(hist)))

    # Plotting the graph
    gt.remove_parallel_edges(graph)  # Removing any superfluous edges

    edgeweights = graph.edge_properties["weight"]
    colorprops = graph.new_vertex_property("string")
    vertexsize = graph.new_vertex_property("double")

    start = -1
    goal = -1

    for v in range(graph.num_vertices()):

        # Color and size vertices by type: start, goal, other
        if pd.isStartVertex(v):
            start = v
            colorprops[graph.vertex(v)] = "cyan"
            vertexsize[graph.vertex(v)] = 10
        elif pd.isGoalVertex(v):
            goal = v
            colorprops[graph.vertex(v)] = "green"
            vertexsize[graph.vertex(v)] = 10
        else:
            colorprops[graph.vertex(v)] = "yellow"
            vertexsize[graph.vertex(v)] = 5

    # default edge color is black with size 0.5:
    edgecolor = graph.new_edge_property("string")
    edgesize = graph.new_edge_property("double")
    for e in graph.edges():
        edgecolor[e] = "black"
        edgesize[e] = 0.5

    # using A* to find shortest path in planner data
    if start != -1 and goal != -1:
        _, pred = gt.astar_search(graph, graph.vertex(start), edgeweights)

        # Color edges along shortest path red with size 3.0
        v = graph.vertex(goal)
        while v != graph.vertex(start):
            p = graph.vertex(pred[v])
            for e in p.out_edges():
                if e.target() == v:
                    edgecolor[e] = "red"
                    edgesize[e] = 2.0
            v = p

    # Writing graph to file:
    # pos indicates the desired vertex positions, and pin=True says that we
    # really REALLY want the vertices at those positions
    gt.graph_draw(
        graph,
        vertex_size=vertexsize,
        vertex_fill_color=colorprops,
        edge_pen_width=edgesize,
        edge_color=edgecolor,
        output="graph.png",
    )
    print("\nGraph written to graph.png")


def mp_to_point(
    env,
    ik_controller_config,
    osc_controller_config,
    pos,
    grasp=False,
    ignore_object_collision=False,
    planning_time=1,
    get_intermediate_frames=False,
):
    og_goal_pos = pos.copy()
    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()
    update_controller_config(env, ik_controller_config)
    ik_ctrl = controller_factory("IK_POSE", ik_controller_config)
    ik_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)

    og_eef_xpos = env._eef_xpos.copy()
    og_eef_xquat = env._eef_xquat.copy()
    if env.name.endswith("Lift"):
        ee_to_object_translation = (
            env.sim.data.body_xpos[env.cube_body_id] - og_eef_xpos
        )
    else:
        ee_to_object_translation = (
            env.sim.data.body_xpos[env.obj_body_id[env.obj_to_use]] - og_eef_xpos
        )
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x, y, z = [], [], []
    col_x, col_y, col_z = [], [], []
    log_dir = logger.get_snapshot_dir()

    def isStateValid(state):
        pos = np.array([state.getX(), state.getY(), state.getZ()])
        quat = np.array(
            [
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
        )
        if all(pos == og_eef_xpos) and all(quat == og_eef_xquat):
            # start state is always valid.
            return True
        else:
            # TODO; if it was grasping before ik and not after automatically set to invalid
            set_robot_based_on_ee_pos(
                env, pos, quat, ik_ctrl, qpos, qvel, ee_to_object_translation, grasp
            )
            valid = not check_robot_collision(
                env, ignore_object_collision=ignore_object_collision
            )

            if valid:
                x.append(state.getX())
                y.append(state.getY())
                z.append(state.getZ())
            else:
                col_x.append(state.getX())
                col_y.append(state.getY())
                col_z.append(state.getZ())
            return valid

    # create an SE3 state space
    space = ob.SE3StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)

    # compare bounds to start state
    bounds_low = env.mp_bounds_low
    bounds_high = env.mp_bounds_high

    bounds_low = np.minimum(env.mp_bounds_low, og_eef_xpos)
    bounds_high = np.maximum(env.mp_bounds_high, og_eef_xpos)
    pos[:3] = np.clip(pos[:3], bounds_low, bounds_high)

    bounds.setLow(0, bounds_low[0])
    bounds.setLow(1, bounds_low[1])
    bounds.setLow(2, bounds_low[2])
    bounds.setHigh(0, bounds_high[0])
    bounds.setHigh(1, bounds_high[1])
    bounds.setHigh(2, bounds_high[2])
    space.setBounds(bounds)

    # construct an instance of space information from this state space
    si = ob.SpaceInformation(space)
    # set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    # create a random start state
    start = ob.State(space)
    start().setXYZ(*og_eef_xpos)
    start().rotation().x = og_eef_xquat[0]
    start().rotation().y = og_eef_xquat[1]
    start().rotation().z = og_eef_xquat[2]
    start().rotation().w = og_eef_xquat[3]
    # create a random goal state
    goal = ob.State(space)
    goal().setXYZ(*pos[:3])
    goal().rotation().x = pos[3]
    goal().rotation().y = pos[4]
    goal().rotation().z = pos[5]
    goal().rotation().w = pos[6]
    goal_valid = isStateValid(goal())
    goal_error = set_robot_based_on_ee_pos(
        env, pos[:3], og_eef_xquat, ik_ctrl, qpos, qvel, ee_to_object_translation, grasp
    )
    print(f"Goal Validity: {goal_valid}")
    print(f"Goal Error {goal_error}")

    if not goal_valid:
        pos = backtracking_search_from_goal(
            env,
            ik_ctrl,
            ignore_object_collision,
            og_eef_xpos,
            pos[:3],
            og_eef_xquat,
            qpos,
            qvel,
            ee_to_object_translation=ee_to_object_translation,
            is_grasped=grasp,
        )
        goal = ob.State(space)
        goal().setXYZ(*pos)
        goal().rotation().x = og_eef_xquat[0]
        goal().rotation().y = og_eef_xquat[1]
        goal().rotation().z = og_eef_xquat[2]
        goal().rotation().w = og_eef_xquat[3]
        goal_error = set_robot_based_on_ee_pos(
            env,
            pos[:3],
            og_eef_xquat,
            ik_ctrl,
            qpos,
            qvel,
            ee_to_object_translation,
            grasp,
        )
        goal_valid = isStateValid(goal())
        print(f"Updated Goal Validity: {goal_valid}")
        print(f"Goal Error {goal_error}")
        print(pos)
    # if grasp:
    #     cv2.imwrite("/home/mdalal/research/mprl/rlkit/test.png", env.get_image())
    #     assert (
    #         env.reward(None) == 1.0
    #     ), f"goal state should have reward 1.0. xpos:{pos[:3]} xquat:{pos[3:]}"
    #     print("Goal state has reward 1.0")
    # create a problem instance
    pdef = ob.ProblemDefinition(si)
    # set the start and goal states
    pdef.setStartAndGoalStates(start, goal)
    # create a planner for the defined space
    planner = og.RRTConnect(si)
    # set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)
    # perform setup steps for the planner
    planner.setup()
    # attempt to solve the problem within planning_time seconds of planning time
    solved = planner.solve(planning_time)

    if get_intermediate_frames:
        ax.scatter3D(x, y, z, c="g",)
        ax.scatter3D(col_x, col_y, col_z, c="r",)
        ax.scatter3D(goal().getX(), goal().getY(), goal().getZ(), c="b",)
        ax.scatter3D(start().getX(), start().getY(), start().getZ(), c="y",)
        log_dir = logger.get_snapshot_dir()
        set_robot_based_on_ee_pos(
                env,
                og_eef_xpos,
                og_eef_xquat,
                ik_ctrl,
                qpos,
                qvel,
                ee_to_object_translation,
                grasp,
            )
        cv2.imwrite(f"{log_dir}/start_{env.num_steps}.png", env.get_image())
        set_robot_based_on_ee_pos(
                env,
                pos[:3],
                og_eef_xquat,
                ik_ctrl,
                qpos,
                qvel,
                ee_to_object_translation,
                grasp,
            )
        cv2.imwrite(f"{log_dir}/goal_{env.num_steps}.png", env.get_image())
        plt.savefig(f"{log_dir}/plot_{env.num_steps}.png")
    intermediate_frames = []
    if solved:
        # Extracting planner data from most recent solve attempt
        pd = ob.PlannerData(si)
        planner.getPlannerData(pd)

        # Computing weights of all edges based on state space distance
        pd.computeEdgeWeights()

        if graphtool:
            useGraphTool(pd)

        path = pdef.getSolutionPath()
        og.PathSimplifier(si).simplify(path, 1)
        if get_intermediate_frames:
            ax.scatter3D(x, y, z, c="g",)
            ax.scatter3D(col_x, col_y, col_z, c="r",)
            ax.scatter3D(goal().getX(), goal().getY(), goal().getZ(), c="b",)
            ax.scatter3D(start().getX(), start().getY(), start().getZ(), c="y",)
            log_dir = logger.get_snapshot_dir()
            plt.savefig(f"{log_dir}/plot_post_shorten_{env.num_steps}.png")
        converted_path = []
        for s, state in enumerate(path.getStates()):
            new_state = [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().x,
                state.rotation().y,
                state.rotation().z,
                state.rotation().w,
            ]
            if env.update_with_true_state:
                # get actual state that we used for collision checking on
                set_robot_based_on_ee_pos(
                    env,
                    new_state[:3],
                    new_state[3:],
                    ik_ctrl,
                    qpos,
                    qvel,
                    ee_to_object_translation,
                    grasp,
                )
                new_state = np.concatenate((env._eef_xpos, env._eef_xquat))
            else:
                new_state = np.array(new_state)
            converted_path.append(new_state)
        # reset env to original qpos/qvel
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos.copy()
        env.sim.data.qvel[:] = qvel.copy()
        env.sim.forward()

        update_controller_config(env, osc_controller_config)
        osc_ctrl = controller_factory("OSC_POSE", osc_controller_config)
        osc_ctrl.update_base_pose(env.robots[0].base_pos, env.robots[0].base_ori)
        osc_ctrl.reset_goal()
        for state in converted_path:
            desired_rot = quat2mat(state[3:])
            for _ in range(100):
                current_rot = quat2mat(env._eef_xquat)
                rot_delta = orientation_error(desired_rot, current_rot)
                pos_delta = state[:3] - env._eef_xpos
                if grasp:
                    grip_ctrl = 1
                else:
                    grip_ctrl = -1
                action = np.concatenate((pos_delta, rot_delta, [grip_ctrl]))
                if np.linalg.norm(action[:-1]) < 1e-5:
                    break
                policy_step = True
                for i in range(int(env.control_timestep / env.model_timestep)):
                    env.sim.forward()
                    apply_controller(osc_ctrl, action, env.robots[0], policy_step)
                    env.sim.step()
                    env._update_observables()
                    policy_step = False
                if hasattr(env, "num_steps"):
                    env.num_steps += 1
                if get_intermediate_frames:
                    intermediate_frames.append(env.get_image())
                # print(env.reward(None), env.check_grasp(), env._check_success())
        env.mp_mse = (
            np.linalg.norm(state - np.concatenate((env._eef_xpos, env._eef_xquat))) ** 2
        )
        env.goal_error = goal_error
    else:
        env._wrapped_env.reset()
        env.sim.data.qpos[:] = qpos.copy()
        env.sim.data.qvel[:] = qvel.copy()
        env.sim.forward()
        env.mp_mse = 0
        env.goal_error = 0
        env.num_failed_solves += 1
        # print(og_goal_pos)
        # exit()
    env.intermediate_frames = intermediate_frames
    return env._get_observations()


class MPEnv(ProxyEnv):
    def __init__(
        self,
        env,
        vertical_displacement,
        teleport_position=True,
        planning_time=1,
        plan_to_learned_goals=False,
        execute_hardcoded_policy_to_goal=False,
        learn_residual=False,
        mp_bounds_low=None,
        mp_bounds_high=None,
        update_with_true_state=False,
    ):
        super().__init__(env)
        for (cam_name, cam_w, cam_h, cam_d) in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
            # self.camera_segmentations,
        ):

            # Add cameras associated to our arrays
            cam_sensors, cam_sensor_names = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                # cam_segs=cam_segs,
                modality="image",
            )
            self.cam_sensor = cam_sensors
        self.num_steps = 0
        self.vertical_displacement = vertical_displacement
        self.teleport_position = teleport_position
        self.planning_time = planning_time
        self.plan_to_learned_goals = plan_to_learned_goals
        self.execute_hardcoded_policy_to_goal = execute_hardcoded_policy_to_goal
        self.learn_residual = learn_residual
        self.mp_bounds_low = mp_bounds_low
        self.mp_bounds_high = mp_bounds_high
        self.update_with_true_state = update_with_true_state

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def get_init_target_pos(self):
        if self.name.endswith("Lift"):
            pos = self.sim.data.body_xpos[self.cube_body_id]
        elif self.name.endswith("PickPlaceBread"):
            pos = self.sim.data.body_xpos[self.obj_body_id[self.obj_to_use]]
            self.target_z_pos = (
                self.sim.data.body_xpos[self.obj_body_id[self.obj_to_use]][-1] + 0.025
            )
        pos += np.array([0, 0, self.vertical_displacement])
        return pos

    def reset(self, get_intermediate_frames=False, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self.ik_controller_config = {
            "type": "IK_POSE",
            "ik_pos_limit": 0.02,
            "ik_ori_limit": 0.05,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        self.osc_controller_config = {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "output_min": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
        self.ep_step_ctr = 0
        self.num_failed_solves = 0
        self.reset_ori = self._eef_xquat.copy()
        if not self.plan_to_learned_goals:
            if self.teleport_position:
                update_controller_config(self, self.ik_controller_config)
                ik_ctrl = controller_factory("IK_POSE", self.ik_controller_config)
                ik_ctrl.update_base_pose(
                    self.robots[0].base_pos, self.robots[0].base_ori
                )
                pos = self.get_init_target_pos()
                set_robot_based_on_ee_pos(
                    self,
                    pos,
                    self._eef_xquat,
                    ik_ctrl,
                    self.sim.data.qpos,
                    self.sim.data.qvel,
                )
                obs, reward, done, info = self._wrapped_env.step(np.zeros(7))
                self.num_steps += 100
            else:
                pos = self.get_init_target_pos()
                pos = np.concatenate((pos, self.reset_ori))
                obs = mp_to_point(
                    self,
                    self.ik_controller_config,
                    self.osc_controller_config,
                    pos,
                    grasp=False,
                    planning_time=self.planning_time,
                    get_intermediate_frames=get_intermediate_frames,
                )
                obs = self._flatten_obs(obs)
        return obs

    def check_grasp(
        self,
    ):
        if self.name.endswith("Lift"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )
        elif self.name.endswith("PickPlaceBread"):
            is_grasped = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.objects[self.object_id],
            )
        return is_grasped

    def get_target_pos(
        self,
    ):
        if self.name.endswith("Lift"):
            pose = np.array([0, 0, 0.05]) + self._eef_xpos
        elif self.name.endswith("PickPlaceBread"):
            pose = np.array(
                [
                    0.25,
                    0.1,
                    self.target_z_pos,
                ]
            )
        return pose

    def clamp_planner_action_mp_space_bounds(self, action):
        # action[0] = (
        #     action[0] * (self.mp_bounds_high[0] - self.mp_bounds_low[0]) / 2
        #     + self.mp_bounds_low[0] / 2
        #     + self.mp_bounds_high[0] / 2
        # )
        # action[1] = (
        #     action[1] * (self.mp_bounds_high[1] - self.mp_bounds_low[1]) / 2
        #     + self.mp_bounds_low[1] / 2
        #     + self.mp_bounds_high[1] / 2
        # )
        # action[2] = (
        #     action[2] * (self.mp_bounds_high[2] - self.mp_bounds_low[2]) / 2
        #     + self.mp_bounds_low[2] / 2
        #     + self.mp_bounds_high[2] / 2
        # )
        # assert (action[:3] >= self.mp_bounds_low).all() and (
        #     action[:3] <= self.mp_bounds_high
        # ).all(), action
        return action

    def step(self, action, get_intermediate_frames=False):
        if self.plan_to_learned_goals:
            if self.ep_step_ctr == 0 or self.ep_step_ctr == self.horizon + 1:
                if self.learn_residual:
                    # TODO: don't add quaternion, use quaternion multiplication rule
                    pos = action + np.concatenate(
                        (self.get_init_target_pos(), self._eef_xquat)
                    )
                else:
                    action = self.clamp_planner_action_mp_space_bounds(action)
                    pos = action
                is_grasped = self.check_grasp()
                o = mp_to_point(
                    self,
                    self.ik_controller_config,
                    self.osc_controller_config,
                    np.concatenate((pos[:3], self.reset_ori)).astype(np.float64),
                    grasp=is_grasped,
                    ignore_object_collision=is_grasped,
                    planning_time=self.planning_time,
                    get_intermediate_frames=get_intermediate_frames,
                )
                o = self._flatten_obs(o)
                r = self.reward(action)
                i = {}
                d = False
            else:
                o, r, d, i = self._wrapped_env.step(action)
                self.num_steps += 1
            self.ep_step_ctr += 1
        else:
            o, r, d, i = self._wrapped_env.step(action)
            self.num_steps += 1
            self.ep_step_ctr += 1
            if self.ep_step_ctr == self.horizon:
                is_grasped = self.check_grasp()
                target_pos = self.get_target_pos()
                if self.teleport_position:
                    for _ in range(50):
                        self._wrapped_env.step(
                            np.concatenate((target_pos - self._eef_xpos, [0, 0, 0, 1]))
                        )
                        self.num_steps += 1
                else:
                    mp_to_point(
                        self,
                        self.ik_controller_config,
                        self.osc_controller_config,
                        np.concatenate((target_pos, self.reset_ori)),
                        grasp=is_grasped,
                        ignore_object_collision=is_grasped,
                        planning_time=self.planning_time,
                        get_intermediate_frames=get_intermediate_frames,
                    )
        i["success"] = float(self._check_success())
        i["grasped"] = float(self.check_grasp())
        i["num_steps"] = self.num_steps
        if not self.teleport_position:
            i["mp_mse"] = self.mp_mse
            i["num_failed_solves"] = self.num_failed_solves
            i["goal_error"] = self.goal_error
        return o, r, d, i


class RobosuiteEnv(ProxyEnv):
    def __init__(self, env):
        super().__init__(env)
        for (cam_name, cam_w, cam_h, cam_d, cam_segs) in zip(
            self.camera_names,
            self.camera_widths,
            self.camera_heights,
            self.camera_depths,
            self.camera_segmentations,
        ):

            # Add cameras associated to our arrays
            cam_sensors, cam_sensor_names = self._create_camera_sensors(
                cam_name,
                cam_w=cam_w,
                cam_h=cam_h,
                cam_d=cam_d,
                cam_segs=cam_segs,
                modality="image",
            )
            self.cam_sensor = cam_sensors
        self.num_steps = 0

    def get_image(self):
        im = self.cam_sensor[0](None)
        im = cv2.flip(im[:, :, ::-1], 0)
        return im

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        o, r, d, i = super().step(action)
        self.num_steps += 1
        i["success"] = float(self._check_success())
        i["grasped"] = float(
            self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cube,
            )
        )
        i["num_steps"] = self.num_steps
        return o, r, d, i
