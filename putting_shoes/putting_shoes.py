import argparse
import time

import omnigibson as og
import numpy as np

from omnigibson.robots import Fetch
from omnigibson.scenes import Scene
from omnigibson.utils.control_utils import IKSolver
from omnigibson.macros import gm 
from omnigibson.object_states import OnTop
from omnigibson.objects import PrimitiveObject
from omnigibson.sensors import VisionSensor
# from omnigibson.utils.usd_utils import SemanticsAPI
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController
from scipy.spatial.transform import Rotation as R

import os
import pickle
import shutil
import yaml
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import carb
import omni

# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True
# gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

# Josiah: Define global action to update periodically
action = None
gripper = True
step = 0

IMG_HEIGHT = 720
IMG_WIDTH = 720


def get_eef_orientation(z_rot_degree=0):
    # Define your initial orientation quaternion
    init_orientation = [0.5, 0.5, -0.5, 0.5]

    # Define the rotation around Z axis
    degrees = z_rot_degree
    radians = np.deg2rad(degrees)

    # Compute the quaternion for this rotation
    rot_quat = [np.cos(radians/2), 0, 0, np.sin(radians/2)]

    # Use scipy's Rotation class to handle the quaternion multiplication (which is equivalent to applying the rotations)
    # Note that scipy's Rotation uses quaternions in the form [x, y, z, w], not [w, x, y, z], so we have to re-arrange.
    r_init = R.from_quat(init_orientation)
    r_rot = R.from_quat(rot_quat)

    # Combine the rotations, apply the Z-rotation first
    r_final = r_init * r_rot

    # Convert back to the [w, x, y, z] form
    # final_quaternion = [r_final.as_quat()[3], r_final.as_quat()[0], r_final.as_quat()[1], r_final.as_quat()[2]]

    # return final_quaternion
    return r_final.as_quat()


def save_image(path, image_array):
    image = Image.fromarray(image_array)
    image.save(path)


def camera_save_obs(data_dir, camera_list):
    global step
    """
    Save rgb and depth_linear modality
    """
    for idx in range(len(camera_list)):
        camera = camera_list[idx]

        camera_name = f"camera_{idx}"
        camera_save_dir = os.path.join(data_dir, camera_name)
        # os.makedirs(camera_save_dir, exist_ok=True)
        
        # camera_params = camera.get_obs()["camera"]
        # with open(os.path.join(camera_save_dir, "camera_params.pkl"), 'wb') as file:
        #     pickle.dump(camera_params, file)

        rgb_dir = os.path.join(camera_save_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_data = camera.get_obs()["rgb"]
        assert len(rgb_data.shape) == 3
        save_image(os.path.join(rgb_dir, f"{step}.png"), rgb_data)

        depth_dir = os.path.join(camera_save_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        depth_data = camera.get_obs()["depth_linear"]
        assert len(depth_data.shape) == 2
        depth_data = depth_data * 1000
        depth_data = depth_data.astype(np.uint16)
        # with open(os.path.join(depth_dir, f"{step}.npy"), 'wb') as depth_file:
        #     np.save(depth_file, depth_data)
        save_image(os.path.join(depth_dir, f"{step}.png"), depth_data)
    
    print(f"saving camera observations..., step {step}.")
    step += 1



def add_world_camera(robot, position, target, idx):
    from omni.isaac.core.utils.viewports import set_camera_view
    parent = "World"
    name = f"camera_{idx}"
    prim_path = f"/{parent}/{name}"
    cam = VisionSensor(
        name=name,
        prim_path=prim_path,
        modalities=["rgb", "depth_linear", "camera"],
        enabled=True,
        noise=None,
        load_config=None,
        image_height=IMG_HEIGHT,
        image_width=IMG_WIDTH,
        viewport_name=None,
    )
    cam.load()
    cam.initialize()
    set_camera_view(position, target, camera_prim_path = prim_path, viewport_api=None)
    # cam.set_position_orientation(position, 1,1,1,1)
    # robot.sensors[name] = cam
    
    # return robot.sensors[name]
    return cam

def og_import_object(obj, pos=np.array([15, 15, 10]), quat=np.array([0, 0, 0, 1])):
    assert obj != None
    og.sim.import_object(obj)
    obj.set_position_orientation(pos, quat)
    og.sim.play()
    og.sim.step()

class Marker:
    def __init__(self, pos=None, quat=None, size=0.05):
        self.pos = pos
        self.quat = quat
        self.size = size
    
    def set_position_orientation(self, pos, quat):
        self.pos = pos
        self.quat = quat
    
    def get_position_orientation(self):
        return self.pos, self.quat
    
    def set_position(self, pos):
        self.pos = pos
    
    def set_orientation(self, quat):
        self.quat = quat


def run_simulation(args=None):

    global action
    global gripper
    global step

######################################### Define the Configuration of Simulation #############################

    cfg = dict()

    # Define scene
    # Traversable Scene
    # As for Rs_int, we change the default breakfast table model from white to brown 
    # to distinguish the table and white tablewares
    # not_load = ["straight_chair", "swivel_chair", "oven", "coffee_table", "laptop", "loudspeaker", "pot_plant", "sofa",
    #             "standing_tv", "stool", "trash_can", "bed", "dishwasher", "fridge", "microwave", "mirror", "shower", "sink", 
    #             "toilet",]
    not_load = ["stool", "pot_plant", "laptop", "oven", "sink", "bed", "dishwasher", "fridge", "microwave", "mirror", 
                "shower", "sink", "toilet", "swivel_chair", "coffee_table", ]
    cfg["scene"] = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "not_load_object_categories": not_load,
    }

    
    # Define robot
    cfg["robots"] = []

    robot0_cfg = dict(
        type="Fetch",
        scale=1.0,
        obs_modalities=["scan"],  # we're just doing a grasping demo so we don't need all observation modalities
        action_type="continuous",
        # Josiah: Do not force action normalize so that the controller config respects our command_input_limits kwarg
        action_normalize=False,
        grasping_mode="sticky",
        fixed_base=True,
        controller_config={
            # Josiah: Fix the controller config so it's the proper names and values desired
            # i.e.: Joint Position control using absolute values and no re-scaling / normalization of commands
            "arm_0": {
                "name": "JointController",
                "motor_type": "position",
                "use_delta_commands": False,
                "command_input_limits": None,
                "command_output_limits": None,
            },
        },
        default_arm_pose="vertical",
    )
    cfg["robots"].append(robot0_cfg)


    # Define tablewares
    cfg["objects"] = []

    # gym shoes
    num_shoes = 3 # number of pairs of shoes
    shoe_models = [ "hwrvmy", "kmcbym", "lqzooz", "wfmrpw", ]
    selected = np.random.choice(shoe_models, num_shoes, replace=False)
    for idx in range(num_shoes):
        model = selected[idx]
        shoe_cfg_right = {
            "type": "DatasetObject",
            "name": f"shoe_{idx}",
            "category": "gym_shoe",
            "model": model,
            # "scale": 1.3*factor,
            "position": [-1.0, 0, 0.3],
        }
        shoe_cfg_left = {
            "type": "DatasetObject",
            "name": f"shoe_{idx + num_shoes}",
            "category": "gym_shoe",
            "model": model,
            # "scale": 1.3*factor,
            "position": [-1.0, 0, 0.3],
        }
        cfg["objects"].append(shoe_cfg_right)
        cfg["objects"].append(shoe_cfg_left)
        print(f"shoe_{idx}_pair model: ", model)


    args.save_dir = os.path.join(args.save_dir, str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))


    # Define task
    cfg["task"] = {
        "type": "DummyTask",
        "termination_config": dict(),
        "reward_config": dict(),
    }


################################################## ENVIRONMENT SETUP ##########################################

    # Create the environment
    env = og.Environment(cfg)


    # Set carpet and table position and orientation
    carpet = env.scene.object_registry("name", "carpet_ctclvd_0")
    carpet.set_position_orientation(np.array([
                        -0.4199644818588833 + 0.5,
                        -1.2938668212900015 - 0.3,
                        0.002372790995
                    ]), np.array([
                        0.0,
                        0.0,
                        0.9999999999999812,
                        1.9470718377062835e-07
                    ]))
    
    # coffee_table = env.scene.object_registry("name", "coffee_table_fqluyq_0")
    # coffee_table.set_position_orientation(np.array([
    #                     -0.4762580692768097 + 0.5,
    #                     -1.2195799350738525 - 0.3,
    #                     0.2763478457927704
    #                 ]), np.array([
    #                     -1.6298145055770874e-08,
    #                     3.725290298461914e-09,
    #                     0.708872377872467,
    #                     0.7053368091583252
    #                 ]))
    # coffee_table.links["base_link"].mass = 200.0
    
    for _ in range(20):
        og.sim.step()


    # Set robot position and orientation
    robot = env.robots[0]
    init_quat = np.array([0, 0, 0, 1])
    t_pos, t_quat = carpet.get_position_orientation()
    robot.set_position_orientation(np.array([t_pos[0] + 0.65, t_pos[1] - 2.4 / 4, 0.0]), T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/180 * 180])), np.array([0, 0, 0, 1])))
    for _ in range(10):
        og.sim.step()
    robot.reset()
    robot.keep_still()
    robot.reset()
    robot.keep_still()
    for _ in range(60):
        og.sim.step()


    ## Initialize shoes
    t_pos, t_quat = carpet.get_position_orientation()
    center = [t_pos[0], t_pos[1] - 2.4 / 4]
    # x_min, x_max = center[0] - 0.2, center[0] + 0.2
    # y_min, y_max = center[1] - 0.25, center[1] + 0.25


    shoe_dict = dict()
    for idx in range(num_shoes * 2):
        if idx == 0 or idx == 3:
            x_min, x_max = center[0] - 0.0, center[0] + 0.3
        elif idx == 1 or idx == 4:
            x_min, x_max = center[0] - 0.3, center[0] - 0.0
        elif idx == 2 or idx == 5:
            x_min, x_max = center[0] - 0.6, center[0] - 0.3
        y_min, y_max = center[1] - 0.25, center[1] + 0.25
        xy = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max])
        shoe = env.scene.object_registry("name", f"shoe_{idx}")
        shoe.links["base_link"].mass = 2.5
        shoe_pos = shoe.get_position().copy()
        shoe_pos[:2] = xy
        shoe_pos[-1] = 0.15
        angle = np.random.randint(0, 360)
        rad = np.pi / 180 * angle
        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
        shoe.set_position_orientation(shoe_pos, quat)
        for _ in range(60):
            og.sim.step()  

        shoe_dict[f"shoe_{idx}"] = env.scene.object_registry("name", f"shoe_{idx}")


####################################################### Camera Setup ##############################################

    # Create and set the 10 cameras 

    target = carpet.get_position()
    # print(target)
    # target[2] = 0.386

    positions = []

    # lower camera: mid mid-left mid-right left right (0 -- 4)
    #positions.append([target[0] + -1.8, target[1] + 0, 1.25])
    #positions.append([target[0] + -1.6, target[1] + 1.5, 1.25])
    #positions.append([target[0] + -1.6, target[1] + -1.5, 1.25])
    #positions.append([target[0] + 0, target[1] + 1.8, 1.25])
    #positions.append([target[0] + 0, target[1] + -2, 1.25])

    # higher camera: mid mid-left mid-right left right (5 -- 9)
    # positions.append([target[0] + 1.9, target[1] + 0, 2.5]) # good

    positions.append([target[0] - 2.65, target[1] + 0, 2.6]) # good
    positions.append([target[0] - +2.2, target[1] + 1.8, 2.8]) # good
    positions.append([target[0] - +2.2, target[1] + -1.8, 2.8]) # good
    positions.append([target[0] + 0, target[1] + 2.8, 2.8]) # good
    positions.append([target[0] + 0, target[1] + -2.8, 2.8]) # good
    # positions.append([target[0] + 0, target[1] + -0.1, 3.2]) # bird eye view cam

    # dummy camera: required for debugging and being destoried/blacked out
    # positions.append([target[0] + 0, target[1] + -2, 2.5])

    camera_ls = []
    for idx, position in enumerate(positions):
        # if idx == 2:
        #     position[-1] = 3.1
        cam = add_world_camera(None, position, target, idx)
        camera_ls.append(cam)

    for _ in range(60):
        og.sim.step()

    # if not gm.HEADLESS:
    #     og.sim.enable_viewer_camera_teleoperation()


################################################# Robot Controller ###########################################

    # Setup the robot controller
    control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    ik_solver = IKSolver(
        robot_description_path=robot.robot_arm_descriptor_yamls[robot.default_arm],
        robot_urdf_path=robot.urdf_path,
        default_joint_pos=robot.get_joint_positions()[control_idx],
        eef_name=robot.eef_link_names[robot.default_arm],
    )
    default_action = np.concatenate([
        np.zeros(2), # base
        np.zeros(2), # camera
        robot.get_joint_positions()[control_idx], # arm
        np.zeros(1), # gripper
    ])
    left_arm_action_idx = np.arange(4, 4 + 8)
    assert default_action.shape[0] == robot.action_space.shape[0]

    def execute_ik(pos=None, quat=None, max_iter=100):
        global action
        global gripper
        nonlocal robot

        pos, quat = T.relative_pose_transform(*(pos, quat), *robot.get_position_orientation())
        # og.log.info("Querying joint configuration to current marker position")
        # Grab the joint positions in order to reach the desired pose target
        joint_pos = ik_solver.solve(
            target_pos=pos,
            target_quat=quat,
            # tolerance_pos=0.000001,
            # tolerance_quat=0.001,
            # weight_pos=1.00,
            # weight_quat=1.00,
            # max_iterations=1000,
        )
        if joint_pos is not None:
            # og.log.info("Solution found. Setting new arm configuration.")
            robot.set_joint_positions(joint_pos, indices=control_idx, drive=False)
            # Josiah: Update the global action as well
            action = np.copy(default_action)
            action[left_arm_action_idx] = joint_pos
            if gripper:
                action[-1] = 1.0
            else:
                action[-1] = -1.0
        else:
            og.log.info("EE position not reachable.")
        og.sim.step()

    # marker = PrimitiveObject(
    #     # prim_path=f"/World/marker",
    #     name="marker",
    #     primitive_type="Cube",
    #     # radius=0.03,
    #     size=0.05,
    #     visual_only=True,
    #     rgba=[1.0, 0, 0, 1.0],
    # )
    # og.sim.import_object(marker)
    marker = Marker(size=0.05)

    # Get initial EE position and set marker to that location
    command_pos = robot.get_eef_position()
    command_ori = robot.get_eef_orientation() # make the gripper facing downwards as always
    marker.set_position_orientation(command_pos, command_ori)
    og.sim.step()


################################################## Task Specs / Targets / Preparations #####################################

    # Setup callbacks for grabbing keyboard inputs from omni
    exit_now = False
    
    t_pos, t_quat = carpet.get_position_orientation()
    t_pos[1]  += 2.4 / 4
    t_pos[-1] += 0.10
    shoe_target = dict()
    for idx in range(num_shoes * 2):
        shoe_target_pos = t_pos.copy()
        if idx == 0:
            shoe_target_pos[1] += 0.3
        elif idx == 1:
            shoe_target_pos[1] += 0.0
        elif idx == 2:
            shoe_target_pos[1] -= 0.3
        elif idx == 3:
            shoe_target_pos[1] += (0.3 - 0.12)
        elif idx == 4:
            shoe_target_pos[1] += (0.0 - 0.12)
        elif idx == 5:
            shoe_target_pos[1] -= (0.3 + 0.12)

        shoe_target_ori = T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * 90]), t_quat)

        shoe_target[f"shoe_{idx}"] = [shoe_target_pos, shoe_target_ori]
    
    shoe_order = ["shoe_0", "shoe_3", "shoe_1", "shoe_4", "shoe_2", "shoe_5"]
    # shoe_order = sorted(shoe_order, key=lambda shoe: shoe_dict[shoe].get_position()[-1], reverse=True)


#################################################### START TASK #################################################
    # # shoes area to the table
    # shoe = "shoe_0"
    # exec_len = 25
    # curr_pos, curr_angle = robot.get_position(), 180
    # print("init_quat: ", init_quat)
    # tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 90
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #     print(curr_angle)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0, shoe_target[shoe][0][1] - robot.get_position()[1], 0]), 90
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #     print(curr_angle)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 180
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #     print(curr_angle)
    # for _ in range(120):
    #     og.sim.step()
    # input()
    # exit()

    # for idx in range(num_shoes * 2):
    #     angle = np.random.randint(0, 360)
    #     rad = np.pi / 180 * angle
    #     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #     shoe_dict[f"shoe_{idx}"].set_orientation(quat)
    #     for _ in range(10):
    #         og.sim.step()

    count = 0
    init_robot_x = robot.get_position()[0]
    for shoe in shoe_order:
        
        for _ in range(6):
            og.sim.step()
        
        exec_len = 25
        curr_pos, curr_ori = robot.get_eef_position(), robot.get_eef_orientation()

        ## g_high_0
        reach_exec_len = 10 if count != 0 else exec_len
        tar_pos = shoe_dict[shoe].get_position() + np.array([0, 0, 0.5])
        tar_ori = get_eef_orientation(z_rot_degree=0)
        delta_pos = (tar_pos - curr_pos) / reach_exec_len
        for i in range(reach_exec_len):
            curr_pos += delta_pos
            inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(reach_exec_len-1))
            execute_ik(curr_pos, inter_ori)
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        curr_ori = tar_ori.copy()

        ## g_low
        tar_pos = shoe_dict[shoe].get_position() + 0.7 * marker.size * np.array([0, 0, 1])
        pos0, quat0 = np.array([-5.51750756e-04,  6.64187087e-05,  1.19998987e-01]), np.array([-0.49998525,  0.50028019,  0.49741334,  0.50230913])
        pos1, quat1 = shoe_dict[shoe].get_position_orientation()
        tar_ori = T.pose_transform(pos1, quat1, pos0, quat0)[1]
        delta_pos = (tar_pos - curr_pos) / exec_len
        for i in range(exec_len):
            curr_pos += delta_pos
            inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
            execute_ik(curr_pos, inter_ori)
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        curr_ori = tar_ori.copy()
        for _ in range(5):
            og.sim.step()
        gripper = False
        action[-1] = -1.0
        for _ in range(int(15)):
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        for _ in range(5):
            og.sim.step()

        ## g_high_1
        tar_pos = shoe_dict[shoe].get_position() + np.array([0, 0, 0.5])
        tar_ori = get_eef_orientation(z_rot_degree=0)
        delta_pos = (tar_pos - curr_pos) / exec_len
        for i in range(exec_len):
            curr_pos += delta_pos
            inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
            execute_ik(curr_pos, inter_ori)
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        curr_ori = tar_ori.copy()

        for _ in range(6):
            og.sim.step()

        # shoes area to the table
        exec_len = 20
        curr_pos, curr_angle = robot.get_position(), 180

        if shoe != "shoe_0" and shoe != "shoe_3":
            tar_pos, tar_angle = robot.get_position() + np.array([init_robot_x - robot.get_position()[0], 0, 0]), 180
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

        tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 90
        delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
        for _ in range(exec_len):
            curr_pos += delta_pos
            curr_angle += delta_angle
            robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
            for _ in range(1):
                og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        
        tar_pos, tar_angle = robot.get_position() + np.array([0, shoe_target[shoe][0][1] - robot.get_position()[1], 0]), 90
        delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
        for _ in range(exec_len):
            curr_pos += delta_pos
            curr_angle += delta_angle
            robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
            for _ in range(1):
                og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        
        tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 180
        delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
        for _ in range(exec_len):
            curr_pos += delta_pos
            curr_angle += delta_angle
            robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
            for _ in range(1):
                og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        
        for _ in range(6):
            og.sim.step()

        
        exec_len = 25
        curr_pos, curr_ori = robot.get_eef_position(), robot.get_eef_orientation()

        ## t_high_0
        reach_exec_len = 10
        tar_pos = shoe_target[shoe][0] + np.array([0, 0, 0.5])
        tar_ori = get_eef_orientation(z_rot_degree=0)
        delta_pos = (tar_pos - curr_pos) / reach_exec_len
        for i in range(reach_exec_len):
            curr_pos += delta_pos
            inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(reach_exec_len-1))
            execute_ik(curr_pos, inter_ori)
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        curr_ori = tar_ori.copy()

        ## t_low
        tar_pos = shoe_target[shoe][0] + 0.0 * marker.size * np.array([0, 0, 1])
        pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
        pos1, quat1 = shoe_dict[shoe].get_position_orientation()
        T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))
        quat = shoe_target[shoe][1]
        T_wu = T.pose2mat((tar_pos, quat))
        T_we = T_wu @ T.pose_inv(T_eu)
        tar_pos, tar_ori = T.mat2pose(T_we)
        delta_pos = (tar_pos - curr_pos) / exec_len
        for i in range(exec_len):
            curr_pos += delta_pos
            inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
            execute_ik(curr_pos, inter_ori)
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        curr_ori = tar_ori.copy()
        for _ in range(5):
            og.sim.step()
        gripper = True
        action[-1] = 1.0
        for _ in range(int(15)):
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

        ## t_high_1
        tar_pos = shoe_target[shoe][0] + np.array([0, 0, 0.5])
        tar_ori = get_eef_orientation(z_rot_degree=0)
        delta_pos = (tar_pos - curr_pos) / exec_len
        for i in range(exec_len):
            curr_pos += delta_pos
            inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
            execute_ik(curr_pos, inter_ori)
            robot.apply_action(action=action)
            og.sim.step()
            if args.save_data:
                camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        curr_ori = tar_ori.copy()

        for _ in range(6):
            og.sim.step()
        
        if count != len(shoe_order) - 1:
            count += 1

            # table to the shoe ares
            exec_len = 20
            curr_pos, curr_angle = robot.get_position(), 180
            init_quat = np.array([0, 0, 0, 1])
            
            tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 270
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
            tar_pos, tar_angle = robot.get_position() + np.array([0, center[1] - robot.get_position()[1], 0]), 270
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
            tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 180
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
            if shoe == "shoe_0":
                diff = np.array([0, 0, 0])
                continue
            elif shoe == "shoe_3" or shoe == "shoe_1":
                diff = np.array([-0.3, 0, 0])
            elif shoe == "shoe_4" or shoe == "shoe_2":
                diff = np.array([-0.6, 0, 0])

            tar_pos, tar_angle = robot.get_position() + diff, 180
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    # def keyboard_event_handler(event, *args, **kwargs):
    #     nonlocal command_pos, command_ori, exit_now
    #     nonlocal shoe
    #     global action, gripper

    #     # print("haha")

    #     # Check if we've received a key press or repeat
    #     if event.type == carb.input.KeyboardEventType.KEY_PRESS \
    #             or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            
    #         if event.input == carb.input.KeyboardInput.ENTER:
    #             # Execute the command
    #             print(f"moving to: pos {command_pos}, quat {command_ori}")
    #             execute_ik(pos=command_pos, quat=command_ori)
    #         elif event.input == carb.input.KeyboardInput.ESCAPE:

    #             # command_pos = shoe.get_position() + marker.size * np.array([0, 0, 1])           
    #             command_pos = shoe.get_position()             
    #             command_ori = get_eef_orientation(z_rot_degree=0)
    #             marker.set_position_orientation(command_pos, command_ori)
    #             og.sim.step()

    #         else:
    #             # We see if we received a valid delta command, and if so, we update our command and visualized
    #             # marker position
    #             delta_cmd = input_to_xyz_delta_command(inp=event.input)
    #             if delta_cmd is not None and delta_cmd.sum() != 0:
    #                 command_pos = command_pos + delta_cmd
    #                 marker.set_position(command_pos)
    #                 og.sim.step()
                
    #             elif delta_cmd is not None and delta_cmd.sum() == 0:
    #                 # Josiah: Toggle the corresponding left gripper action value
    #                 if gripper:
    #                     action[-1] = -1.0
    #                     gripper = False
    #                 else:
    #                     action[-1] = 1.0
    #                     gripper = True
    #                 og.sim.step()

    #     # Callback must return True if valid
    #     return True

    # # Hook up the callback function with omni's user interface
    # appwindow = omni.appwindow.get_default_app_window()
    # input_interface = carb.input.acquire_input_interface()
    # keyboard = appwindow.get_keyboard()
    # sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)

    # # Print out helpful information to the user
    # print_message()

    # # Loop until the user requests an exit
    # print("start!!!!!!!!!!!!!!!!!!!!!")

    # if args.save_data:
    #     for idx in range(len(camera_ls)):
    #         camera = camera_ls[idx]

    #         camera_name = f"camera_{idx}"
    #         camera_save_dir = os.path.join(args.save_dir, camera_name)
    #         os.makedirs(camera_save_dir, exist_ok=True)
            
    #         camera_params = camera.get_obs()["camera"]
    #         with open(os.path.join(camera_save_dir, "camera_params.pkl"), 'wb') as file:
    #             pickle.dump(camera_params, file)

    # count = 0
    # while not exit_now:
    #     og.sim.step()
    #     # Josiah: Always take a robot action if it exists
    #     if action is not None:
    #         robot.apply_action(action=action)
        
    #     if count % 20 == 0 and args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #     count += 1
    #     # pos0, quat0 = shoe.get_position_orientation()
    #     # pos1, quat1 = robot.get_eef_position(), robot.get_eef_orientation()
    #     # print("relative transform: ", T.relative_pose_transform(pos1, quat1, pos0, quat0))
    #     # print("quat_dist: ", T.quat_distance(robot.get_eef_orientation(), knife.get_orientation()))
    #     # print("knife quat: ", knife.get_orientation())
    #     # print("spoon quat: ", spoon.get_orientation())
    #     # print("fork  quat: ", fork.get_orientation())

    # # Always shut the simulation down cleanly at the end
    # og.app.close()


    # if args.save_data:
    #     for idx in range(len(camera_ls)):
    #         camera = camera_ls[idx]

    #         camera_name = f"camera_{idx}"
    #         camera_save_dir = os.path.join(args.save_dir, camera_name)
    #         os.makedirs(camera_save_dir, exist_ok=True)
            
    #         camera_params = camera.get_obs()["camera"]
    #         with open(os.path.join(camera_save_dir, "camera_params.pkl"), 'wb') as file:
    #             pickle.dump(camera_params, file)


    # ## mid to left
    # curr_pos, curr_angle = robot.get_position(), 0
    # exec_len = 15
    
    # tar_pos, tar_angle = robot.get_position() + np.array([-0.3, 0, 0]), 0
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), -90
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len - 5):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0, -(0.6 + 0.3), 0]), -90
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 0
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len - 5):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0.0 - robot.get_position()[0], 0, 0]), 0
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    # tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 90
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len - 5):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    # tar_pos, tar_angle = np.array([0.0, -1.235, 0.0]), 90
    # delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    # for _ in range(exec_len):
    #     curr_pos += delta_pos
    #     curr_angle += delta_angle
    #     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #     for _ in range(1):
    #         og.sim.step()
    #     if args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    
    # ## main loop
    # for idx in range(num_plates):

    #     for _ in range(3):
    #         og.sim.step()

    #     exec_len = 25
    #     curr_pos, curr_ori = robot.get_eef_position(), robot.get_eef_orientation()
    #     tableware_ls = []
    #     tableware_ls += tablewares[f"plate_{idx}"]["not_on_plate"]
    #     tableware_ls += tablewares[f"plate_{idx}"]["on_plate"]

    #     for tableware in tableware_ls:

    #         ## g_high_0
    #         tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 3.5])
    #         tar_ori = get_eef_orientation(z_rot_degree=0)
    #         delta_pos = (tar_pos - curr_pos) / exec_len
    #         for i in range(exec_len):
    #             curr_pos += delta_pos
    #             inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
    #             execute_ik(curr_pos, inter_ori)
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         curr_ori = tar_ori.copy()
            

    #         ## g_low
    #         if tableware in not_on_plate:
    #             tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.6])
    #         else:
    #             tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.68])
    #         if tableware == "knife":
    #             pos0, quat0 = np.array([ 0.00072538, -0.0050252 ,  0.05978444]), np.array([-0.70199821, -0.02527567,  0.71092023,  0.03394224])
    #         if tableware == "spoon":
    #             if is_spoon_90[f"spoon_{idx}"]:
    #                 pos0, quat0 = np.array([-0.00147704, -0.0005369 ,  0.04997578]), np.array([ 0.5076075 , -0.48696841, -0.49809182, -0.50705121])
    #             else:
    #                 pos0, quat0 = np.array([ 0.00055356, -0.00012311,  0.04999698]), np.array([-0.70316285,  0.00368559,  0.71099893,  0.00537952])
    #         if tableware == "fork":
    #             if is_usd_fork[f"fork_{idx}"]:
    #                 pos0, quat0 = np.array([-1.85259592e-03,  6.43737602e-06,  3.99573354e-02]), np.array([ 0.0104983 ,  0.69046573, -0.01013086,  0.72321796])
    #             else:
    #                 pos0, quat0 = np.array([ 0.00025444, -0.00052758,  0.03999597]), np.array([-7.04789855e-01, -1.29435355e-04,  7.09356952e-01,  9.16291966e-03])
    #         pos1, quat1 = tablewares[f"plate_{idx}"][tableware]["object"].get_position_orientation()
    #         tar_ori = T.pose_transform(pos1, quat1, pos0, quat0)[1]
    #         delta_pos = (tar_pos - curr_pos) / exec_len
    #         for i in range(exec_len):
    #             curr_pos += delta_pos
    #             inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
    #             execute_ik(curr_pos, inter_ori)
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         curr_ori = tar_ori.copy()
    #         for _ in range(3):
    #             og.sim.step()
    #         gripper = False
    #         action[-1] = -1.0
    #         for _ in range(int(15)):
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         for _ in range(3):
    #             og.sim.step()

            
    #         ## g_high_1
    #         tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 3.5])
    #         tar_ori = get_eef_orientation(z_rot_degree=0)
    #         delta_pos = (tar_pos - curr_pos) / exec_len
    #         for i in range(exec_len):
    #             curr_pos += delta_pos
    #             inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
    #             execute_ik(curr_pos, inter_ori)
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         curr_ori = tar_ori.copy()
            
            
    #         ## t_high_0
    #         tar_pos = tablewares[f"plate_{idx}"][tableware]["target"] + marker.size * np.array([0, 0, 3.5])
    #         tar_ori = get_eef_orientation(z_rot_degree=0)
    #         delta_pos = (tar_pos - curr_pos) / exec_len
    #         for i in range(exec_len):
    #             curr_pos += delta_pos
    #             inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
    #             execute_ik(curr_pos, inter_ori)
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         curr_ori = tar_ori.copy()
            

    #         ## t_low
    #         tar_pos = tablewares[f"plate_{idx}"][tableware]["target"] + marker.size * np.array([0, 0, 0.8])
    #         pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
    #         pos1, quat1 = tablewares[f"plate_{idx}"][tableware]["object"].get_position_orientation()
    #         T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))
    #         if idx == 0:
    #             if tableware == "knife":
    #                 rad = np.pi / 180 * 180
    #                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #             if tableware == "spoon":
    #                 if is_spoon_90["spoon_0"]:
    #                     rad = np.pi / 180 * 90
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                 else:
    #                     rad = np.pi / 180 * 0
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #             if tableware == "fork":
    #                 if is_usd_fork["fork_0"]:
    #                     rad = np.pi / 180 * 0
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                 else:
    #                     rad = np.pi / 180 * 180
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #         else:
    #             if tableware == "knife":
    #                 rad = np.pi / 180 * 0
    #                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #             if tableware == "spoon":
    #                 if is_spoon_90["spoon_1"]:
    #                     rad = np.pi / 180 * -90
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                 else:
    #                     rad = np.pi / 180 * -180
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #             if tableware == "fork":
    #                 if is_usd_fork["fork_1"]:
    #                     rad = np.pi / 180 * -180
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                 else:
    #                     rad = np.pi / 180 * 0
    #                     quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)

    #         T_wu = T.pose2mat((tar_pos - 0.3 * marker.size * np.array([0, 0, 1.0]), quat))
    #         T_we = T_wu @ T.pose_inv(T_eu)
    #         tar_pos, tar_ori = T.mat2pose(T_we)
    #         delta_pos = (tar_pos - curr_pos) / exec_len
    #         for i in range(exec_len):
    #             curr_pos += delta_pos
    #             inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
    #             execute_ik(curr_pos, inter_ori)
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         curr_ori = tar_ori.copy()
    #         for _ in range(3):
    #             og.sim.step()
    #         gripper = True
    #         action[-1] = 1.0
    #         for _ in range(int(15)):
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        

    #         ## t_high_1
    #         tar_pos = tablewares[f"plate_{idx}"][tableware]["target"] + marker.size * np.array([0, 0, 3.5])
    #         tar_ori = get_eef_orientation(z_rot_degree=0)
    #         delta_pos = (tar_pos - curr_pos) / exec_len
    #         for i in range(exec_len):
    #             curr_pos += delta_pos
    #             inter_ori = T.quat_slerp(curr_ori, tar_ori, fraction=i/(exec_len-1))
    #             execute_ik(curr_pos, inter_ori)
    #             robot.apply_action(action=action)
    #             og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #         curr_ori = tar_ori.copy()
        

    #     # left to right
    #     if idx == 0:
    #         for _ in range(3):
    #             og.sim.step()
    #         curr_pos, curr_angle = robot.get_position(), 90
    #         exec_len = 15
            
    #         tar_pos, tar_angle = robot.get_position() + np.array([0, -0.3, 0]), 90
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
    #         tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 180
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len - 5):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
    #         tar_pos, tar_angle = robot.get_position() + np.array([-(0.35 + 0.45 + 0.3), 0, 0]), 180
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
    #         tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 90
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len - 5):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
    #         tar_pos, tar_angle = robot.get_position() + np.array([0, +(1.2 + 0.4 * 2), 0]), 90
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
    #         tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 0
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len - 5):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    #         tar_pos, tar_angle = robot.get_position() + np.array([0.0 - robot.get_position()[0], 0, 0]), 0
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    #         tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), -90
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len - 5):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    #         tar_pos, tar_angle = np.array([0.0, 0.235, 0.0]), -90
    #         delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #         for _ in range(exec_len):
    #             curr_pos += delta_pos
    #             curr_angle += delta_angle
    #             robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #             for _ in range(1):
    #                 og.sim.step()
    #             if args.save_data:
    #                 camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    while True:
        og.sim.step()




def input_to_xyz_delta_command(inp, delta=0.01):
    mapping = {
        carb.input.KeyboardInput.W: np.array([delta, 0, 0]),
        carb.input.KeyboardInput.S: np.array([-delta, 0, 0]),
        carb.input.KeyboardInput.E: np.array([0, 0, -delta]),
        carb.input.KeyboardInput.Q: np.array([0, 0, delta]),
        carb.input.KeyboardInput.A: np.array([0, delta, 0]),
        carb.input.KeyboardInput.D: np.array([0, -delta, 0]),
        carb.input.KeyboardInput.Z: np.array([0, 0, 0]),
    }

    return mapping.get(inp)


def print_message():
    print("*" * 80)
    print("Move the marker to a desired position to query IK and press ENTER")
    print("W/S: move marker further away or closer to the robot")
    print("A/D: move marker to the left or the right of the robot")
    print("Q/E: move marker up and down")
    print("Z: open/close the gripper")
    print("ESC: move marker to the waypoint for grasping")
    print("ENTER: IK solver to move arm and eef to the desired position and quaternion")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_data', action='store_true', help="whether to save the omnigibson data")
    parser.add_argument('--save_dir', default='og_shoe_data', type=str, help="the root directory of saving og tableware data")
    args = parser.parse_args()
    
    run_simulation(args=args)
