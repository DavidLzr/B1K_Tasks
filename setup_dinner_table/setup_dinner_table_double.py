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

    ################################### Define the Configuration of Simulation #######################
    cfg = dict()

    # Define scene
    # Traversable Scene
    # As for Rs_int, we change the default breakfast table model from white to brown 
    # to distinguish the table and white tablewares
    not_load = ["straight_chair", "swivel_chair", "oven", "coffee_table", "laptop", "loudspeaker", "pot_plant", "sofa",
                "standing_tv", "stool", "trash_can", "bed", "dishwasher", "fridge", "microwave", "mirror", "shower", "sink", 
                "toilet",]
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
    factor = 1.35

    # Define table plates
    plate_scale = {
        "plate_1": 0.005*1.3, #size good
        "plate_2": 0.1*1.3,    # size good
        "plate_10": 0.025*1.3, #size good
    }
    model_plates = ["iawoof"] + list(plate_scale.keys())
    num_plates = np.random.randint(2, args.max_num_plate + 1)
    for idx in range(num_plates):
        model = np.random.choice(model_plates)
        if model in ["iawoof"]:
            plate_cfg = {
                "type": "DatasetObject",
                "name": f"plate_{idx}",
                "category": "plate", 
                "model": model,
                "scale": 0.0055*1.3,
                "position": [1, 1, 0.1 + idx]
            }
        else:
            plate_cfg = {
                "type": "USDObject",
                "name": f"plate_{idx}",
                "usd_path": f"{gm.ASSET_PATH}/models/tablewares/plate/{model}/{model}/{model}.usd",
                "category": "plate", 
                "scale": plate_scale[model],
                "position": [1, 1, 0.1 + idx]
            }
        cfg["objects"].append(plate_cfg)
        print(f"plate_{idx} model: ", model)


    # Define table knives
    num_knives = np.random.randint(2, args.max_num_knife + 1)
    # model_knives = ["lrdmpf", "nmkend"]
    model_knives = ["lrdmpf"]
    for idx in range(num_knives):
        model = np.random.choice(model_knives)
        knife_cfg = {
            "type": "DatasetObject",
            "name": f"knife_{idx}",
            "category": "table_knife",
            "model": model,
            "scale": 1.3*factor,
            "position": [1, 1, 0.1 + idx]
        }
        cfg["objects"].append(knife_cfg)
        print(f"knife_{idx} model: ", model)
    

    # Define table spoons
    spoon_scale = {
        "spoon_0": 0.0015*factor, # 90 orientation
        "spoon_2": 0.10*factor, # good
        "spoon_4": 0.026*factor, # good
        "spoon_5": 0.008*factor, # 90 orientation
        "spoon_6": 0.008*factor, # 90 orientation
    }
    is_spoon_90 = dict()
    num_spoons = np.random.randint(2, args.max_num_spoon + 1)
    for idx in range(num_spoons):
        model = np.random.choice(list(spoon_scale.keys()))
        if model in ["spoon_2", "spoon_4"]:
            is_spoon_90[f"spoon_{idx}"] = False
        else:
            is_spoon_90[f"spoon_{idx}"] = True
        spoon_cfg = {
            "type": "USDObject",
            "name": f"spoon_{idx}",
            "usd_path": f"{gm.ASSET_PATH}/models/tablewares/spoon/{model}/{model}/{model}.usd",
            "category": "spoon",
            "scale": spoon_scale[model],
            "position": [1, 1, 0.1 + idx]
        }
        cfg["objects"].append(spoon_cfg)
        print(f"spoon_{idx} model: ", model)
    

    # Define table forks
    fork_scale = {
        "fork_3": 0.10*factor,
    }
    is_usd_fork = dict()
    num_forks = np.random.randint(2, args.max_num_fork + 1)
    model_forks = ["flexrc", "pqrtkc"] + list(fork_scale.keys())
    for idx in range(num_forks):
        model = np.random.choice(model_forks)
        if model in ["flexrc", "pqrtkc"]:
            fork_cfg = {
                "type": "DatasetObject",
                "name": f"fork_{idx}",
                "category": "tablefork",
                "model": model,
                "scale": 1.4*factor,
                "position": [1, 1, 0.1 + idx]
            }
            is_usd_fork[f"fork_{idx}"] = False
        else:
            fork_cfg = {
                "type": "USDObject",
                "name": f"fork_{idx}",
                "category": "tablefork",
                "usd_path": f"{gm.ASSET_PATH}/models/tablewares/fork/{model}/{model}/{model}.usd",
                "scale": fork_scale[model],
                "position": [1, 1, 0.1 + idx]
            }
            is_usd_fork[f"fork_{idx}"] = True
        cfg["objects"].append(fork_cfg)
        print(f"fork_{idx} model: ", model)


    args.save_dir = os.path.join(args.save_dir, str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # Define task
    cfg["task"] = {
        "type": "DummyTask",
        "termination_config": dict(),
        "reward_config": dict(),
    }



    ########################################### Create the Simulation ################################

    # Create the environment
    env = og.Environment(cfg)


    robot = env.robots[0]
    robot.set_position_orientation(np.array([-0.55, -0.5, 0.0]), np.array([0, 0, 0, 1]))
    # robot.set_position_orientation(np.array([-0.75, -0.5, 0.0]), np.array([0, 0, 0, 1]))
    # robot.set_position_orientation(np.array([0.0, -1.235, 0.0]), T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/180 * 90])), np.array([0, 0, 0, 1])))
    # robot.set_position_orientation(np.array([0.0, 0.235, 0.0]), T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/180 * -90])), np.array([0, 0, 0, 1])))
    ## facing directly to the positive direction of the X-axis
    # np.array([0, 0, 0, 1])
    ## facing inside the screen
    # np.array([0.,     0.,     0.70710678, 0.70710678])
    ## facing outside the screen
    # np.array([0.,     0.,     -0.70710678, 0.70710678]) 
    for _ in range(10):
        og.sim.step()
    robot.reset()
    robot.keep_still()
    robot.reset()
    robot.keep_still()
    for _ in range(10):
        og.sim.step()


    # Set tablewares randomly on the TOP of the table
    breakfast_table = env.scene.object_registry("name", "breakfast_table_skczfi_0")
    # breakfast_table = env.scene.object_registry("name", "table_0")
    breakfast_table.set_position_orientation(np.array([
                        # 0.8725971221923828 - 0.35,
                        0.0,
                        # 0.8725971221923828 + 1.0,
                        -0.5,
                        0.5881626009941101
                    ]), np.array([
                        -1.8670980352908373e-06,
                        -1.91313779396296e-06,
                        -0.7024190425872803,
                        0.7117636799812317
                    ]))
    for _ in range(20):
        og.sim.step()

    breakfast_table.links["base_link"].mass = 200.0
    for _ in range(20):
        og.sim.step()


    ## Initialize the tablewares 
    t_pos, t_quat = breakfast_table.get_position_orientation()


    plate_dict = dict()
    for idx in range(num_plates):
        plate_dict[f"plate_{idx}"] = [env.scene.object_registry("name", f"plate_{idx}"), 0]
        plate_dict[f"plate_{idx}"][0].links["base_link"].mass = 15.0
        success = plate_dict[f"plate_{idx}"][0].states[OnTop].set_value(breakfast_table, True)
        print(f"plate_{idx} mass: ", plate_dict[f"plate_{idx}"][0].links["base_link"].mass)

        pos, _ = plate_dict[f"plate_{idx}"][0].get_position_orientation()
        pos[:2] = t_pos[:2]
        if idx == 0:
            pos[1] -= 0.35
        elif idx == 1:
            pos[1] += 0.35
        plate_dict[f"plate_{idx}"][0].set_position_orientation(pos, t_quat)

        for _ in range(60):
            og.sim.step()
    

    ## Remove OnTop Feature
    # Sample positions in a rectangular region to spawn the utensils
    # With 1. COM distance threshold trick 2. Segmentation non-overlapping trick (TODO)
    knife_dict = dict()
    spoon_dict = dict()
    fork_dict = dict()
    for idx in range(num_plates):
        
        plate_pos = plate_dict[f"plate_{idx}"][0].get_position()
        if idx == 0:
            x_min, x_max = plate_pos[0] - 0.35, plate_pos[0] + 0.35
            y_min, y_max = plate_pos[1] - 0.2, plate_pos[1] + 0.3
        elif idx == 1:
            x_min, x_max = plate_pos[0] - 0.35, plate_pos[0] + 0.35
            y_min, y_max = plate_pos[1] - 0.3, plate_pos[1] + 0.2

        xy_ls = []
        while len(xy_ls) < 3:
            xy = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max])
            close = False
            for p in xy_ls:
                if np.linalg.norm(xy - p) < np.sqrt(0.5**2 + 0.7**2) / 3:
                    close = True
            if not close:
                xy_ls.append(xy)
        
        # knife
        knife = env.scene.object_registry("name", f"knife_{idx}")
        knife.links["base_link"].mass = 1.0
        print(f"knife_{idx} mass: ", knife.links["base_link"].mass)
        knife_pos = knife.get_position().copy()
        knife_pos[:2] = xy_ls[0]
        knife_pos[-1] = plate_pos[-1] + 0.05
        angle = np.random.randint(0, 360)
        rad = np.pi / 180 * angle
        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
        knife.set_position_orientation(knife_pos, quat)
        for _ in range(30):
            og.sim.step()

        # spoon
        spoon = env.scene.object_registry("name", f"spoon_{idx}")
        spoon.links["base_link"].mass = 1.0
        print(f"spoon_{idx} mass: ", spoon.links["base_link"].mass)
        spoon_pos =  spoon.get_position().copy()
        spoon_pos[:2] = xy_ls[1]
        spoon_pos[-1] = plate_pos[-1] + 0.05
        angle = np.random.randint(0, 360)
        rad = np.pi / 180 * angle
        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
        spoon.set_position_orientation(spoon_pos, quat)
        for _ in range(30):
            og.sim.step()

        # fork
        fork = env.scene.object_registry("name", f"fork_{idx}")
        fork.links["base_link"].mass = 1.0
        print(f"fork_{idx} mass: ", fork.links["base_link"].mass)
        fork_pos = fork.get_position().copy()
        fork_pos[:2] = xy_ls[2]
        fork_pos[-1] = plate_pos[-1] + 0.05
        angle = np.random.randint(0, 360)
        rad = np.pi / 180 * angle
        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
        fork.set_position_orientation(fork_pos, quat)
        for _ in range(30):
            og.sim.step()


        knife_dict[f"knife_{idx}"] = env.scene.object_registry("name", f"knife_{idx}")
        spoon_dict[f"spoon_{idx}"] = env.scene.object_registry("name", f"spoon_{idx}")
        fork_dict[f"fork_{idx}"] = env.scene.object_registry("name", f"fork_{idx}")
        
        plate_dict[f"plate_{idx}"][0].fixed_base = True



    # Create and set the 10 cameras 

    target = breakfast_table.get_position()
    # print(target)
    target[2] = 0.75

    positions = []

    # lower camera: mid mid-left mid-right left right (0 -- 4)
    #positions.append([target[0] + -1.8, target[1] + 0, 1.25])
    #positions.append([target[0] + -1.6, target[1] + 1.5, 1.25])
    #positions.append([target[0] + -1.6, target[1] + -1.5, 1.25])
    #positions.append([target[0] + 0, target[1] + 1.8, 1.25])
    #positions.append([target[0] + 0, target[1] + -2, 1.25])

    # higher camera: mid mid-left mid-right left right (5 -- 9)
    # positions.append([target[0] + 1.9, target[1] + 0, 2.5]) # good

    positions.append([target[0] + 2.0, target[1] + 0, 2.7]) # good
    positions.append([target[0] + +1.7, target[1] + 1.6, 2.6]) # good
    positions.append([target[0] + +1.7, target[1] + -1.6, 2.6]) # good
    positions.append([target[0] + 0, target[1] + 2.0, 2.6]) # good
    positions.append([target[0] + 0, target[1] + -2.0, 2.6]) # good
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


######################################### Robot Controller #################################

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


    # Setup callbacks for grabbing keyboard inputs from omni
    exit_now = False

    ## Split the setup table task into subtasks
    #  Each subtask is to set up for a specific plate
    tablewares = dict()
    
    for idx in range(num_plates):

        plate_pos    = plate_dict[f"plate_{idx}"][0].get_position()

        knife_target = plate_pos.copy()
        if idx == 0:
            knife_target[0]  += 0.18
        elif idx == 1:
            knife_target[0]  -= 0.18

        spoon_target = plate_pos.copy()
        if idx == 0:
            spoon_target[0]  -= 0.18
        elif idx == 1:
            spoon_target[0]  += 0.18
        if "spoon_4" in spoon_dict[f"spoon_{idx}"].usd_path or "spoon_2" in spoon_dict[f"spoon_{idx}"].usd_path:
            if idx == 0:
                spoon_target[1]  += 0.05
            elif idx == 1:
                spoon_target[1]  -= 0.05

        fork_target = plate_pos.copy()
        if idx == 0:
            fork_target[0] -= 0.24
        elif idx == 1:
            fork_target[0] += 0.24

        tablewares[f"plate_{idx}"] = {
                        "knife": {"object": knife_dict[f"knife_{idx}"], "g_high_0": False, "g_low": False, "g_high_1": False, "t_high_0": False, "t_low": False, "t_high_1": False, "target": knife_target, "done": False}, 
                        "spoon": {"object": spoon_dict[f"spoon_{idx}"], "g_high_0": False, "g_low": False, "g_high_1": False, "t_high_0": False, "t_low": False, "t_high_1": False, "target": spoon_target, "done": False}, 
                        "fork" : {"object": fork_dict[f"fork_{idx}"], "g_high_0": False, "g_low": False, "g_high_1": False, "t_high_0": False, "t_low": False, "t_high_1": False, "target": fork_target, "done": False}
                        }

        # Split the objects into: 1. not on the plate (we will start with them first). 2. on the plate (in descending Z-axis order)
        not_on_plate = []
        on_plate = []

        plate_pos = plate_dict[f"plate_{idx}"][0].get_position()
        for tableware in tablewares[f"plate_{idx}"]:
            object_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position()
            if object_pos[-1] > plate_pos[-1] and np.linalg.norm(object_pos[:2] - plate_pos[:2]) < 0.1:
                on_plate.append(tableware)
            else:
                not_on_plate.append(tableware)
        on_plate = sorted(on_plate, key=lambda obj:tablewares[f"plate_{idx}"][obj]["object"].get_position()[-1], reverse=True)
        
        tablewares[f"plate_{idx}"]["on_plate"] = on_plate
        tablewares[f"plate_{idx}"]["not_on_plate"] = not_on_plate
        tablewares[f"plate_{idx}"]["done"] = False


    robot_move_mid_left, robot_move_left_right = False, False
    init_quat = robot.get_orientation()


    ################################## START TASK #############################
    if args.save_data:
        for idx in range(len(camera_ls)):
            camera = camera_ls[idx]

            camera_name = f"camera_{idx}"
            camera_save_dir = os.path.join(args.save_dir, camera_name)
            os.makedirs(camera_save_dir, exist_ok=True)
            
            camera_params = camera.get_obs()["camera"]
            with open(os.path.join(camera_save_dir, "camera_params.pkl"), 'wb') as file:
                pickle.dump(camera_params, file)


    ## mid to left
    curr_pos, curr_angle = robot.get_position(), 0
    exec_len = 15
    
    tar_pos, tar_angle = robot.get_position() + np.array([-0.3, 0, 0]), 0
    delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    for _ in range(exec_len):
        curr_pos += delta_pos
        curr_angle += delta_angle
        robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
        for _ in range(1):
            og.sim.step()
        if args.save_data:
            camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), -90
    delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    for _ in range(exec_len - 5):
        curr_pos += delta_pos
        curr_angle += delta_angle
        robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
        for _ in range(1):
            og.sim.step()
        if args.save_data:
            camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    tar_pos, tar_angle = robot.get_position() + np.array([0, -(0.6 + 0.3), 0]), -90
    delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    for _ in range(exec_len):
        curr_pos += delta_pos
        curr_angle += delta_angle
        robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
        for _ in range(1):
            og.sim.step()
        if args.save_data:
            camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 0
    delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    for _ in range(exec_len - 5):
        curr_pos += delta_pos
        curr_angle += delta_angle
        robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
        for _ in range(1):
            og.sim.step()
        if args.save_data:
            camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    
    tar_pos, tar_angle = robot.get_position() + np.array([0.0 - robot.get_position()[0], 0, 0]), 0
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
    for _ in range(exec_len - 5):
        curr_pos += delta_pos
        curr_angle += delta_angle
        robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
        for _ in range(1):
            og.sim.step()
        if args.save_data:
            camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    tar_pos, tar_angle = np.array([0.0, -1.235, 0.0]), 90
    delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    for _ in range(exec_len):
        curr_pos += delta_pos
        curr_angle += delta_angle
        robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
        for _ in range(1):
            og.sim.step()
        if args.save_data:
            camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

    
    ## main loop
    for idx in range(num_plates):

        for _ in range(3):
            og.sim.step()

        exec_len = 25
        curr_pos, curr_ori = robot.get_eef_position(), robot.get_eef_orientation()
        tableware_ls = []
        tableware_ls += tablewares[f"plate_{idx}"]["not_on_plate"]
        tableware_ls += tablewares[f"plate_{idx}"]["on_plate"]

        for tableware in tableware_ls:

            ## g_high_0
            tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 3.5])
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
            

            ## g_low
            if tableware in not_on_plate:
                tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.6])
            else:
                tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.68])
            if tableware == "knife":
                pos0, quat0 = np.array([ 0.00072538, -0.0050252 ,  0.05978444]), np.array([-0.70199821, -0.02527567,  0.71092023,  0.03394224])
            if tableware == "spoon":
                if is_spoon_90[f"spoon_{idx}"]:
                    pos0, quat0 = np.array([-0.00147704, -0.0005369 ,  0.04997578]), np.array([ 0.5076075 , -0.48696841, -0.49809182, -0.50705121])
                else:
                    pos0, quat0 = np.array([ 0.00055356, -0.00012311,  0.04999698]), np.array([-0.70316285,  0.00368559,  0.71099893,  0.00537952])
            if tableware == "fork":
                if is_usd_fork[f"fork_{idx}"]:
                    pos0, quat0 = np.array([-1.85259592e-03,  6.43737602e-06,  3.99573354e-02]), np.array([ 0.0104983 ,  0.69046573, -0.01013086,  0.72321796])
                else:
                    pos0, quat0 = np.array([ 0.00025444, -0.00052758,  0.03999597]), np.array([-7.04789855e-01, -1.29435355e-04,  7.09356952e-01,  9.16291966e-03])
            pos1, quat1 = tablewares[f"plate_{idx}"][tableware]["object"].get_position_orientation()
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
            for _ in range(3):
                og.sim.step()
            gripper = False
            action[-1] = -1.0
            for _ in range(int(15)):
                robot.apply_action(action=action)
                og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            for _ in range(3):
                og.sim.step()

            
            ## g_high_1
            tar_pos = tablewares[f"plate_{idx}"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 3.5])
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
            
            
            ## t_high_0
            tar_pos = tablewares[f"plate_{idx}"][tableware]["target"] + marker.size * np.array([0, 0, 3.5])
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
            

            ## t_low
            tar_pos = tablewares[f"plate_{idx}"][tableware]["target"] + marker.size * np.array([0, 0, 0.8])
            pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
            pos1, quat1 = tablewares[f"plate_{idx}"][tableware]["object"].get_position_orientation()
            T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))
            if idx == 0:
                if tableware == "knife":
                    rad = np.pi / 180 * 180
                    quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                if tableware == "spoon":
                    if is_spoon_90["spoon_0"]:
                        rad = np.pi / 180 * 90
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                    else:
                        rad = np.pi / 180 * 0
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                if tableware == "fork":
                    if is_usd_fork["fork_0"]:
                        rad = np.pi / 180 * 0
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                    else:
                        rad = np.pi / 180 * 180
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
            else:
                if tableware == "knife":
                    rad = np.pi / 180 * 0
                    quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                if tableware == "spoon":
                    if is_spoon_90["spoon_1"]:
                        rad = np.pi / 180 * -90
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                    else:
                        rad = np.pi / 180 * -180
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                if tableware == "fork":
                    if is_usd_fork["fork_1"]:
                        rad = np.pi / 180 * -180
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
                    else:
                        rad = np.pi / 180 * 0
                        quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)

            T_wu = T.pose2mat((tar_pos - 0.3 * marker.size * np.array([0, 0, 1.0]), quat))
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
            for _ in range(3):
                og.sim.step()
            gripper = True
            action[-1] = 1.0
            for _ in range(int(15)):
                robot.apply_action(action=action)
                og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
        

            ## t_high_1
            tar_pos = tablewares[f"plate_{idx}"][tableware]["target"] + marker.size * np.array([0, 0, 3.5])
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
        

        # left to right
        if idx == 0:
            for _ in range(3):
                og.sim.step()
            curr_pos, curr_angle = robot.get_position(), 90
            exec_len = 15
            
            tar_pos, tar_angle = robot.get_position() + np.array([0, -0.3, 0]), 90
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
            for _ in range(exec_len - 5):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
            tar_pos, tar_angle = robot.get_position() + np.array([-(0.35 + 0.45 + 0.3), 0, 0]), 180
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
            for _ in range(exec_len - 5):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
            tar_pos, tar_angle = robot.get_position() + np.array([0, +(1.2 + 0.4 * 2), 0]), 90
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
            
            tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 0
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len - 5):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

            tar_pos, tar_angle = robot.get_position() + np.array([0.0 - robot.get_position()[0], 0, 0]), 0
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

            tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), -90
            delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
            for _ in range(exec_len - 5):
                curr_pos += delta_pos
                curr_angle += delta_angle
                robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
                for _ in range(1):
                    og.sim.step()
                if args.save_data:
                    camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)

            tar_pos, tar_angle = np.array([0.0, 0.235, 0.0]), -90
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
    #     # nonlocal waypoints, plate_done, knife_done, spoon_done, fork_done
    #     nonlocal tablewares, not_on_plate, on_plate, plate_dict
    #     nonlocal is_usd_fork, is_spoon_90, t_quat, t_pos
    #     nonlocal robot, robot_move_mid_left, robot_move_left_right, init_quat
    #     global action, gripper

    #     # print("haha")

    #     # Check if we've received a key press or repeat
    #     if event.type == carb.input.KeyboardEventType.KEY_PRESS \
    #             or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            
    #         if event.input == carb.input.KeyboardInput.ENTER:
    #             # Execute the command
    #             print(f"moving to: pos {command_pos}, quat {command_ori}")
    #             command_pos_, command_ori_ = T.relative_pose_transform(*(command_pos, command_ori), *robot.get_position_orientation())
    #             execute_ik(pos=command_pos_, quat=command_ori_)

    #         elif event.input == carb.input.KeyboardInput.ESCAPE:
    #             # Get the waypoint
    #             ## Some Logic:
    #             #  1. First grasp and place the tablewares NOT ON the plate.
    #             #  2. Then proceed to the tablewares ON the plate (from top to bottom, by Z-axis).
    #             #     This needs 3 waypoints: tableware_high --> tableware_low (grasp) --> tableware_high.
    #             #     Move to the target location then open the gripper to place the tableware.
    #             #     This needs 2 waypoints: --> target_high --> target_low (open gripper) --> target_high.

    #             # if len(not_on_plate)
    #             if not robot_move_mid_left:
    #                 curr_pos, curr_angle = robot.get_position(), 0
    #                 exec_len = 15
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([-0.3, 0, 0]), 0
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), -90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, -(0.6 + 0.3), 0]), -90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 0
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0.0 - robot.get_position()[0], 0, 0]), 0
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()

    #                 tar_pos, tar_angle = np.array([0.0, -1.235, 0.0]), 90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 robot_move_mid_left = True


    #             elif not tablewares["plate_0"]["done"]:

    #                 if len(tablewares["plate_0"]["not_on_plate"]) > 0:
    #                     tableware = tablewares["plate_0"]["not_on_plate"][0]
    #                     print("not on plate: ", tableware)
    #                     if not tablewares["plate_0"][tableware]["g_high_0"]:
    #                         command_pos = tablewares["plate_0"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["g_high_0"] = True
    #                     elif not tablewares["plate_0"][tableware]["g_low"]:
    #                         command_pos = tablewares["plate_0"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.6])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["g_low"] = True
    #                     elif not tablewares["plate_0"][tableware]["g_high_1"]:
    #                         command_pos = tablewares["plate_0"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["g_high_1"] = True
    #                     elif not tablewares["plate_0"][tableware]["t_high_0"]:
    #                         command_pos = tablewares["plate_0"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["t_high_0"] = True
    #                     elif not tablewares["plate_0"][tableware]["t_low"]:
    #                         command_pos = tablewares["plate_0"][tableware]["target"] + marker.size * np.array([0, 0, 1.0])

    #                         print("transformation calculation starts.....")
    #                         pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
    #                         pos1, quat1 = tablewares["plate_0"][tableware]["object"].get_position_orientation()
    #                         T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))

    #                         if tableware == "knife":
    #                             rad = np.pi / 180 * 180
    #                             quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "spoon":
    #                             if is_spoon_90["spoon_0"]:
    #                                 rad = np.pi / 180 * 90
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * 0
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "fork":
    #                             if is_usd_fork["fork_0"]:
    #                                 rad = np.pi / 180 * 0
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * 180
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)

    #                         T_wu = T.pose2mat((command_pos - 0.1 * marker.size * np.array([0, 0, 1.0]), quat))
    #                         T_we = T_wu @ T.pose_inv(T_eu)
    #                         command_pos, command_ori = T.mat2pose(T_we)
    #                         marker.set_position_orientation(command_pos, command_ori)

    #                         tablewares["plate_0"][tableware]["t_low"] = True
    #                     elif not tablewares["plate_0"][tableware]["t_high_1"]:
    #                         command_pos = tablewares["plate_0"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["t_high_1"] = True
    #                         tablewares["plate_0"][tableware]["done"] = True
    #                         tablewares["plate_0"]["not_on_plate"] = tablewares["plate_0"]["not_on_plate"][1:]
                    
    #                 elif len(tablewares["plate_0"]["on_plate"]) > 0:
    #                     tableware = tablewares["plate_0"]["on_plate"][0]
    #                     print("on plate: ", tableware)
    #                     if not tablewares["plate_0"][tableware]["g_high_0"]:
    #                         command_pos = tablewares["plate_0"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["g_high_0"] = True
    #                     elif not tablewares["plate_0"][tableware]["g_low"]:
    #                         command_pos = tablewares["plate_0"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.68])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["g_low"] = True
    #                     elif not tablewares["plate_0"][tableware]["g_high_1"]:
    #                         command_pos = tablewares["plate_0"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["g_high_1"] = True
    #                     elif not tablewares["plate_0"][tableware]["t_high_0"]:
    #                         command_pos = tablewares["plate_0"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["t_high_0"] = True
    #                     elif not tablewares["plate_0"][tableware]["t_low"]:
    #                         command_pos = tablewares["plate_0"][tableware]["target"] + marker.size * np.array([0, 0, 1.0])

    #                         print("transformation calculation starts.....")
    #                         pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
    #                         pos1, quat1 = tablewares["plate_0"][tableware]["object"].get_position_orientation()
    #                         T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))

    #                         if tableware == "knife":
    #                             rad = np.pi / 180 * 180
    #                             quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "spoon":
    #                             if is_spoon_90["spoon_0"]:
    #                                 rad = np.pi / 180 * 90
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * 0
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "fork":
    #                             if is_usd_fork["fork_0"]:
    #                                 rad = np.pi / 180 * 0
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * 180
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)

    #                         T_wu = T.pose2mat((command_pos - 0.1 * marker.size * np.array([0, 0, 1.0]), quat))
    #                         T_we = T_wu @ T.pose_inv(T_eu)
    #                         command_pos, command_ori = T.mat2pose(T_we)
    #                         marker.set_position_orientation(command_pos, command_ori)

    #                         tablewares["plate_0"][tableware]["t_low"] = True
    #                     elif not tablewares["plate_0"][tableware]["t_high_1"]:
    #                         command_pos = tablewares["plate_0"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_0"][tableware]["t_high_1"] = True
    #                         tablewares["plate_0"][tableware]["done"] = True
    #                         tablewares["plate_0"]["on_plate"] = tablewares["plate_0"]["on_plate"][1:]

    #                 if len(tablewares["plate_0"]["not_on_plate"]) == 0 and len(tablewares["plate_0"]["on_plate"]) == 0:
    #                     tablewares["plate_0"]["done"] = True
                
    #             elif not robot_move_left_right:

    #                 curr_pos, curr_angle = robot.get_position(), 90
    #                 exec_len = 15
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, -0.3, 0]), 90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 180
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([-(0.35 + 0.45 + 0.3), 0, 0]), 180
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, +(1.2 + 0.4 * 2), 0]), 90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()
                    
    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), 0
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()

    #                 tar_pos, tar_angle = robot.get_position() + np.array([0.0 - robot.get_position()[0], 0, 0]), 0
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()

    #                 tar_pos, tar_angle = robot.get_position() + np.array([0, 0, 0]), -90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()

    #                 tar_pos, tar_angle = np.array([0.0, 0.235, 0.0]), -90
    #                 delta_pos, delta_angle = (tar_pos - curr_pos) / exec_len, (tar_angle - curr_angle) / exec_len
    #                 for _ in range(exec_len):
    #                     curr_pos += delta_pos
    #                     curr_angle += delta_angle
    #                     robot.set_position_orientation(curr_pos, T.quat_multiply(T.euler2quat([0, 0, np.pi / 180 * curr_angle]), init_quat))
    #                     for _ in range(1):
    #                         og.sim.step()

    #                 robot_move_left_right = True


    #             elif not tablewares["plate_1"]["done"]:

    #                 if len(tablewares["plate_1"]["not_on_plate"]) > 0:
    #                     tableware = tablewares["plate_1"]["not_on_plate"][0]
    #                     print("not on plate: ", tableware)
    #                     if not tablewares["plate_1"][tableware]["g_high_0"]:
    #                         command_pos = tablewares["plate_1"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["g_high_0"] = True
    #                     elif not tablewares["plate_1"][tableware]["g_low"]:
    #                         command_pos = tablewares["plate_1"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.6])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["g_low"] = True
    #                     elif not tablewares["plate_1"][tableware]["g_high_1"]:
    #                         command_pos = tablewares["plate_1"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["g_high_1"] = True
    #                     elif not tablewares["plate_1"][tableware]["t_high_0"]:
    #                         command_pos = tablewares["plate_1"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["t_high_0"] = True
    #                     elif not tablewares["plate_1"][tableware]["t_low"]:
    #                         command_pos = tablewares["plate_1"][tableware]["target"] + marker.size * np.array([0, 0, 1.0])

    #                         print("transformation calculation starts.....")
    #                         pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
    #                         pos1, quat1 = tablewares["plate_1"][tableware]["object"].get_position_orientation()
    #                         T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))

    #                         if tableware == "knife":
    #                             rad = np.pi / 180 * 0
    #                             quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "spoon":
    #                             if is_spoon_90["spoon_1"]:
    #                                 rad = np.pi / 180 * -90
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * -180
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "fork":
    #                             if is_usd_fork["fork_1"]:
    #                                 rad = np.pi / 180 * -180
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * 0
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)

    #                         T_wu = T.pose2mat((command_pos - 0.1 * marker.size * np.array([0, 0, 1.0]), quat))
    #                         T_we = T_wu @ T.pose_inv(T_eu)
    #                         command_pos, command_ori = T.mat2pose(T_we)
    #                         marker.set_position_orientation(command_pos, command_ori)

    #                         tablewares["plate_1"][tableware]["t_low"] = True
    #                     elif not tablewares["plate_1"][tableware]["t_high_1"]:
    #                         command_pos = tablewares["plate_1"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["t_high_1"] = True
    #                         tablewares["plate_1"][tableware]["done"] = True
    #                         tablewares["plate_1"]["not_on_plate"] = tablewares["plate_1"]["not_on_plate"][1:]
                    
    #                 elif len(tablewares["plate_1"]["on_plate"]) > 0:
    #                     tableware = tablewares["plate_1"]["on_plate"][0]
    #                     print("on plate: ", tableware)
    #                     if not tablewares["plate_1"][tableware]["g_high_0"]:
    #                         command_pos = tablewares["plate_1"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["g_high_0"] = True
    #                     elif not tablewares["plate_1"][tableware]["g_low"]:
    #                         command_pos = tablewares["plate_1"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 0.68])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["g_low"] = True
    #                     elif not tablewares["plate_1"][tableware]["g_high_1"]:
    #                         command_pos = tablewares["plate_1"][tableware]["object"].get_position() + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["g_high_1"] = True
    #                     elif not tablewares["plate_1"][tableware]["t_high_0"]:
    #                         command_pos = tablewares["plate_1"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["t_high_0"] = True
    #                     elif not tablewares["plate_1"][tableware]["t_low"]:
    #                         command_pos = tablewares["plate_1"][tableware]["target"] + marker.size * np.array([0, 0, 1.0])

    #                         print("transformation calculation starts.....")
    #                         pos0, quat0 = robot.get_eef_position(), robot.get_eef_orientation()
    #                         pos1, quat1 = tablewares["plate_1"][tableware]["object"].get_position_orientation()
    #                         T_eu = T.pose2mat(T.relative_pose_transform(pos1, quat1, pos0, quat0))

    #                         if tableware == "knife":
    #                             rad = np.pi / 180 * 0
    #                             quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "spoon":
    #                             if is_spoon_90["spoon_1"]:
    #                                 rad = np.pi / 180 * -90
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * -180
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                         if tableware == "fork":
    #                             if is_usd_fork["fork_1"]:
    #                                 rad = np.pi / 180 * -180
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)
    #                             else:
    #                                 rad = np.pi / 180 * 0
    #                                 quat = T.quat_multiply(T.euler2quat([0, 0, rad]), t_quat)

    #                         T_wu = T.pose2mat((command_pos - 0.1 * marker.size * np.array([0, 0, 1.0]), quat))
    #                         T_we = T_wu @ T.pose_inv(T_eu)
    #                         command_pos, command_ori = T.mat2pose(T_we)
    #                         marker.set_position_orientation(command_pos, command_ori)

    #                         tablewares["plate_1"][tableware]["t_low"] = True
    #                     elif not tablewares["plate_1"][tableware]["t_high_1"]:
    #                         command_pos = tablewares["plate_1"][tableware]["target"] + marker.size * np.array([0, 0, 4])
    #                         command_ori = get_eef_orientation(z_rot_degree=0)
    #                         marker.set_position_orientation(command_pos, command_ori)
    #                         tablewares["plate_1"][tableware]["t_high_1"] = True
    #                         tablewares["plate_1"][tableware]["done"] = True
    #                         tablewares["plate_1"]["on_plate"] = tablewares["plate_1"]["on_plate"][1:]

    #                 if len(tablewares["plate_1"]["not_on_plate"]) == 0 and len(tablewares["plate_1"]["on_plate"]) == 0:
    #                     tablewares["plate_1"]["done"] = True


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
    #                     # action[-2] = -1.0
    #                     action[-1] = -1.0
    #                     gripper = False
    #                 else:
    #                     # action[-2] = 1.0
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
        
    #     if count % 10 == 0 and args.save_data:
    #         camera_save_obs(data_dir=args.save_dir, camera_list=camera_ls)
    #     count += 1

    # # Always shut the simulation down cleanly at the end
    # og.app.close()



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
    parser.add_argument('--save_dir', default='og_tableware_data', type=str, help="the root directory of saving og tableware data")
    # parser.add_argument('--get_overview', action='store_true', help="get an overview of all the camera outputs")
    # parser.add_argument('--multi_category',  action='store_true', help="whether to load objects from multiple categories")
    # parser.add_argument('--single_category',  action='store_true', help="whether to load objects from a single category")
    # parser.add_argument('--max_num_plate', default=0, type=int, help="the maximum number of plates to load")
    # parser.add_argument('--max_num_knife', default=0, type=int, help="the maximum number of knives to load")
    # parser.add_argument('--max_num_spoon', default=0, type=int, help="the maximum number of spoons to load")
    # parser.add_argument('--max_num_fork', default=0, type=int, help="the maximum number of forks to load")
    args = parser.parse_args()

    # assert args.multi_category != args.single_category

    # if not args.multi_category and args.single_category:
    #     max_num_objects = np.array([args.max_num_plate, args.max_num_knife, args.max_num_spoon, args.max_num_fork])
    #     assert sum(max_num_objects != 0) == 1
    #     run_simulation(args=args)

    # elif args.multi_category and not args.single_category:
    #     args.max_num_plate = 1
    #     args.max_num_knife = 1
    #     args.max_num_spoon = 1
    #     args.max_num_fork  = 1
    args.max_num_plate = 2
    args.max_num_knife = 2
    args.max_num_spoon = 2
    args.max_num_fork  = 2
    run_simulation(args=args)