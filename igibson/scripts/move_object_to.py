import logging
import os
import platform

import numpy as np
import yaml
from collections import OrderedDict
import time
import random
import sys
import pybullet as p
import argparse
import igibson
from igibson.robots import REGISTERED_ROBOTS, ManipulationRobot
from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import quat_from_euler,euler_from_quat
from igibson.objects.articulated_object import URDFObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import let_user_pick, l2_distance, parse_config, restoreState
from igibson.robots.fetch import Fetch
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.external.pybullet_tools.utils import (
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    joints_from_names,
    set_joint_positions,
)
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from scipy.spatial.transform import Rotation   

log = logging.getLogger(__name__)

def main(headless=False, short_exec=False):
    """
    This script takes object to a desired location using fetch arm. object should be specified in the yaml file. 
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # take object yaml file and location as input
    parser = argparse.ArgumentParser()
    parser.add_argument('object')
    parser.add_argument('x')
    parser.add_argument('y')
    parser.add_argument('z')
    args = parser.parse_args()
    params = vars(args)
    
    # Load the robot, scene, etc configuration
    config_filename = os.path.join(igibson.configs_path, "fetch_motion_planning_empty.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # Improving visuals in the example (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True
    
    env = iGibsonEnv(
        config_file=config_data,
        mode="gui_interactive" if not headless else "headless",
        action_timestep=1.0 / 120.0,
        physics_timestep=1.0 / 120.0
    )
    s = env.simulator
    # Load robot
    fetch = env.robots[0]
    body_ids = fetch.get_body_ids()
    assert len(body_ids) == 1, "Fetch robot is expected to be single-body."
    robot_id = body_ids[0]

    # set camera position
    s.viewer.initial_pos = [1.0, -1.4, 1.2]
    s.viewer.initial_view_direction = [-0.7, 0.7, -0.2]
    s.viewer.reset_viewer()

    # Reset the robot
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, -1])
    fetch.reset()

    #file naem of the object to load
    objects_to_load = ["table_1"]

    grabbing_object_file_name = os.path.join(igibson.configs_path, params["object"]+".yaml")
    grabbing_object_config_data = yaml.load(open(grabbing_object_file_name, "r"), Loader=yaml.FullLoader)
    objects_to_load.append(grabbing_object_config_data["name"])
    table_objects_to_load = {}
    for object_name in objects_to_load:
        object_file_name = os.path.join(igibson.configs_path, object_name+".yaml")
        object_config_data = yaml.load(open(object_file_name, "r"), Loader=yaml.FullLoader)
        object_attribute = {}
        object_attribute["category"] = object_config_data["category"]
        object_attribute["model"] = object_config_data["model"]
        object_attribute["pos"] = object_config_data["pos"]
        object_attribute["orn"] = object_config_data["orn"]
        table_objects_to_load[object_name] = object_attribute
    # Objects to load: two tables, the first one is predefined model, the second, random for the same category
    # table_objects_to_load = {
    #     "table_1": {
    #         "category": "coffee_table",
    #         "model": "19203",
    #         "pos": [0.6, -0.7, 0.4],
    #         "orn": [0, 0, 0],
    #     },
    #     "bowl": {
    #         "category": "bowl",
    #         "model": "68_0",
    #         "pos": [0.6, -0.7, 0.52],
    #         "orn": [0, 0, 0],
    #     },        
    # }

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_ig_avg_category_specs()
    
    scene_objects = {}
    for obj in table_objects_to_load.values():
        category = obj["category"]
        if category in scene_objects:
            scene_objects[category] += 1
        else:
            scene_objects[category] = 1

        # Get the path for all models of this category
        category_path = get_ig_category_path(category)

        # If the specific model is given, we use it. If not, we select one randomly
        if "model" in obj:
            model = obj["model"]
        else:
            model = np.random.choice(os.listdir(category_path))

        # Create the full path combining the path for all models and the name of the model
        model_path = get_ig_model_path(category, model)
        filename = os.path.join(model_path, model + ".urdf")

        # Create a unique name for the object instance
        obj_name = "{}_{}".format(category, scene_objects[category])
        # Create and import the object
        simulator_obj = URDFObject(
            filename,
            name=obj_name,
            category=category,
            model_path=model_path,
            avg_obj_dims=avg_category_spec.get(category),
            fit_avg_dim_volume=True,
            texture_randomization=False,
            overwrite_inertial=True,
        )
        s.import_object(simulator_obj)
        simulator_obj.set_position_orientation(obj["pos"], quat_from_euler(obj["orn"]))


    
    # where is the bowl located???? 
    #print(env.scene.objects_by_name)
    #print("bowl body id {}".format(env.scene.objects_by_name["bowl_1"].get_body_ids()[0]))
    max_steps = -1 if not short_exec else 100
    
    full_observability_2d_planning = True
    collision_with_pb_2d_planning = True
    motion_planner = MotionPlanningWrapper(
        env,
        optimize_iter=10,
        full_observability_2d_planning=full_observability_2d_planning,
        collision_with_pb_2d_planning=collision_with_pb_2d_planning,
        visualize_2d_planning=not headless,
        visualize_2d_result=not headless,
    )
    # Reset the robot
    fetch.reset()
    fetch.keep_still()
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, -1])
    env.scene.reset_scene_objects()
    # let environment settle down
    for _ in range(100):
        s.step()

    step = 0    
    success = False
    while(step != max_steps):
        if success: 
            break
        
        object = env.scene.objects_by_name[grabbing_object_config_data['name']]
        object_id = object.get_body_ids()[0]
        object_pos = object.get_position()
        object_orn = euler_from_quat(object.get_orientation())
        object_offset = grabbing_object_config_data['offset']
        print(grabbing_object_config_data)
        #move arm to position object's pos with offset in config file
        print("object offset",object_offset)
        print("object orientation",object_orn)
        print("object position",object_pos)
        target_pos = target_with_offset(object_orn, object_pos,object_offset)

        max_attempts=30
        obj_to_grasp = env.scene.objects_by_name[grabbing_object_config_data['name']]
        obj_to_grasp = obj_to_grasp.get_body_ids()[0]
        moved_to_object_success = find_and_execute_motion_plan(motion_planner, target_pos, max_attempts)
        # TODO: grasp the object
        if(moved_to_object_success):
            grasp_object(fetch,motion_planner,obj_to_grasp,s)
            # TODO: move arm to location and release the object 
            drop_pos = np.array([float(params["x"]),float(params["y"]),float(params["z"])])
            success = find_and_execute_motion_plan(motion_planner, drop_pos, max_attempts)
            if(success):
                drop_object(fetch,motion_planner,obj_to_grasp,s)
            fetch.keep_still()
        for _ in range(30):
            s.step()
        
        
    s.disconnect()
def grasp_object(fetch,motion_planner,object_id,simulator):
    close_gripper_action = np.zeros(fetch.action_dim)
    close_gripper_action[10] = -1
    is_grasping = fetch.is_grasping()

    while is_grasping != 1:
        fetch.apply_action(close_gripper_action)
        is_grasping = fetch.is_grasping()
        for _ in range(10):
            simulator.step()
        print("closing gripper")
    print("grasped!")
    if is_grasping == 1:
        if(motion_planner.mp_obstacles.count(object_id) > 0):
            motion_planner.mp_obstacles.remove(object_id)
    else:
        if(motion_planner.mp_obstacles.count(object_id) == 0):
            motion_planner.mp_obstacles.append(object_id)

def drop_object(fetch,motion_planner,object_id,simulator):
    open_gripper_action = np.zeros(fetch.action_dim)
    open_gripper_action[10] = 1
    is_grasping = fetch.is_grasping()

    while is_grasping == 1:
        fetch.apply_action(open_gripper_action)
        is_grasping = fetch.is_grasping()
        for _ in range(10):
            simulator.step()
        print("opening gripper")
    print("dropped!")
    if is_grasping == 1:
        if(motion_planner.mp_obstacles.count(object_id) > 0):
            motion_planner.mp_obstacles.remove(object_id)
    else:
        if(motion_planner.mp_obstacles.count(object_id) == 0):
            motion_planner.mp_obstacles.append(object_id)

def find_and_execute_motion_plan(motion_planner, target_pos, max_attempts=30):
    success = False
    for attempt in range(1, max_attempts + 1):
        if success: 
            break
        # get goal config using motion_planning_wrapper ik solver
        joint_positions = motion_planner.get_arm_joint_positions(target_pos)
        print("joint_positions {}".format(joint_positions))
        if joint_positions is  None:
            print("IK Solver failed to find a configuration")
            continue
        motion_planner.simulator_sync()
        reach_plan = motion_planner.plan_arm_motion(joint_positions,override_fetch_collision_links=True)
        if reach_plan is not None:
            print("Executing planned arm reaching")
            motion_planner.cus_dry_run_arm_plan(reach_plan)                    
            print("End of the reach execution")
            success = True
        else:
            if max_attempts%attempt ==30:
                logging.error(
                    "MP couldn't find path to the arm pushing location. Attempt {} of {}".format(attempt, max_attempts)
                )

    if not success:
        logging.error("MP failed after {} attempts. Exiting".format(max_attempts))
    return success

def target_with_offset(euler,pos,offset):
    r = Rotation.from_euler("xyz",euler,degrees=False).as_matrix()

    print("rotation matrix : ", r)
    print("previous offset", offset)
    new_offset= np.matmul(r,np.transpose(np.matrix(offset)))
    print("new_offset :", new_offset)
    print("previous pos: ",pos)
    new_target_pos = pos + np.array(np.transpose(new_offset))
    print("new_position  :", new_target_pos)
    return new_target_pos[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
