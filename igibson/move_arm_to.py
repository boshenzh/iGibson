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
from igibson.external.pybullet_tools.utils import quat_from_euler
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

log = logging.getLogger(__name__)

def main(selection="user", headless=False, short_exec=False):
    """
    This script takes arm to desired position x, y, z in the scene. 
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # For the second and further selections, we either as the user or randomize
    # If the we are exhaustively testing the first selection, we randomize the rest
    
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

    s.viewer.initial_pos = [1.0, -1.4, 1.2]
    s.viewer.initial_view_direction = [-0.7, 0.7, -0.2]
    s.viewer.reset_viewer()

    # Reset the robot
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, -1])
    fetch.reset()

    # Objects to load: two tables, the first one is predefined model, the second, random for the same category
    table_objects_to_load = {
        "table_1": {
            "category": "coffee_table",
            "model": "19203",
            "pos": (0.6, -0.7, 0.4),
            "orn": (0, 0, 0),
        },
        "bowl": {
            "category": "bowl",
            "model": "68_0",
            "pos": (0.6, -0.7, 0.6),
            "orn": (0, 0, 0),
        },        
    }

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

    # bowl = YCBObject("024_bowl")
    # s.import_object(bowl)
    # table1 = table_objects_to_load["table_1"]
    # bowl.set_position_orientation((table1["pos"][0], table1["pos"][1], table1["pos"][2] + 0.2), [0, 0, 0, 1])

    def accurate_calculate_inverse_kinematics(robot_id, eef_link_id, target_pos, threshold, max_iter):
        

        print("IK solution to end effector position {}".format(target_pos))
        # Save initial robot pose
        state_id = p.saveState()

        max_attempts = 5
        solution_found = False
        joint_poses = None
        for attempt in range(1, max_attempts + 1):
            print("Attempt {} of {}".format(attempt, max_attempts))
            # Get a random robot pose to start the IK solver iterative process
            # We attempt from max_attempt different initial random poses
            sample_fn = get_sample_fn(robot_id, arm_joint_ids)
            sample = np.array(sample_fn())
            # Set the pose of the robot there
            set_joint_positions(robot_id, arm_joint_ids, sample)

            it = 0
            # Query IK, set the pose to the solution, check if it is good enough repeat if not
            while it < max_iter:

                joint_poses = p.calculateInverseKinematics(
                    robot_id,
                    eef_link_id,
                    target_pos,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                )
                joint_poses = np.array(joint_poses)[robot_arm_indices]

                set_joint_positions(robot_id, arm_joint_ids, joint_poses)

                dist = l2_distance(fetch.get_eef_position(), target_pos)
                if dist < threshold:
                    solution_found = True
                    break
                logging.debug("Dist: " + str(dist))
                it += 1

            if solution_found:
                print("Solution found at iter: " + str(it) + ", residual: " + str(dist))
                break
            else:
                print("Attempt failed. Retry")
                joint_poses = None

        restoreState(state_id)
        p.removeState(state_id)
        return joint_poses

    
    # where is the bowl located???? 
    # print(env.scene.objects_by_id)
    print("bowl body id {}".format(env.scene.objects_by_name["bowl_1"].get_body_ids()[0]))
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
    fetch.reset()
    fetch.keep_still()
    fetch.set_position_orientation([0, 0, 0], [0, 0, 0, -1])
    env.scene.reset_scene_objects()
    # let environment settle down
    for _ in range(100):
        s.step()
    close_gripper_action = np.zeros(fetch.action_dim)
    close_gripper_action[10] = -1
    open_gripper_action = np.zeros(fetch.action_dim)
    open_gripper_action[10] = 1

    step = 0
    mode = "gripper"
    
    while(step != max_steps):
        action_input = input("Enter target x, y, z  or enter open/close if want to grab: ")
        bowl = env.scene.objects_by_name["bowl_1"]
        bowl_id = bowl.get_body_ids()[0]

        if action_input == "close":
            fetch.apply_action(close_gripper_action)

            
            is_grasping = fetch.is_grasping()
            if is_grasping == 1:
                if(motion_planner.mp_obstacles.count(bowl_id) > 0):
                    motion_planner.mp_obstacles.remove(bowl_id)
            else:
                if(motion_planner.mp_obstacles.count(bowl_id) == 0):
                    motion_planner.mp_obstacles.append(bowl_id)
            for _ in range(10):
                s.step()
            print("is grasping:", fetch.is_grasping())
            print(motion_planner.mp_obstacles.count(bowl_id))

            continue
        elif action_input == "open":
            fetch.apply_action(open_gripper_action)
            is_grasping = fetch.is_grasping()
            if is_grasping == 1:
                if(motion_planner.mp_obstacles.count(bowl_id) > 0):
                    motion_planner.mp_obstacles.remove(bowl_id)
            else:
                if(motion_planner.mp_obstacles.count(bowl_id) == 0):
                    motion_planner.mp_obstacles.append(bowl_id)
            for _ in range(10):
                s.step()
            print("is grasping:", fetch.is_grasping())
            print(motion_planner.mp_obstacles.count(bowl_id))

            continue
        else:
            #move arm to position x y z
            x, y, z = action_input.split()
            target_pos = np.array([float(x), float(y), float(z)])
            
            
            success = False
            max_attempts=30
            print("bowl position {}".format(env.scene.objects_by_name["bowl_1"].get_position()))

            for attempt in range(1, max_attempts + 1):
                if success: 
                    break
                # get goal config using motion_planning_wrapper ik solver
                # TODO: optimize code. store joint positions in a array. 
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


            for _ in range(30):
                s.step()
        
    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
