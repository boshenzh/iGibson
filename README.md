# Script for Picking Up / Dropping Objects in iGibson
<img src="./docs/images/knife_grab_drop.gif" width="200"> 

This project is built based on [iGibson](https://github.com/StanfordVL/iGibson), a Simulation Environment designed by [Stanford Vision and Learning Lab (SVL)](https://svl.stanford.edu/) for training robots in large, realistic, interactive scenes.

I have written scripts to control the Fetch robot, enabling it to pick up an object from a table and drop it at a specified location.

To use the script, please follow the instructions in the [iGibson Documentation](http://svl.stanford.edu/igibson/docs/) to quickly start and download the required assets and datasets for an interactive scene.


## Installing

### Building from source
it is useful to setup a conda environment with Python 3.8

```
git clone https://github.com/boshenzh/iGibson.git --recursive
cd iGibson

# if you didn't create the conda environment before:
conda create -y -n igibson python=3.8
conda activate igibson

pip install -e . # This step takes about 4 minutes
```

## Code Structure Overview 

`igibson/scripts` contains:

- `move_object_to.py`: instantiates a Fetch robot in an iGibson interactive scene. It plans and executes a trajectory generated by RRT to a graspable object location. Then, it closes the gripper, grasps the object, plans and executes another trajectory to the target location, and drops the object.

- `move_arm_to_object.py`: move Fetch robot arm to specific position using user keyboard command. Also enabling gripper control.

`igibson/configs` contains:

- `objectname_id.yaml`:configuration file for each object to be loaded in the scene and to be grabed/droped using `move_object_to.py`.

`igibson/motion_planning_wrapper.py` : contains motion planing helper methods including IK and RRT

## Usage

to use the script, first initialize the object in the scene by specifying object yaml file like following:

```
name: bowl_1  #same as file name
category: bowl 
model: "68_0" 
pos: [0.6, -0.7, 0.535]
orn: [0, 0, 0]
offset: [-0.05, 0, 0.04]  #grasp point offset from pos, differ by object,
```

offset of each object can be obtained by playing with `move_arm_to_object.py` with keyboard to find a graspable position.

```
python -m igibson.script.move_arm_to_object --keyboard 
```

after creating object configuration file, we can use script `move_object_to.py` with target object and specified location x, y, and z

```
python -m igibson.script.move_object_to bowl1 0.5 0.5 0.5 
```

