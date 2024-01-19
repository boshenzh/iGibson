# Script for Picking Up / Dropping Objects in iGibson

This project is built based on [iGibson](https://github.com/StanfordVL/iGibson), a Simulation Environment designed by [Stanford Vision and Learning Lab (SVL)](https://svl.stanford.edu/) for training robots in large, realistic, interactive scenes.

I have written scripts to control the Fetch robot, enabling it to pick up an object from a table and drop it at a specified location.

To use the script, please follow the instructions in the [iGibson Documentation](http://svl.stanford.edu/igibson/docs/) to quickly start and download the required assets and datasets for an interactive scene.

### igibson/move_object_to.py
<img src="./docs/images/knife_grab_drop.gif" width="250"> 

This script instantiates a Fetch robot in an iGibson interactive scene. It plans and executes a trajectory generated by RRT to a graspable object location. Then, it closes the gripper, grasps the object, plans and executes another trajectory to the target location, and drops the object.

In this script, users can create an object in the scene by creating a *.yaml file. The script takes four arguments: the configuration file name and the target position (x, y, and z).
```
python -m igibson.move_object_to bowl_1 0.5 0.5 0.5 
```
the object yaml file should contain the following: 
```
name: bowl_1  #same as file name
category: bowl 
model: "68_0" 
pos: [0.6, -0.7, 0.535]
orn: [0, 0, 0]
offset: [-0.05, 0, 0.04]  #grasp point offset from pos, differ by object,
```

