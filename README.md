# Pybullet Playground

This repository is for pybullet development of visual-based manipulation tasks.

## About

### Scene Probing

The panda arm moves its wrist joint to capture depth images from the scene and convert these into the scene point cloud. 

### panda_w_camera.py

The new point clouds are added to the buffer as the camera moves. The points are generated from only the depth images, not color images.

This shows simple robot camera trajectory. Moves joint 4 by a certain degree every time step.
<img src="gifs/probing_movement.gif" width="350" height="350"/>

Capture point cloud every 20 timesteps. This shows the progress of scene point cloud developing. As the camera looks toward the shelf, the shelf point cloud gets filled. Caveat: the ratio between the point clouds in this gif need to be fixed.

<img src="gifs/pc_progress.gif" width="350" height="350"/>


### pointcloud_test.ipynb

Test code to convert a pair of color+depth images into a single point cloud data using open3D. The camera to robot base transformation is currently not working. (stopped developing; chose to output pc directly from depth image and doing post-processing in pybullet)

## Installation

Create a python3 virtual environment and install the dependencies.

Note that the example command below runs on a machine where python3 is located at ```/usr/bin/python3.8```. Please change this accordingly based on the specific python3 path on your machine.

```
virtualenv -p /usr/bin/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```