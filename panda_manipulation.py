import numpy as np
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)
p.setTimeStep(0.01)

# Add plane
plane_id = p.loadURDF("plane_transparent.urdf")
# duck_id = p.loadURDF("objects/mug.urdf", [1, 0, 0.1])
p.loadURDF("random_urdfs/002/002.urdf", [1, 0, 0.1])
p.loadURDF("random_urdfs/001/001.urdf", [1.1, 0, 0.1])
p.loadURDF("random_urdfs/003/003.urdf", [0.9, 0, 0.1])
p.loadURDF("random_urdfs/004/004.urdf", [1.1, 0.1, 0.1])
p.loadURDF("random_urdfs/005/005.urdf", [0.8, -0.1, 0.1])
# p.loadURDF("kiva_shelf/model.sdf", [0.5, 0, 0.1], useFixedBase=True)
p.loadURDF("assets/shelf/shelf.urdf", [1, 0.5, 0.01], [0,0,1,1], useFixedBase=True, globalScaling=0.7)

def create_black_mat():
    # Define the region size and position
    region_half_extents = [0.6, 0.6, 0.01]  # x, y, and z half extents of the region
    region_position = [0.7, 0.0, 0.0]  # x, y, and z position of the region

    # Create a custom collision shape for the region (box shape)
    region_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=region_half_extents)

    # Create a multi-body (no mass) to represent the region
    region_visual = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=region_shape)

    # Set the position of the region
    p.resetBasePositionAndOrientation(region_visual, region_position, [0, 0, 1, 1])

    # Set the color of the region to gray (RGBA format: R, G, B, A)
    gray_color = [0.5, 0.5, 0.5, 1]
    p.changeVisualShape(region_visual, -1, rgbaColor=gray_color)

def reset_camera_visualizer():
    # Change the camera view
    camera_distance = 2.0
    camera_yaw = 0
    camera_pitch = -20
    camera_target_position = [0.5, 0, 0.15] 
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
create_black_mat()
reset_camera_visualizer()
# Add panda bot
start_pos = [0, 0, 0.001]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
panda_id = p.loadURDF("assets/franka_panda/panda.urdf", start_pos, start_orientation, useFixedBase=True)
camera_link_ind = 12

# initialize joint position
initial_pos = [-0., 0.4, 0, -1.2, 0, 2, 0.8]
for i in range(7):
    p.resetJointState(panda_id, i, initial_pos[i])

fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

for i in range(p.getNumJoints(panda_id)):
    print(p.getJointInfo(panda_id, i))


def panda_camera(): 
    # Center of mass position and orientation (of link-7)
    com_p, com_o, _, _, _, _ = p.getLinkState(panda_id, camera_link_ind)
    p.addUserDebugPoints([p.getLinkState(panda_id, camera_link_ind)[0]], [[1,0,0]], 5)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 0, 1) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p, com_p + 0.5 * camera_vector, up_vector)
    p.addUserDebugLine(com_p, com_p + 0.5 * camera_vector, [0, 0, 1], 3, 5)
    img = p.getCameraImage(256, 256, view_matrix, projection_matrix)
    return img


# Main loop
while True:
    p.stepSimulation()
    panda_camera()