import pybullet
import time
import pybullet_data

# Connect to PyBullet with GUI
pybullet.connect(pybullet.GUI)

# Load the Panda robot URDF
panda = pybullet.loadURDF("/Users/easoplee/Desktop/pybullet_playground/assets/franka_panda/panda.urdf", useFixedBase=True)

# Set gravity
pybullet.setGravity(0, 0, -9.81)

# Set additional search path for PyBullet
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane URDF
pybullet.loadURDF("plane.urdf", [0, 0, -0.01])

# Create sliders for controlling joint angles
num_joints = pybullet.getNumJoints(panda)
sliders = []
for i in range(num_joints):
    joint_info = pybullet.getJointInfo(panda, i)
    joint_name = joint_info[1].decode('UTF-8')
    joint_min = joint_info[8]
    joint_max = joint_info[9]

    # Check if the joint is not fixed
    if joint_max > joint_min:
        sliders.append(pybullet.addUserDebugParameter(joint_name, joint_min, joint_max, 0))

# Store the indices of non-fixed joints
non_fixed_joints = [i for i in range(num_joints) if pybullet.getJointInfo(panda, i)[8] < pybullet.getJointInfo(panda, i)[9]]

while True:
    # Update joint angles based on slider values
    for i, slider in enumerate(sliders):
        joint_angle = pybullet.readUserDebugParameter(slider)
        pybullet.setJointMotorControl2(panda, non_fixed_joints[i], pybullet.POSITION_CONTROL, targetPosition=joint_angle)

    # Step the simulation
    pybullet.stepSimulation()
    time.sleep(1./240.)
