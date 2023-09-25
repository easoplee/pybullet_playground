import pybullet
import numpy as np
import math
import cv2

class PandaGripperD435(object):
    def __init__(self, bullet_client):
        super(PandaGripperD435, self).__init__()
        
        self.bullet_client = bullet_client

        self.hand_idx = 8
        self.writst_camera_idx = 12
    
        start_pos = [0, 0, 0]
        start_orientation = self.bullet_client.getQuaternionFromEuler([0, 0, 0])
        
        self.panda_id = self.bullet_client.loadURDF("assets/franka_panda/panda.urdf", start_pos, start_orientation, useFixedBase=True)
        self._movable_joints = self.get_movable_joints()
        print(self._movable_joints)

        self.reset()

    def reset(self):
        initial_pos = [-0., 0.6, 0, -1.2, 0, 2, 0.8]
        for i in range(7):
            self.bullet_client.resetJointState(self.panda_id, i, initial_pos[i])
        for each in [9, 10]:
            self.bullet_client.resetJointState(self.panda_id, each, 0.04)

    # returns id's of all the movable robot joints
    def get_movable_joints(self):
        movable_joint_ids = []
        for idx in range(self.bullet_client.getNumJoints(self.panda_id)):
            joint_info = self.bullet_client.getJointInfo(self.panda_id, idx)
            q_index = joint_info[3]
            if q_index > -1:
                movable_joint_ids.append(idx)
    
        return movable_joint_ids

    def get_joint_state(self, joint_id=None):
        """
        :return: joint positions, velocity, reaction forces, joint efforts as given from bullet physics
        :rtype: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if joint_id is None:

            joint_angles = []
            joint_velocities = []
            joint_reaction_forces = []
            joint_efforts = []

            for idx in self._movable_joints:
                joint_state = self.bullet_client.getJointState(self.panda_id, idx)
                joint_angles.append(joint_state[0])
                joint_velocities.append(joint_state[1])
                joint_reaction_forces.append(joint_state[2])
                joint_efforts.append(joint_state[3])

            return np.array(joint_angles), np.array(joint_velocities), np.array(joint_reaction_forces), np.array(joint_efforts)

        else:
            joint_state = self.bullet_client.getJointState(self.panda_id, joint_id)
            joint_angle = joint_state[0]
            joint_velocity = joint_state[1]
            joint_reaction_forces = joint_state[2]
            joint_effort = joint_state[3]

            return joint_angle, joint_velocity, np.array(joint_reaction_forces), joint_effort

    def get_link_pose(self, link_id):
        """
        :return: Pose of link (Cartesian positionof center of mass, Cartesian orientation of center of mass in quaternion [x,y,z,w]) 
        :rtype: [np.ndarray, np.quaternion]
        :param link_id: optional parameter to specify the link id. If not provided, will return pose of end-effector
        :type link_id: int
        """

        link_state = self.bullet_client.getLinkState(self.panda_id, link_id)
        pos = np.asarray(link_state[0])
        ori = np.array([link_state[5][0], link_state[5][1], link_state[5][2], link_state[5][3], ])  # hamilton convention

        return pos, ori

    def get_hand_pose(self):
        """
        :return: end-effector pose of this robot in the format (position,orientation)
        .. note: orientation is a quaternion, i.e. (x, y, z, w)
        """
        return self.get_link_pose(link_id=self.hand_idx)
    
    def get_camera_pos(self):
        """
        :return: wrist mounted camera pose on this robot in the format (position,orientation)
        .. note: orientation is a quaternion, i.e. (x, y, z, w)
        """
        return self.get_link_pose(link_id=self.writst_camera_idx)
    
    def get_camera_transformation_matrix(self, visualize=False):
        """
        :return: transformation matrix of the wrist mounted camera on this robot
        """
        transformation_matrix = np.zeros((4, 4))
        transformation_matrix[:3, :3] = np.array(self.bullet_client.getMatrixFromQuaternion(self.get_camera_pos()[1])).reshape(3, 3)
        transformation_matrix[:3, 3] = self.get_camera_pos()[0]
        transformation_matrix[3, 3] = 1

        if visualize: # turns on the camera coordinate frame
            cam_pose = self.get_camera_pos()

            base_to_camera_position, base_orientation_quaternion = cam_pose[0], cam_pose[1]
            base_to_camera_rot_matrix = np.array(self.bullet_client.getMatrixFromQuaternion(base_orientation_quaternion)).reshape(3,3)

            arrow_length = 0.5
            self.bullet_client.addUserDebugLine(base_to_camera_position, np.dot(base_to_camera_rot_matrix, np.array([arrow_length, 0, 0])) + base_to_camera_position, [0, 1, 0], 5, 0)
            self.bullet_client.addUserDebugLine(base_to_camera_position, np.dot(base_to_camera_rot_matrix, np.array([0, arrow_length, 0])) + base_to_camera_position, [1, 0, 0], 5, 0)
            self.bullet_client.addUserDebugLine(base_to_camera_position, np.dot(base_to_camera_rot_matrix, np.array([0, 0, arrow_length])) + base_to_camera_position, [0, 0, 1], 5, 0)

        return transformation_matrix
    
    def movej_newpos_ik(self, idx, new_pos):
        joint_pos = self.bullet_client.calculateInverseKinematics(self.panda_id, idx, new_pos)
        # print(joint_pos)
        # print(self._movable_joints)
        for i in range(len(self._movable_joints)):
            self.bullet_client.setJointMotorControl2(self.panda_id, i, self.bullet_client.POSITION_CONTROL, targetPosition=joint_pos[i])

    def close_gripper(self, frame_idx):
        # close the gripper
        for _ in range(50):
            self.bullet_client.setJointMotorControl2(self.panda_id, 9, self.bullet_client.POSITION_CONTROL, targetPosition=-0.03)
            self.bullet_client.setJointMotorControl2(self.panda_id, 10, self.bullet_client.POSITION_CONTROL, targetPosition=-0.03)
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            self.bullet_client.stepSimulation()

        return frame_idx

    def open_gripper(self, frame_idx):
        # open the gripper
        for _ in range(30):
            self.bullet_client.setJointMotorControl2(self.panda_id, 9, self.bullet_client.POSITION_CONTROL, targetPosition=0.04)
            self.bullet_client.setJointMotorControl2(self.panda_id, 10, self.bullet_client.POSITION_CONTROL, targetPosition=0.04)
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            self.bullet_client.stepSimulation()
        return frame_idx