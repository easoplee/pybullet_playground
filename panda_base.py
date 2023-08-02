import pybullet
import numpy as np

class PandaGripperD435(object):
    def __init__(self, bullet_client):
        super(PandaGripperD435, self).__init__()
        
        self.bullet_client = bullet_client

        self.hand_idx = 8
        self.writst_camera_idx = 12

        start_pos = [0, 0, 0.001]
        start_orientation = self.bullet_client.getQuaternionFromEuler([0, 0, 0])
        self.panda_id = self.bullet_client.loadURDF("assets/franka_panda/panda.urdf", start_pos, start_orientation, useFixedBase=True)
        self.reset()

    def reset(self):
        initial_pos = [-0., 0.4, 0, -1.2, 0, 2, 0.8]
        for i in range(7):
            self.bullet_client.resetJointState(self.panda_id, i, initial_pos[i])

    # returns id's of all the movable robot joints
    # def get_movable_joints(self):
    #     movable_joint_ids = []
    #     for idx in self._all_joints_ids:
    #         joint_info = self.bullet_client.getJointInfo(self.robot_id, idx)
    #         q_index = joint_info[3]
    #         if q_index > -1:
    #             movable_joint_ids.append(idx)

    #     return movable_joint_ids

    # def get_joint_state(self, joint_id=None):
    #     """
    #     :return: joint positions, velocity, reaction forces, joint efforts as given from bullet physics
    #     :rtype: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    #     """
    #     if joint_id is None:

    #         joint_angles = []
    #         joint_velocities = []
    #         joint_reaction_forces = []
    #         joint_efforts = []

    #         for idx in self._movable_joints:
    #             joint_state = self.bullet_client.getJointState(self.robot_id, idx)
    #             joint_angles.append(joint_state[0])
    #             joint_velocities.append(joint_state[1])
    #             joint_reaction_forces.append(joint_state[2])
    #             joint_efforts.append(joint_state[3])

    #         return np.array(joint_angles), np.array(joint_velocities), np.array(joint_reaction_forces), np.array(joint_efforts)

    #     else:
    #         joint_state = self.bullet_client.getJointState(self.robot_id, joint_id)
    #         joint_angle = joint_state[0]
    #         joint_velocity = joint_state[1]
    #         joint_reaction_forces = joint_state[2]
    #         joint_effort = joint_state[3]

    #         return joint_angle, joint_velocity, np.array(joint_reaction_forces), joint_effort

    # def get_link_pose(self, link_id):
    #     """
    #     :return: Pose of link (Cartesian positionof center of mass, Cartesian orientation of center of mass in quaternion [x,y,z,w]) 
    #     :rtype: [np.ndarray, np.quaternion]
    #     :param link_id: optional parameter to specify the link id. If not provided, will return pose of end-effector
    #     :type link_id: int
    #     """

    #     link_state = self.bullet_client.getLinkState(self.robot_id, link_id)
    #     pos = np.asarray(link_state[0])
    #     ori = np.array([link_state[1][3], link_state[1][0], link_state[1][1], link_state[1][2]])  # hamilton convention

    #     return pos, ori

    # def get_ee_pose(self):
    #     """
    #     :return: end-effector pose of this robot in the format (position,orientation)
    #     .. note: orientation is a quaternion following Hamilton convention, i.e. (w, x, y, z)
    #     """
    #     return self.get_link_pose(link_id=self.ee_idx)
    
    # def get_camera_pos(self):
    #     """
    #     :return: wrist mounted camera pose on this robot in the format (position,orientation)
    #     .. note: orientation is a quaternion following Hamilton convention, i.e. (w, x, y, z)
    #     """
    #     return self.get_camera_pos(link_id=self.writst_camera_idx)