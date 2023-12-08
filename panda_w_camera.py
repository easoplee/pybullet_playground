from panda_base import PandaGripperD435
from pybullet_base import PyBulletBase
import numpy as np
import time
import open3d as o3d
import matplotlib.pyplot as plt
import pybullet
from PIL import Image
import os
import pyvista as pv
import pybullet
import cv2
import random

class PandaCamera(PyBulletBase):
    def __init__(self, gui_enabled):
        super().__init__(gui_enabled)
        self.initialize_sim()
        self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        self.create_manipulation_scene()

        self.panda = PandaGripperD435(self.bullet_client)
        self.bullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,1)
        self.side_to_side_joint = 4

        # Camera parameters
        self.fov, self.aspect, self.nearplane, self.farplane = 60, 1.0, 0.01, 100
        self.init_camera_vector = (0, 0, 1) # z-axis
        self.init_up_vector = (0, 1, 0) # y-axis
        self.img_size = 256
        self.intrinsic_matrix = np.array([[952.828,     0.,     646.699 ],
                                          [0.,      952.828,     342.637 ],
                                          [0.,         0.,         1.  ]]) 
        self._movable_joints = self.get_movable_joints()
        
        # change time step to 1/60
        self.bullet_client.setTimeStep(1/45)
        
        self.camera_data_idx = 0
        self.camera_capture_interval = 20
        self.scene_pc = []

        self.crop_size = 0
        self.item = 4

        if not os.path.exists('data'):
            os.makedirs('data')

        
        # Define the camera's eye position (x, y, z)
        eye_pos = [0.1, -0.15, 0.4]  # Example position, you can modify this to change the distance and angle
        target_pos = [1, 0., 0.15]  # Looking at the origin (0, 0, 0)
        up_vector = [0, 0, 1]  # Z-axis is up

        self.viewMatrix = pybullet.computeViewMatrix(
            cameraEyePosition=eye_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector
        )

        # Define the projection matrix (for the field of view of the camera)
        fov = 60  # Field of view, in degrees
        aspect = 640 / 480  # Aspect ratio
        near = 0.02  # Near clipping plane
        far = 100  # Far clipping plane

        self.projectionMatrix = pybullet.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )

        self.frame_idx = 0
    
    def panda_camera(self, save_data=False): 
        # Center of mass position and orientation (of wrist camera index)
        com_p, com_o, _, _, _, _ = self.bullet_client.getLinkState(self.panda.panda_id, self.panda.writst_camera_idx)
        self.bullet_client.addUserDebugPoints([self.bullet_client.getLinkState(self.panda.panda_id, self.panda.writst_camera_idx)[0]], [[1,0,0]], 5)
        rot_matrix = self.bullet_client.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Rotated vectors
        camera_vector = rot_matrix.dot(self.init_camera_vector)
        up_vector = rot_matrix.dot(self.init_up_vector)
        view_matrix = self.bullet_client.computeViewMatrix(com_p, com_p + 0.5 * camera_vector, up_vector)
        self.cam_lookat = com_p + 0.5 * camera_vector
        self.bullet_client.addUserDebugLine(com_p, self.cam_lookat, [0, 0, 1], 3, 5)

        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)
        width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(self.img_size, self.img_size, view_matrix, projection_matrix)

        # # points is a 3D numpy array (n_points, 3) coordinates
        # depth_img = img[3]
        # # crop the top part of the depth image to get rid of showing the EE 
        # depth_img = depth_img[self.crop_size:, self.crop_size:]
        # # pc = self.get_point_cloud(depth_img, projection_matrix, view_matrix, self.camera_data_idx)

        frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'frames/frame_{self.frame_idx:04d}.png', frame)
        frame_mask = np.zeros_like(segImg, dtype=np.uint8)  # Create an empty black image with the same shape as segImg

        # Set pixels with value 2 to white (255)
        frame_mask[segImg == self.item] = 255
        cv2.imwrite(f'frames_mask/frame_{self.frame_idx:04d}.png', frame_mask)
        self.frame_idx += 1

        # if save_data:
        #     self._save_color_depth_transform(img, self.camera_data_idx)
        # self.camera_data_idx += 1

        # rgb = img[2][:, :, :3]

        # return rgb, depth_img, self.cam_lookat
        return 0
    
    def _save_color_depth_transform(self, img, idx):
        # process color and depth images
        rgb_opengl = (np.reshape(img[2], (self.img_size, self.img_size, 4)))
        rgbim = Image.fromarray(rgb_opengl)
        rgbim_no_alpha = rgbim.convert('RGB')

        depth_buffer_opengl = np.reshape(img[3], [self.img_size, self.img_size])
        depth_opengl = depth_buffer_opengl*(self.farplane-self.nearplane)+self.nearplane
        depth = depth_opengl.astype(np.uint16)
        new_p = Image.fromarray(depth)
        
        # create directory and save the images
        if not os.path.exists('data'):
            os.makedirs('data')
        rgbim_no_alpha.save(f'data/color{idx}.jpg')
        new_p.save(f'data/depth{idx}.png')

        # save the camera transformation matrix
        transformation_matrix = self.panda.get_camera_transformation_matrix()
        np.save(f'data/transformation_matrix{idx}.npy', transformation_matrix)

    def get_point_cloud(self, depth_img, projection_matrix, view_matrix, idx):
        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        crop_img_size = self.img_size - self.crop_size
        y, x = np.mgrid[-1:1:2 / crop_img_size, -1:1:2 / crop_img_size]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth_img.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]
        
        if idx % self.camera_capture_interval == 0:
            if idx == 0:
                self.scene_pc = points
            else:
                self.scene_pc = np.concatenate((self.scene_pc, points), axis=0)
            np.save('data/scene_pc.npy', self.scene_pc)
            cloud = pv.PolyData(self.scene_pc)
            cloud.plot(eye_dome_lighting=True)
            cloud.save('data/scene_pc.ply')
        return points
    
    def get_movable_joints(self):
        movable_joint_ids = []
        for idx in range(self.bullet_client.getNumJoints(self.panda.panda_id)):
            joint_info = self.bullet_client.getJointInfo(self.panda.panda_id, idx)
            q_index = joint_info[3]
            if q_index > -1:
                movable_joint_ids.append(idx)
    
        return movable_joint_ids

    def move_wrist(self, direction=1, speed = 0.5, camera=True):
        # wait for the scene to stablize
        for _ in range(100):
            self.bullet_client.stepSimulation()

        curr_speed = 0.3 * direction
        turn = 0
        initial_pos = self.panda.get_joint_state(self.side_to_side_joint)[0]
        while True:
            panda.panda.get_camera_transformation_matrix()
            self.bullet_client.stepSimulation()
            # get RGB-D image and camera position/orientation relative to the world/robot base frame
            if self.camera_data_idx % self.camera_capture_interval == 0:
                save_data = False
            else:
                save_data = False
            if camera:
                self.panda_camera(save_data=save_data)
            curr_pos = self.panda.get_joint_state(self.side_to_side_joint)[0]
            # to stablize the wrist from not chaning the direction
            if self.cam_lookat[2] > 0.65 and turn == 0:
                curr_speed *= -1  
                turn += 1
            if np.abs(initial_pos - curr_pos) < 0.01 and turn == 1:
                break
            
            theta = curr_pos + curr_speed

            joint_angle = theta  
            # Set the joint position to move the wrist side to side
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, self.side_to_side_joint, self.bullet_client.POSITION_CONTROL, targetPosition=joint_angle)

    def close_gripper(self):
        # close the gripper
        for _ in range(3):
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, 9, self.bullet_client.POSITION_CONTROL, targetPosition=-0.1)
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, 10, self.bullet_client.POSITION_CONTROL, targetPosition=-0.1)
            self.panda_camera()
            self.bullet_client.stepSimulation()

    def open_gripper(self):
        # open the gripper
        for _ in range(30):
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, 9, self.bullet_client.POSITION_CONTROL, targetPosition=0.04)
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, 10, self.bullet_client.POSITION_CONTROL, targetPosition=0.04)
            self.panda_camera()
            self.bullet_client.stepSimulation()

    def move_hand_forward(self, dist_step, direction="x"):
        curr_hand_pos = self.panda.get_hand_pose()[0]
        if direction == "x":
            new_hand_pos = curr_hand_pos + np.array([ dist_step, 0, 0])
        elif direction == "y":
            new_hand_pos = curr_hand_pos + np.array([ 0, dist_step, 0])
        while True:
            self.panda.movej_newpos_ik(self.panda.hand_idx, new_hand_pos)
            self.bullet_client.stepSimulation()
            self.panda_camera()
            curr_hand_pos = self.panda.get_hand_pose()[0]
            if np.abs(curr_hand_pos[1] - new_hand_pos[1]) < 0.1:
                break
            

    def _lerp(self, A, B, num_points):
        """
        Linearly interpolate between A and B.
        :param A: Starting point
        :param B: Ending point
        :param num_points: Number of interpolated points to return
        :return: A list of interpolated points between A and B
        """
        points = []
        for t in range(num_points + 1):
            t /= num_points
            C = [a + t * (b - a) for a, b in zip(A, B)]
            points.append(C)
        return points
    
    def expert_demo_push(self):
        # wait for the scene to stablize
        for _ in range(100):
            self.bullet_client.stepSimulation()

        # keypoints to reach the object
        EE_loc = self.panda.get_hand_pose()[0]
        initial_EE_loc = EE_loc.copy()
        object_loc = np.array([0.65, 0.2, 0.13])
        
        keypoints = self._lerp(EE_loc, object_loc, 20)

        # inverse kinematics to reach the keypoints
        for keypoint in keypoints:
            self.panda.movej_newpos_ik(self.panda.hand_idx, keypoint)
            self.bullet_client.stepSimulation()
            time.sleep(0.01)

        # keep the hand in the same position
        for _ in range(100):
            self.bullet_client.stepSimulation()

        # close the gripper
        self.panda.close_gripper()

        # push the object
        keypoints = self._lerp(object_loc, np.array([0.65, -0.0, 0.13]), 20)
        for keypoint in keypoints:
            self.panda.movej_newpos_ik(self.panda.hand_idx, keypoint)
            self.bullet_client.stepSimulation()
            time.sleep(0.01)

        # record object location
        object_loc,_ = self.bullet_client.getBasePositionAndOrientation(self.object)
        print(object_loc)

    def movej_newpos_ik(self, idx, new_pos):
        joint_pos = self.bullet_client.calculateInverseKinematics(self.panda.panda_id, idx, new_pos)
        for i in range(len(self._movable_joints)):
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, i, self.bullet_client.POSITION_CONTROL, targetPosition=joint_pos[i])
            self.panda_camera()


    def expert_demo_pick_and_place(self):
        # wait for the scene to stablize
        for _ in range(100):
            # rgb, depth, _ = panda.panda_camera()
            self.bullet_client.stepSimulation()

        # keypoints to reach the object
        EE_loc = self.panda.get_hand_pose()[0]
        initial_EE_loc = EE_loc.copy()
        object_loc = np.array([0.62, 0.0, 0.01])

        keypoints = self._lerp(EE_loc, object_loc, 5)

        # inverse kinematics to reach the keypoints
        for keypoint in keypoints:
            self.movej_newpos_ik(self.panda.hand_idx, keypoint)
            self.bullet_client.stepSimulation()

        # keep the hand in the same position
        for _ in range(5):
            self.panda_camera()
            self.bullet_client.stepSimulation()
        
        # close the gripper
        self.close_gripper()

        # move up the object
        keypoints = self._lerp(object_loc, EE_loc, 5)
        for keypoint in keypoints:
            self.movej_newpos_ik(self.panda.hand_idx, keypoint)
            self.bullet_client.stepSimulation()
            # time.sleep(0.1)

        # rotate the wrist
        curr_speed = 2
        for _ in range(15):
            self.panda_camera()
            self.bullet_client.stepSimulation()
            # get RGB-D image and camera position/orientation relative to the world/robot base frame
            curr_pos = self.panda.get_joint_state(self.side_to_side_joint)[0]
            # to stablize the wrist from not chaning the direction
            theta = curr_pos + curr_speed
            joint_angle = theta  
            # Set the joint position to move the wrist side to side
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, self.side_to_side_joint, self.bullet_client.POSITION_CONTROL, targetPosition=joint_angle)

        # move the hand forward
        self.frame_idx = self.move_hand_forward(0.18, direction="y")

        # open the gripper
        self.frame_idx = self.open_gripper()

        # move the hand back
        self.frame_idx = self.move_hand_forward(-0.18, direction="y")

    def open_drawer(self):
        pass
    
if __name__ == '__main__':
    panda = PandaCamera(gui_enabled=True)
    # panda.expert_demo_push()
    panda.expert_demo_pick_and_place()
    # panda.move_wrist(direction=-1)
    # panda.move_hand_forward(0.15)
    # panda.move_wrist()
    # panda.move_wrist(direction=-1)
    # while True:
    #     # rgb, depth, _ = panda.panda_camera()
    #     panda.bullet_client.stepSimulation()
    #     # time.sleep(0.01)