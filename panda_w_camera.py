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
        
        self.camera_data_idx = 0
        self.camera_capture_interval = 20
        self.scene_pc = []

        self.crop_size = 0

        if not os.path.exists('data'):
            os.makedirs('data')
    
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
        img = self.bullet_client.getCameraImage(self.img_size, self.img_size, view_matrix, projection_matrix)

        # points is a 3D numpy array (n_points, 3) coordinates
        depth_img = img[3]
        # crop the top part of the depth image to get rid of showing the EE 
        depth_img = depth_img[self.crop_size:, self.crop_size:]
        # pc = self.get_point_cloud(depth_img, projection_matrix, view_matrix, self.camera_data_idx)

        if save_data:
            self._save_color_depth_transform(img, self.camera_data_idx)
        self.camera_data_idx += 1

        rgb = img[2][:, :, :3]

        return rgb, depth_img, self.cam_lookat
    
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


    def move_hand_forward(self, dist_step, frame_idx, direction="x"):
        curr_hand_pos = self.panda.get_hand_pose()[0]
        if direction == "x":
            new_hand_pos = curr_hand_pos + np.array([ dist_step, 0, 0])
        elif direction == "y":
            new_hand_pos = curr_hand_pos + np.array([ 0, dist_step, 0])
        while True:
            self.panda.movej_newpos_ik(self.panda.hand_idx, new_hand_pos)
            self.bullet_client.stepSimulation()
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            curr_hand_pos = self.panda.get_hand_pose()[0]
            if np.abs(curr_hand_pos[1] - new_hand_pos[1]) < 0.1:
                return frame_idx
            

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


    def expert_demo_pick_and_place(self):
        # wait for the scene to stablize
        frame_idx = 0
        for _ in range(100):
            # rgb, depth, _ = panda.panda_camera()
            self.bullet_client.stepSimulation()

        # keypoints to reach the object
        EE_loc = self.panda.get_hand_pose()[0]
        initial_EE_loc = EE_loc.copy()
        object_loc = np.array([0.68, 0.0, 0.13])

        keypoints = self._lerp(EE_loc, object_loc, 50)

        # inverse kinematics to reach the keypoints
        for keypoint in keypoints:
            self.panda.movej_newpos_ik(self.panda.hand_idx, keypoint)
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            self.bullet_client.stepSimulation()

        # keep the hand in the same position
        for _ in range(20):
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            self.bullet_client.stepSimulation()
        
        # close the gripper
        frame_idx = self.panda.close_gripper(frame_idx)

        # move up the object
        keypoints = self._lerp(object_loc, EE_loc, 15)
        for keypoint in keypoints:
            self.panda.movej_newpos_ik(self.panda.hand_idx, keypoint)
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            self.bullet_client.stepSimulation()
            time.sleep(0.1)

        # rotate the wrist
        curr_speed = 0.3
        for _ in range(50):
            width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=640, height=480) 
            frame = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'frames/frame_{frame_idx:04d}.png', frame)
            frame_idx += 1
            self.bullet_client.stepSimulation()
            # get RGB-D image and camera position/orientation relative to the world/robot base frame
            curr_pos = self.panda.get_joint_state(self.side_to_side_joint)[0]
            # to stablize the wrist from not chaning the direction
            theta = curr_pos + curr_speed
            joint_angle = theta  
            # Set the joint position to move the wrist side to side
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, self.side_to_side_joint, self.bullet_client.POSITION_CONTROL, targetPosition=joint_angle)

        # move the hand forward
        frame_idx = self.move_hand_forward(0.18, frame_idx, direction="y")

        # open the gripper
        frame_idx = self.panda.open_gripper(frame_idx)

        # move the hand back
        frame_idx = self.move_hand_forward(-0.18, frame_idx, direction="y")

if __name__ == '__main__':
    # panda = PandaCamera(gui_enabled=True)
    # # panda.expert_demo_push()
    # panda.expert_demo_pick_and_place()
    # # panda.move_wrist(direction=-1)
    # # panda.move_hand_forward(0.15)
    # # panda.move_wrist()
    # # panda.move_wrist(direction=-1)
    # while True:
    #     # rgb, depth, _ = panda.panda_camera()
    #     panda.bullet_client.stepSimulation()

    import cv2
    import os
    import glob

    img_array = []
    for filename in sorted(glob.glob('frames/*.png')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # Use 'mp4v' and .mp4 extension for the output video file
    out = cv2.VideoWriter('pickandplace_drawer.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
