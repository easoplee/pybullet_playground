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

class panda_camera(PyBulletBase):
    def __init__(self, gui_enabled):
        super().__init__(gui_enabled)
        self.initialize_sim()
        self.create_manipulation_scene()

        self.panda = PandaGripperD435(self.bullet_client)
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
        self.camera_capture_interval = 40
        self.scene_pc = []
    
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
        pc = self.get_point_cloud(img[3], projection_matrix, view_matrix, self.camera_data_idx)

        if save_data:
            self._save_color_depth_transform(img, self.camera_data_idx)
        self.camera_data_idx += 1

        rgb = img[2][:, :, :3]
        depth = img[3]

        return rgb, depth, self.cam_lookat
    
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
        y, x = np.mgrid[-1:1:2 / self.img_size, -1:1:2 / self.img_size]
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
            cloud = pv.PolyData(self.scene_pc)
            cloud.plot()
        return points
    
    def move_wrist(self):
        curr_speed = 0.1
        turn = 0
        while True:
            panda.panda.get_camera_transformation_matrix()
            self.bullet_client.stepSimulation()
            # get RGB-D image and camera position/orientation relative to the world/robot base frame
            if self.camera_data_idx % self.camera_capture_interval == 0:
                save_data = True
            else:
                save_data = False
            self.panda_camera(save_data=save_data)
            curr_pos = self.panda.get_joint_state(self.side_to_side_joint)[0]

            # to stablize the wrist from not chaning the direction
            if self.cam_lookat[2] > 0.65 and turn == 0:
                curr_speed *= -1  
                turn += 1
            if self.cam_lookat[2] > 0.75 and turn == 1:
                break
            
            theta = curr_pos + curr_speed
            joint_angle = theta  

            # Set the joint position to move the wrist side to side
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, self.side_to_side_joint, self.bullet_client.POSITION_CONTROL, targetPosition=joint_angle)

if __name__ == '__main__':
    panda = panda_camera(gui_enabled=True)
    panda.move_wrist()
    
    while True:
        panda.bullet_client.stepSimulation()
        # get RGB-D image and camera position/orientation relative to the world/robot base frame
        rgb, depth, _ = panda.panda_camera()
        
        # get point cloud from RGB-D image
        rgb = np.array(rgb, dtype=np.float32)
        depth = np.array(depth, dtype=np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth))
        pinholeCamera = o3d.camera.PinholeCameraIntrinsic(panda.img_size, 
                                                          panda.img_size, 
                                                          panda.intrinsic_matrix[0][0], 
                                                          panda.intrinsic_matrix[1][1], 
                                                          panda.intrinsic_matrix[0][2], 
                                                          panda.intrinsic_matrix[1][2])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinholeCamera)
        panda.panda.get_camera_transformation_matrix()
        # o3d.visualization.draw_geometries([pcd])
        panda.panda.reset()