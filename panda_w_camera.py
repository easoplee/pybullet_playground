from panda_base import PandaGripperD435
from pybullet_base import PyBulletBase
import numpy as np
import time
import open3d as o3d
import matplotlib.pyplot as plt
import pybullet
from PIL import Image

class panda_camera(PyBulletBase):
    def __init__(self, gui_enabled):
        super().__init__(gui_enabled)
        self.initialize_sim()
        self.create_manipulation_scene()

        self.panda = PandaGripperD435(self.bullet_client)
    
    def panda_camera(self): 
        # Center of mass position and orientation (of link-7)
        com_p, com_o, _, _, _, _ = self.bullet_client.getLinkState(self.panda.panda_id, self.panda.writst_camera_idx)
        self.bullet_client.addUserDebugPoints([self.bullet_client.getLinkState(self.panda.panda_id, self.panda.writst_camera_idx)[0]], [[1,0,0]], 5)
        rot_matrix = self.bullet_client.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (0, 0, 1) # z-axis
        init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self.bullet_client.computeViewMatrix(com_p, com_p + 0.5 * camera_vector, up_vector)
        self.cam_lookat = com_p + 0.5 * camera_vector
        self.bullet_client.addUserDebugLine(com_p, self.cam_lookat, [0, 0, 1], 3, 5)

        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
        img = self.bullet_client.getCameraImage(256, 256, view_matrix, projection_matrix)
        rgb, depth = img[2][:,:,:3], img[3]
        
        rgb_opengl = (np.reshape(img[2], (256,256, 4)))
        rgbim = Image.fromarray(rgb_opengl)
        rgbim_no_alpha = rgbim.convert('RGB')
        rgbim_no_alpha.save('color.jpg')

        far = 1000.
        near = 0.01
        depth_buffer_opengl = np.reshape(img[3], [256, 256])
        depth_opengl = depth_buffer_opengl*(far-near)+near
        depth = depth_opengl.astype(np.uint16)
        new_p = Image.fromarray(depth)
        
        new_p.save('depth.png')
        return rgb, depth, self.cam_lookat
    
    def move_wrist(self):
        curr_speed = 0.1
        turn = 0
        while True:
            self.bullet_client.stepSimulation()
            self.panda_camera()
            curr_pos = self.panda.get_joint_state(4)[0]
            if self.cam_lookat[2] > 0.65 and turn == 0:
                curr_speed *= -1  
                turn += 1
            if self.cam_lookat[2] > 0.75 and turn == 1:
                break
            
            theta = curr_pos + curr_speed
            joint_angle = theta  

            # Set the joint position to move the wrist
            self.bullet_client.setJointMotorControl2(self.panda.panda_id, 4, self.bullet_client.POSITION_CONTROL, targetPosition=joint_angle)

if __name__ == '__main__':
    panda = panda_camera(gui_enabled=True)
    # panda.move_wrist()
    
    while True:
        panda.bullet_client.stepSimulation()
        rgb, depth, _ = panda.panda_camera()
        
        rgb = np.array(rgb, dtype=np.float32)
        depth = np.array(depth, dtype=np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth))
        width, height = 264, 264
        intrinsic_matrix = np.array([[952.828,     0.,     646.699 ],[0.,      952.828,     342.637 ], [0.,         0.,         1.  ]]) 
        pinholeCamera = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0][2], intrinsic_matrix[1][2])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            pinholeCamera)
        # o3d.visualization.draw_geometries([pcd])
        # panda.panda.reset()
        # time.sleep(10000)