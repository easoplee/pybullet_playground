from panda_base import PandaGripperD435
from pybullet_base import PyBulletBase
import numpy as np

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
        self.bullet_client.addUserDebugLine(com_p, com_p + 0.5 * camera_vector, [0, 0, 1], 3, 5)

        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
        img = self.bullet_client.getCameraImage(256, 256, view_matrix, projection_matrix)
        return img

if __name__ == '__main__':
    panda = panda_camera(gui_enabled=True)
    while True:
        panda.bullet_client.stepSimulation()
        panda.panda_camera()