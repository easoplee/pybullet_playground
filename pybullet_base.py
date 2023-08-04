import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

# Base class to initialize pybullet environment
class PyBulletBase():
    def __init__(self, gui_enabled):
        super(PyBulletBase, self).__init__()
        self.gui_enabled = gui_enabled

        # Change the camera view
        self.camera_distance = 2.0
        self.camera_yaw = 0
        self.camera_pitch = -20
        self.camera_target_position = [0.5, 0, 0.15] 

        # connect to pybullet
        self.connect_to_pybullet()

    def connect_to_pybullet(self):
        if self.gui_enabled:
            self.bullet_client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)

    def initialize_sim(self):

        self.bullet_client.resetSimulation()
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.setGravity(0, 0, -9.8)
        self.bullet_client.setTimeStep(1. / 240.)
        self._plane_id = self.bullet_client.loadURDF("plane.urdf", [0, 0, -0.01], useFixedBase=True) # slightly below the base of the robot

        self.create_black_mat()
        self.reset_camera_visualizer()
        
    def create_black_mat(self):
        region_half_extents = [0.6, 0.6, 0.01]  # x, y, and z half extents of the region
        region_position = [0.7, 0.0, 0.0]  # x, y, and z position of the region

        region_shape = self.bullet_client.createCollisionShape(p.GEOM_BOX, halfExtents=region_half_extents)

        # Create a multi-body (no mass) to represent the region
        region_visual = self.bullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=region_shape)

        self.bullet_client.resetBasePositionAndOrientation(region_visual, region_position, [0, 0, 1, 1])

        gray_color = [0.5, 0.5, 0.5, 1]
        self.bullet_client.changeVisualShape(region_visual, -1, rgbaColor=gray_color)

    def reset_camera_visualizer(self):
        self.bullet_client.resetDebugVisualizerCamera(self.camera_distance, 
                                                      self.camera_yaw, 
                                                      self.camera_pitch, 
                                                      self.camera_target_position)
    
    def create_manipulation_scene(self):
        self.bullet_client.loadURDF("random_urdfs/002/002.urdf", [1, 0, 0.1])
        self.bullet_client.loadURDF("random_urdfs/001/001.urdf", [1.1, 0, 0.1])
        # self.bullet_client.loadURDF("random_urdfs/003/003.urdf", [0.9, 0, 0.1])
        self.bullet_client.loadURDF("random_urdfs/004/004.urdf", [1.1, 0.1, 0.1])
        self.bullet_client.loadURDF("random_urdfs/005/005.urdf", [0.8, -0.1, 0.1])
        self.bullet_client.loadURDF("assets/shelf/shelf.urdf", [1, 0.5, 0.01], [0,0,1,1], useFixedBase=True, globalScaling=0.7)