import numpy as np

class Camera():
    def __init__(self, filename='data/camera.dat') -> None:
        with open(filename, 'r') as file:
            next(file)  
            self.K = np.array([[float(x) for x in next(file).split()] for _ in range(3)])
            next(file)  
            self.T = np.array([[float(x) for x in next(file).split()] for _ in range(4)])
            #next(file)
            self.z_near = float(next(file).split()[-1])
            self.z_far = float(next(file).split()[-1])
            self.width = int(next(file).split()[-1])
            self.height = int(next(file).split()[-1])
    
    def camera_pose_from_odometry_pose(self, odo_pose):
        theta = odo_pose[2]
        x = odo_pose[0]
        y = odo_pose[1]
        odo_pose_3d = np.array([
            [np.cos(theta), -np.sin(theta), 0, x],
            [np.sin(theta), np.cos(theta), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        return odo_pose_3d @ self.T