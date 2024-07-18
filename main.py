from utils import *
from plotting import *
from ls_solver import LS_solver
from triangulation import triangulate_points
from camera import Camera
from compute_error import *
import os

def main():
        
        os.mkdir('plots/')

        cam = Camera()

        measurements = load_measurements(trajectory = 'odom_pose') #load the measurements for each pose
        gt_traj, odo_traj = load_trajectory() 

        landmarks = load_landmarks() #load landmarks 3d position from world.dat

        triangulated_points, history = triangulate_points(measurements)

        # if you want to see the result of triangulation for each step uncomment break
        for frame_id in history.keys():
            break
            plot_3d(history[frame_id]['pose'], history[frame_id]['visible_landmarks'], history[frame_id]['triangulated_points'], history[frame_id]['filtered_points'])
                
        solver = LS_solver()
        XR, XL = solver.doBundleAdjustment(damping=1e-11, XR=np.array(odo_traj), XL=triangulated_points, Z=measurements, cam=cam, num_poses=200, num_landmarks=1000, num_iterations=50,  gt = gt_traj,odo= odo_traj,landmarks= landmarks)
        
        err = landmark_error(landmarks, XL)
        print(f"Landmark_error: {err}")
        
        predicted_l = []
        for l in XL.keys():
            predicted_l.append(XL[l])
        
        print("DONE")
        
        animate_trajectories(gt_traj, odo_traj, XR, landmarks[list(XL.keys())], np.array(predicted_l), animate_bool=True)    


if __name__ == "__main__":
    main()