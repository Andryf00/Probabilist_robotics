from utils import *
from plotting import *
from ls_solver import LS_solver
from triangulation import triangulate_points
from camera import Camera
def main():

    cam = Camera('data/camera.dat')
    measurements = load_measurements(trajectory = 'gt_pose')
    gt_traj, odo_traj = load_trajectory()
    landmarks = load_landmarks()
    triangulated_points, history = triangulate_points(measurements)


    for frame_id in history.keys():
        break
        plot_3d(history[frame_id]['pose'], history[frame_id]['visible_landmarks'], history[frame_id]['triangulated_points'], history[frame_id]['filtered_points'])
            
    solver = LS_solver(poses = np.array(gt_traj), landmarks = triangulated_points)
    XR, XL = solver.doBundleAdjustment(XR=np.array(gt_traj), XL=triangulated_points, Z=measurements, cam=cam, num_poses=200, num_landmarks=1000, num_iterations=5, damping=1)
    for i in range(200):
        print(gt_traj[i], XR[i])
    
    animate_trajectories(gt_traj, XR, landmarks)    
    print("DONE")

if __name__ == "__main__":
    main()