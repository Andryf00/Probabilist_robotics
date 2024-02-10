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
    solver.doBundleAdjustment(np.array(gt_traj), triangulated_points, measurements, cam, 200, triangulated_points.shape[0], 10, 1)

if __name__ == "__main__":
    main()