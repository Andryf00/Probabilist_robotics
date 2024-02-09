from utils import *
from plotting import *
from triangulation import triangulate_points

def main():
    measurements = load_measurements(trajectory = 'gt_pose')
    gt_traj, odo_traj = load_trajectory()
    landmarks = load_landmarks()

    triangulated_points, history = triangulate_points(measurements)

    for frame_id in history.keys():
        plot_3d(history[frame_id]['pose'], history[frame_id]['visible_landmarks'], history[frame_id]['triangulated_points'], history[frame_id]['filtered_points'])
            


if __name__ == "__main__":
    main()