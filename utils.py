import numpy as np

def load_trajectory():
    ground_truth_poses = []
    odometry_poses = []
    with open("data/trajectory.dat", "r") as f:
        for line in f:
            splitted_line = line.split()
            ground_truth_poses.append([float(value) for value in splitted_line[4:]])
            odometry_poses.append([float(value) for value in splitted_line[1:4]])
    gt_array = np.array(ground_truth_poses)
    odo_array = np.array(odometry_poses)

    return gt_array, odo_array

def load_landmarks():
    landmark_coords_3D = []
    with open("data/world.dat", "r") as f:
        for line in f:
            splitted_line = line.split()
            landmark_coords_3D.append([float(value) for value in splitted_line[1:]])
    landmark_array = np.array(landmark_coords_3D)

    return landmark_array


def load_measurements(trajectory = 'gt_pose'):
    counter = "00000"
    new_meas = True
    measurements = {}
    while new_meas:
        dict = {}
        points = {}

        try: 
            with open(f"data/meas-{counter}.dat", "r") as f:
                for line in f:
                    splitted_line = line.split()
                    if len(splitted_line) == 0: 
                        break
                    
                    if splitted_line[0] == trajectory+':':
                        dict["pose"] = [float(x) for x in splitted_line[1:]]
                    elif splitted_line[0] == 'point':
                        points[int(splitted_line[2])] = [float(x) for x in splitted_line[3:]]
            dict['points'] = points
            measurements[int(counter)] = dict
        except: 
            return measurements
        counter = str(int(counter) + 1).zfill(len(counter))