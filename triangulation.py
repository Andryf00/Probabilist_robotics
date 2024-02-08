import numpy as np
from camera import Camera
from collections import defaultdict

from plotting import *






def triangulate(p1, p2, K, T1, T2):
    m1 = K @ np.linalg.inv(T1)[:3]
    m2 = K @ np.linalg.inv(T2)[:3]
    A = np.array([
         p1[1] * m1[2, :].T - m1[1, :].T ,
        m1[0, :].T - p1[0]*m1[2,:].T,
        p2[1] * m2[2, :].T - m2[1, :].T,
        m2[0, :].T - p2[0]*m2[2,:].T
    ])

    _,_, V = np.linalg.svd(A)
    X_homogeneous = V[-1]  # Homogeneous coordinates of 3D point
    
    # Convert homogeneous coordinates to 3D point
    X_homogeneous /= X_homogeneous[-1]
    return X_homogeneous[:-1]

def load_measurements():
    counter = "00000"
    new_meas = True
    measurements = {}
    while new_meas:
        dict = {}
        points = {}

        try: 
            with open(f"data/measurements/meas-{counter}.dat", "r") as f:
                for line in f:
                    splitted_line = line.split()
                    if len(splitted_line) == 0: 
                        break
                
                    if splitted_line[0] == 'odom_pose:':
                        dict["pose"] = [float(x) for x in splitted_line[1:]]
                    elif splitted_line[0] == 'point':
                        points[int(splitted_line[2])] = [float(x) for x in splitted_line[3:]]
            dict['points'] = points
            measurements[int(counter)] = dict
        except: 
            return measurements
        counter = str(int(counter) + 1).zfill(len(counter))
    
m = load_measurements()
cam = Camera('data/camera.dat')

triangulated_points = defaultdict(list)


for frame_id in range(len(m)-1):
    current_frame_measurements = m[frame_id]
    next_frame_measurements = m[frame_id + 1]
    current_triangs = []
    for point_id in current_frame_measurements['points'].keys():
            try:
                triangulated_point = triangulate(current_frame_measurements['points'][point_id], 
                                             next_frame_measurements['points'][point_id],
                                             cam.K,
                                             cam.camera_pose_from_odometry_pose(current_frame_measurements['pose']),
                                             cam.camera_pose_from_odometry_pose(next_frame_measurements['pose'])
                                             )
                triangulated_points[point_id].append(triangulated_point)
                current_triangs.append(triangulated_point)

                print(point_id, triangulated_point)
                
                
            except:
                continue

    if frame_id%5==0:
        plot_3d(cam.camera_pose_from_odometry_pose(current_frame_measurements['pose']), 0.5, 0.4, 0.5, seen_frames=list(current_frame_measurements['points'].keys()), triangulated_points=current_triangs)
    
