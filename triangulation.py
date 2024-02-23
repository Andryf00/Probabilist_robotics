import numpy as np
from camera import Camera
from collections import defaultdict

from plotting import *
from utils import *


import numpy as np
from scipy.optimize import minimize

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

    
def reprojection_error(point_3d, point_2d, K, T):
    
    P = K @  np.linalg.inv(T)[:3]
    projected_point = P @ np.append(point_3d, 1)
    projected_point /= projected_point[2]
    
    # Compute Euclidean distance between projected and ground truth image points
    reprojection_error = np.sqrt(np.sum((projected_point[:2] - point_2d)**2, axis=0))
    # Return mean reprojection error
    return reprojection_error

def outlier(triangulated_point, point_2d, K, T, z_max):
    #first of all check if the triangulated point is closer than z_max
    if (T[:3,:3].T @ (triangulated_point - T[:3, 3]))[2]> z_max:
        #print("ABOVE Z_FAR")      
        
        return True
    if (T[:3,:3].T @ (triangulated_point - T[:3, 3]))[2] < 0:
        #print("BEHIND CAM")
        return True
    elif reprojection_error(triangulated_point, point_2d, K, T) > 25:
        #print("rEPROJECTION ERROR")
        return True
    return False

def triangulate_points(m):
    cam = Camera('data/camera.dat')
    history = {}

    triangulated_points = defaultdict(list)


    for frame_id in range(len(m)-1):
        current_frame_measurements = m[frame_id]
        next_frame_measurements = m[frame_id + 1]
        current_triangs = []
        filtered_points = []
        for point_id in current_frame_measurements['points'].keys():
                if point_id==8:
                    print(frame_id)
                    print(frame_id in next_frame_measurements['points'].keys())
                try:
                    triangulated_point = triangulate(current_frame_measurements['points'][point_id], 
                                                next_frame_measurements['points'][point_id],
                                                cam.K,
                                                cam.camera_pose_from_odometry_pose(current_frame_measurements['pose']),
                                                cam.camera_pose_from_odometry_pose(next_frame_measurements['pose'])
                                                )
                    if not outlier(triangulated_point,
                                np.array(current_frame_measurements['points'][point_id]),
                                cam.K,
                                cam.camera_pose_from_odometry_pose(current_frame_measurements['pose']),
                                cam.z_far):
                        triangulated_points[point_id].append(triangulated_point)
                        current_triangs.append(triangulated_point)
                    else:
                        filtered_points.append(triangulated_point)
                except KeyError:
                    continue
                except Exception as e:
                    print(e)
                    continue
        history[frame_id] = {
            'pose': cam.camera_pose_from_odometry_pose(current_frame_measurements['pose']),
            'visible_landmarks': list(current_frame_measurements['points'].keys()),
            'triangulated_points': current_triangs,
            'filtered_points': filtered_points
        }
        if frame_id%5==0:
            #plot_3d(history[frame_id]['pose'], history[frame_id]['visible_landmarks'], history[frame_id]['triangulated_points'], history[frame_id]['filtered_points'])
            pass
    averaged_triangulated_points = {}
    for point in sorted(triangulated_points.keys()):
        points_array = np.array(triangulated_points[point])
        averaged_point = np.array([np.mean(points_array[:,0]), np.mean(points_array[:,1]), np.mean(points_array[:,2]) ])
        averaged_triangulated_points[point] = averaged_point

    return averaged_triangulated_points, history