from utils import *
from plotting import *
from ls_solver import LS_solver
from triangulation import triangulate_points
from camera import Camera
from compute_error import *
def main():
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
    XR, XL = solver.doBundleAdjustment(damping=1e-12, XR=np.array(odo_traj), XL=triangulated_points, Z=measurements, cam=cam, num_poses=200, num_landmarks=1000, num_iterations=50,  gt = gt_traj,odo= odo_traj,landmarks= landmarks)
    
    err = landmark_error(landmarks, XL)
    print(f"Landmark_error: {err}")
    
    for i in range(200):
        break
        print(gt_traj[i], XR[i])
    
    predicted_l = []
    for l in XL.keys():
        predicted_l.append(XL[l])
    
    print("DONE")

    f = open("final_log.txt", 'w')

    for pose_index in range(200):
        pose = XR[pose_index]
        visible_l = measurements[pose_index]['points']
        T = cam.camera_pose_from_odometry_pose(pose)
        f.write(f"\n\n\n POSE {pose_index}, gt: {gt_traj[pose_index]}, pred: {XR[pose_index]}, odom: {odo_traj[pose_index]} \n\n {T} \n\n")
        for l_index in visible_l.keys():
            try: l_3d = XL[l_index]
            except: continue
            z_hat, _ = solver.project(l_3d, cam.K, T)
            z = visible_l[l_index]
            e =  z-z_hat
            f.write(f"{l_index}: z {z} , z_hat {z_hat}, error {e}, gt_l {landmarks[l_index]}, pred_l {l_3d}\n")
            #print(z,z_hat,e)
            if np.sum(np.abs(e))>1:
                #pass
                print("ERROR")

    #animate_trajectories(gt_traj, odo_traj, XR, landmarks[list(XL.keys())], np.array(predicted_l), animate_bool=True)    


if __name__ == "__main__":
    main()