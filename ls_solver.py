import numpy as np
import sympy as sp
from plotting import *


class LS_solver():
    def __init__(self) -> None:
        self.pos_dim = 3
        self.l_dim = 3

    def project(self, point_3d, K, T):
        behind = False
        projected_point = K@(np.linalg.inv(T[:3,:3]) @ point_3d - np.linalg.inv(T[:3,:3]) @ T[:3,3])
        if projected_point[2]<0: behind = True
        projected_point /= projected_point[2]

        return projected_point[:2], behind
    
    def normalize(self,theta):
        #return theta
        s = np.sin(theta)
        c = np.cos(theta)
        return np.arctan2(s,c)
     
    def J_mat(self, pose, p, cam):
        x = pose[0] 
        y = pose[1]
        theta = pose[2]
        xl = p[0]
        yl = p[1]
        zl = p[2]
        cos = np.cos
        sin = np.sin
        J_r = np.array([[-sin(theta), cos(theta), xl*cos(theta) - x*cos(theta) - y*sin(theta) + yl*sin(theta)],
                        [0, 0, 0],
                        [-cos(theta), -sin(theta), yl*cos(theta) - y*cos(theta) + x*sin(theta) - xl*sin(theta)]])
        J_l = np.array([[sin(theta), -cos(theta), 0],
                        [0, 0,-1],
                        [cos(theta), sin(theta), 0]]
                    )
        return J_r, J_l
    

    def errorAndJacobian(self, Xr, Xl, z, cam):
        T= cam.camera_pose_from_odometry_pose(Xr)
        z_hat,_ = self.project(Xl, cam.K, T) #prediction
        p_camera_frame = np.linalg.inv(T[:3,:3]) @ Xl - np.linalg.inv(T[:3,:3]) @ T[:3,3]
        p_projected = cam.K @ p_camera_frame
        e=z_hat-z

        inverse_z = 1/p_projected[2]
        inverse_square_z = inverse_z * inverse_z
        J_proj = np.array([[inverse_z, 0, -p_projected[0] * inverse_square_z],
                   [ 0, inverse_z, -p_projected[1] * inverse_square_z]])

        Jicp_r, J_icp_l = self.J_mat( Xr, Xl, cam)

        Jr = J_proj @ cam.K @ Jicp_r
        Jl = J_proj @ cam.K @ J_icp_l

        return e, Jr, Jl
    

# implementation of the boxplus
# applies a perturbation to a set of landmarks and robot poses
# input:
    def boxPlus(self, XR, XL, num_poses, num_landmarks, dx, gt, landmarks, err):
        for pose_index in range(num_poses):
            dxr=dx[pose_index*self.pos_dim:(pose_index+1)*self.pos_dim]
            if np.sum(np.abs(dxr))>1: 
                pass
                #print("Pose", pose_index, XR[pose_index],gt[pose_index], dxr)
            XR[pose_index, :][:2]+=dxr[:2]
            XR[pose_index, :][2]=self.normalize(XR[pose_index, :][2]+dxr[2]) 
        for landmark_index in range(num_landmarks):
            
            dxl=dx[num_poses*self.pos_dim+landmark_index*self.l_dim:num_poses*self.pos_dim+(landmark_index+1)*self.l_dim]
            try: XL[landmark_index]
            except: 
                #print(dxl) it's alway 0, because if the lm is not observed there is no update
                continue
            if np.sum(np.abs(dxl))>10: 
                pass
                #print("Landmark", landmark_index, XL[landmark_index],landmarks[landmark_index], dxl, err[landmark_index])
            else:
                pass
                #print("Landmark_SMALL", landmark_index, XL[landmark_index],landmarks[landmark_index], dxl, err[landmark_index])
            XL[landmark_index]+=dxl
        
        return XR, XL

    def doBundleAdjustment(self,
                           XR, XL, Z, cam,
                                num_poses, 
                                num_landmarks, 
                                num_iterations, 
                                damping,
                                gt, odo, landmarks,
                                kernel_threshold = 25
                                ):
        patience = 5
        no_imp = 0
        breakable = False
        chi_min = np.inf
        # size of the linear system
        system_size=self.pos_dim*num_poses + self.l_dim*num_landmarks; 
        import tqdm
        f = open("iteration_log.txt", 'w')
        for iteration in tqdm.tqdm(range(num_iterations)):
            #input("...")
            log = False
            if iteration == 39 and False:
                log = True
            from collections import defaultdict
            errors = defaultdict(int)
            chi_tot = 0
            H=np.zeros((system_size, system_size))
            b=np.zeros((system_size))
            skipped = 0
            not_skipped = 0
            
            for pose_index in Z.keys():
                Xr=XR[pose_index, :]
                T= cam.camera_pose_from_odometry_pose(Xr)
                if log:
                    f.write(f"\n\n\n POSE {pose_index}, {XR[pose_index]} \n\n {T} \n\n")
                for landmark_index in Z[pose_index]['points'].keys():
                    z = Z[pose_index]['points'][landmark_index]
                    try:Xl=XL[landmark_index]
                    except: continue

                    e,Jr,Jl = self.errorAndJacobian(Xr, Xl, z, cam)
                    chi=e.T@e
                    errors[landmark_index] += chi
                    z_hat, behind = self.project(Xl, cam.K, T) #prediction
                    if behind:
                        print(pose_index, landmark_index)
                        continue
                    if log:
                        f.write(f"{landmark_index}: z {z} , z_hat {z_hat}, error {e}, gt_l {landmarks[landmark_index]}, pred_l {Xl}\n")

                    #print(z, z_hat, e)
                    if z_hat[0]>(cam.width) or z_hat[1]>(cam.height):
                        #print("skipping ", landmark_index)
                        skipped += 1
                        continue
                    not_skipped += 1
                    chi_tot += chi
                    w = 1
                    if (chi>kernel_threshold):
                        #pass
                        w = w/(kernel_threshold*3)
                    omega = w* np.eye(2)
                    Hrr = Jr.T @ omega @ Jr
                    Hrl = Jr.T @ omega @ Jl
                    Hll = Jl.T @ omega @ Jl
                    br = Jr.T @ omega @ e
                    bl = Jl.T @ omega @ e

                    pose_matrix_index=self.pos_dim*pose_index
                    landmark_matrix_index=self.pos_dim*num_poses + self.l_dim*landmark_index

                    H[pose_matrix_index:pose_matrix_index+self.pos_dim,
                    pose_matrix_index:pose_matrix_index+self.pos_dim]+=Hrr

                    H[pose_matrix_index:pose_matrix_index+self.pos_dim,
                    landmark_matrix_index:landmark_matrix_index+self.l_dim]+=Hrl
                    H[landmark_matrix_index:landmark_matrix_index+self.l_dim,
                    landmark_matrix_index:landmark_matrix_index+self.l_dim]+=Hll

                    H[landmark_matrix_index:landmark_matrix_index+self.l_dim,
                    pose_matrix_index:pose_matrix_index+self.pos_dim]+=Hrl.T

                    b[pose_matrix_index:pose_matrix_index+self.pos_dim]+=br
                    b[landmark_matrix_index:landmark_matrix_index+self.l_dim]+=bl
            if chi_tot < chi_min:
                chi_min = chi_tot
                no_imp = 0
            else: no_imp+=1
            if chi_tot<1:
                breakable = True
            if no_imp>patience and breakable:
                break
            H+=np.eye(system_size)*damping
            dx=np.zeros(system_size)
            print("skip/non skip: ", skipped, not_skipped)
            if log: break  #stop before updating
            
            # we solve the linear system, blocking the first pose
            # this corresponds to "remove" from H and b the locks
            # of the 1st pose, while solving the system
            dx[self.pos_dim:] = np.linalg.solve(H[self.pos_dim:, self.pos_dim:], -b[self.pos_dim:])

            XR, XL=self.boxPlus(XR,XL,num_poses, num_landmarks, dx, gt, landmarks, errors)
            predicted_l = []
            for l in XL.keys():
                predicted_l.append(XL[l])
            print(chi_tot)
            animate_trajectories(gt,odo , XR, landmarks[list(XL.keys())], np.array(predicted_l), iteration)  
   
        return XR, XL
