import numpy as np

class LS_solver():
    def __init__(self, poses, landmarks) -> None:
        self.Xr = np.array(poses)
        self.Xl = np.array(landmarks)
        self.pos_dim = 3
        self.l_dim = 3

    def project(self, point_3d, K, T):
    
        P = K @  T[:3]
        projected_point = P @ np.append(point_3d, 1)
        projected_point /= projected_point[2]

        return projected_point[:2]
    
    def normalize(self,theta):
        s = np.sin(theta)
        c = np.cos(theta)
        return np.arctan2(s,c)
    
    def rotation2Dgradient(self, theta):
        s=np.sin(theta)
        c=np.cos(theta)
        Rp=np.array([[-s, -c],
        [c, -s,]])
        Rp = np.array([
            [s, -c, 0],
            [0,0,0],
            [c,s,0]
        ])
        return Rp

    def skew(self, p):
        m = np.zeros((3,3))
        m[0,1] = -p[2]
        m[0,2] = p[1]
        m[1,0] = p[2]
        m[1,2] = -p[0]
        m[2,0] = -p[1]
        m[2,1] = p[0]
        return m

    def J_icp(self, p, theta, cam):
        J=np.zeros((3,3))
        J[:,:2]=np.array([[1,0],[0,1],[0,0]])
        J[:,2]=self.rotation2Dgradient(theta) @ p
        return J
    

    # error and jacobian of a measured landmark
    # input:
    #   Xr: the robot pose (4x4 homogeneous matrix)
    #   Xl: the landmark pose (3x1 vector, 3d pose in world frame)
    #   z:  measured position of landmark
    # output:
    #   e: 2x1 is the difference between prediction and measurement
    #   Jr: 2x6 derivative w.r.t a the error and a perturbation on the
    #       pose
    #   Jl: 2x3 derivative w.r.t a the error and a perturbation on the
    #       landmark
    def errorAndJacobian(self, Xr, Xl, z, cam):
        
        T = np.linalg.inv(cam.camera_pose_from_odometry_pose(Xr))
        z_hat = self.project(Xl, cam.K, T); #prediction
        e = np.zeros((2,1))
        Jl = np.zeros((2,3))
        Jr = np.zeros((2,6))
        p_camera_frame = T[:3,:3] @ Xl + T[:3,3]
        p_projected = cam.K @ p_camera_frame
        e=z_hat-z
        # TODO: fill the jacobians but it's already done later no???
        inverse_z = 1/p_projected[2]
        inverse_square_z = inverse_z * inverse_z
        J_proj = np.array([[inverse_z, 0, -p_projected[1] * inverse_square_z],
                   [ 0, inverse_z, -p_projected[2] * inverse_square_z]])

        Jicp = self.J_icp(p_camera_frame, Xr[2], cam)

        Jr = J_proj @ cam.K @ Jicp
        Jl = J_proj @ cam.K @ T[:3,:3]

        return e, Jr, Jl
    

# implementation of the boxplus
# applies a perturbation to a set of landmarks and robot poses
# input:
#   XR: the robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the landmark pose (3xnum_landmarks matrix of landmarks)
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   dx: the perturbation vector of appropriate dimensions
#       the poses come first, then the landmarks
# output:
#   XR: the robot poses obtained by applying the perturbation
#   XL: the landmarks obtained by applying the perturbation
    def boxPlus(self, XR, XL, num_poses, num_landmarks, dx):
        for pose_index in range(num_poses):
            dxr=dx[pose_index*self.pos_dim:(pose_index+1)*self.pos_dim]
            XR[pose_index, :][:2]+=dxr[:2]
            XR[pose_index, :][2]+=dxr[2] #TODO normalize this
        for landmark_index in range(num_landmarks):
            
            dxl=dx[num_poses*self.pos_dim+landmark_index*self.l_dim:num_poses*self.pos_dim+(landmark_index+1)*self.l_dim]
            if type(XL[landmark_index]) == list: continue
            XL[landmark_index]+=dxl
        
        return XR, XL


# implementation of the optimization loop with robust kernel
# applies a perturbation to a set of lanpositiondmarks and robot poses
# input:
#   XR: the initial robot poses (4x4xnum_poses: array of homogeneous matrices)
#   XL: the initial landmark estimates (3xnum_landmarks matrix of landmarks)
#   Z:  the measurements (3xnum_measurements)
#   associations: 2xnum_measurements. 
#                 associations(:,k)=[p_idx,l_idx]' means the kth measurement
#                 refers to an observation made from pose p_idx, that
#                 observed landmark l_idx
#   num_poses: number of poses in XR (added for consistency)
#   num_landmarks: number of landmarks in XL (added for consistency)
#   num_iterations: the number of iterations of least squares
#   damping:      damping factor (in case system not spd)
#   kernel_threshod: robust kernel threshold

# output:
#   XR: the robot poses after optimization
#   XL: the landmarks after optimization
#   chi_stats: array 1:num_iterations, containing evolution of chi2
#   num_inliers: array 1:num_iterations, containing evolution of inliers
    def doBundleAdjustment(self,
                           XR, XL, Z, cam,
                                num_poses, 
                                num_landmarks, 
                                num_iterations, 
                                damping
                                ):
        # size of the linear system
        system_size=self.pos_dim*num_poses + self.l_dim*num_landmarks; 
        import tqdm
        for iteration in tqdm.tqdm(range(num_iterations)):
            H=np.zeros((system_size, system_size))
            b=np.zeros((system_size))
            for pose_index in Z.keys():
                for landmark_index in Z[pose_index]['points'].keys():
                    z = Z[pose_index]['points'][landmark_index]
                    Xr=XR[pose_index, :]
                    Xl=XL[landmark_index]
                    if type(Xl) == list: continue
                    e,Jr,Jl = self.errorAndJacobian(Xr, Xl, z, cam)
                    
                    omega =  np.eye(2)
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

            H+=np.eye(system_size)*damping
            dx=np.zeros(system_size)
            
            # we solve the linear system, blocking the first pose
            # this corresponds to "remove" from H and b the locks
            # of the 1st pose, while solving the system
            dx[self.pos_dim + 1:] = -np.linalg.solve(H[self.pos_dim + 1:, self.pos_dim + 1:], b[self.pos_dim + 1:])

            XR, XL=self.boxPlus(XR,XL,num_poses, num_landmarks, dx)
        return XR, XL
