We provided, though, a ready to use dataset. It is composed by the following files:

- world.dat (you need this only for evaluation)
  It contains information about the map.
  Every row contains:
   LANDMARK_ID (1) POSITION (2:4)

- camera.dat
  It contains information about the camera used to gather data:
  - camera matrix
  - cam_transform: pose of the camera w.r.t. robot
  - z_near/z_far how close/far the camera can perceive stuff
  - width/height of images

- trajectory.dat
  POSE_ID (1) ODOMETRY_POSE (2:4) GROUNDTRUTH_POSE (5:7)
  the ODOMETRY_POSE is obtained by adding Gaussian Noise (0; 0.001) to the actual robot commands

- meas-XXXX.dat
  Every measurement contains a sequence number, groundtruth (of the robot) and odometry pose and measurement information:
  - point POINT_ID_CURRENT_MESUREMENT (1) ACTUAL_POINT_ID (2) IMAGE_POINT (3:4)

  The Image_Point represents the pair [col;row] where the landmark is observed in the image

=====================================================================================================================
 
What we expect you to do in Planar Monocular SLAM?
- get an estimate of points in the world from images correspondences (use the points id for data association) using the noisy odometry, you can achieve this in two ways:
	- use always pair of images where correspondence appear, find correspondences, triangulate (you will have duplication of 3D points due to noise in image features, if this is the case merge same 3D points in position) 
	- use all the correspondences from all the images, to find the set of world points that minimizes the distance among all features directions
- once you have an initial estimate of 3D points, SE(2) poses (your odometry), you can perform Bundle Adjustment, see Total Least Squares but remember that your motion is planar


Evaluation of your poses:
Evaluate your solution comparing it with the gt using the delta between poses

For example, suppose we these two consecutive poses
T_0
T_1 

- compute the relative motion rel_T = inv(T_0)*T_1
- compute the relative gt motion rel_GT = inv(GT_0)*GT_1
- compute the SE(2) error error_T = inv(rel_T)*rel_GT
- rotation part:
        - atan2(error_T(2, 1), error_T(1, 1))
- translation part
	- compute the RMSE on translations error_T(1:3, 4) 

Evaluation of your map: 
- compute the whole RMSE    

Hints:
- develop each component independently and use the groudtruth for testing
- when everything is working put all together and create your Planar Monocular SLAM
- the better will be the inital guess the better you final estimate after BA

