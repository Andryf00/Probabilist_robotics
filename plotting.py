import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from compute_error import compute_error
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import math





def plot_3d(T, length, width, height, seen_frames, triangulated_points):

    matplotlib.use('TkAgg') 
    landmark_coords_3D = []
    ground_truth_poses = []
    with open("data/trajectory.dat", "r") as f:
        for line in f:
            splitted_line = line.split()
            ground_truth_poses.append([float(value) for value in splitted_line[4:]])
    ground_truth_array = np.array(ground_truth_poses)
    triang_array = np.array(triangulated_points)

    with open("data/world.dat", "r") as f:
        for line in f:
            splitted_line = line.split()
            landmark_coords_3D.append([float(value) for value in splitted_line[1:]])
    landmark_array = np.array(landmark_coords_3D)
    seen_points = landmark_array[seen_frames]
    # Define the vertices of the rectangle in its local coordinate system
    vertices_local = np.array([
        [-length/2, -width/2, 0],
        [length/2, -width/2, 0],
        [length/2, width/2, 0],
        [-length/2, width/2, 0],
        [-length/2, -width/2, height],
        [length/2, -width/2, height],
        [length/2, width/2, height],
        [-length/2, width/2, height]
    ])

    # Transform vertices to world coordinates using the pose matrix T
    vertices_world = np.dot(T[:3, :3], vertices_local.T).T + T[:3, 3]

    # Define the edges of the rectangle
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connections between top and bottom faces
    ]

    fig = plt.figure()
    plt.suptitle("Press Q to move to next frame, ESC to exit full screen mode")
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter3D(0,0,0)

    # Plot edges
    for edge in edges:
        ax.plot3D(*zip(*vertices_world[edge]), color='b')

    # Plot vertices
    ax.scatter3D(*zip(*vertices_world), c='b')
    ax.scatter3D(seen_points[:, 0], seen_points[:, 1],seen_points[:, 2], c='r')
    ax.scatter3D(triang_array[:, 0], triang_array[:, 1],triang_array[:, 2], c='g')

    # Set axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 10)

    ax.view_init(elev=0, azim=90)
    

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter3D(0,0,0)

    # Plot edges
    for edge in edges:
        ax2.plot3D(*zip(*vertices_world[edge]), color='b')

    # Plot vertices
    ax2.scatter3D(*zip(*vertices_world), c='b')
    ax2.scatter3D(seen_points[:, 0], seen_points[:, 1],seen_points[:, 2], c='r')
    ax2.scatter3D(triang_array[:, 0], triang_array[:, 1],triang_array[:, 2], c='g')

    # Set axes labels
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Set plot limits
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_zlim(0, 10)

    ax2.view_init(elev=0, azim=180)


    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter3D(0,0,0)

    # Plot edges
    for edge in edges:
        ax3.plot3D(*zip(*vertices_world[edge]), color='b')

    # Plot vertices
    ax3.scatter3D(*zip(*vertices_world), c='b')
    ax3.scatter3D(seen_points[:, 0], seen_points[:, 1],seen_points[:, 2], c='r')
    ax3.scatter3D(triang_array[:, 0], triang_array[:, 1],triang_array[:, 2], c='g')

    # Set axes labels
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Set plot limits
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    ax3.set_zlim(0, 10)
    ax3.plot(ground_truth_array[:, 0], ground_truth_array[:, 1], [0 for _ in range(len(ground_truth_poses))])
    ax3.view_init(elev=90, azim=0)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    def key_event(event):
        if event.key == 'q':  # Change 'q' to any key you want to use for closing the figure
            plt.close()
        if event.key == 'escape':
            manager.full_screen_toggle()



    # Connect the key press event to the figure
    fig.canvas.mpl_connect('key_press_event', key_event)

    plt.show()

def animated_trajectories():
    ground_truth_poses = []
    odometry_poses = []

    with open("data/trajectory.dat", "r") as f:
        for line in f:
            splitted_line = line.split()
            ground_truth_poses.append([float(value) for value in splitted_line[4:]])
            odometry_poses.append([float(value) for value in splitted_line[1:4]])

    landmark_coords_2D = []
    with open("data/world.dat", "r") as f:
        for line in f:
            splitted_line = line.split()
            landmark_coords_2D.append([float(value) for value in splitted_line[1:]])
    landmark_array = np.array(landmark_coords_2D)
    gt_array = np.array(ground_truth_poses)
    odo_array = np.array(odometry_poses)

    rotation_errors, translation_errors = compute_error(odo_array, gt_array)

    patch = Rectangle((0,0),0.3,0.3,
                        angle=math.degrees(0),
                        edgecolor='red',
                        facecolor='none',
                        lw=4)

    patch_max_distance = Circle((0,0), radius=5, facecolor='none', edgecolor='red')

    fig = plt.figure()
    ax = fig.add_subplot(1,2, 1)
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.add_patch(patch)
    ax.add_patch(patch_max_distance)



    ax.plot(gt_array[:,0], gt_array[:, 1])
    ax.plot(odo_array[:,0], odo_array[:, 1])
    ax.scatter(landmark_array[:, 0], landmark_array[:, 1],
            facecolors = 'none', edgecolors='g')

    ax2 = fig.add_subplot(2,2,2)
    ax2.set_xlim(0, len(gt_array))
    ax2.set_ylim(min(rotation_errors), max(rotation_errors))
    ax2.set_title("Relative rotational error")

    ax3 = fig.add_subplot(2,2,4)
    ax3.set_xlim(0, len(gt_array))
    ax3.set_ylim(min(translation_errors), max(translation_errors))
    ax3.set_title("Relative ranslational error")


    rotation_plot, = ax2.plot([], [])

    translation_plot, = ax3.plot([], [])

    def animate(i):
        angle_rad = gt_array[i,2]
        dx = 0.15*np.cos(angle_rad) - 0.15*np.sin(angle_rad)
        dy = 0.15*np.sin(angle_rad) + 0.15*np.cos(angle_rad)
        x = gt_array[i,0] - dx
        y = gt_array[i,1] - dy
        patch.set_xy((x, y))
        patch.set_angle(math.degrees(angle_rad))
        patch_max_distance.set_center((gt_array[i,0], gt_array[i,1]))

        rotation_plot.set_data([x for x in range(i)],rotation_errors[:i])
        translation_plot.set_data([x for x in range(i)],translation_errors[:i])
        
        #print(patch._angle)
        return patch, patch_max_distance, rotation_plot, translation_plot

    anim = animation.FuncAnimation(fig, animate,
                                frames=len(ground_truth_poses),
                                interval=500,
                                blit=False)
    plt.show()