import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from compute_error import compute_error
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import math

from utils import *

def plot_sphere(center, radius, ax, color='b', num_points=100):
    # Create spherical coordinates
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)

    # Create the meshgrid for spherical coordinates
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere
    ax.plot_surface(x, y, z, color=color, alpha=0.2)


def plot_3d(T, seen_frames, triangulated_points, filtered_points, width = 0.5, length = 0.4, height = 0.5, ):
    matplotlib.use('TkAgg') 
    ground_truth_array, _ = load_trajectory()
    triang_array = np.array(triangulated_points)
    filtered_array = np.array(filtered_points)

    landmark_array = load_landmarks()
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
    try: ax.scatter3D(triang_array[:, 0], triang_array[:, 1],triang_array[:, 2], c='g')
    except: pass
    try: ax.scatter3D(filtered_array[:, 0], filtered_array[:, 1],filtered_array[:, 2], c='y')
    except: pass

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
    try: ax2.scatter3D(triang_array[:, 0], triang_array[:, 1],triang_array[:, 2], c='g')
    except: pass
    try: ax2.scatter3D(filtered_array[:, 0], filtered_array[:, 1],filtered_array[:, 2], c='y')
    except: pass

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
    try: ax3.scatter3D(triang_array[:, 0], triang_array[:, 1],triang_array[:, 2], c='g')
    except: pass
    try: ax3.scatter3D(filtered_array[:, 0], filtered_array[:, 1],filtered_array[:, 2], c='y')
    except: pass

    # Set axes labels
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Set plot limits
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    ax3.set_zlim(0, 10)
    ax3.plot(ground_truth_array[:, 0], ground_truth_array[:, 1], [0 for _ in range(len(ground_truth_array))])
    ax3.view_init(elev=90, azim=0)
    plot_sphere(center = T[:3, 3], radius=5, ax=ax)
    plot_sphere(center = T[:3, 3], radius=5, ax=ax2)
    plot_sphere(center = T[:3, 3], radius=5, ax=ax3)

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

def animate_trajectories(gt_array, odo_array, corrected, landmark_array, pred_land, iteration = 'latest', animate_bool=False):



    patch = Rectangle((0,0),0.3,0.3,
                        angle=math.degrees(0),
                        edgecolor='red',
                        facecolor='none',
                        lw=4)

    patch_max_distance = Circle((0,0), radius=5, facecolor='none', edgecolor='red')

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2,2, 1)
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.add_patch(patch)
    ax.add_patch(patch_max_distance)



    ax.plot(gt_array[:,0], gt_array[:, 1], color = 'c', label = 'gt_traj')
    ax.plot(odo_array[:,0], odo_array[:, 1], color = 'orange', label = 'odom_traj')
    ax.plot(corrected[:,0], corrected[:, 1], color = 'green', label = 'corrected_traj')
    ax.legend()


    ax4 = fig.add_subplot(2,2, 3)
    ax4.set_aspect('equal')
    ax4.scatter(landmark_array[:, 0], landmark_array[:, 1],
            facecolors = 'none', edgecolors='g', label = 'gt_lm')
    ax4.scatter(pred_land[:, 0], pred_land[:, 1],
            facecolors = 'none', edgecolors='r', label = 'pred_lm')
    plt.legend()

    plt.savefig(f'plots/plot_{str(iteration)}.png')
    plt.close()
    if not animate_bool: return

    rotation_errors, translation_errors = compute_error(odo_array, gt_array)
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_xlim(0, len(gt_array))
    ax2.set_ylim(min(rotation_errors)-2, max(rotation_errors)+2)
    ax2.set_title("Relative rotational error")

    ax3 = fig.add_subplot(2,2,4)
    ax3.set_xlim(0, len(gt_array))
    ax3.set_ylim(min(translation_errors)-2, max(translation_errors)+2)
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
                                frames=len(gt_array),
                                interval=500,
                                blit=False)  
    anim.save(f"plots/animation.gif",writer='pillow',fps=20, dpi=200)
    plt.show()
