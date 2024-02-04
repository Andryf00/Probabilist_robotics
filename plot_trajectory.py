import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import math
import numpy as np
from compute_error import compute_error

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
