import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import math
import numpy as np

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


patch = Rectangle((0,0),0.2,0.3,
                    angle=math.degrees(0),
                    edgecolor='red',
                    facecolor='none',
                    lw=4)

fig = plt.figure()
plt.axis('equal')
plt.grid()
ax = fig.add_subplot()
ax.add_patch(patch)

ax.plot(gt_array[:,0], gt_array[:, 1])
ax.plot(odo_array[:,0], odo_array[:, 1])
#ax.scatter(landmark_array[:, 0], landmark_array[:, 1])

def animate(i):
    patch.set_xy((gt_array[i,0], gt_array[i,1]))
    patch.set_angle(math.degrees(gt_array[i,2]))
    #print(patch._angle)
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               frames=len(ground_truth_poses),
                               interval=500,
                               blit=True)
plt.show()
