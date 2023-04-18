# This is a simple script to visualize a training example
import matplotlib.pyplot as plt
import numpy as np
import os

from os.path import join
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.usetex'] = True
np.random.seed(1024)

jnt_color = [[[0.25, 0.00, 0.00], [0.40, 0.00, 0.00], [0.55, 0.00, 0.00], [0.70, 0.00, 0.00],
			  [0.85, 0.00, 0.00], [1.00, 0.00, 0.00]], # thumb
			 [[0.00, 0.25, 0.00], [0.00, 0.40, 0.00], [0.00, 0.55, 0.00], [0.00, 0.70, 0.00],
			  [0.00, 0.85, 0.00], [0.00, 1.00, 0.00]], # index
			 [[0.00, 0.00, 0.25], [0.00, 0.00, 0.40], [0.00, 0.00, 0.55], [0.00, 0.00, 0.70],
			  [0.00, 0.00, 0.85], [0.00, 0.00, 1.00]], # middle
			 [[0.00, 0.25, 0.25], [0.00, 0.40, 0.40], [0.00, 0.55, 0.55], [0.00, 0.70, 0.70],
			  [0.00, 0.85, 0.85], [0.00, 1.00, 1.00]], # ring
			 [[0.25, 0.00, 0.25], [0.40, 0.00, 0.40], [0.55, 0.00, 0.55], [0.70, 0.00, 0.70],
			  [0.85, 0.00, 0.85], [1.00, 0.00, 1.00]], # little
			 [[0.25, 0.25, 0.00], [0.40, 0.40, 0.00], [0.55, 0.55, 0.00], [0.70, 0.70, 0.00],
			  [0.85, 0.85, 0.00], [1.00, 1.00, 0.00]]] # palm


jnt_color_gray = [[[0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00],
			  [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]], # thumb
			 [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00],
			  [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]], # index
			 [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00],
			  [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]], # middle
			 [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00],
			  [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]], # ring
			 [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00],
			  [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]], # little
			 [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, 0.00],
			  [0.00, 0.00, 0.00], [0.00, 0.00, 0.00]]] # palm


def show_joint_skeleton(ax,
						jnt_xyz,
						jnt_color=jnt_color,
						alpha=None,
						ticks=True):
	# plot
	root_jnt_idx = 29
	line_width = 2
	s_size = 30
	for f in range(6):
		if f < 5:
			for bone in range(5):
				ax.plot([jnt_xyz[f * 6 + bone, 0], jnt_xyz[f * 6 + bone + 1, 0]],
						[jnt_xyz[f * 6 + bone, 1], jnt_xyz[f * 6 + bone + 1, 1]],
						[jnt_xyz[f * 6 + bone, 2], jnt_xyz[f * 6 + bone + 1, 2]],
						color=jnt_color[f][bone], linewidth=line_width, alpha=alpha)
			if f == 4:
				ax.plot([jnt_xyz[f * 6 + bone + 1, 0], jnt_xyz[root_jnt_idx, 0]],
						[jnt_xyz[f * 6 + bone + 1, 1], jnt_xyz[root_jnt_idx, 1]],
						[jnt_xyz[f * 6 + bone + 1, 2], jnt_xyz[root_jnt_idx, 2]],
						color=jnt_color[f][bone], linewidth=line_width, alpha=alpha)
			else:
				ax.plot([jnt_xyz[f * 6 + bone + 1, 0], jnt_xyz[root_jnt_idx, 0]],
						[jnt_xyz[f * 6 + bone + 1, 1], jnt_xyz[root_jnt_idx, 1]],
						[jnt_xyz[f * 6 + bone + 1, 2], jnt_xyz[root_jnt_idx, 2]],
						color=jnt_color[f][bone], linewidth=line_width, alpha=alpha)

			for jnt in range(6):
				ax.scatter(jnt_xyz[f * 6 + jnt, 0],
						   jnt_xyz[f * 6 + jnt, 1],
						   jnt_xyz[f * 6 + jnt, 2],
						   s=s_size,
						   color=jnt_color[f][jnt],
						   alpha=alpha)
		else:
			# plam coordinate
			for jnt in range(6):
				ax.scatter(jnt_xyz[root_jnt_idx, 0],
						   jnt_xyz[root_jnt_idx, 1],
						   jnt_xyz[root_jnt_idx, 2],
						   s=s_size,
						   color=jnt_color[f][jnt],
						   alpha=alpha)

	# ax.set_xlabel('$x$')
	# ax.set_ylabel('$y$')
	# ax.set_zlabel('$z$')
	ax.view_init(elev=30, azim=45)
	# ax.set_title("3D hand joints")

	if ticks == False:
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])