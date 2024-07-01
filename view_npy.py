import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--npy_file', type=str, default='./results/2024-07-01-14-29-49/out.npy')
args = parser.parse_args()

# 加载.npy文件
point_cloud = np.load(args.npy_file)

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 将点云数据添加到图形中
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

# 显示图形
plt.show()