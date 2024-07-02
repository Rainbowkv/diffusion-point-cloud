import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


parser = argparse.ArgumentParser()
# parser.add_argument('--npy_file', type=str, default='./point_cloud_1.npy')
# parser.add_argument('--npy_file', type=str, default='./results/2024-07-02-12-56-42/out.npy')  # airplane
parser.add_argument('--npy_file', type=str, default='./results/chair_2024-07-02-16-16-59/out.npy')  # chair
args = parser.parse_args()

# 加载.npy文件
point_cloud = np.load(args.npy_file)
print(point_cloud.shape)


# # 将第一维消除，也可以point_cloud = point_cloud.squeeze(0)，这里下面的写法更合适
for i in range(point_cloud.shape[0]):
    tmp = point_cloud[i, :, :]
    print(tmp.shape)

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将点云数据添加到图形中
    colors = np.array([128, 0, 128]).reshape(1, -1) / 255  # 紫色
    ax.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2], c=colors)

    # 关闭坐标轴
    ax.set_axis_off()

    # 显示图形
    plt.show()