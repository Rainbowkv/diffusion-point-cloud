import h5py
import numpy as np
import torch
import argparse
from utils.dataset import Dataset

from utils.dataset import ShapeNetCore


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/AE_airplane.pt')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

ckpt = torch.load(args.ckpt)

# Datasets and loaders
print('Loading datasets...')
test_dset = ShapeNetCore(
    path="./data/shapenet.hdf5",
    cates=["airplane"],
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
batch_size = 1
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, num_workers=0)

group_data = next(iter(test_loader))  # 每次都取batch_size个

pointcloud = group_data['pointcloud']
shift = group_data['shift']
scale = group_data['scale']

# 还原点云数据
real_pointcloud = pointcloud * scale + shift
# print(real_pointcloud)

np.save("./point_cloud_1.npy", real_pointcloud.numpy())
print(real_pointcloud.size())