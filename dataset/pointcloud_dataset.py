"""
Module: pointcloud_dataset.py

功能：
    定义用于点云变化检测的 PyTorch 数据集，
    从指定的 tile 文件夹中加载每个 tile 的点云数据和对应的变化标签，
    并支持多种数据增强（如随机旋转、随机缩放、随机抖动）和归一化处理。

    此数据集模块与数据预处理模块输出的 .npy 文件接口一致，
    为后续深度学习模型（如 PointNet、PointNet++ 等）的训练提供输入数据。

作者: Aurora
日期: 2025-04-05
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudChangeDataset(Dataset):
    def __init__(self, tile_folder: str, num_points: int = 1024, augment: bool = False,
                 augmentation_params: dict = None):
        """
        初始化点云变化检测数据集

        参数:
            tile_folder (str): 存储 tile 文件的目录。目录下应包含形如
                               tile_{tile_id:04d}_points.npy 和 tile_{tile_id:04d}_labels.npy 的文件。
            num_points (int): 每个样本采样的点数。若 tile 中点数超过此值，则随机采样。
            augment (bool): 是否启用数据增强。
            augmentation_params (dict): 数据增强参数，可以包含如下键：
                - 'rotation': bool，是否随机旋转（默认 True）
                - 'scaling': bool，是否随机缩放（默认 False）
                - 'jitter': bool，是否随机抖动（默认 False）
                - 'scale_range': tuple(float, float)，缩放范围，如 (0.9, 1.1)
                - 'jitter_std': float，抖动标准差，如 0.01
        """
        self.tile_folder = tile_folder
        # 获取所有 tile 前缀（即文件名去掉 '_points.npy' 部分）
        self.tiles = sorted([f[:-11] for f in os.listdir(tile_folder) if f.endswith('_points.npy')])
        self.num_points = num_points
        self.augment = augment
        # 设置默认增强参数
        self.augmentation_params = {
            'rotation': True,
            'scaling': False,
            'jitter': False,
            'scale_range': (0.9, 1.1),
            'jitter_std': 0.01
        }
        if augmentation_params is not None:
            self.augmentation_params.update(augmentation_params)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int):
        """
        获取指定 tile 的数据，返回经过随机采样、数据增强与归一化处理后的点云和标签。

        返回:
            points: Tensor, shape (num_points, 3)
            labels: Tensor, shape (num_points,)
        """
        tile_name = self.tiles[idx]
        points_path = os.path.join(self.tile_folder, tile_name + '_points.npy')
        labels_path = os.path.join(self.tile_folder, tile_name + '_labels.npy')
        points = np.load(points_path)  # 形状 (N, 3)
        labels = np.load(labels_path)  # 形状 (N,)
        # 加入：过滤非法标签
        valid_mask = (labels == 0) | (labels == 1)
        points = points[valid_mask]
        labels = labels[valid_mask]

        # 如果筛完后太少点了（不够采样），返回空样本避免报错
        if points.shape[0] < 10:
            return torch.zeros((self.num_points, 3)), torch.zeros((self.num_points,), dtype=torch.long)

        # 如果点数大于 num_points，则随机采样 num_points 个点
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
            labels = labels[indices]
        else:
            # 重复采样补足
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]
            labels = labels[indices]

        # 数据增强：随机旋转、缩放、抖动（归一化可以放在后面统一处理）
        if self.augment:
            if self.augmentation_params.get('rotation', True):
                points = self.random_rotate(points)
            if self.augmentation_params.get('scaling', False):
                points = self.random_scale(points, self.augmentation_params.get('scale_range', (0.9, 1.1)))
            if self.augmentation_params.get('jitter', False):
                points = self.random_jitter(points, self.augmentation_params.get('jitter_std', 0.01))
        # 数据归一化：每个 tile 单独归一化
        points = self.normalize(points)

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def random_rotate(self, pts: np.ndarray) -> np.ndarray:
        """
        对点云进行随机旋转（绕 Z 轴）
        参数:
            pts (np.ndarray): 输入点云, shape (N, 3)
        返回:
            np.ndarray: 旋转后的点云
        """
        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        return pts @ R.T

    def random_scale(self, pts: np.ndarray, scale_range: tuple) -> np.ndarray:
        """
        对点云进行随机缩放
        参数:
            pts (np.ndarray): 输入点云, shape (N, 3)
            scale_range (tuple): 缩放范围，如 (0.9, 1.1)
        返回:
            np.ndarray: 缩放后的点云
        """
        scale = np.random.uniform(*scale_range)
        return pts * scale

    def random_jitter(self, pts: np.ndarray, jitter_std: float) -> np.ndarray:
        """
        对点云进行随机抖动（添加高斯噪声）
        参数:
            pts (np.ndarray): 输入点云, shape (N, 3)
            jitter_std (float): 抖动标准差
        返回:
            np.ndarray: 抖动后的点云
        """
        noise = np.random.normal(0, jitter_std, pts.shape)
        return pts + noise

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        """
        对点云进行归一化处理，使其均值为 0，标准差为 1。
        注意：这里是局部归一化（每个 tile 独立），后续可以考虑全局归一化。

        参数:
            pts (np.ndarray): 输入点云, shape (N, 3)
        返回:
            np.ndarray: 归一化后的点云
        """
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0) + 1e-8
        return (pts - mean) / std


if __name__ == "__main__":
    # 测试数据集模块
    tile_folder = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\tile_data"  # 修改为你的实际 tile 文件夹路径
    dataset = PointCloudChangeDataset(tile_folder, num_points=1024, augment=True)
    points, labels = dataset[0]
    print("点云 shape:", points.shape)
    print("标签 shape:", labels.shape)
