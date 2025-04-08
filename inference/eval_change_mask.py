"""
Module: eval_change_mask.py

功能：
    1. 加载训练好的 PointNetSeg 模型 checkpoint；
    2. 遍历指定 tile 文件夹中的所有 tile 数据，
       对每个 tile 进行推理，得到预测的变化标签；
    3. 计算评价指标，如分类准确率和 Intersection over Union (IoU)；
    4. 输出全局平均指标，并对部分 tile 进行 3D 可视化，
       直观展示预测结果与真实标签的对比。

衔接说明：
    该模块与数据预处理、数据集构造、模型模块无缝衔接，
    符合当前点云变化检测领域中对全局评估与可视化展示的最新要求。

作者: Aurora
日期: 2025-04-05
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 可视化
from tqdm import tqdm

# 导入数据集和模型模块
from dataset.pointcloud_dataset import PointCloudChangeDataset
from models.pointnet_model import PointNetSeg

def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    计算分类准确率

    参数:
        preds (np.ndarray): 预测标签数组
        labels (np.ndarray): 真实标签数组
    返回:
        float: 准确率
    """
    return np.mean(preds == labels)

def compute_iou(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    计算 Intersection over Union (IoU) 指标，仅考虑正类 1（变化）

    参数:
        preds (np.ndarray): 预测标签数组
        labels (np.ndarray): 真实标签数组
    返回:
        float: IoU 指标
    """
    intersection = np.sum((preds == 1) & (labels == 1))
    union = np.sum((preds == 1) | (labels == 1))
    return 1.0 if union == 0 else intersection / union

def visualize_tile(points: np.ndarray, true_labels: np.ndarray, pred_labels: np.ndarray, title: str = "Tile Prediction"):
    """
    使用 matplotlib 进行 3D 可视化展示

    参数:
        points (np.ndarray): 点云坐标，形状 (N, 3)
        true_labels (np.ndarray): 真实标签数组，形状 (N,)
        pred_labels (np.ndarray): 预测标签数组，形状 (N,)
        title (str): 图标题
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 根据预测标签上色：预测为 1（变化）为红色，预测为 0（未变化）为蓝色
    colors = ['red' if p == 1 else 'blue' for p in pred_labels]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def evaluate_all_tiles(tile_folder: str, checkpoint_file: str, num_points: int = 1024) -> None:
    """
    遍历指定 tile 文件夹中的所有 tile，加载训练好的模型，对每个 tile 进行推理评估，
    并计算全局平均准确率和 IoU。

    参数：
        tile_folder (str): 存储 tile 数据的文件夹路径
        checkpoint_file (str): 模型 checkpoint 文件的绝对路径
        num_points (int): 每个 tile 采样的点数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型结构并加载权重（建议使用 weights_only=True 提高安全性）
    model = PointNetSeg(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    model.eval()

    # 构造数据集（关闭数据增强）
    dataset = PointCloudChangeDataset(tile_folder, num_points=num_points, augment=False)

    total_acc = 0.0
    total_iou = 0.0
    count = 0

    for idx in tqdm(range(len(dataset)), desc="Evaluating tiles"):
        points_tensor, labels_tensor = dataset[idx]  # points_tensor: (num_points, 3), labels_tensor: (num_points,)
        # 加入 batch 维度
        points_tensor = points_tensor.unsqueeze(0).to(device)  # (1, N, 3)
        labels_tensor = labels_tensor.unsqueeze(0).to(device)  # (1, N)

        with torch.no_grad():
            outputs = model(points_tensor)  # 输出: (1, N, 2)
            predictions = torch.argmax(outputs, dim=-1)  # (1, N)

        pred_labels = predictions.cpu().numpy().reshape(-1)
        true_labels = labels_tensor.cpu().numpy().reshape(-1)
        points_np = points_tensor.cpu().numpy().reshape(-1, 3)

        acc = compute_accuracy(pred_labels, true_labels)
        iou = compute_iou(pred_labels, true_labels)
        total_acc += acc
        total_iou += iou
        count += 1

        # 可视化前 2 个 tile 的预测结果
        if idx < 2:
            visualize_tile(points_np, true_labels, pred_labels, title=f"Tile {idx} Prediction")

    avg_acc = total_acc / count if count > 0 else 0
    avg_iou = total_iou / count if count > 0 else 0
    print(f"Global Average Accuracy over {count} tiles: {avg_acc:.4f}")
    print(f"Global Average IoU over {count} tiles: {avg_iou:.4f}")

if __name__ == "__main__":
    # 修改 tile_folder 为你的 tile 数据存储路径
    tile_folder = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\tile_data"
    # 修改 checkpoint_file 为你训练好的模型权重文件的绝对路径
    checkpoint_file = r"D:\BaiduNetdiskWorkspace\Polyu\开发\python\Lidar-Change-Detection\train\checkpoint\model_epoch_20.pth"
    evaluate_all_tiles(tile_folder, checkpoint_file, num_points=1024)
