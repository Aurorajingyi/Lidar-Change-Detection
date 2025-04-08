"""
Script: train_change_detection.py
功能：加载数据集、训练 PointNetSeg 模型进行点云变化检测，
     使用 CrossEntropyLoss 进行监督，记录训练损失，并保存模型 checkpoint。

     该脚本与数据预处理和模型模块无缝衔接，
     模块化设计便于扩展到更复杂的模型（例如 PointNet++ 或 Transformer）以及后续验证、评估与后处理。

作者: Aurora
日期: 2025-04-05
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.pointcloud_dataset import PointCloudChangeDataset
from models.pointnet_model import PointNetSeg
from tqdm import tqdm


def train_model(tile_folder: str, num_points: int = 1024, batch_size: int = 16, num_epochs: int = 20,
                learning_rate: float = 0.001, checkpoint_dir: str = "checkpoint"):
    """
    训练点云变化检测模型

    参数：
        tile_folder (str): 存储 tile 数据的目录
        num_points (int): 每个样本采样的点数
        batch_size (int): 每个批次的样本数量
        num_epochs (int): 训练的 epoch 数量
        learning_rate (float): 学习率
        checkpoint_dir (str): 模型 checkpoint 保存目录

    返回：
        None
    """
    # 创建 checkpoint 目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 构造数据集与 DataLoader
    dataset = PointCloudChangeDataset(tile_folder, num_points=num_points, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化设备、模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetSeg(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()  # 对于点级二分类（标签 0/1）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for points_batch, labels_batch in pbar:
            points_batch = points_batch.to(device)  # (B, N, 3)
            labels_batch = labels_batch.to(device)  # (B, N)
            optimizer.zero_grad()
            outputs = model(points_batch)  # (B, N, 2)
            outputs = outputs.reshape(-1, 2)
            labels_batch = labels_batch.reshape(-1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        scheduler.step()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        # 保存当前 epoch 的模型 checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    print("训练结束！")


if __name__ == "__main__":
    # 修改 tile_folder 为你的 tile 数据存储路径
    tile_folder = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\tile_data"
    train_model(tile_folder, num_points=1024, batch_size=16, num_epochs=20, learning_rate=0.001,
                checkpoint_dir="checkpoint")
