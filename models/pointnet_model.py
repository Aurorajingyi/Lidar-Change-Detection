"""
Module: pointnet_model.py

功能：
    定义用于点云变化检测的简化版 PointNet 分割模型。
    该模型接受形状为 (B, N, 3) 的点云作为输入，输出 (B, N, num_classes) 的预测，
    适用于点级二分类任务（例如变化 vs 不变化）。

    该模型作为基线实现，采用了一维卷积层（MLP），批归一化和 ReLU 激活，
    并通过全局 max pooling 提取全局特征，与局部特征拼接后再进行分类。
    为提高模型泛化能力，引入了 dropout 层。

    注意：这是一个基础模型，后续可以进一步扩展为 PointNet++ 或引入 Transformer 结构，
         以更好地捕捉局部几何信息，符合最新点云变化检测的前沿方向。

作者: Aurora
日期: 2025-04-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSeg(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_prob: float = 0.3):
        """
        初始化 PointNet 分割模型

        参数:
            num_classes (int): 分类数，对于变化检测任务通常为 2（变化/不变化）。
            dropout_prob (float): Dropout 概率，用于提高模型泛化能力，默认 0.3。
        """
        super(PointNetSeg, self).__init__()
        # 第一阶段：逐点特征提取
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # 第二阶段：全局特征与局部特征融合
        # 这里将初级特征 feat1 (B x 64 x N) 与全局特征（B x 1024 x N）拼接，得到 1088 维特征
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)

        # 添加 dropout 层以防止过拟合
        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv6 = nn.Conv1d(256, 128, 1)
        self.bn6 = nn.BatchNorm1d(128)

        # 最后一层输出每个点的分类得分
        self.conv7 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x (torch.Tensor): 输入点云，形状 (B, N, 3)，其中 B 是批次大小，N 是点数。

        返回:
            torch.Tensor: 模型输出，形状 (B, N, num_classes)，表示每个点的分类得分。
        """
        # 转换输入形状为 B x 3 x N
        x = x.transpose(2, 1)

        # 逐点特征提取
        feat1 = F.relu(self.bn1(self.conv1(x)))  # B x 64 x N
        feat2 = F.relu(self.bn2(self.conv2(feat1)))  # B x 128 x N
        feat3 = F.relu(self.bn3(self.conv3(feat2)))  # B x 1024 x N

        # 全局特征提取：对每个点取最大值
        global_feat, _ = torch.max(feat3, 2, keepdim=True)  # B x 1024 x 1
        # 将全局特征扩展到每个点
        global_feat_expanded = global_feat.repeat(1, 1, feat1.size(2))  # B x 1024 x N

        # 融合局部特征和全局特征
        combined = torch.cat([feat1, global_feat_expanded], dim=1)  # B x 1088 x N

        # 后续 MLP 层
        x = F.relu(self.bn4(self.conv4(combined)))  # B x 512 x N
        x = F.relu(self.bn5(self.conv5(x)))  # B x 256 x N
        x = self.dropout(x)  # 应用 dropout
        x = F.relu(self.bn6(self.conv6(x)))  # B x 128 x N

        # 分类层
        x = self.conv7(x)  # B x num_classes x N
        x = x.transpose(2, 1)  # 转换为 B x N x num_classes
        return x


if __name__ == "__main__":
    # 测试模型模块
    model = PointNetSeg(num_classes=2, dropout_prob=0.3)
    # 模拟 2 个样本，每个样本 1024 个点，3 维坐标
    x = torch.randn(2, 1024, 3)
    outputs = model(x)
    print("输出形状：", outputs.shape)  # 期望输出形状为 (2, 1024, 2)
