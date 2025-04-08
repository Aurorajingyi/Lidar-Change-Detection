import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Step 1: 数据加载与预处理
# --------------------------

# 假设你已经用 CloudCompare 得到带变化掩码的 CSV 文件
csv_path = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\DenseMap_Change_with_mask.csv"

# 读取 CSV 文件，注意这里 CSV 文件应为逗号分隔且包含标题行
df = pd.read_csv(csv_path, sep=',', header=0, index_col=False, low_memory=False)

# 观察一下 CSV 文件的标题行（如有需要重命名，可参考下面代码）
# 例如 CSV 原始标题可能为: "//X", "Y", "Z", "R", "G", "B", "Approx. distances", "Temp. approx. distances"
df.rename(columns={
    '//X': 'X',
    'Approx. distances': 'ApproxDistance',
    'Temp. approx. distances': 'ChangeDistance'
}, inplace=True)

# 此时 CSV 中应包含至少以下字段：X, Y, Z, ChangeDistance, ChangeMask
# 如果你已经生成了 ChangeMask（0/1/-1），直接使用即可；否则参考之前的代码生成
# 这里假设 ChangeMask 已经存在

# 输出数据预览
print("CSV 文件预览：")
print(df.head(10))

# --------------------------
# Step 2: 划分 Tile（数据已切分，这里直接加载 tile_0001）
# --------------------------
# 假设你已将 tile 切分为 .npy 文件
tile_folder = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\tile_data"
tile_idx = "0001"  # 使用 tile_0001

points_path = os.path.join(tile_folder, f"tile_{tile_idx}_points.npy")
labels_path = os.path.join(tile_folder, f"tile_{tile_idx}_labels.npy")

# 加载 tile 数据
points = np.load(points_path)  # 形状: (N, 3)
labels = np.load(labels_path)  # 形状: (N,)

# 过滤掉不确定区域（ChangeMask == -1）
mask = labels != -1
points = points[mask]
labels = labels[mask]

print(f"Tile_{tile_idx} 点数: {points.shape[0]}")
print("变化标签统计：", np.bincount(labels))


# --------------------------
# Step 3: 构建 PyTorch Dataset 和 DataLoader
# --------------------------
class PointCloudChangeDataset(Dataset):
    def __init__(self, points, labels):
        # points: (N, 3), labels: (N,)
        # 为方便深度学习训练，通常需要固定点数 (比如采样 1024 个点)
        self.points = points
        self.labels = labels
        self.num_points = 1024  # 可调参数，若点数太多则随机采样

    def __len__(self):
        # 这里按 tile 作为样本，每个 tile 做一遍（实际项目可考虑分多次采样）
        return 1

    def __getitem__(self, idx):
        pts = self.points
        lbl = self.labels
        # 若当前 tile 点数大于 self.num_points，则随机采样 self.num_points 个点
        if pts.shape[0] > self.num_points:
            indices = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts = pts[indices]
            lbl = lbl[indices]
        else:
            # 如果点数不足，则可以进行重复采样或者填充（这里简单处理为直接使用）
            pass
        # 返回点云 (num_points, 3) 和标签 (num_points,)
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


# 构造数据集和 DataLoader
dataset = PointCloudChangeDataset(points, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # 每个 tile 单独作为一个 batch


# --------------------------
# Step 4: 构建 PointNet 分割模型（轻量版，用于二分类变化检测）
# --------------------------
class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSeg, self).__init__()
        # PointNet segmentation: 用于点级二分类
        # 输入：B x N x 3, 输出：B x N x num_classes
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # 获取全局特征
        # 为 segmentation 使用局部特征（conv1 输出）和全局特征拼接
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)

        self.conv6 = nn.Conv1d(256, 128, 1)
        self.bn6 = nn.BatchNorm1d(128)

        self.conv7 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        # x: B x N x 3, 需要转为 B x 3 x N
        x = x.transpose(2, 1)  # B x 3 x N
        feat1 = nn.functional.relu(self.bn1(self.conv1(x)))  # B x 64 x N
        feat2 = nn.functional.relu(self.bn2(self.conv2(feat1)))  # B x 128 x N
        feat3 = nn.functional.relu(self.bn3(self.conv3(feat2)))  # B x 1024 x N

        # 全局特征
        global_feat, _ = torch.max(feat3, 2, keepdim=True)  # B x 1024 x 1
        global_feat = global_feat.repeat(1, 1, feat1.size(2))  # B x 1024 x N

        # 拼接局部特征（feat1）和全局特征（global_feat）
        combined = torch.cat([feat1, global_feat], dim=1)  # B x (64+1024)=B x 1088 x N

        x = nn.functional.relu(self.bn4(self.conv4(combined)))  # B x 512 x N
        x = nn.functional.relu(self.bn5(self.conv5(x)))  # B x 256 x N
        x = nn.functional.relu(self.bn6(self.conv6(x)))  # B x 128 x N
        x = self.conv7(x)  # B x num_classes x N
        x = x.transpose(2, 1)  # B x N x num_classes
        return x


# 创建模型
model = PointNetSeg(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------------------------
# Step 5: 定义损失函数与优化器，并训练模型（先用 tile_0001 测试流程）
# --------------------------
criterion = nn.CrossEntropyLoss()  # 对于点级二分类（标签 0/1）
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for points_batch, labels_batch in dataloader:
        # points_batch: (B, N, 3), labels_batch: (B, N)
        points_batch = points_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(points_batch)  # 输出 shape: (B, N, 2)
        # 将输出和标签reshape为 (B*N, 2) 和 (B*N,)
        outputs = outputs.view(-1, 2)
        labels_batch = labels_batch.view(-1)

        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("训练结束！")
