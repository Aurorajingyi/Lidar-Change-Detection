import pandas as pd
import numpy as np
import os

# 参数设置
csv_file = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\DenseMap_Change_with_mask.csv"  #  CSV 文件路径
output_dir = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\tile_data"  # 输出 tile 文件夹
tile_size = 10.0  # 每个 tile 的边长（单位：米）

# 如果输出文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 1. 读取 CSV 文件
# 注意：此处我们假设 CSV 文件中已经包含了标题行，且标题为 'X', 'Y', 'Z', 'R', 'G', 'B', 'ChangeMask'
df = pd.read_csv(csv_file, sep=',', header=0, index_col=False, low_memory=False)

# 检查数据的前几行
print("数据前10行：")
print(df.head(10))

# 2. 计算点云在 X 和 Y 方向的边界
min_x, max_x = df['X'].min(), df['X'].max()
min_y, max_y = df['Y'].min(), df['Y'].max()
print(f"X 范围: {min_x:.2f} - {max_x:.2f}")
print(f"Y 范围: {min_y:.2f} - {max_y:.2f}")

# 3. 划分网格（tile）
# 生成从 min 到 max 的 bin 边界，步长为 tile_size
x_bins = np.arange(min_x, max_x + tile_size, tile_size)
y_bins = np.arange(min_y, max_y + tile_size, tile_size)

tile_count = 0

# 遍历所有网格
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        # 定义当前 tile 的边界
        x_min_tile = x_bins[i]
        x_max_tile = x_bins[i + 1]
        y_min_tile = y_bins[j]
        y_max_tile = y_bins[j + 1]

        # 筛选落在当前 tile 内的点
        tile_df = df[(df['X'] >= x_min_tile) & (df['X'] < x_max_tile) &
                     (df['Y'] >= y_min_tile) & (df['Y'] < y_max_tile)]

        # 如果当前 tile 没有点，则跳过
        if tile_df.empty:
            continue

        # 如果当前 tile 内所有点的变化掩码都是 -1（不确定区域），也可以选择跳过
        if (tile_df['ChangeMask'] == -1).all():
            continue

        # 提取点坐标和变化掩码
        points = tile_df[['X', 'Y', 'Z']].values  # 点云坐标，形状 (N, 3)
        labels = tile_df['ChangeMask'].values  # 变化掩码，形状 (N,)

        # 保存为 .npy 文件（便于后续深度学习使用）
        tile_filename = f"tile_{tile_count:04d}_points.npy"
        label_filename = f"tile_{tile_count:04d}_labels.npy"
        np.save(os.path.join(output_dir, tile_filename), points)
        np.save(os.path.join(output_dir, label_filename), labels)
        tile_count += 1

print(f"共保存了 {tile_count} 个 tile。")
