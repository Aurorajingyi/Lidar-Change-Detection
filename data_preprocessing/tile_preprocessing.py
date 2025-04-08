"""
Module: tile_preprocessing.py

功能：
    1. 读取 CloudCompare 导出的 CSV 文件（包含点云坐标和变化掩码）。
    2. 计算点云在 X 和 Y 方向的边界。
    3. 根据给定的 tile 大小（tile_size）将整个点云切分为多个 tile。
    4. 对每个 tile，提取落在该区域内的点云数据（X, Y, Z）和变化掩码（ChangeMask）。
    5. 如果 tile 数据为空或所有点均为不确定标签 (-1)，则跳过该 tile。
    6. 将有效的 tile 数据分别保存为 .npy 文件，方便后续深度学习训练使用。

作者: Aurora
日期: 2025-04-05
"""

import os
import numpy as np
import pandas as pd
import logging

# 配置 logging 输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_process_csv(csv_file: str) -> pd.DataFrame:
    """
    读取 CSV 文件并进行初步处理，返回 DataFrame。

    参数:
        csv_file (str): CSV 文件的路径。该 CSV 文件应包含标题行，
                        标题行应至少包含 'X', 'Y', 'Z', 'R', 'G', 'B', 'ChangeMask'。

    返回:
        pd.DataFrame: 处理后的 DataFrame，其中列名称已经被重命名为：
                      'X', 'Y', 'Z', 'R', 'G', 'B', 'ChangeMask'.
    """
    logging.info(f"读取 CSV 文件: {csv_file}")
    df = pd.read_csv(csv_file, sep=',', header=0, index_col=False, low_memory=False)

    # 根据 CSV 文件实际标题重命名（根据你的文件调整）
    df.rename(columns={
        '//X': 'X',
        'Approx. distances': 'ApproxDistance',
        'Temp. approx. distances': 'ChangeDistance'
    }, inplace=True)

    # 检查必要的字段是否存在
    required_columns = ['X', 'Y', 'Z', 'ChangeMask']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"缺少必要字段: {col}")
            raise ValueError(f"CSV 文件中缺少必要字段: {col}")

    logging.info("CSV 文件加载成功")
    return df


def split_into_tiles(df: pd.DataFrame, tile_size: float, output_dir: str) -> int:
    """
    根据 DataFrame 中的 X 和 Y 坐标范围，按照 tile_size 划分网格，
    并将每个 tile 内的点云数据和变化掩码分别保存为 .npy 文件。

    参数:
        df (pd.DataFrame): 包含点云数据的 DataFrame，必须包含 'X', 'Y', 'Z', 'ChangeMask' 列。
        tile_size (float): 每个 tile 的边长（单位：米）。
        output_dir (str): 输出文件夹路径，用于保存生成的 tile 数据。

    返回:
        int: 成功保存的 tile 数量。
    """
    logging.info(f"开始划分 tile，tile 大小 = {tile_size} 米")
    os.makedirs(output_dir, exist_ok=True)

    # 计算 X 和 Y 方向的边界
    min_x, max_x = df['X'].min(), df['X'].max()
    min_y, max_y = df['Y'].min(), df['Y'].max()
    logging.info(f"X 范围: {min_x:.2f} - {max_x:.2f}")
    logging.info(f"Y 范围: {min_y:.2f} - {max_y:.2f}")

    # 生成从 min 到 max 的 bin 边界，步长为 tile_size
    x_bins = np.arange(min_x, max_x + tile_size, tile_size)
    y_bins = np.arange(min_y, max_y + tile_size, tile_size)

    tile_count = 0
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

            # 如果当前 tile 内所有点的变化掩码都是 -1（不确定区域），也跳过
            if (tile_df['ChangeMask'] == -1).all():
                continue

            # 提取点坐标和变化掩码
            points = tile_df[['X', 'Y', 'Z']].values  # 形状 (N, 3)
            labels = tile_df['ChangeMask'].values  # 形状 (N,)

            # 构造保存文件名
            tile_filename = f"tile_{tile_count:04d}_points.npy"
            label_filename = f"tile_{tile_count:04d}_labels.npy"

            # 保存为 .npy 文件
            np.save(os.path.join(output_dir, tile_filename), points)
            np.save(os.path.join(output_dir, label_filename), labels)

            logging.info(f"保存 tile {tile_count:04d}: {points.shape[0]} 点")
            tile_count += 1

    logging.info(f"共保存了 {tile_count} 个 tile。")
    return tile_count


if __name__ == "__main__":
    # 示例参数，修改为你的实际路径
    csv_file = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\DenseMap_Change_with_mask.csv"
    output_dir = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\tile_data"
    tile_size = 10.0

    # 读取并处理 CSV 数据
    df = load_and_process_csv(csv_file)
    print("CSV 文件预览：")
    print(df.head(10))

    # 划分 tile 并保存
    tile_count = split_into_tiles(df, tile_size, output_dir)
    print(f"共保存了 {tile_count} 个 tile。")
