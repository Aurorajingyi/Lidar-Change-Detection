import pandas as pd
from tabulate import tabulate

# 1. 读取 CSV 文件（使用逗号分隔，header=0 表示使用文件第一行作为标题）
csv_path = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\DenseMap_Change.csv"
df = pd.read_csv(csv_path, sep=',', header=0, index_col=False, low_memory=False)

# 2. 重命名列（根据CSV 文件实际标题改动）
df.rename(columns={
    '//X': 'X',
    'Approx. distances': 'ApproxDistance',
    'Temp. approx. distances': 'ChangeDistance'
}, inplace=True)

# 检查前10行数据，确认读取正确
print(tabulate(df.head(10), headers='keys', tablefmt='pretty'))

# 3. 生成变化掩码（Change Mask）
# 这里使用了双阈值策略：如果变化距离大于 T_high 则认为发生了变化，
# 小于 T_low 则认为没有变化，其间则视为不确定区域（-1）
T_low = 0.3   # 比如小于 0.3 米认为没变化
T_high = 0.5  # 大于 0.5 米认为有明确变化

def mask_rule(d):
    if d > T_high:
        return 1    # 表示发生变化
    elif d < T_low:
        return 0    # 表示没有变化
    else:
        return -1   # 表示不确定，可以在训练时忽略

df['ChangeMask'] = df['ChangeDistance'].apply(mask_rule)

# 输出变化掩码的统计信息
print("Change mask counts:")
print(df['ChangeMask'].value_counts())

# 4. 保存含有变化掩码的新数据文件，方便后续使用（例如用作深度学习训练的标签）
output_path = r"D:\3D-demo-data\3D_Data\VMMS_LiDAR\DenseMap_Change_with_mask.csv"
df.to_csv(output_path, index=False)

# -------------------------------------------------------------
# 以下部分是对比 ChangeDet 的思路，后续扩展参考：
#
# ChangeDet 项目中，他们通常不仅仅使用单一阈值，而是采用“双阈值策略”
# 来区分显著变化、不变和不确定区域。这里就简单实现了这种策略，
# 并将不确定区域标记为 -1（在深度学习训练中，可以 mask 掉这部分区域）。
#
# 此外，ChangeDet 还会对提取出的变化掩码做后处理（例如通过连通域分析
# 去除小面积噪声），可以考虑后续使用 OpenCV 或 scikit-image 来实现类似功能，
# 这部分可以作为后续论文中进一步精细化变化掩码生成的方法展开讨论。
# -------------------------------------------------------------

print("数据处理完成，可以进行下一步的可视化或模型训练准备。")
