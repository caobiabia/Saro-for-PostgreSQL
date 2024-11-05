import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 定义要查看的 pkl 文件路径
pkl_file_path = r'D:\Saro\records\plans_dict_test_TPCH.pkl'

# 加载 pkl 文件
with open(pkl_file_path, 'rb') as file:
    plans_dict = pickle.load(file)
print(len(plans_dict.keys()))

# 字典用于统计每个索引最小值出现的次数
index_count = defaultdict(int)

# 遍历所有sql_file，查找每个文件的time列表中的最小值索引
for sql_file, plans_list in plans_dict.items():
    if plans_list:
        exe_times = [item["time"] for item in plans_list]
        exe_plans = [item["plan"] for item in plans_list]
        min_time = min(exe_times)
        min_index = exe_times.index(min_time)
        index_count[min_index] += 1

# 准备绘图数据
indices = list(index_count.keys())
counts = list(index_count.values())

# 补全索引以包含从0到48的所有值（假设有49个计划）
all_indices = list(range(49))
all_counts = [0] * len(all_indices)

# 填入出现次数
for index, count in zip(indices, counts):
    all_counts[index] = count

# 创建柱状图
plt.figure(figsize=(12, 6))
plt.bar(all_indices, all_counts, color='skyblue')
plt.xlabel('Generated Physical Execution Plan Index')
plt.ylabel('Occurrences of Shortest Execution Time Plans')
plt.title('Distribution of Shortest Execution Time Plans Among 49 Plans')
plt.xticks(all_indices)  # 设置x轴刻度

# 设置y轴为整数，并减少刻度数量
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))  # 控制Y轴刻度数量

# 保存图像
plt.savefig('draws/hints_distribution_test_TPCH.png', dpi=300)
# 显示图像
plt.show()
