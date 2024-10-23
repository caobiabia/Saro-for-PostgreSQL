import pickle
from collections import defaultdict
import matplotlib.pyplot as plt


# 定义要查看的 pkl 文件路径
pkl_file_path = r'D:\Saro\records\plans_dict_job.pkl'
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
        exe_plans = [item["plan"]["Plan"] for item in plans_list]
        # 找出time列表中最小值的索引
        min_time = min(exe_times)
        # print(min_time)
        min_index = exe_times.index(min_time)
        # print(exe_plans[0])
        # print(exe_times[0])
        # 统计每个最小值索引出现的次数
        index_count[min_index] += 1

# Prepare data for plotting
indices = list(index_count.keys())
counts = list(index_count.values())

# Complete the indices to include all from 0 to 48 (assuming 49 plans)
all_indices = list(range(49))
all_counts = [0] * len(all_indices)

# Fill in the occurrences
for index, count in zip(indices, counts):
    all_counts[index] = count

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(all_indices, all_counts, color='skyblue')
plt.xlabel('Generated Physical Execution Plan Index')
plt.ylabel('Occurrences of Shortest Execution Time Plans')
plt.title('Distribution of Shortest Execution Time Plans Among 49 Plans')
plt.xticks(all_indices)  # Set x-axis ticks

# Set y-axis to show integer values
plt.yticks(range(0, max(all_counts) + 1))

plt.grid(axis='y')
plt.savefig('draws/hints_distribution.png', dpi=300)
# Show the plot
plt.show()
