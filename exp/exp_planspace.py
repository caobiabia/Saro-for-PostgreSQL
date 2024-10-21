import pickle
from collections import defaultdict

# 定义要查看的 pkl 文件路径
pkl_file_path = r'D:\Saro\records\plans_dict_job.pkl'

# 加载 pkl 文件
with open(pkl_file_path, 'rb') as file:
    plans_dict = pickle.load(file)

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
        print(exe_plans[0])
        print(exe_times[0])
        # 统计每个最小值索引出现的次数
        index_count[min_index] += 1

# # 输出每个索引及其出现的次数
# for index, count in index_count.items():
#     print(f"Index {index} occurs {count} time(s)")
