import pickle

pkl_file_path = r'D:\Saro\records\plans_dict_stats.pkl'
# 加载 pkl 文件
with open(pkl_file_path, 'rb') as file:
    plans_dict = pickle.load(file)
print(len(plans_dict.keys()))
# sql_file = "1d.sql"
# x = []  # [{plan1},{plan2}...]
# if sql_file in plans_dict and plans_dict[sql_file]:
#     exe_times = [item["time"] for item in plans_dict[sql_file]]
#     plans = [item["plan"] for item in plans_dict[sql_file]]
#     print(plans[0])