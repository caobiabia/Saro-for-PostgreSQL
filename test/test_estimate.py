import pickle

# 从指定路径加载 plans_dict
pkl_path = r"D:\Saro\records\plans_dict_test_TPCH.pkl"
with open(pkl_path, "rb") as f:
    plans_dict = pickle.load(f)

# 统计满足条件的 file_name 数量
count = 0

for file_name, plans in plans_dict.items():
    # 计算当前 file_name 中 "time" >= 60 的数量
    num_time_over_60 = sum(1 for plan in plans if plan["time"] >= 60)

    # 判断是否满足 "time >= 60" 的数量超过总计划数量的一半
    if num_time_over_60 > len(plans) / 4:
        count += 1

print(f"满足条件的 file_name 数量: {count}")
