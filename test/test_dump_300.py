import pickle

# 加载字典
with open(r"D:\Saro\records\plans_dict_stats_copy.pkl", "rb") as file:
    plans_dict = pickle.load(file)

# 处理字典
keys_to_delete = []
for file_name, plans in plans_dict.items():
    if all(plan["time"] > 300 for plan in plans):
        keys_to_delete.append(file_name)

# 删除满足条件的键值对
delete_count = len(keys_to_delete)  # 记录删除的数量
for key in keys_to_delete:
    del plans_dict[key]

# 保存处理后的字典
with open(r"D:\Saro\records\plans_dict_stats_copy.pkl", "wb") as file:
    pickle.dump(plans_dict, file)

print(f"处理完成，已删除 {delete_count} 个符合条件的键值对。")
