import pickle

# 文件路径
file_path = r"D:\\Saro\\records\\plans_dict_train_JOB.pkl"

# 加载字典
with open(file_path, "rb") as f:
    plans_dict = pickle.load(f)

# 删除键为 "q620" 的键值对
if "q620" in plans_dict:
    del plans_dict["q620"]

# 将修改后的字典保存回文件
with open(file_path, "wb") as f:
    pickle.dump(plans_dict, f)