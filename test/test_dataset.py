import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np  # 引入 numpy


# 自定义 Dataset 类
class PlansDataset(Dataset):
    def __init__(self, plans_dict):
        self.data = []

        # 遍历 plans_dict
        for file_name, plans in plans_dict.items():
            if len(plans) < 2:
                continue  # 跳过少于两个计划的情况

            # 获取最后一个计划及其执行时间
            last_plan = plans[-1]['plan']
            last_plan_time = plans[-1]['time']

            # 将前 n-1 个计划与最后一个计划配对
            for i in range(len(plans) - 1):
                plan1 = plans[i]['plan']
                time_diff = last_plan_time - plans[i]['time']

                # 使用 numpy 处理 time_diff，提前转换为 tensor
                self.data.append((plan1, last_plan, np.float32(time_diff)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        plan1, plan2, label = self.data[idx]
        return plan1, plan2, torch.tensor(label, dtype=torch.float)  # 将 label 转换为 tensor


# 自定义 collate_fn 函数，用于批量化处理数据
def custom_collate_fn(batch):
    # 将 plan1 和 plan2 按照批次转化为列表，减少 Python 循环的开销
    plan1_batch = [item[0] for item in batch]
    plan2_batch = [item[1] for item in batch]

    # 使用 torch.stack 批量处理 label，避免使用列表转换
    labels_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return plan1_batch, plan2_batch, labels_batch


# 读取 plans_dict 文件
with open(r"D:\Saro\records\plans_dict_stats.pkl", "rb") as f:
    plans_dict = pickle.load(f)

# 创建 Dataset 和 DataLoader
plans_dataset = PlansDataset(plans_dict)
data_loader = DataLoader(plans_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

# 测试 DataLoader
for batch in data_loader:
    plan1, plan2, label = batch
    print(plan1)
    print(plan2)
    print("Label:", label)
    break  # 仅打印一个 batch
