import torch
from torch.utils.data import DataLoader

from src.Dataset import PlansDataset, custom_collate_fn
from src.featurizer import pre_evaluate_process
from src.nets.Sub_MCD_net import MCDTCNN
import pickle

# 加载 pkl 文件
with open(r'D:\Saro\records\plans_dict_test_job.pkl', 'rb') as file:
    plans_dict = pickle.load(file)

# 创建 Dataset 和 DataLoader
plans_dataset = PlansDataset(plans_dict)
data_loader = DataLoader(plans_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)

# 初始化模型
in_channels = 9
model = MCDTCNN(in_channels=in_channels, dropout_prob=0.5)

# 在评估模式下运行一次，作为基准
model.eval()

for batch in data_loader:
    plan1, plan2, label = batch

    # 假设 pre_evaluate_process 返回的是 Tensor 类型的输入
    v_plans1 = pre_evaluate_process(plan1)
    v_plans2 = pre_evaluate_process(plan2)

    # print(f"Processed plan1: {v_plans1}")
    # print(f"Processed plan2: {v_plans2}")

    # 不使用 MC Dropout 进行一次推理作为基准
    with torch.no_grad():
        score_no_dropout = model(v_plans1, v_plans2, mc_dropout=False)
    print("Score without MC Dropout:", score_no_dropout)

    # 使用 MC Dropout 进行多次推理并计算方差
    mc_dropout_samples = 10
    scores_with_mc_dropout = []

    for _ in range(mc_dropout_samples):
        model.train()  # MC Dropout 模式（保持 Dropout active）
        with torch.no_grad():
            score = model(v_plans1, v_plans2, mc_dropout=True)  # 启用MC Dropout
        scores_with_mc_dropout.append(score)

    # 将所有结果堆叠为 tensor 并计算均值和方差
    scores_with_mc_dropout = torch.stack(scores_with_mc_dropout)
    mean_score = scores_with_mc_dropout.mean(dim=0)
    std_score = scores_with_mc_dropout.std(dim=0)

    print("Mean Score with MC Dropout:", mean_score)
    print("Standard Deviation with MC Dropout:", std_score)
