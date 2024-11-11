import logging
import pickle
import time

import torch
from tqdm import tqdm

from recordAndExecutePlan import get_hints_by_arm_idx
from src.PGconnector import PostgresDB, read_sql_file
from src.featurizer import pre_evaluate_process
from src.nets.SATCNN import SATCNN_Extend
from pprint import pprint

# 定义要查看的 pkl 文件路径
# # print(plans_dict.keys())
# sql_file = '10a.sql'
# exe_plan_and_time = 0
# 检查键是否存在
# if sql_file in plans_dict:
#     print(plans_dict[sql_file][exe_plan_and_time]['plan'])
#     print(type(plans_dict[sql_file][exe_plan_and_time]['plan']))  # 打印对应键的内容
# else:
#     print(f"Key '{sql_file}' not found in the data.")


# pkl_file_path = r'D:\Saro\records\plans_dict_job.pkl'
# # 加载 pkl 文件
# with open(pkl_file_path, 'rb') as file:
#     plans_dict = pickle.load(file)
#
# sql_file = "12c.sql"
# x = []  # [{plan1},{plan2}...]
# if sql_file in plans_dict and plans_dict[sql_file]:
#     exe_times = [item["time"] for item in plans_dict[sql_file]]
#     plans = [item["plan"] for item in plans_dict[sql_file]]
#     # print(plans)
#     x = pre_evaluate_process(plans)
#     # print(x[0][1][1][0])
#     # print(exe_times[6])

DBParam = {
    "dbname": "imdbload",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

sql_path = "datasets/JOB/12c.sql"
db_job = PostgresDB(**DBParam)
db_job.connect()

plans = []
for arm in tqdm(range(0, 49), unit="arm"):
    hints = get_hints_by_arm_idx(arm)
    for hint in hints:
        try:
            db_job.execute_query("BEGIN;")  # 开始新的事务
            db_job.execute_query(hint)  # 执行提示
        except Exception as e:
            logging.error(f"Error executing query hint: {e}")

    # 获取执行计划
    plan = None
    try:
        plan = db_job.get_execution_plan_from_file(file_path=sql_path)
        plans.extend(plan)
        db_job.execute_query("COMMIT;")  # 提交事务
    except Exception as e:
        logging.error(f"Error getting execution plan: {e}")
        db_job.execute_query("ROLLBACK;")  # 回滚事务
print(type(plans))
print(len(plans))
v_plans = pre_evaluate_process(plans)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SATCNN_Extend(9).to(device)
net.load_state_dict(torch.load(r'D:\Saro\outputs\finetune.pt'))

net.eval()
scores = net(v_plans)
scores = torch.reshape(scores, (-1, 49))  # [num_sql, 49]
sorted_, indices = torch.sort(scores, dim=-1, descending=False)
indices = torch.reshape(indices, (-1, 49))
index = indices[:, 0].cpu().numpy()  # nparray 只有一个元素
hints = get_hints_by_arm_idx(index[0])
print(hints)


start_time = time.time()
for hint in hints:
    try:
        db_job.execute_query(hint)  # 执行提示

    except Exception as e:
        logging.error(f"Error executing query hint: {e}")
print("提示集执行完毕")

result = db_job.execute_query(read_sql_file(sql_path))
print(f"执行时间：{time.time() - start_time} s")
print(result)
