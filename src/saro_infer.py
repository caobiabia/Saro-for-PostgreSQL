import logging
import time

import torch
from tqdm import tqdm

from recordAndExecutePlan import get_hints_by_arm_idx
from src.PGconnector import PostgresDB, read_sql_file
from src.featurizer import pre_evaluate_process
from src.nets.SATCNN import SATCNN_Extend


class Saro_infer:
    def __init__(self, db_param, sql_path, model_path):
        self.db_param = db_param
        self.sql_path = sql_path
        self.model_path = model_path
        self.db_job = None
        self.net = None

    def connect_db(self):
        self.db_job = PostgresDB(**self.db_param)
        self.db_job.connect()

    def execute_sql_with_plans(self):
        plans = []
        for arm in tqdm(range(0, 49), unit="arm"):
            hints = get_hints_by_arm_idx(arm)
            for hint in hints:
                try:
                    self.db_job.execute_query("BEGIN;")  # 开始新的事务
                    self.db_job.execute_query(hint)  # 执行提示
                except Exception as e:
                    logging.error(f"Error executing query hint: {e}")

            # 获取执行计划
            try:
                plan = self.db_job.get_execution_plan_from_file(file_path=self.sql_path)
                plans.extend(plan)
                self.db_job.execute_query("COMMIT;")  # 提交事务
            except Exception as e:
                logging.error(f"Error getting execution plan: {e}")
                self.db_job.execute_query("ROLLBACK;")  # 回滚事务

        return plans

    def process_plans(self, plans):
        v_plans = pre_evaluate_process(plans)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SATCNN_Extend(9).to(device)
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()
        scores = self.net(v_plans)
        scores = torch.reshape(scores, (-1, 49))  # [num_sql, 49]
        sorted_, indices = torch.sort(scores, dim=-1, descending=False)
        indices = torch.reshape(indices, (-1, 49))
        index = indices[:, 0].cpu().numpy()  # nparray 只有一个元素
        return index

    def run_inference(self):
        self.connect_db()
        plans = self.execute_sql_with_plans()
        index = self.process_plans(plans)

        hints = get_hints_by_arm_idx(index[0])
        print("提示集执行完毕")

        start_time = time.time()
        for hint in hints:
            try:
                self.db_job.execute_query(hint)  # 执行提示
            except Exception as e:
                logging.error(f"Error executing query hint: {e}")

        result = self.db_job.execute_query(read_sql_file(self.sql_path))
        exe_time = time.time() - start_time
        self.db_job.close()
        return index, exe_time, result

    def cmp_to_pg(self):
        self.connect_db()
        start_time = time.time()
        self.db_job.execute_query(read_sql_file(self.sql_path))
        exe_time = time.time() - start_time
        self.db_job.close()
        return exe_time


if __name__ == '__main__':
    DBParam = {
        "dbname": "STATS",
        "user": "postgres",
        "password": "postgres",
        "port": 5432
    }

    sql_path = r"D:\Saro\datasets\STATS\q0.sql"
    model_path = r'D:\Saro\outputs\finetune.pt'
    saro_infer = Saro_infer(DBParam, sql_path, model_path)

    # index, execution_time, result = saro_infer.run_inference()
    # print(index, execution_time, result)
    pg_time = saro_infer.cmp_to_pg()
    print(pg_time)
