import os
import pickle
import time

from tqdm import tqdm

from src.PGconnector import PostgresDB
from src.config import get_args
from src.utils import list_files_in_directory

args = get_args()

PG_CONNECTION_STR_JOB = {
    "dbname": "imdbload",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

_ALL_OPTIONS = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
]
# https://rmarcus.info/appendix.html
all_48_hint_sets = '''hashjoin,indexonlyscan
hashjoin,indexonlyscan,indexscan
hashjoin,indexonlyscan,indexscan,mergejoin
hashjoin,indexonlyscan,indexscan,mergejoin,nestloop
hashjoin,indexonlyscan,indexscan,mergejoin,seqscan
hashjoin,indexonlyscan,indexscan,nestloop
hashjoin,indexonlyscan,indexscan,nestloop,seqscan
hashjoin,indexonlyscan,indexscan,seqscan
hashjoin,indexonlyscan,mergejoin
hashjoin,indexonlyscan,mergejoin,nestloop
hashjoin,indexonlyscan,mergejoin,nestloop,seqscan
hashjoin,indexonlyscan,mergejoin,seqscan
hashjoin,indexonlyscan,nestloop
hashjoin,indexonlyscan,nestloop,seqscan
hashjoin,indexonlyscan,seqscan
hashjoin,indexscan
hashjoin,indexscan,mergejoin
hashjoin,indexscan,mergejoin,nestloop
hashjoin,indexscan,mergejoin,nestloop,seqscan
hashjoin,indexscan,mergejoin,seqscan
hashjoin,indexscan,nestloop
hashjoin,indexscan,nestloop,seqscan
hashjoin,indexscan,seqscan
hashjoin,mergejoin,nestloop,seqscan
hashjoin,mergejoin,seqscan
hashjoin,nestloop,seqscan
hashjoin,seqscan
indexonlyscan,indexscan,mergejoin
indexonlyscan,indexscan,mergejoin,nestloop
indexonlyscan,indexscan,mergejoin,nestloop,seqscan
indexonlyscan,indexscan,mergejoin,seqscan
indexonlyscan,indexscan,nestloop
indexonlyscan,indexscan,nestloop,seqscan
indexonlyscan,mergejoin
indexonlyscan,mergejoin,nestloop
indexonlyscan,mergejoin,nestloop,seqscan
indexonlyscan,mergejoin,seqscan
indexonlyscan,nestloop
indexonlyscan,nestloop,seqscan
indexscan,mergejoin
indexscan,mergejoin,nestloop
indexscan,mergejoin,nestloop,seqscan
indexscan,mergejoin,seqscan
indexscan,nestloop
indexscan,nestloop,seqscan
mergejoin,nestloop,seqscan
mergejoin,seqscan
nestloop,seqscan'''
all_48_hint_sets = all_48_hint_sets.split('\n')
all_48_hint_sets = [["enable_" + j for j in i.split(',')] for i in all_48_hint_sets]


def get_hints_by_arm_idx(arm_idx):
    hints = []
    for option in _ALL_OPTIONS:
        hints.append(f"SET {option} TO off")

    if -1 < arm_idx < 48:
        for i in all_48_hint_sets[arm_idx]:
            hints.append(f"SET {i} TO on")

    elif arm_idx == 48:
        for option in _ALL_OPTIONS:
            hints.append(f"SET {option} TO on")  # default PG setting
    else:
        print('48 hint set error')
        exit(0)
    return hints


def save_plans_dict(plans_dict, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(plans_dict, file)


def load_plans_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return {}


def recordAndExecuteSQL(DBParam, sqlPath, ARMS, save_path="plans_dict_job.pkl"):
    # 从文件加载已有的 plans_dict
    plans_dict = load_plans_dict(save_path)

    # 创建数据库对象并连接
    db_job = PostgresDB(**DBParam)
    db_job.connect()

    if not db_job.connection:
        print("Failed to connect to the database.")
        return

    sql_files = list_files_in_directory(sqlPath)

    for sql_file in sql_files:
        # 提取文件名（保留 .sql 扩展名）
        file_name = os.path.basename(sql_file)

        # 检查计划列表是否存在以及其长度是否小于 ARMS
        if file_name in plans_dict and len(plans_dict[file_name]) >= ARMS:
            print(f"Skipping {file_name}, already processed.")
            continue  # 跳过已处理的文件

        # 如果文件名未出现在 plans_dict 中，则初始化一个空列表
        if file_name not in plans_dict:
            plans_dict[file_name] = []

        # 获取已经处理的臂数
        processed_arms = len(plans_dict[file_name])

        for arm in tqdm(range(processed_arms, ARMS), desc=f"Processing {file_name}", unit="arm"):
            # print(f"Executing arm {arm + 1}/{ARMS} for file: {file_name}")
            hints = get_hints_by_arm_idx(arm)
            for hint in hints:
                db_job.execute_query(hint)  # 使用 execute_query 执行提示

            start_time = time.time()  # 记录开始时间
            db_job.execute_sql_file(sql_file)  # 执行 SQL 文件
            end_time = time.time()  # 记录结束时间

            execution_time = end_time - start_time  # 计算执行时间

            # 获取执行计划并将其添加到字典中的列表中
            plan = db_job.get_execution_plan_from_file(file_path=sql_file)
            plans_dict[file_name].append({"plan": plan[0], "time": execution_time})  # 存储执行计划和时间

            # 每次更新 plans_dict 后都保存到文件
            save_plans_dict(plans_dict, save_path)

    # 关闭数据库连接
    db_job.close()


if __name__ == '__main__':
    recordAndExecuteSQL(PG_CONNECTION_STR_JOB, args.fp, args.ARMS)
