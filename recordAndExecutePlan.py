import logging
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

# 设置日志记录
log_dir = r"/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "SQLexecute.log")

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
        logging.error("Failed to connect to the database.")
        return

    sql_files = list_files_in_directory(sqlPath)

    for sql_file in sql_files:
        file_name = os.path.basename(sql_file)

        # 检查计划列表是否存在以及其长度是否小于 ARMS
        if file_name in plans_dict and len(plans_dict[file_name]) >= ARMS:
            logging.info(f"Skipping {file_name}, already processed.")
            continue

        if file_name not in plans_dict:
            plans_dict[file_name] = []

        processed_arms = len(plans_dict[file_name])

        for arm in tqdm(range(processed_arms, ARMS), desc=f"Processing {file_name}", unit="arm"):
            hints = get_hints_by_arm_idx(arm)
            for hint in hints:
                try:
                    db_job.execute_query(hint)  # 执行提示
                except Exception as e:
                    logging.error(f"Error executing query hint: {e}")

            # 获取执行计划
            plan = None
            try:
                db_job.execute_query("BEGIN;")  # 开始新的事务
                plan = db_job.get_execution_plan_from_file(file_path=sql_file)
                if plan is None:
                    plan = ["Plan Not Available"]  # 占位符
                db_job.execute_query("COMMIT;")  # 提交事务
            except Exception as e:
                logging.error(f"Error getting execution plan: {e}")
                plan = ["Plan Not Available"]  # 占位符
                db_job.execute_query("ROLLBACK;")  # 回滚事务

            # 执行 SQL 文件
            start_time = time.time()  # 记录开始时间
            try:
                db_job.execute_query("BEGIN;")  # 开始新的事务
                db_job.execute_query("SET statement_timeout TO 180000")  # 增加超时时间
                db_job.execute_sql_file(sql_file)  # 执行 SQL 文件
                db_job.execute_query("COMMIT;")  # 提交事务
            except Exception as e:
                logging.error(f"Error executing SQL file {file_name}: {e}")
                db_job.execute_query("ROLLBACK;")  # 回滚事务
                continue  # 继续处理下一个文件
            finally:
                end_time = time.time()  # 记录结束时间
                execution_time = end_time - start_time  # 计算执行时间

                # 添加到 plans_dict 中
                plans_dict[file_name].append({"plan": plan[0], "time": execution_time})

                # 每次更新 plans_dict 后都保存到文件
                save_plans_dict(plans_dict, save_path)

    # 关闭数据库连接
    db_job.close()


if __name__ == '__main__':
    recordAndExecuteSQL(PG_CONNECTION_STR_JOB, args.fp, args.ARMS)