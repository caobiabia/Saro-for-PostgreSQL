import os
import glob

from src.PGconnector import PostgresDB

# 创建数据库对象并连接
PG_CONNECTION_STR_JOB = {
    "dbname": "imdbload",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

db_job = PostgresDB(**PG_CONNECTION_STR_JOB)
db_job.connect()

# 获取目录下所有的 SQL 文件
sql_directory = r"D:\Saro\datasets\JOB"
sql_files = glob.glob(os.path.join(sql_directory, "*.sql"))

# 遍历并执行所有 SQL 文件
for sql_file in sql_files:
    print(f"Executing file: {sql_file}")
    db_job.execute_sql_file(sql_file)

# 关闭连接
db_job.close()
