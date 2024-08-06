import json
import time
import psycopg2


def read_sql_file(file_path):
    """
    读取 SQL 文件中的查询。

    :param file_path: SQL 文件的路径
    :return: 文件中的 SQL 查询字符串
    """
    try:
        with open(file_path, 'r') as file:
            sql = file.read()
        return sql
    except IOError as e:
        print(f"Error reading SQL file {file_path}: {e}")
        return None


class PostgresDB:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print(f"Connected to database {self.dbname} successfully.")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            self.connection = None

    def close(self):
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def execute_query(self, query):
        if not self.connection:
            print("No active database connection.")
            return None

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                self.connection.commit()
                # print("Query executed successfully.")
                return cursor.fetchall() if cursor.description else None
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return None

    def execute_sql_file(self, file_path):
        if not self.connection:
            print("No active database connection.")
            return None

        sql = read_sql_file(file_path)
        if not sql:
            print(f"Failed to read SQL file: {file_path}")
            return None

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                self.connection.commit()
                return cursor.fetchall() if cursor.description else None
        except (psycopg2.Error, IOError) as e:
            print(f"Error executing SQL file: {e}")

    def get_execution_plan(self, query):
        """
        获取查询的物理执行计划，并以 JSON 格式返回。

        :param query: 要执行的查询
        :return: 查询的执行计划（JSON 格式）
        """
        if not self.connection:
            print("No active database connection.")
            return None

        try:
            explain_query = f"EXPLAIN (FORMAT JSON) {query}"
            with self.connection.cursor() as cursor:
                cursor.execute(explain_query)
                execution_plan = cursor.fetchall()
            return execution_plan[0][0] if execution_plan else None
        except psycopg2.Error as e:
            print(f"Error getting execution plan: {e}")
            return None

    def get_execution_plan_from_file(self, file_path):
        """
        从 SQL 文件中获取查询的物理执行计划（JSON 格式）。

        :param file_path: SQL 文件的路径
        :return: 查询的执行计划（JSON 格式）
        """
        query = read_sql_file(file_path)
        if query:
            return self.get_execution_plan(query)
        else:
            print(f"Failed to read SQL file: {file_path}")
            return None


# PG_CONNECTION_STR_JOB = {
#     "dbname": "imdbload",
#     "user": "postgres",
#     "password": "postgres",
#     "port": 5432
# }

# # 创建数据库对象并连接
# db_job = PostgresDB(**PG_CONNECTION_STR_JOB)
# db_job.connect()
#
# # 从 SQL 文件中获取查询的物理执行计划（JSON 格式）
# sql_file_path = r"D:\Saro\datasets\JOB\1a.sql"
# execution_plan = db_job.get_execution_plan_from_file(sql_file_path)
#
# if execution_plan:
#     # 打印格式化的 JSON 执行计划
#     print(json.dumps(execution_plan, indent=2))
#
# # 关闭连接
# db_job.close()
