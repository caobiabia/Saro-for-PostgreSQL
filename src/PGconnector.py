import json
import time
import psycopg2
import re


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

    @staticmethod
    def extract_table_names(sql_query):
        # 改进正则表达式，捕获 FROM, JOIN, INTO, UPDATE 后的表名
        # 排除 SQL 函数或常见关键字
        regex = r"(FROM|JOIN|INTO|UPDATE)\s+([`'\[\"]?[\w\.]+[`'\]\"]?)|(?:,)\s*([`'\[\"]?[\w\.]+[`'\]\"]?)"

        # 常见的SQL聚合函数和关键字列表
        sql_keywords = {
            'SELECT', 'WHERE', 'GROUP', 'ORDER', 'BY', 'MIN', 'MAX', 'COUNT', 'AVG', 'SUM',
            'DISTINCT', 'ON', 'AND', 'OR', 'NOT', 'AS', 'DESC', 'ASC', 'LEFT', 'RIGHT', 'INNER',
            'OUTER', 'FULL', 'JOIN', 'UNION', 'EXISTS', 'IN', 'LIKE', 'BETWEEN', 'HAVING', 'LIMIT',
            'OFFSET', 'TOP', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ALL', 'ANY', 'INTO',
            'INSERT', 'UPDATE', 'DELETE', 'FROM', 'VALUES'
        }

        # 查找所有匹配的表名
        matches = re.findall(regex, sql_query, re.IGNORECASE)

        # 提取所有可能的表名
        table_names = set()
        for match in matches:
            # 表名可能出现在第二个或第三个捕获组
            table_name = match[1] if match[1] else match[2]
            if table_name:
                # 清理表名去掉可能的包裹符号（`'`、`"`、[] 等）
                cleaned_table_name = re.sub(r"[`'\[\]\"]", "", table_name)
                # 排除SQL关键字
                if cleaned_table_name.upper() not in sql_keywords:
                    table_names.add(cleaned_table_name)

        return list(table_names)

    def get_table_cardinalities(self, tables):
        # 将表名转换为SQL中的IN查询
        table_names_str = ', '.join([f"'{table}'" for table in tables])
        query = f"""
        SELECT relname AS table_name, reltuples AS row_count
        FROM pg_class
        WHERE relname IN ({table_names_str});
        """
        result = self.execute_query(query)

        # 返回表名和基数的字典
        if result:
            table_cardinalities = {row[0]: row[1] for row in result}
            return table_cardinalities
        return {}


if __name__ == "__main__":
    PG_CONNECTION_STR_JOB = {
        "dbname": "imdbload",
        "user": "postgres",
        "password": "postgres",
        "port": 5432
    }

    # 创建数据库对象并连接
    db_job = PostgresDB(**PG_CONNECTION_STR_JOB)
    db_job.connect()

    # 从 SQL 文件中获取查询的物理执行计划（JSON 格式）
    sql_file_path = r"D:\Saro\datasets\JOB\1a.sql"
    sql = read_sql_file(sql_file_path)

    start_time = time.time()
    # 提取表名
    table_names = db_job.extract_table_names(sql)
    print("Extracted table names:", table_names)  # 输出提取到的表名

    # 获取表的基数
    table_cardinalities = db_job.get_table_cardinalities(table_names)
    print("Table cardinalities:", table_cardinalities)  # 输出表名及其基数
    end_time = time.time()
    time = end_time - start_time
    print("exe time:", time)
    # 关闭连接
    db_job.close()

