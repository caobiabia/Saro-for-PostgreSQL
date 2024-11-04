import json
import time

import numpy as np
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

    def count_total_joins(self, sql_query):
        # 计算显式连接数量
        explicit_joins = re.findall(r'\bJOIN\b', sql_query, re.IGNORECASE)
        explicit_join_count = len(explicit_joins)

        # 提取所有表名，匹配如 "table_name AS alias" 或 "table_name"
        tables = re.findall(r'(\w+)\s+AS\s+(\w+)|(\w+)', sql_query)

        # 提取表名，去除 None、空字符串和 SQL 关键字
        unique_tables = set()
        for match in tables:
            unique_tables.update(filter(lambda x: x and x not in {
                'SELECT', 'FROM', 'WHERE', 'AS', 'AND', 'MIN', 'LIKE', 'character', 'movie_with_american_producer'
            }, match))

        # # 打印唯一表名
        # print("唯一表名:", unique_tables)

        # 在 WHERE 子句中查找连接条件
        where_clause = re.search(r'WHERE(.*?);', sql_query, re.DOTALL)
        if where_clause:
            where_conditions = where_clause.group(1).strip()
        else:
            where_conditions = ''

        # # 打印 WHERE 条件
        # print("WHERE 条件:", where_conditions)

        # 初始化隐式连接计数
        implicit_join_count = 0
        found_connections = set()

        # 查找隐式连接条件
        for table1 in unique_tables:
            for table2 in unique_tables:
                if table1 != table2:
                    # 查找隐式连接条件
                    pattern = re.compile(
                        rf'\b{table1}\.(\w+)\s*=\s*{table2}\.(\w+)|\b{table2}\.(\w+)\s*=\s*{table1}\.(\w+)')
                    matches = pattern.findall(where_conditions)
                    if matches:
                        # 只记录一次连接
                        connection = tuple(sorted([table1, table2]))
                        if connection not in found_connections:
                            found_connections.add(connection)
                            implicit_join_count += 1  # 计数加一
                            # print(f"找到隐式连接: {table1} <-> {table2}")

        # 返回显式连接数量和隐式连接数量的总和
        total_join_count = explicit_join_count + implicit_join_count
        return total_join_count

    def extract_rows_from_plan(self, data: list):
        """
        从查询计划中提取 Plan Rows 数据，并返回处理后的 rows 数组。

        参数:
        - data (list): 查询计划的列表，其中每个计划是一个字典

        返回:
        - rows (np.array): 提取并取对数的 rows 数组
        """
        rows = []

        def recurse(n):
            # 提取当前节点的 Plan Rows
            if "Plan Rows" in n:
                rows.append(n["Plan Rows"])

            # 递归处理子节点
            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)

        # 遍历所有计划
        for plan in data:
            recurse(plan["Plan"])

        # 将提取的行数转换为 numpy 数组并取对数
        rows = np.array(rows)
        return rows

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


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
