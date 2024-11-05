import re
import numpy as np
from src.PGconnector import PostgresDB
import logging

logging.basicConfig(level=logging.WARNING)


class SimpleQueryChecker:
    def __init__(self, db_instance, max_tables=7, max_joins=5, max_filter_conditions=10, max_rows=1000 * 10,
                 max_subqueries=1):
        """
        初始化简单查询检查器
        :param db_instance: PostgresDB 的实例，用于执行 SQL 操作
        :param max_tables, max_joins, max_filter_conditions, max_rows, max_subqueries: 定义简单查询的限制条件
        """
        self.db = db_instance
        self.max_tables = max_tables
        self.max_joins = max_joins
        self.max_filter_conditions = max_filter_conditions
        self.max_rows = max_rows
        self.max_subqueries = max_subqueries

    def is_simple_query(self, query):
        """判断给定 SQL 查询是否为简单查询"""
        try:
            table_count = len(self.db.extract_table_names(query))
            logging.debug(f"表的数量为 {table_count}")
            if table_count > self.max_tables:
                logging.info("复杂查询：表数量超过限制")
                return False

            join_conditions = self.db.count_total_joins(query)
            logging.debug(f"连接条件数量为 {join_conditions}")
            if join_conditions > self.max_joins:
                logging.info("复杂查询：连接条件数量超过限制")
                return False

            filter_conditions = len(re.findall(r'\b(?:WHERE|AND|OR)\b', query, re.IGNORECASE)) - 1
            logging.debug(f"筛选条件数量为 {filter_conditions}")
            if filter_conditions > self.max_filter_conditions:
                logging.info("复杂查询：筛选条件数量超过限制")
                return False

            if bool(re.search(r'\bGROUP BY\b|\bORDER BY\b', query, re.IGNORECASE)):
                logging.info("复杂查询：包含 GROUP BY 或 ORDER BY")
                return False

            subqueries = len(re.findall(r'\bSELECT\b', query, re.IGNORECASE)) - 1
            logging.debug(f"子查询数量为 {subqueries}")
            if subqueries > self.max_subqueries:
                logging.info("复杂查询：子查询数量超过限制")
                return False

            self.db.connect()
            plan = self.db.get_execution_plan(query)
            rows = self.db.extract_rows_from_plan(plan)
            self.db.close()

            total_rows = np.sum(rows) if rows.size > 0 else 0
            logging.debug(f"扫描总行数为 {total_rows}")
            if total_rows > self.max_rows:
                logging.info("复杂查询：扫描行数超过限制")
                return False

            if bool(re.search(r'\bWITH RECURSIVE\b', query, re.IGNORECASE)):
                logging.info("复杂查询：包含递归 CTE")
                return False

            logging.info("简单查询")
            return True

        except Exception as e:
            logging.error(f"查询分析失败: {e}")
            return False


# 使用示例
if __name__ == '__main__':
    PG_CONNECTION_STR_JOB = {
        "dbname": "imdbload",
        "user": "postgres",
        "password": "postgres",
        "port": 5432
    }

    # 创建数据库对象并连接
    db_instance = PostgresDB(**PG_CONNECTION_STR_JOB)
    checker = SimpleQueryChecker(db_instance)

    test_query = """SELECT     MIN(chn.name) AS character,     MIN(t.title) AS movie_with_american_producer FROM     
    char_name AS chn,     cast_info AS ci,     company_name AS cn,     company_type AS ct,     movie_companies AS mc, 
        role_type AS rt,     title AS t WHERE     ci.note like '%r%'     AND cn.country_code = '[suhh]'     AND 
        t.production_year > 1974     AND t.id = mc.movie_id     AND t.id = ci.movie_id     AND ci.movie_id = 
        mc.movie_id     AND chn.id = ci.person_role_id     AND rt.id = ci.role_id     AND cn.id = mc.company_id     
        AND ct.id = mc.company_type_id;"""

    if checker.is_simple_query(test_query):
        print("该查询被定义为简单查询。")
    else:
        print("该查询被定义为复杂查询。")
