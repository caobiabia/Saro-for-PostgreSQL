import re


def count_total_joins(sql_query):
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

    # 打印唯一表名
    print("唯一表名:", unique_tables)

    # 在 WHERE 子句中查找连接条件
    where_clause = re.search(r'WHERE(.*?);', sql_query, re.DOTALL)
    if where_clause:
        where_conditions = where_clause.group(1).strip()
    else:
        where_conditions = ''

    # 打印 WHERE 条件
    print("WHERE 条件:", where_conditions)

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
                        print(f"找到隐式连接: {table1} <-> {table2}")

    # 返回显式连接数量和隐式连接数量的总和
    total_join_count = explicit_join_count + implicit_join_count
    return total_join_count


# 示例 SQL 查询
test_query = """SELECT     MIN(chn.name) AS character,     MIN(t.title) AS movie_with_american_producer FROM     
char_name AS chn,     cast_info AS ci,     company_name AS cn,     company_type AS ct,     movie_companies AS mc, 
    role_type AS rt,     title AS t WHERE     ci.note like '%r%'     AND cn.country_code = '[suhh]'     AND 
    t.production_year > 1974     AND t.id = mc.movie_id     AND t.id = ci.movie_id     AND ci.movie_id = 
    mc.movie_id     AND chn.id = ci.person_role_id     AND rt.id = ci.role_id     AND cn.id = mc.company_id     
    AND ct.id = mc.company_type_id;"""

# 调用函数并打印连接数量
total_join_count = count_total_joins(test_query)
print(f"总连接操作数量: {total_join_count}")
