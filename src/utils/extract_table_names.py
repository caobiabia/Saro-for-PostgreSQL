import re


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


# 测试SQL查询
sql_query = """
SELECT MIN(mi_idx.info) AS rating,
       MIN(t.title) AS movie_title
FROM info_type AS it,
     keyword AS k,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     title AS t
WHERE it.info ='rating'
  AND k.keyword LIKE '%sequel%'
  AND mi_idx.info > '5.0'
  AND t.production_year > 2005
  AND t.id = mi_idx.movie_id
  AND t.id = mk.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it.id = mi_idx.info_type_id;
"""

# 调用函数
tables = extract_table_names(sql_query)
print(tables)
