import time
import psycopg2


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
                print("Query executed successfully.")
                return cursor.fetchall() if cursor.description else None
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return None

    def execute_sql_file(self, file_path):
        if not self.connection:
            print("No active database connection.")
            return None

        try:
            with open(file_path, 'r') as file:
                sql = file.read()

            with self.connection.cursor() as cursor:
                start_time = time.time()
                cursor.execute(sql)
                self.connection.commit()
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Executed SQL file {file_path} successfully.")
                print(f"Execution time: {elapsed_time:.2f} seconds.")
                return cursor.fetchall() if cursor.description else None
        except (psycopg2.Error, IOError) as e:
            print(f"Error executing SQL file: {e}")


# PG_CONNECTION_STR_JOB = {
#     "dbname": "imdbload",
#     "user": "postgres",
#     "password": "postgres",
#     "port": 5432
# }
#
# db_job = PostgresDB(**PG_CONNECTION_STR_JOB)
# db_job.connect()
#
# sql_file_path = r"D:\Saro\datasets\JOB\1a.sql"
# result = db_job.execute_sql_file(sql_file_path)
# if result:
#     print(result)
