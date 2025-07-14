class Core:
    def __str__(self):
        return f"{self.name}"

class SQLiteCore(Core):
    name = 'sqlite'
    queries_file = 'sql_queries.json'

class DuckDBCore(Core):
    name = 'duckdb'
    queries_file = 'sql_queries.json'

class RocksDBCore(Core):
    name = 'rocksdb'
    queries_file = 'nosql_queries.json'
