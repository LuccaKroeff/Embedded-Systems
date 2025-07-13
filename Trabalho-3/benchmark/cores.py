class Core:
    def __str__(self):
        return f"{self.name}"

class SQLiteCore(Core):
    name = 'sqlite'

class DuckDBCore(Core):
    name = 'duckdb'

class RocksDBCore(Core):
    name = 'rocksdb'
