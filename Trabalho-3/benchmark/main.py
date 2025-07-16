from benchmark import BaseBenchmark
from platforms import Linux_x86, ESP32, Android

class BenchMarkDBs(BaseBenchmark):
    tests = [
        {
            "platform": Linux_x86,
            "cwd": "code/linux_x86/",
            "target": "sqlite_bench",
            "app": "sqlite",
            "input_file": "inputs/sql_queries.json",
        },
        {
            "platform": Linux_x86,
            "cwd": "code/linux_x86/",
            "target": "duckdb_bench",
            "app": "duckdb",
            "input_file": "inputs/sql_queries.json",
        },
        {
            "platform": Linux_x86,
            "cwd": "code/linux_x86/",
            "target": "rocksdb_bench",
            "app": "rocksdb",
            "input_file": "inputs/nosql_queries.json",
        },
    ]
    output_file = "results/results_linux_x86.csv"
    output_columns = ["time", "energy", "edp"]


class BenchMarkSQLite(BaseBenchmark):
    tests = [
        {
            "platform": ESP32,
            "cwd": "code/esp32",
            "app": "sqlite",
            "target": "sqlite",
            "input_file": "inputs/sql_queries_reduced.json",
        },
        {
            "platform": Linux_x86,
            "cwd": "code/linux_x86/",
            "app": "sqlite",
            "target": "sqlite_bench",
            "input_file": "inputs/sql_queries_reduced.json",
        },
        {
            "platform": Android,
            "cwd": "code/android/",
            "app": "sqlite",
            "target": "sqlite_bench",
            "input_file": "inputs/sql_queries_reduced.json",
        },
    ]
    output_file = "results/results_sqlite.csv"
    output_columns = ["time", "energy", "edp"]

#BenchMarkDBs().run()
BenchMarkSQLite().run()