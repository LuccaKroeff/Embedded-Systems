#include <stdio.h>
#include "sqlite_benchmark.h"
#include "duckdb_benchmark.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path_to_queries.json>\n", argv[0]);
        return 1;
    }

    const char *json_path = argv[1];

    printf("Starting embedded database benchmarks...\n\n");

    // Run SQLite benchmarks
    run_sqlite_benchmarks(json_path, ":memory:");

    // Run DuckDB benchmarks
    run_duckdb_benchmarks(json_path, ":memory:");

    printf("Benchmarking complete.\n");

    return 0;
}