#include <stdio.h>
#include "sqlite_benchmark.h"
#include "duckdb_benchmark.h"
#include "duckdb.h"


int main() {
    printf("SQLITE:\n");
    sqlite_benchmark();
    printf("DuckDB:\n");
    duckdb_benchmark();
    return 0;
}
