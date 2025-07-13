#ifndef DUCKDB_BENCHMARK_H
#define DUCKDB_BENCHMARK_H

#include "benchmark_utils.h"

void run_duckdb_benchmarks(const BenchmarkQuery queries[], int query_count);

#endif // DUCKDB_BENCHMARK_H