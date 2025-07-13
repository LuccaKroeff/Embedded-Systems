#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <duckdb.h>
#include "duckdb_benchmark.h"

#define DUCKDB_DB_NAME ":memory:"

// Helper to execute a simple, non-timed query
void exec_ddb_sql(duckdb_connection conn, const char *sql) {
    duckdb_result result;
    if (duckdb_query(conn, sql, &result) != DuckDBSuccess) {
        fprintf(stderr, "DuckDB query error: %s\n", duckdb_result_error(&result));
    }
    duckdb_destroy_result(&result);
}

void run_duckdb_benchmarks(const BenchmarkQuery queries[], int query_count) {
    printf("--- Running DuckDB Benchmarks ---\n");

    duckdb_database db;
    duckdb_connection conn;
    if (duckdb_open(DUCKDB_DB_NAME, &db) != DuckDBSuccess || duckdb_connect(db, &conn) != DuckDBSuccess) {
        fprintf(stderr, "Error connecting to DuckDB\n");
        return;
    }

    for (int i = 0; i < query_count; i++) {
        const char *name = queries[i].name;
        const char *sql = queries[i].query;
        int iterations = queries[i].iterations;
        
        printf("  Executing '%s'... ", name);
        fflush(stdout);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (iterations > 1) { // Prepared statement loop
            duckdb_prepared_statement stmt;
            if (duckdb_prepare(conn, sql, &stmt) != DuckDBSuccess) {
                fprintf(stderr, "Error preparing statement: %s\n", duckdb_prepare_error(stmt));
                continue;
            }

            exec_ddb_sql(conn, "BEGIN TRANSACTION;");
            for (int j = 1; j <= iterations; j++) {
                char name_buf[50], email_buf[50];
                sprintf(name_buf, "user%d", j);
                sprintf(email_buf, "user%d@example.com", j);
                duckdb_bind_int32(stmt, 1, j);
                duckdb_bind_varchar(stmt, 2, name_buf);
                duckdb_bind_varchar(stmt, 3, email_buf);
                if (duckdb_execute_prepared(stmt, NULL) != DuckDBSuccess) fprintf(stderr, "Execution failed.\n");
            }
            exec_ddb_sql(conn, "COMMIT;");
            duckdb_destroy_prepare(&stmt);
        } else { // Simple query
            exec_ddb_sql(conn, sql);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Done in %.4f seconds.\n", time_spent);
    }

    duckdb_disconnect(&conn);
    duckdb_close(&db);
    printf("--------------------------------\n\n");
}