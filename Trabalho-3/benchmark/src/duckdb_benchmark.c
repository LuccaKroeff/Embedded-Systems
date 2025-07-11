#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jansson.h>
#include <duckdb.h>
#include "duckdb_benchmark.h"

#define DUCKDB_DB_NAME "benchmark.duckdb"

// Helper to execute a simple, non-timed query
void exec_ddb_sql(duckdb_connection conn, const char *sql) {
    duckdb_result result;
    if (duckdb_query(conn, sql, &result) != DuckDBSuccess) {
        fprintf(stderr, "DuckDB query error: %s\n", duckdb_result_error(&result));
    }
    duckdb_destroy_result(&result);
}

void run_duckdb_benchmarks(const char *json_path, char *db_name) {
    printf("--- Running DuckDB Benchmarks ---\n");

    json_error_t error;
    json_t *root = json_load_file(json_path, 0, &error);
    if (!root) {
        fprintf(stderr, "Error: on line %d: %s\n", error.line, error.text);
        return;
    }

    db_name = (db_name == NULL) ? DUCKDB_DB_NAME : db_name;

    duckdb_database db;
    duckdb_connection conn;
    if (duckdb_open(db_name, &db) != DuckDBSuccess || duckdb_connect(db, &conn) != DuckDBSuccess) {
        fprintf(stderr, "Error connecting to DuckDB\n");
        return;
    }

    size_t index;
    json_t *value;
    json_array_foreach(root, index, value) {
        const char *name = json_string_value(json_object_get(value, "name"));
        const char *sql = json_string_value(json_object_get(value, "query"));
        json_t *iter_json = json_object_get(value, "iterations");
        int iterations = iter_json ? json_integer_value(iter_json) : 1;
        
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
            for (int i = 1; i <= iterations; i++) {
                char name_buf[50];
                char email_buf[50];
                sprintf(name_buf, "user%d", i);
                sprintf(email_buf, "user%d@example.com", i);

                duckdb_bind_int32(stmt, 1, i);
                duckdb_bind_varchar(stmt, 2, name_buf);
                duckdb_bind_varchar(stmt, 3, email_buf);
                
                if (duckdb_execute_prepared(stmt, NULL) != DuckDBSuccess) {
                    fprintf(stderr, "Execution failed: %s\n", duckdb_result_error(NULL));
                }
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
    json_decref(root);
    remove(DUCKDB_DB_NAME);
    //remove(strcat(strdup(DUCKDB_DB_NAME), ".wal"));
    printf("--------------------------------\n\n");
}