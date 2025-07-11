#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jansson.h>
#include <sqlite3.h>
#include "sqlite_benchmark.h"

#define SQLITE_DB_NAME "benchmark.sqlite"

// Helper to execute a simple, non-timed query
void exec_sql(sqlite3 *db, const char *sql) {
    char *err_msg = 0;
    if (sqlite3_exec(db, sql, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    }
}

void run_sqlite_benchmarks(const char *json_path, char *db_name) {
    printf("--- Running SQLite Benchmarks ---\n");

    json_error_t error;
    json_t *root = json_load_file(json_path, 0, &error);
    if (!root) {
        fprintf(stderr, "Error: on line %d: %s\n", error.line, error.text);
        return;
    }

    db_name = (db_name == NULL) ? SQLITE_DB_NAME : db_name;

    sqlite3 *db;
    if (sqlite3_open(db_name, &db)) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
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
            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) != SQLITE_OK) {
                fprintf(stderr, "Error preparing statement: %s\n", sqlite3_errmsg(db));
                continue;
            }

            exec_sql(db, "BEGIN TRANSACTION;");
            for (int i = 1; i <= iterations; i++) {
                char name_buf[50];
                char email_buf[50];
                sprintf(name_buf, "user%d", i);
                sprintf(email_buf, "user%d@example.com", i);

                sqlite3_bind_int(stmt, 1, i);
                sqlite3_bind_text(stmt, 2, name_buf, -1, SQLITE_STATIC);
                sqlite3_bind_text(stmt, 3, email_buf, -1, SQLITE_STATIC);

                if (sqlite3_step(stmt) != SQLITE_DONE) {
                    fprintf(stderr, "Execution failed: %s\n", sqlite3_errmsg(db));
                }
                sqlite3_reset(stmt);
            }
            exec_sql(db, "COMMIT;");
            sqlite3_finalize(stmt);

        } else { // Simple query
            exec_sql(db, sql);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Done in %.4f seconds.\n", time_spent);
    }

    sqlite3_close(db);
    json_decref(root);
    remove(SQLITE_DB_NAME);
    printf("---------------------------------\n\n");
}