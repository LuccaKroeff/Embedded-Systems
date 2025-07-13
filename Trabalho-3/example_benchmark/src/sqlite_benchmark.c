#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sqlite3.h>
#include "sqlite_benchmark.h"

#define SQLITE_DB_NAME ":memory:"

// Helper to execute a simple, non-timed query
void exec_sql(sqlite3 *db, const char *sql) {
    char *err_msg = 0;
    if (sqlite3_exec(db, sql, 0, 0, &err_msg) != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    }
}

void run_sqlite_benchmarks(const BenchmarkQuery queries[], int query_count) {
    printf("--- Running SQLite Benchmarks ---\n");

    sqlite3 *db;
    if (sqlite3_open(SQLITE_DB_NAME, &db)) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return;
    }

    for (int i = 0; i < query_count; i++) {
        const char *name = queries[i].name;
        const char *sql = queries[i].query;
        int iterations = queries[i].iterations;
        int slots = queries[i].slots;

        printf("  Executing '%s'... ", name);
        fflush(stdout);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (slots > 1) { // Prepared statement
            sqlite3_stmt *stmt;
            if (sqlite3_prepare_v2(db, sql, -1, &stmt, 0) != SQLITE_OK) {
                fprintf(stderr, "Error preparing statement: %s\n", sqlite3_errmsg(db));
                continue;
            }

            exec_sql(db, "BEGIN TRANSACTION;");
            for (int j = 1; j <= iterations; j++) {
                // Filling templates and binding slots
                for (int slot=0; slot < queries[i].slots; slot++) {
                    QueryTemplate template = queries[i].templates[slot];
                    char buf[50];
                    sprintf(buf, template.template, j);
                    
                    if (strcmp("int", template.type) == 0) {
                        sqlite3_bind_int(stmt, slot + 1, atoi(buf));
                    } else if (strcmp("str", template.type) == 0) {
                        sqlite3_bind_text(stmt, slot + 1, buf, -1, SQLITE_STATIC);
                    } else {
                        printf("Error: %s\n", template.type);
                    }
                }
    
                if (sqlite3_step(stmt) != SQLITE_DONE) fprintf(stderr, "Execution failed: %s\n", sqlite3_errmsg(db));
                sqlite3_reset(stmt);
            }
            exec_sql(db, "COMMIT;");
            sqlite3_finalize(stmt);

        } else { // Simple query
            for (int j = 1; j <= iterations; j++) {
                exec_sql(db, sql);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Done in %.4f seconds.\n", time_spent);
    }

    sqlite3_close(db);
    printf("---------------------------------\n\n");
}