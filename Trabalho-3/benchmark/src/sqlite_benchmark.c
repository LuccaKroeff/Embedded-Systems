#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sqlite3.h"
#include "utils.h"
#define N 10000  // number of rows to insert


void run_query(sqlite3 *db, const char *sql) {
    char *err = NULL;
    if (sqlite3_exec(db, sql, 0, 0, &err) != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err);
        sqlite3_free(err);
        exit(1);
    }
}

int sqlite_benchmark() {
    sqlite3 *db;
    sqlite3_stmt *stmt;
    double start, end;

    sqlite3_open(":memory:", &db);

    run_query(db, "CREATE TABLE t1(a INTEGER PRIMARY KEY, b TEXT, c TEXT);");

    // Bulk insert
    run_query(db, "BEGIN TRANSACTION;");
    sqlite3_prepare_v2(db, "INSERT INTO t1(a, b, c) VALUES (?, ?, ?);", -1, &stmt, 0);

    start = get_time_sec();
    for (int i = 1; i <= N; i++) {
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_text(stmt, 2, "value_b", -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, "value_c", -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    end = get_time_sec();
    sqlite3_finalize(stmt);
    run_query(db, "COMMIT;");
    printf("Insert %d rows: %.3f sec\n", N, end - start);

    // Point SELECT
    sqlite3_prepare_v2(db, "SELECT * FROM t1 WHERE a = ?;", -1, &stmt, 0);
    start = get_time_sec();
    for (int i = 1; i <= N; i += 1000) {
        sqlite3_bind_int(stmt, 1, i);
        while (sqlite3_step(stmt) == SQLITE_ROW) {}
        sqlite3_reset(stmt);
    }
    end = get_time_sec();
    sqlite3_finalize(stmt);
    printf("Point SELECTs: %.3f sec\n", end - start);

    // Range SELECT
    sqlite3_prepare_v2(db, "SELECT * FROM t1 WHERE a BETWEEN ? AND ?;", -1, &stmt, 0);
    start = get_time_sec();
    for (int i = 1; i <= N; i += 2000) {
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, i + 500);
        while (sqlite3_step(stmt) == SQLITE_ROW) {}
        sqlite3_reset(stmt);
    }
    end = get_time_sec();
    sqlite3_finalize(stmt);
    printf("Range SELECTs: %.3f sec\n", end - start);

    // COUNT(*)
    start = get_time_sec();
    run_query(db, "SELECT COUNT(*) FROM t1;");
    end = get_time_sec();
    printf("COUNT(*): %.3f sec\n", end - start);

    // UPDATE
    run_query(db, "BEGIN TRANSACTION;");
    sqlite3_prepare_v2(db, "UPDATE t1 SET b = 'updated' WHERE a = ?;", -1, &stmt, 0);
    start = get_time_sec();
    for (int i = 1; i <= N; i += 1000) {
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    end = get_time_sec();
    sqlite3_finalize(stmt);
    run_query(db, "COMMIT;");
    printf("UPDATEs: %.3f sec\n", end - start);

    // DELETE
    run_query(db, "BEGIN TRANSACTION;");
    sqlite3_prepare_v2(db, "DELETE FROM t1 WHERE a = ?;", -1, &stmt, 0);
    start = get_time_sec();
    for (int i = 1; i <= N; i += 2000) {
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    end = get_time_sec();
    sqlite3_finalize(stmt);
    run_query(db, "COMMIT;");
    printf("DELETEs: %.3f sec\n", end - start);

    sqlite3_close(db);
    return 0;
}
