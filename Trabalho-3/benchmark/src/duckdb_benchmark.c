#include <stdio.h>
#include <stdlib.h>
#include "duckdb.h"
#include <time.h>
#include "utils.h"

#define N 10000


void exec_or_die(duckdb_connection conn, const char *sql) {
    duckdb_result res;
    if (duckdb_query(conn, sql, &res) != DuckDBSuccess) {
        fprintf(stderr, "DuckDB error: %s\n", duckdb_result_error(&res));
        duckdb_destroy_result(&res);
        exit(1);
    }
    duckdb_destroy_result(&res);
}

int duckdb_benchmark() {
    duckdb_database db;
    duckdb_connection conn;

    if (duckdb_open(":memory:", &db) != DuckDBSuccess || duckdb_connect(db, &conn) != DuckDBSuccess) {
        fprintf(stderr, "Failed to open DuckDB.\n");
        return 1;
    }

    exec_or_die(conn, "CREATE TABLE t1(a INTEGER PRIMARY KEY, b TEXT, c TEXT);");

    // Bulk inserts
    double start = get_time_sec();
    exec_or_die(conn, "BEGIN TRANSACTION;");
    for (int i = 1; i <= N; i++) {
        char sql[256];
        snprintf(sql, sizeof(sql),
                 "INSERT INTO t1(a, b, c) VALUES (%d, 'value_b', 'value_c');", i);
        duckdb_query(conn, sql, NULL);
    }
    exec_or_die(conn, "COMMIT;");
    double end = get_time_sec();
    printf("Insert %d rows: %.3f sec\n", N, end - start);

    // Point SELECTs
    start = get_time_sec();
    for (int i = 1; i <= N; i += 1000) {
        char sql[128];
        snprintf(sql, sizeof(sql), "SELECT * FROM t1 WHERE a = %d;", i);
        exec_or_die(conn, sql);
    }
    end = get_time_sec();
    printf("Point SELECTs: %.3f sec\n", end - start);

    // Range SELECTs
    start = get_time_sec();
    for (iThe sorting columns are serialized into a fixed-size byte representation that is naturally sortable (i.e. memcmp will give the correct order). For variable length sorting columns (e.g. strings) we serialize a fixed-size prefix. An index is appended to this sorting column.
    nt i = 1; i <= N; i += 2000) {
        char sql[128];
        snprintf(sql, sizeof(sql), "SELECT * FROM t1 WHERE a BETWEEN %d AND %d;", i, i + 500);
        exec_or_die(conn, sql);
    }
    end = get_time_sec();
    printf("Range SELECTs: %.3f sec\n", end - start);

    // Aggregation
    start = get_time_sec();
    exec_or_die(conn, "SELECT COUNT(*) FROM t1;");
    end = get_time_sec();
    printf("COUNT(*): %.3f sec\n", end - start);

    // Updates
    start = get_time_sec();
    exec_or_die(conn, "BEGIN TRANSACTION;");
    for (int i = 1; i <= N; i += 1000) {
        char sql[128];
        snprintf(sql, sizeof(sql),
                 "UPDATE t1 SET b = 'updated' WHERE a = %d;", i);
        exec_or_die(conn, sql);
    }
    exec_or_die(conn, "COMMIT;");
    end = get_time_sec();
    printf("UPDATEs: %.3f sec\n", end - start);

    // Deletes
    start = get_time_sec();
    exec_or_die(conn, "BEGIN TRANSACTION;");
    for (int i = 1; i <= N; i += 2000) {
        char sql[128];
        snprintf(sql, sizeof(sql),
                 "DELETE FROM t1 WHERE a = %d;", i);
        exec_or_die(conn, sql);
    }
    exec_or_die(conn, "COMMIT;");
    end = get_time_sec();
    printf("DELETEs: %.3f sec\n", end - start);

    duckdb_disconnect(&conn);
    duckdb_close(&db);
    return 0;
}