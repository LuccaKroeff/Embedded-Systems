#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <duckdb.h>
#include <time.h>
#include "energia.h"

/**
 * @brief Removes the trailing newline character from a string read by fgets.
 * @param str The string to modify.
 */
void remove_newline(char *str) {
    str[strcspn(str, "\r\n")] = 0;
}

int main() {
    rapl_init();

    duckdb_database db = NULL;
    duckdb_connection con = NULL;

    // Open an in-memory DuckDB database and create a connection.
    // Passing NULL as the path to duckdb_open creates an in-memory database.
    if (duckdb_open(NULL, &db) == DuckDBError) {
        fprintf(stderr, "Failed to open DuckDB database.\n");
        return 1;
    }
    if (duckdb_connect(db, &con) == DuckDBError) {
        fprintf(stderr, "Failed to connect to DuckDB.\n");
        duckdb_close(&db);
        return 1;
    }

    // Infinite loop to process user commands
    while (1) {
        char query[1024];
        char iters_str[32];
        int iterations;
        char slots_str[32];
        int slots; // Unused in this implementation, but read as per requirement

        // 1. Get the SQL query from the user
        printf("Q:\n");
        fflush(stdout); // Ensure the prompt is displayed before waiting for input
        if (fgets(query, sizeof(query), stdin) == NULL) {
            break; // Exit loop on EOF (Ctrl+D)
        }
        remove_newline(query);
        if (strlen(query) == 0) continue; // Skip empty input

        // 2. Get the number of iterations
        printf("I:\n");
        fflush(stdout);
        if (fgets(iters_str, sizeof(iters_str), stdin) == NULL) {
            break; // Exit loop on EOF
        }
        iterations = atoi(iters_str);
        if (iterations <= 0) {
            fprintf(stderr, "Error: Please enter a positive number for iterations.\n");
            continue;
        }
        
        // 3. Get the number of slots
        printf("S:\n");
        fflush(stdout);
        if (fgets(slots_str, sizeof(slots_str), stdin) == NULL) {
            break; // Exit loop on EOF
        }
        slots = atoi(slots_str);
        if (slots < 0) { // slots can be 0 if there are no placeholders
            fprintf(stderr, "Error: Please enter a non-negative number for slots.\n");
            continue;
        }


        // 4. Execute the query and measure execution time
        struct timespec start, end;
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_rapl_sysfs();

        // Use duckdb_query to execute statements.
        // For simple commands, the result object can be NULL.
        duckdb_query(con, "BEGIN TRANSACTION;", NULL);
        
        // This buffer will hold the query with placeholders replaced by the iteration number
        char formatted_query[4096];

        for (int i = 0; i < iterations; i++) {
            // --- NEW LOGIC START ---
            // Build the formatted query for the current iteration
            char *fq_ptr = formatted_query;     // Pointer to the current position in the output buffer
            const char *q_ptr = query;          // Pointer to the current position in the input query

            while (*q_ptr != '\0' && (fq_ptr - formatted_query) < sizeof(formatted_query) - 1) {
                // Check for the '%d' placeholder
                if (*q_ptr == '%' && *(q_ptr + 1) == 'd') {
                    // Append the current iteration number 'i' to the formatted query
                    int written = snprintf(fq_ptr, sizeof(formatted_query) - (fq_ptr - formatted_query), "%d", i);
                    if (written > 0) {
                         fq_ptr += written;
                    }
                    q_ptr += 2; // Move the input pointer past '%d'
                } else {
                    // Copy the character from the original query
                    *fq_ptr++ = *q_ptr++;
                }
            }
            *fq_ptr = '\0'; // Null-terminate the new string
            // --- NEW LOGIC END ---

            duckdb_result result;
            if (duckdb_query(con, formatted_query, &result) == DuckDBError) {
                // If the query fails, the error is in the result object.
                fprintf(stderr, "E: %s\n", duckdb_result_error(&result));
            }
            // IMPORTANT: Always destroy the result object to avoid memory leaks.
            duckdb_destroy_result(&result);
        }
        
        duckdb_query(con, "COMMIT;", NULL);
        double joule = end_rapl_sysfs();

        clock_gettime(CLOCK_MONOTONIC, &end);

        // 5. Calculate and print the total time spent
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Result: %.4f,%.4f,%.4f\n", time_spent, joule, time_spent*joule);
        fflush(stdout);
    }

    // Clean up DuckDB resources
    printf("\nExiting.\n");
    duckdb_disconnect(&con);
    duckdb_close(&db);
    return 0;
}