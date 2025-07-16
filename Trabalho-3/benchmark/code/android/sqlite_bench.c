#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sqlite3.h" // We still include the header, but will link to the system library

/**
 * @brief Removes the trailing newline character from a string read by fgets.
 * @param str The string to modify.
 */
void remove_newline(char *str) {
    if (str == NULL) return;
    str[strcspn(str, "\r\n")] = 0;
}

int main() {
    sqlite3 *db;
    char *errMsg = NULL;
    int rc;

    // Open an in-memory SQLite database
    // This function is part of the system's SQLite library we will link against.
    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    // Infinite loop to process user commands
    while (1) {
        char query[1024];
        char iters_str[32];
        int iterations;
        char slots_str[32];
        int slots; // Unused, but kept for compatibility with input format

        // 1. Get the SQL query from the user
        printf("Q:\n");
        fflush(stdout);
        if (fgets(query, sizeof(query), stdin) == NULL) {
            break; // Exit loop on EOF (Ctrl+D)
        }
        remove_newline(query);
        if (strlen(query) == 0) continue;

        // 2. Get the number of iterations
        printf("I:\n");
        fflush(stdout);
        if (fgets(iters_str, sizeof(iters_str), stdin) == NULL) {
            break;
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
            break;
        }
        slots = atoi(slots_str);
        if (slots < 0) {
            fprintf(stderr, "Error: Please enter a non-negative number for slots.\n");
            continue;
        }

        // 4. Execute the query and measure execution time
        struct timespec start, end;
        
        // Time measurement using clock_gettime is compatible with Android
        clock_gettime(CLOCK_MONOTONIC, &start);

        sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, &errMsg);
        
        char formatted_query[4096];

        for (int i = 0; i < iterations; i++) {
            char *fq_ptr = formatted_query;
            const char *q_ptr = query;

            while (*q_ptr != '\0' && (fq_ptr - formatted_query) < sizeof(formatted_query) - 1) {
                if (*q_ptr == '%' && *(q_ptr + 1) == 'd') {
                    int written = snprintf(fq_ptr, sizeof(formatted_query) - (fq_ptr - formatted_query), "%d", i);
                    if (written > 0) {
                         fq_ptr += written;
                    }
                    q_ptr += 2;
                } else {
                    *fq_ptr++ = *q_ptr++;
                }
            }
            *fq_ptr = '\0';

            rc = sqlite3_exec(db, formatted_query, NULL, NULL, &errMsg);
            if (rc != SQLITE_OK) {
                fprintf(stderr, "E: %s\n", errMsg);
                sqlite3_free(errMsg);
                errMsg = NULL;
            }
        }
        
        sqlite3_exec(db, "COMMIT;", NULL, NULL, &errMsg);
        
        clock_gettime(CLOCK_MONOTONIC, &end);

        // 5. Calculate and print the total time spent
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        double energy = time_spent * 6; // Estimated
        
        // We only print time now. Energy will be measured externally.
        printf("Result: %.4f,%.4f,%.4f\n", time_spent, energy, time_spent * energy);
        fflush(stdout);
    }

    // Clean up
    printf("\nExiting.\n");
    sqlite3_close(db);
    return 0;
}