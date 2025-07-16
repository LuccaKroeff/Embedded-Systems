#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "libs/sqlite3.h"
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
    sqlite3 *db;
    char *errMsg = NULL;
    int rc;

    // Open an in-memory SQLite database
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

        sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, &errMsg);
        
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

            rc = sqlite3_exec(db, formatted_query, NULL, NULL, &errMsg);
            if (rc != SQLITE_OK) {
                fprintf(stderr, "E: %s\n", errMsg);
                sqlite3_free(errMsg);
                errMsg = NULL; // Reset errMsg to prevent issues
            }
        }
        
        sqlite3_exec(db, "COMMIT;", NULL, NULL, &errMsg);
        double joule = end_rapl_sysfs();
        clock_gettime(CLOCK_MONOTONIC, &end);

        // 5. Calculate and print the total time spent
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Result: %.4f,%.4f,%.4f\n", time_spent, joule, time_spent*joule);
        fflush(stdout);
    }

    // Clean up
    printf("\nExiting.\n");
    sqlite3_close(db);
    return 0;
}