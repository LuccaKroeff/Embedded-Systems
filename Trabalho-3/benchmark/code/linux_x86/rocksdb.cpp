#include <cstdio> // Replaced iostream with cstdio
#include <string>
// <sstream> is no longer needed
#include <vector>
#include <ctime>
#include <cstring>
#include "libs/energia.h"

#include "rocksdb/db.h"

/**
 * @brief Removes the trailing newline character from a string read by fgets.
 * @param str The string to modify.
 */
void remove_newline(char *str) {
    str[strcspn(str, "\r\n")] = 0;
}

/**
 * @brief Splits a string into tokens based on whitespace.
 * @param input The string to split.
 * @return A vector of string tokens.
 */
std::vector<std::string> split(const std::string &input) {
    std::vector<std::string> tokens;
    // strtok modifies the string, so we create a mutable copy.
    char* str_copy = new char[input.length() + 1];
    strcpy(str_copy, input.c_str());

    char* token = strtok(str_copy, " \t\n");
    while (token != NULL) {
        tokens.push_back(std::string(token));
        token = strtok(NULL, " \t\n");
    }

    delete[] str_copy;
    return tokens;
}

// The format_string function is no longer needed, as we'll use snprintf directly.

int main() {
    rapl_init();
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;
    
    // Using a temporary directory for the database
    const char* db_path = "/tmp/rocksdb_test";

    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
    if (!status.ok()) {
        fprintf(stderr, "Failed to open RocksDB: %s\n", status.ToString().c_str());
        return 1;
    }

    while (true) {
        char command_line[1024];
        char iters_str[32];
        int iterations = 1;
        char slots_str[32];
        int slots; // Unused, but read for consistency with other tools

        printf("Q:\n");
        fflush(stdout);
        if (!fgets(command_line, sizeof(command_line), stdin)) break;
        remove_newline(command_line);

        if (strlen(command_line) == 0) continue;

        std::vector<std::string> parts = split(command_line);
        if (parts.empty()) continue;

        printf("I:\n");
        fflush(stdout);
        if (!fgets(iters_str, sizeof(iters_str), stdin)) break;
        iterations = atoi(iters_str);
        if (iterations <= 0) {
            fprintf(stderr, "Error: Please enter a positive number for iterations.\n");
            continue;
        }

        printf("S:\n");
        fflush(stdout);
        if (!fgets(slots_str, sizeof(slots_str), stdin)) break;
        slots = atoi(slots_str);
        if (slots < 0) {
            fprintf(stderr, "Error: Please enter a non-negative number for slots.\n");
            continue;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        start_rapl_sysfs();

        for (int i = 0; i < iterations; i++) {
            if (parts[0] == "put" && parts.size() >= 3) {
                // --- NEW: Format key and value with iteration number using snprintf ---
                char key_buf[1024];
                char val_buf[1024];
                snprintf(key_buf, sizeof(key_buf), parts[1].c_str(), i);
                snprintf(val_buf, sizeof(val_buf), parts[2].c_str(), i);
                
                status = db->Put(rocksdb::WriteOptions(), key_buf, val_buf);
                if (!status.ok()) {
                    fprintf(stderr, "E: %s\n", status.ToString().c_str());
                    break;
                }
            } else if (parts[0] == "get" && parts.size() >= 2) {
                // --- NEW: Format key with iteration number using snprintf ---
                char key_buf[1024];
                snprintf(key_buf, sizeof(key_buf), parts[1].c_str(), i);
                
                std::string value;
                status = db->Get(rocksdb::ReadOptions(), key_buf, &value);
                if (status.ok()) {
                    if (i == iterations - 1) // Only print on last iteration
                        printf("V: %s\n", value.c_str());
                } else if (!status.IsNotFound()) { // Don't print an error if key is just not found
                    fprintf(stderr, "E: %s\n", status.ToString().c_str());
                    break;
                }
            } else if (parts[0] == "del" && parts.size() >= 2) {
                // --- NEW: Format key with iteration number using snprintf ---
                char key_buf[1024];
                snprintf(key_buf, sizeof(key_buf), parts[1].c_str(), i);

                status = db->Delete(rocksdb::WriteOptions(), key_buf);
                if (!status.ok()) {
                    fprintf(stderr, "E: %s\n", status.ToString().c_str());
                    break;
                }
            } else {
                fprintf(stderr, "E: Unknown command. Use put/get/del.\n");
                break;
            }
        }
        double joule = end_rapl_sysfs();
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("T: %.4f\n", time_spent);
        printf("En: %.4f\n", joule);
        fflush(stdout);
    }

    printf("\nExiting.\n");
    delete db;
    // Clean up the database files
    rocksdb::DestroyDB(db_path, rocksdb::Options());
    return 0;
}