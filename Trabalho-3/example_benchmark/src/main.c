#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jansson.h>
#include "benchmark_utils.h"
#include "sqlite_benchmark.h"
#include "duckdb_benchmark.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path_to_queries.json>\n", argv[0]);
        return 1;
    }

    // 1. Parse JSON file (Jansson-dependent code)
    json_error_t error;
    json_t *root = json_load_file(argv[1], 0, &error);
    if (!root) {
        fprintf(stderr, "JSON Error: on line %d: %s\n", error.line, error.text);
        return 1;
    }

    size_t query_count = json_array_size(root);
    BenchmarkQuery *queries = malloc(query_count * sizeof(BenchmarkQuery));
    if (!queries) {
        fprintf(stderr, "Failed to allocate memory for queries.\n");
        json_decref(root);
        return 1;
    }

    // 2. Translate JSON to internal representation
    for (size_t i = 0; i < query_count; i++) {
        json_t *obj = json_array_get(root, i);
        const char *name = json_string_value(json_object_get(obj, "name"));
        const char *sql = json_string_value(json_object_get(obj, "query"));
        json_t *iter_json = json_object_get(obj, "iterations");
        json_t *slots = json_object_get(obj, "slots");

        queries[i].name = strdup(name);
        queries[i].query = strdup(sql);
        queries[i].iterations = iter_json ? json_integer_value(iter_json) : 1;
        queries[i].slots = slots ? json_integer_value(slots) : 0;
        
        json_t *templates = json_object_get(obj, "templates");
        for (int slot = 0; slot < queries[i].slots; slot++) {
            json_t *template_obj = json_array_get(templates, slot);
            char* template_pattern = json_string_value(json_object_get(template_obj, "template"));
            char* template_type = json_string_value(json_object_get(template_obj, "type"));
            queries[i].templates[slot].template = strdup(template_pattern);
            queries[i].templates[slot].type = strdup(template_type);
        }
    }
    json_decref(root); // Jansson objects are no longer needed

    // 3. Run benchmarks with the internal representation
    printf("Starting embedded database benchmarks...\n\n");
    run_sqlite_benchmarks(queries, query_count);
    //run_duckdb_benchmarks(queries, query_count);
    printf("Benchmarking complete.\n");

    // 4. Clean up allocated memory
    for (size_t i = 0; i < query_count; i++) {
        free(queries[i].name);
        free(queries[i].query);
    }
    free(queries);

    return 0;
}