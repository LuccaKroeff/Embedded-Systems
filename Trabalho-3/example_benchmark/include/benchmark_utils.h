#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H
#define MAX_SLOTS 10

typedef struct {
    char *type;
    char *template;
} QueryTemplate;

typedef struct {
    char *name;                    // Name of the benchmark
    char *query;                   // SQL query string
    int slots;                     // Number of slots to fill
    QueryTemplate templates[MAX_SLOTS];    // Templates for each slot
    int iterations;                // Number of times to run (defaults to 1)
} BenchmarkQuery;

#endif // BENCHMARK_UTILS_H