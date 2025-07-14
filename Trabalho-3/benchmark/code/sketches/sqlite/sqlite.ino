#include <sqlite3.h>
#include <time.h>

sqlite3 *db;
char *errMsg = NULL;
int success = -1;

void setup() {
  delay(1000);
  Serial.begin(115200);
  delay(5000);
  while(!Serial);
  success = sqlite3_open(":memory:", &db);
}

void loop() {
  Serial.println("Q:");
  String query = "";

  while (query.length() == 0) {
    query = Serial.readString();
  }
  //Serial.readString(); // Consume trailing newline
  Serial.println("I:");
  String iters = "";
  while (iters.length() == 0) {
    iters = Serial.readString();
  }
  int iterations = atoi(iters.c_str());

  //Serial.readString(); // Consume trailing newline
  Serial.println("S:");
  String slots = "";
  while (slots.length() == 0) {
    slots = Serial.readString();
  }
  int s = atoi(slots.c_str());
  //Serial.readString(); // Consume trailing newline
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);
  sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
  for (int i = 0; i < iterations; i++) {
    // --- NEW LOGIC START ---
    // Create a new string for this iteration and replace placeholders
    String formatted_query = query;
    formatted_query.replace("%d", String(i));
    // --- NEW LOGIC END ---

    char *errMsg = NULL;
    if (sqlite3_exec(db, formatted_query.c_str(), NULL, NULL, &errMsg) != SQLITE_OK) {
      Serial.printf("E: %s\n", errMsg);
      sqlite3_free(errMsg);
    }
  }
  sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  Serial.printf("T: %.4f\n", time_spent);
  Serial.printf("En: %.4f\n", time_spent * 0.5);
}