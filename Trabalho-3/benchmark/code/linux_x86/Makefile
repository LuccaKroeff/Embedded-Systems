sqlite_bench: sqlite.c
	gcc -O2 -Ilibs sqlite.c libs/energia.c -o sqlite_bench -Llibs -Wl,-rpath=code/linux_x86/libs -lsqlite3 -Wno-unused-result

duckdb_bench: duckdb.c
	gcc -O2 -Ilibs duckdb.c libs/energia.c -o duckdb_bench -Llibs -Wl,-rpath=code/linux_x86/libs -lduckdb -Wno-unused-result

rocksdb_bench: rocksdb.cpp
	g++ -std=c++17 -o rocksdb_bench rocksdb.cpp libs/energia.c -lrocksdb -lz -lbz2 -lsnappy -llz4 -lzstd -pthread -Wno-unused-result -Wno-format-overflow

clean:
	rm -f sqlite_bench duckdb_bench rocksdb_bench
