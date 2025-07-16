# Makefile for cross-compiling for Android

# --- Configuration ---
# Target architecture. Common values: arm64-v8a, armeabi-v7a, x86_64, x86
TARGET_ARCH := aarch64

# Android API level. 21 is a good minimum for 64-bit.
API_LEVEL := 21

# Output binary name
TARGET := sqlite_bench

# --- Toolchain Definition ---
# Check if NDK_HOME is set
ifndef NDK_HOME
    $(error NDK_HOME is not set. Please set it to your Android NDK root directory)
endif

# Define the toolchain path based on the host system (assuming linux-x86_64)
TOOLCHAIN := $(NDK_HOME)/toolchains/llvm/prebuilt/linux-x86_64

# Define the compiler
# The format is {target-triple}{api-level}-clang
CC := $(TOOLCHAIN)/bin/$(TARGET_ARCH)-linux-android$(API_LEVEL)-clang

# --- Build Flags ---
# CFLAGS: Compiler flags
# -I. tells the compiler to look for headers (sqlite3.h) in the current directory
CFLAGS := -I.

# LDFLAGS: Linker flags
# -lsqlite3: Link against the system's native SQLite library
# -pie: Required for executables on modern Android versions
LDFLAGS := -pie

# --- Build Rules ---
# Default rule, executed when you just run `make`
all: $(TARGET)

# Rule to build the target executable
$(TARGET): sqlite_bench.c
	@echo "--- Compiling for $(TARGET_ARCH) ---"
	$(CC) $(CFLAGS) -o $@ $< sqlite3.c $(LDFLAGS)
	@echo "--- Build successful: $(TARGET) ---"

# Rule to clean up build artifacts
clean:
	rm -f $(TARGET)

# Phony targets are not files
.PHONY: all clean
