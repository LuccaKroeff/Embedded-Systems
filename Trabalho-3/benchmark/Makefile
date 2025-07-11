# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -Llib -Wl,-rpath=lib

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
LIB_DIR := lib

# Auto-detect source files and object files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Auto-detect libraries in lib/
LIBFILES := $(wildcard $(LIB_DIR)/lib*.so $(LIB_DIR)/lib*.a)
LIBNAMES := $(notdir $(LIBFILES))
LIBS := $(foreach lib, $(LIBNAMES), -l$(patsubst lib%.so,%, $(patsubst lib%.a,%, $(lib))))

# Final executable
TARGET := $(BIN_DIR)/app

# Default target
.PHONY: all
all: $(TARGET)

# Link final binary with auto-detected libs
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

# Compile .c into .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
