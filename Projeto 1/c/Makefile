# Créditos ao usuário aleatório do StackOverflow :D

# Makefile - compile all .c files and link into one executable (default: main)
# Usage examples:
#   make                # build (default target)
#   make debug          # build with -g (no optimizations)
#   make release        # build with -O3 and -march=native
#   make clean          # remove objects and executable
#   make run            # build then run ./main
#   make TARGET=other   # override executable name

# Toolchain / flags (override on the make command line if you want)
CC      ?= gcc
STD     ?= -std=c11
CFLAGS  ?= -O2 -fopenmp -Wall -Wextra $(STD)
LDFLAGS ?= -fopenmp
LDLIBS  ?=

# Default executable name (can be overridden: make TARGET=myprog)
TARGET  ?= main

# Build directory
BUILD_DIR := build

# Collect all .c files in the directory
SOURCES := $(wildcard *.c)
OBJECTS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCES))

.PHONY: all debug release clean run $(BUILD_DIR)

all: $(BUILD_DIR) $(TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link rule: link all object files into $(TARGET)
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Generic compile rule: .c -> build/.o
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build: no optimization, include debug symbols
debug: CFLAGS := -g -O0 -Wall -Wextra $(STD)
debug: clean $(TARGET)

# Release build: optimized (tweak as you like)
release: CFLAGS := -O3 -march=native -funroll-loops -Wall $(STD)
release: clean $(TARGET)

# Remove build products
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Build then run the default target
run: all
	./$(TARGET)
