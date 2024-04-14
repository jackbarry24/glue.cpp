# Makefile for building glue.cpp with libbert.so

# Compiler settings
CXX = g++
CXXFLAGS = -Wall -std=c++17

# Directories
LIB_DIR = ./bert.cpp/build
INCLUDE_DIR = ./bert.cpp

# Target executable name
TARGET = gluex

# Source files
SOURCES = glue.cpp

# Include and library flags
LDFLAGS = -L$(LIB_DIR) -Wl,-rpath=$(LIB_DIR) -lbert
CPPFLAGS = -I$(INCLUDE_DIR)

# Build target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)

.PHONY: clean