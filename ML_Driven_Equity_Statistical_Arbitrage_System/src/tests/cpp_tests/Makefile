# Makefile for C++ Order Book Tests
# Simple Makefile for systems without CMake

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -pthread
INCLUDES = -I../../cpp -I/usr/local/include
LIBS = -pthread

# Source files
ORDER_BOOK_SOURCES = ../../cpp/order_book/OrderBook.cpp
TEST_SOURCES = test_order_book.cpp
OBJECTS = $(ORDER_BOOK_SOURCES:.cpp=.o) $(TEST_SOURCES:.cpp=.o)

# Google Test (assumes installed system-wide)
GTEST_LIBS = -lgtest -lgtest_main -lgmock -lgmock_main

# Target executable
TARGET = OrderBookTests

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(GTEST_LIBS)

# Object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run tests
test: $(TARGET)
	./$(TARGET)

# Run tests with XML output
test-xml: $(TARGET)
	./$(TARGET) --gtest_output=xml:test_results.xml

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) test_results.xml

# Install Google Test (Ubuntu/Debian)
install-gtest:
	sudo apt-get update
	sudo apt-get install libgtest-dev libgmock-dev
	sudo apt-get install cmake
	cd /usr/src/gtest && sudo cmake . && sudo make && sudo cp *.a /usr/lib

# Install Google Test (CentOS/RHEL)
install-gtest-rhel:
	sudo yum install epel-release
	sudo yum install gtest-devel gmock-devel

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Release build
release: CXXFLAGS += -O3 -DNDEBUG
release: $(TARGET)

# Memory check with Valgrind
memcheck: $(TARGET)
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TARGET)

# Performance profiling
profile: CXXFLAGS += -pg
profile: $(TARGET)

# Code coverage
coverage: CXXFLAGS += --coverage
coverage: LIBS += --coverage
coverage: $(TARGET)
	./$(TARGET)
	gcov $(ORDER_BOOK_SOURCES) $(TEST_SOURCES)
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory coverage_report

# Static analysis with cppcheck
static-analysis:
	cppcheck --enable=all --std=c++17 --suppress=missingIncludeSystem $(ORDER_BOOK_SOURCES) $(TEST_SOURCES)

# Format code with clang-format
format:
	clang-format -i $(ORDER_BOOK_SOURCES) $(TEST_SOURCES)

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the test executable"
	@echo "  test         - Run tests"
	@echo "  test-xml     - Run tests with XML output"
	@echo "  clean        - Remove build artifacts"
	@echo "  debug        - Build with debug symbols"
	@echo "  release      - Build optimized release version"
	@echo "  memcheck     - Run with Valgrind memory checker"
	@echo "  coverage     - Generate code coverage report"
	@echo "  static-analysis - Run static code analysis"
	@echo "  format       - Format source code"
	@echo "  install-gtest - Install Google Test (Ubuntu/Debian)"
	@echo "  install-gtest-rhel - Install Google Test (CentOS/RHEL)"

# Phony targets
.PHONY: all test test-xml clean debug release memcheck profile coverage static-analysis format help install-gtest install-gtest-rhel
