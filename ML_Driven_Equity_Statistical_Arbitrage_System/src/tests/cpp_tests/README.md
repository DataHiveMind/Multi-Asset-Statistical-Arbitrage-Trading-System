# C++ Order Book Unit Tests

This directory contains comprehensive unit tests for the C++ OrderBook implementation using Google Test framework.

## 📁 Files Structure

```
src/tests/cpp_tests/
├── test_order_book.cpp    # Main test file with all test cases
├── CMakeLists.txt         # CMake configuration for building tests
├── Makefile              # Simple Makefile for Unix systems
├── build.bat             # Windows build script
└── README.md             # This file
```

## 🧪 Test Coverage

The test suite covers all major OrderBook functionality:

### **Core Order Management**
- ✅ Order addition (buy/sell, limit/market orders)
- ✅ Order cancellation and modification
- ✅ Order quantity and price updates
- ✅ Order status tracking and validation

### **Order Matching Engine**
- ✅ Simple order matching (full fills)
- ✅ Partial order matching
- ✅ Multi-level order matching
- ✅ Price-time priority (FIFO) matching
- ✅ Market order execution
- ✅ Insufficient liquidity handling

### **Market Data & Statistics**
- ✅ Best bid/ask price and quantity queries
- ✅ Market depth calculation (multiple levels)
- ✅ Spread and mid-price calculations
- ✅ Volume-weighted average price (VWAP)
- ✅ Market impact simulation
- ✅ Order book statistics and metrics

### **Advanced Features**
- ✅ Client order tracking and queries
- ✅ Volume at price level queries
- ✅ Volume around price calculations
- ✅ Book validation and integrity checks
- ✅ Event callbacks (order, trade, depth updates)

### **Performance & Stress Testing**
- ✅ Large number of orders (1000+ orders)
- ✅ Frequent order modifications
- ✅ Concurrent order operations
- ✅ Memory usage validation
- ✅ Threading safety tests

### **Edge Cases & Error Handling**
- ✅ Zero quantity orders
- ✅ Negative price orders
- ✅ Non-existent order operations
- ✅ Empty book operations
- ✅ Book clearing and reset

### **OrderBookManager Tests**
- ✅ Multi-symbol book management
- ✅ Book creation and retrieval
- ✅ Book removal and cleanup
- ✅ Cross-book statistics

## 🛠️ Building and Running Tests

### **Option 1: Using CMake (Recommended)**

#### Windows (Visual Studio):
```cmd
cd src/tests/cpp_tests
build.bat
```

#### Windows (Manual):
```cmd
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
ctest --output-on-failure
```

#### Linux/Mac:
```bash
cd src/tests/cpp_tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
ctest --output-on-failure
```

### **Option 2: Using Makefile (Unix systems)**

```bash
cd src/tests/cpp_tests

# Install Google Test (Ubuntu/Debian)
make install-gtest

# Build and run tests
make
make test

# Run with XML output
make test-xml
```

### **Option 3: Manual Compilation**

```bash
# Compile (adjust paths as needed)
g++ -std=c++17 -pthread -I../../cpp -lgtest -lgtest_main -lgmock \
    ../../cpp/order_book/OrderBook.cpp test_order_book.cpp \
    -o OrderBookTests

# Run tests
./OrderBookTests
```

## 📊 Test Categories

Tests are organized into logical groups:

### **Basic Tests** (`OrderBookTest` fixture)
- Initial state validation
- Single order operations
- Basic market data queries

### **Concurrency Tests** (`OrderBookConcurrencyTest` fixture)
- Multi-threaded order addition
- Concurrent order operations
- Thread safety validation

### **Manager Tests** (`OrderBookManagerTest` fixture)
- Multi-symbol book management
- Symbol-based operations
- Cross-book statistics

## 🔧 Configuration Options

### **CMake Build Types**
```bash
# Debug build (default)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build (optimized)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Enable code coverage
cmake .. -DENABLE_COVERAGE=ON
make coverage
```

### **Test Filtering**
```bash
# Run specific test categories
./OrderBookTests --gtest_filter="OrderBookTest.*"
./OrderBookTests --gtest_filter="*Concurrency*"
./OrderBookTests --gtest_filter="*Performance*"

# Exclude certain tests
./OrderBookTests --gtest_filter="-*Stress*"
```

### **Output Options**
```bash
# XML output for CI/CD
./OrderBookTests --gtest_output=xml:test_results.xml

# Verbose output
./OrderBookTests --gtest_verbose

# List all tests
./OrderBookTests --gtest_list_tests
```

## 📈 Performance Testing

The test suite includes performance benchmarks:

```bash
# Build release version for performance testing
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Run only performance tests
./OrderBookTests --gtest_filter="*Performance*:*Stress*"
```

Example performance metrics:
- **Order Addition**: ~1000 orders in <100μs
- **Order Modification**: ~1000 operations in <200μs
- **Market Data Queries**: <1μs per query
- **Concurrent Operations**: 4 threads × 100 ops each

## 🧬 Memory Testing

### **With Valgrind (Linux)**
```bash
# Using CMake target
make memory_tests

# Manual Valgrind
valgrind --tool=memcheck --leak-check=full ./OrderBookTests
```

### **With AddressSanitizer**
```bash
# Compile with AddressSanitizer
cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=address -g"
make
./OrderBookTests
```

## 📋 Test Results Interpretation

### **Successful Output**
```
[==========] Running 50 tests from 4 test fixtures.
[----------] Global test environment set-up.
[----------] 25 tests from OrderBookTest
[ RUN      ] OrderBookTest.InitialStateIsEmpty
[       OK ] OrderBookTest.InitialStateIsEmpty (0 ms)
...
[----------] Global test environment tear-down.
[==========] 50 tests from 4 test fixtures ran. (125 ms total)
[  PASSED  ] 50 tests.
```

### **Performance Benchmarks**
```
Added 1000 orders in 89 microseconds
Performed 1000 operations in 156 microseconds
```

### **Coverage Report** (if enabled)
```
Lines executed: 95.2% of 2840
Creating 'coverage_report/index.html'
```

## 🐛 Troubleshooting

### **Common Build Issues**

1. **Google Test not found**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libgtest-dev libgmock-dev
   
   # CentOS/RHEL
   sudo yum install gtest-devel gmock-devel
   
   # Windows (vcpkg)
   vcpkg install gtest
   ```

2. **C++17 support required**:
   ```bash
   # Update compiler or specify explicitly
   cmake .. -DCMAKE_CXX_STANDARD=17
   ```

3. **Threading issues**:
   ```bash
   # Ensure pthread is linked
   cmake .. -DCMAKE_CXX_FLAGS="-pthread"
   ```

### **Test Failures**

1. **Timing-sensitive tests**: Some tests may occasionally fail on heavily loaded systems
2. **Concurrency tests**: May require multiple runs to catch race conditions
3. **Performance tests**: Results vary by hardware and system load

### **Memory Issues**

1. **Leaks detected**: Usually indicates missing cleanup in test teardown
2. **Race conditions**: Use ThreadSanitizer for detection:
   ```bash
   cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
   ```

## 🔄 Continuous Integration

For CI/CD pipelines, use:

```yaml
# Example GitHub Actions workflow
- name: Build and Test C++
  run: |
    cd src/tests/cpp_tests
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make
    ./OrderBookTests --gtest_output=xml:test_results.xml
    
- name: Upload Test Results
  uses: actions/upload-artifact@v2
  with:
    name: cpp-test-results
    path: src/tests/cpp_tests/build/test_results.xml
```

## 📚 Adding New Tests

To add new test cases:

1. **Create test fixture** (if needed):
   ```cpp
   class MyNewTest : public ::testing::Test {
   protected:
       void SetUp() override { /* setup code */ }
       void TearDown() override { /* cleanup code */ }
   };
   ```

2. **Add test cases**:
   ```cpp
   TEST_F(MyNewTest, TestSomething) {
       // Test implementation
       EXPECT_EQ(expected, actual);
       ASSERT_NE(nullptr, pointer);
   }
   ```

3. **Use appropriate assertions**:
   - `EXPECT_*` - continues on failure
   - `ASSERT_*` - stops on failure
   - `EXPECT_NEAR` - for floating-point comparisons
   - `EXPECT_THAT` - for complex matchers

4. **Follow naming conventions**:
   - Test names should be descriptive
   - Use CamelCase for test names
   - Group related tests in fixtures

## 📖 Additional Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Mock Documentation](https://google.github.io/googletest/gmock_cook_book.html)
- [CMake Documentation](https://cmake.org/documentation/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

The test suite ensures the OrderBook implementation is robust, performant, and suitable for production use in high-frequency trading applications.
