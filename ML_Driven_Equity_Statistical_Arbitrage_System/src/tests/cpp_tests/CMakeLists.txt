# CMakeLists.txt for C++ Order Book Tests
cmake_minimum_required(VERSION 3.16)
project(OrderBookTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release for performance, Debug for testing
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /WX)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Werror -pedantic)
endif()

# Find required packages
find_package(Threads REQUIRED)

# Google Test setup
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9fd7188aa8447e3e3e1e5e0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Order Book source files
set(ORDER_BOOK_SOURCES
    ../../cpp/order_book/OrderBook.cpp
    ../../cpp/order_book/OrderBook.h
)

# Create OrderBook library
add_library(OrderBookLib ${ORDER_BOOK_SOURCES})
target_link_libraries(OrderBookLib Threads::Threads)

# Set up include directories for the library
target_include_directories(OrderBookLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp
)

# Test executable
add_executable(OrderBookTests
    test_order_book.cpp
)

# Link libraries
target_link_libraries(OrderBookTests
    OrderBookLib
    gtest_main
    gmock_main
    Threads::Threads
)

# Include Google Test directories
target_include_directories(OrderBookTests PRIVATE
    ${gtest_SOURCE_DIR}/include
    ${gmock_SOURCE_DIR}/include
)

# Enable testing
enable_testing()

# Add test to CTest
add_test(NAME OrderBookUnitTests COMMAND OrderBookTests)

# Custom target for running tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS OrderBookTests
)

# Custom target for running tests with detailed output
add_custom_target(run_tests_detailed
    COMMAND $<TARGET_FILE:OrderBookTests> --gtest_output=xml:test_results.xml
    DEPENDS OrderBookTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Performance testing (release build)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_target(performance_tests
        COMMAND $<TARGET_FILE:OrderBookTests> --gtest_filter="*Performance*:*Stress*" --gtest_output=xml:performance_results.xml
        DEPENDS OrderBookTests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# Memory testing with Valgrind (Linux only)
if(UNIX AND NOT APPLE)
    find_program(VALGRIND_PROGRAM NAMES valgrind)
    if(VALGRIND_PROGRAM)
        add_custom_target(memory_tests
            COMMAND ${VALGRIND_PROGRAM} --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose $<TARGET_FILE:OrderBookTests>
            DEPENDS OrderBookTests
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

# Code coverage (GCC/Clang only)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
    if(ENABLE_COVERAGE)
        target_compile_options(OrderBookLib PRIVATE --coverage)
        target_link_options(OrderBookLib PRIVATE --coverage)
        target_compile_options(OrderBookTests PRIVATE --coverage)
        target_link_options(OrderBookTests PRIVATE --coverage)
        
        add_custom_target(coverage
            COMMAND lcov --directory . --capture --output-file coverage.info
            COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND lcov --list coverage.info
            COMMAND genhtml -o coverage_report coverage.info
            DEPENDS run_tests
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

# Install targets
install(TARGETS OrderBookTests
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(VALGRIND_PROGRAM)
    message(STATUS "  Valgrind: Found")
endif()
message(STATUS "")
