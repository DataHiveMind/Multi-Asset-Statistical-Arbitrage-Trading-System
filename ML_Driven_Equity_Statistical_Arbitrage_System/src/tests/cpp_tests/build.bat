@echo off
REM Build script for C++ Order Book tests on Windows
REM Requires CMake and a C++ compiler (Visual Studio or MinGW)

echo Building C++ Order Book Tests...
echo.

REM Check if CMake is available
cmake --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: CMake not found. Please install CMake and add it to PATH.
    pause
    exit /b 1
)

REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure the project
echo Configuring project...
cmake .. -DCMAKE_BUILD_TYPE=Debug

if %errorlevel% neq 0 (
    echo Error: CMake configuration failed.
    pause
    exit /b 1
)

REM Build the project
echo.
echo Building project...
cmake --build . --config Debug

if %errorlevel% neq 0 (
    echo Error: Build failed.
    pause
    exit /b 1
)

REM Run the tests
echo.
echo Running tests...
ctest --output-on-failure

if %errorlevel% neq 0 (
    echo Warning: Some tests failed.
) else (
    echo All tests passed!
)

REM Run tests with detailed output
echo.
echo Running tests with detailed output...
if exist "Debug\OrderBookTests.exe" (
    Debug\OrderBookTests.exe --gtest_output=xml:test_results.xml
) else if exist "OrderBookTests.exe" (
    OrderBookTests.exe --gtest_output=xml:test_results.xml
) else (
    echo Warning: Test executable not found.
)

echo.
echo Build and test completed!
if exist "test_results.xml" (
    echo Test results saved to test_results.xml
)

echo.
pause
