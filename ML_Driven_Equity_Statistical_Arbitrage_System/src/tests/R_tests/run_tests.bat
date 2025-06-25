@echo off
REM R Test Runner for Windows
REM Executes R statistical model tests with proper environment setup

echo R Statistical Models Test Runner
echo ==================================
echo.

REM Check if R is available
where R >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: R not found in PATH
    echo Please install R and add it to your system PATH
    echo Download from: https://cran.r-project.org/
    pause
    exit /b 1
)

REM Check if Rscript is available  
where Rscript >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Rscript not found in PATH
    echo R installation may be incomplete
    pause
    exit /b 1
)

echo R and Rscript found successfully
echo.

REM Set working directory
cd /d "%~dp0"

REM Install dependencies first
echo Installing R package dependencies...
Rscript run_r_tests.R --install-deps
if %errorlevel% neq 0 (
    echo Warning: Some dependencies may not have installed correctly
    echo Continuing with test execution...
)
echo.

REM Run the tests
echo Running statistical model tests...
Rscript run_r_tests.R --verbose --html

REM Check exit status
if %errorlevel% eq 0 (
    echo.
    echo ==============================
    echo All tests completed successfully!
    echo ==============================
    
    REM Check if HTML report was generated
    if exist "test_results.html" (
        echo.
        echo HTML test report generated: test_results.html
        set /p openfile="Open test report in browser? (y/n): "
        if /i "%openfile%"=="y" start test_results.html
    )
) else (
    echo.
    echo ==============================
    echo Some tests failed or errors occurred
    echo Exit code: %errorlevel%
    echo ==============================
)

echo.
echo Test run completed.
pause
