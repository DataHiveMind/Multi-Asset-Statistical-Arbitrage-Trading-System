# R Statistical Models Unit Tests

This directory contains comprehensive unit tests for the R statistical models used in the statistical arbitrage system, implemented using the `testthat` framework.

## 📁 Files Structure

```
src/tests/R_tests/
├── test_stat_models.R    # Main test file with all test cases
├── run_r_tests.R         # R script for executing tests
├── run_tests.bat         # Windows batch file for easy execution
└── README.md             # This documentation file
```

## 🧪 Test Coverage

The test suite provides comprehensive coverage of all statistical models:

### **Data Loading and Preprocessing (4 tests)**
- ✅ Package loading functionality
- ✅ Data preprocessing with different return types (log, simple, both)
- ✅ Outlier detection and cleaning
- ✅ Data validation and error handling

### **Stationarity and Unit Root Tests (3 tests)**
- ✅ ADF (Augmented Dickey-Fuller) test
- ✅ PP (Phillips-Perron) test  
- ✅ KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
- ✅ Correct identification of stationary vs non-stationary series
- ✅ Individual test type execution

### **Cointegration Analysis (2 tests)**
- ✅ Engle-Granger cointegration test
- ✅ Detection of cointegrated relationships
- ✅ Handling of non-cointegrated series
- ✅ Parameter estimation (beta coefficients)

### **Time Series Modeling (4 tests)**

#### **ARIMA Models:**
- ✅ Automatic model selection with `auto.arima`
- ✅ Manual model selection with grid search
- ✅ Model diagnostics (Ljung-Box, Jarque-Bera, ARCH tests)
- ✅ Forecasting capabilities

#### **GARCH Models:**
- ✅ Different GARCH specifications (sGARCH, eGARCH, gjrGARCH)
- ✅ Multiple error distributions (normal, Student-t, skewed Student-t)
- ✅ Volatility estimation and standardized residuals
- ✅ Model diagnostics for volatility clustering

#### **VAR Models:**
- ✅ Multiple time series modeling
- ✅ Optimal lag selection (AIC, BIC, HQ)
- ✅ Granger causality testing
- ✅ Impulse response functions
- ✅ Forecast error variance decomposition

### **Regression Analysis (2 tests)**
- ✅ Robust regression with heteroskedasticity-consistent standard errors
- ✅ Different robust SE types (HC0, HC1, HC2, HC3, HAC)
- ✅ Diagnostic tests (Breusch-Pagan, Durbin-Watson, RESET)
- ✅ Model validation and assumptions testing

### **Factor Analysis (1 test)**
- ✅ Principal Component Analysis (PCA)
- ✅ Factor loading estimation
- ✅ Variance explained calculations
- ✅ Factor score generation

### **Statistical Arbitrage Models (2 tests)**

#### **Pairs Trading:**
- ✅ Spread calculation and z-score normalization
- ✅ Entry and exit signal generation
- ✅ Beta coefficient estimation
- ✅ Threshold-based trading signals

#### **Mean Reversion:**
- ✅ Ornstein-Uhlenbeck process modeling
- ✅ Autoregressive mean reversion
- ✅ Half-life calculation
- ✅ Mean reversion signal generation

### **Edge Cases and Error Handling (3 tests)**
- ✅ Missing data (NA) handling
- ✅ Short time series edge cases
- ✅ Constant and highly volatile series
- ✅ Input parameter validation

### **Integration Tests (2 tests)**
- ✅ Complete statistical arbitrage workflow
- ✅ Model comparison and selection
- ✅ End-to-end pipeline validation

## 🛠️ Running Tests

### **Option 1: Windows Batch File (Easiest)**
```cmd
cd src\tests\R_tests
run_tests.bat
```

### **Option 2: R Script with Options**
```bash
# Basic test run
Rscript run_r_tests.R

# Verbose output
Rscript run_r_tests.R --verbose

# Generate HTML report
Rscript run_r_tests.R --html

# Generate code coverage
Rscript run_r_tests.R --coverage

# Install dependencies and run tests
Rscript run_r_tests.R --install-deps --verbose

# Generate JUnit XML for CI/CD
Rscript run_r_tests.R --junit
```

### **Option 3: Interactive R Session**
```r
# Load test environment
source("test_stat_models.R")

# Run all tests
run_statistical_tests()

# Run specific test categories
test_that("ARIMA models work correctly", { ... })

# Generate test data
test_data <- generate_test_data(n = 100, n_assets = 2)
```

## 📊 Test Data Generation

The test suite includes sophisticated synthetic data generators:

### **Financial Data Generator**
```r
generate_test_data(n = 252, n_assets = 2, volatility = 0.02, trend = 0.0005)
```
- Creates realistic price series with returns
- Supports multiple assets with correlation
- Configurable volatility and drift parameters

### **Stationary/Non-stationary Data**
```r
generate_stationary_data(n = 100, ar_coef = 0.7)
generate_nonstationary_data(n = 100)  # Random walk
```

### **Cointegrated Series**
```r
generate_cointegrated_data(n = 200)
```
- Creates two series with common stochastic trend
- Ensures cointegration relationship exists
- Includes stationary error correction term

## 🎯 Test Validation

Each test validates specific aspects:

### **Statistical Properties**
- Parameter estimates within expected ranges
- P-values indicating correct statistical inference
- Residual properties (white noise, normality)
- Model fit statistics (AIC, BIC, R²)

### **Functional Correctness**
- Function return values have expected structure
- Output dimensions match input specifications
- Signal generation produces valid trading signals
- Model diagnostics provide meaningful results

### **Edge Case Handling**
- Graceful handling of missing data
- Appropriate error messages for invalid inputs
- Robust behavior with extreme market conditions
- Proper validation of input parameters

## 📈 Performance Benchmarks

Tests include implicit performance validation:

### **Execution Time Expectations**
- **Data preprocessing**: < 1 second for 252 observations
- **Unit root tests**: < 2 seconds per test
- **ARIMA fitting**: < 5 seconds for automatic selection
- **GARCH models**: < 10 seconds for standard specifications
- **VAR models**: < 5 seconds for 2-3 variables
- **Pairs trading signals**: < 1 second for 252 observations

### **Memory Usage**
- Tests designed to work with realistic dataset sizes
- Efficient handling of time series objects
- Proper cleanup of large model objects

## 🔧 Configuration Options

### **Test Runner Options**
```bash
# Available command-line options
--verbose, -v          # Detailed test output
--coverage, -c         # Generate coverage report  
--html                # HTML test report
--junit               # JUnit XML for CI/CD
--install-deps, -i    # Install missing packages
--filter=PATTERN      # Run specific tests only
--help, -h            # Show help message
```

### **Test Environment Setup**
The test suite automatically:
- Installs missing R packages
- Sets up proper working directories
- Configures random seeds for reproducibility
- Handles package namespace conflicts

## 📋 Dependencies

### **Core Testing Packages**
- `testthat`: Main testing framework
- `covr`: Code coverage analysis  
- `devtools`: Development tools

### **Statistical Analysis Packages**
- `forecast`: Time series forecasting
- `tseries`: Time series analysis
- `lmtest`: Linear model testing
- `rugarch`: GARCH modeling
- `vars`: Vector autoregression
- `sandwich`: Robust standard errors
- `car`: Companion to applied regression

### **Data Manipulation**
- `xts`, `zoo`: Time series objects
- `dplyr`, `tidyr`: Data manipulation
- `PerformanceAnalytics`: Financial analysis

## 🧬 Test Results Interpretation

### **Successful Test Run**
```
R Statistical Models Test Runner
===============================

Running R Statistical Model Tests
=================================
Test directory: .../src/tests/R_tests
Reporter: progress

✓ | 25 test_stat_models.R

Test Results Summary:
====================
Passed: 25
Failed: 0

Test execution time: 15.3 seconds
```

### **Coverage Report**
```
Code Coverage Summary:
=====================
src/R/statistical_analysis/stat_models.R: 87.6%

Coverage report saved to: coverage_report.html
```

### **Failed Test Example**
```
✗ | 1 F [0.1s] stationarity_tests correctly identifies stationary data
  Error: P-value 0.15 not less than 0.1
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Package Installation Failures**:
   ```bash
   # Install packages manually
   install.packages(c("testthat", "forecast", "rugarch"))
   
   # Or use the auto-installer
   Rscript run_r_tests.R --install-deps
   ```

2. **Missing System Dependencies** (Linux):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install r-base-dev libcurl4-openssl-dev libxml2-dev
   
   # CentOS/RHEL
   sudo yum install R-devel libcurl-devel libxml2-devel
   ```

3. **Memory Issues with Large Models**:
   ```r
   # Reduce test data size
   test_data <- generate_test_data(n = 100)  # Instead of 252
   
   # Clear objects after tests
   rm(large_object)
   gc()
   ```

4. **Numerical Precision Issues**:
   ```r
   # Use appropriate tolerances
   expect_equal(result, expected, tolerance = 1e-6)
   expect_near(result, expected, 0.001)
   ```

### **Platform-Specific Issues**

#### **Windows**:
- Ensure R is in system PATH
- May need Rtools for package compilation
- Use forward slashes in file paths

#### **macOS**:
- Install Xcode command line tools
- May need gfortran for some packages
- Use Homebrew for system dependencies

#### **Linux**:
- Install development packages (-dev/-devel)
- May need additional system libraries
- Check R repository configuration

## 🔄 Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Setup R
  uses: r-lib/actions/setup-r@v2
  
- name: Install dependencies
  run: Rscript src/tests/R_tests/run_r_tests.R --install-deps
  
- name: Run tests
  run: Rscript src/tests/R_tests/run_r_tests.R --junit --coverage
  
- name: Upload test results
  uses: actions/upload-artifact@v2
  with:
    name: r-test-results
    path: src/tests/R_tests/test_results.xml
```

## 📚 Adding New Tests

To add new test cases:

1. **Create test function**:
   ```r
   test_that("new model works correctly", {
     # Setup test data
     test_data <- generate_test_data(n = 100)
     
     # Run function
     result <- new_model_function(test_data)
     
     # Assertions
     expect_true("output" %in% names(result))
     expect_equal(length(result$output), 100)
     expect_gt(result$statistic, 0)
   })
   ```

2. **Use appropriate expectations**:
   - `expect_equal()` - exact equality
   - `expect_near()` - floating-point comparison
   - `expect_true/false()` - logical conditions
   - `expect_error()` - error conditions
   - `expect_warning()` - warning conditions

3. **Follow naming conventions**:
   - Descriptive test names
   - Group related tests together
   - Use consistent data generation

## 📖 Additional Resources

- [testthat Documentation](https://testthat.r-lib.org/)
- [R Testing Best Practices](https://r-pkgs.org/tests.html)
- [Statistical Model Validation](https://CRAN.R-project.org/view=Econometrics)
- [Time Series Analysis in R](https://cran.r-project.org/view=TimeSeries)

The test suite ensures all statistical models are robust, accurate, and suitable for production use in statistical arbitrage applications.
