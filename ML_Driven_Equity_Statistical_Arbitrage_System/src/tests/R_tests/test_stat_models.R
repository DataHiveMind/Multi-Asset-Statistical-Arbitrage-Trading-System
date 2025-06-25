# =============================================================================
# Unit Tests for Statistical Models and Analysis
# =============================================================================
# Purpose: Comprehensive tests for R statistical models used in statistical
#          arbitrage system using testthat framework
# Author: Statistical Arbitrage Testing Team
# Date: 2025-06-25
# =============================================================================

# Load required libraries
library(testthat)
library(xts)
library(zoo)
library(stats)
library(forecast)
library(tseries)
library(lmtest)
library(rugarch)
library(vars)
library(sandwich)
library(car)

# Source the main statistical models file
source("../../R/statistical_analysis/stat_models.R")

# =============================================================================
# Helper Functions for Testing
# =============================================================================

#' Generate synthetic financial data for testing
#' @param n Number of observations
#' @param n_assets Number of assets
#' @param volatility Volatility parameter
#' @param trend Trend parameter
#' @return xts object with synthetic price data
generate_test_data <- function(n = 252, n_assets = 2, volatility = 0.02, trend = 0.0005) {
  set.seed(42)  # For reproducible tests
  
  dates <- seq.Date(Sys.Date() - n + 1, Sys.Date(), by = "day")
  returns <- matrix(rnorm(n * n_assets, mean = trend, sd = volatility), ncol = n_assets)
  
  # Create correlated returns for pairs trading tests
  if (n_assets >= 2) {
    returns[, 2] <- 0.8 * returns[, 1] + sqrt(1 - 0.8^2) * returns[, 2]
  }
  
  # Generate prices from returns
  prices <- matrix(NA, nrow = n, ncol = n_assets)
  prices[1, ] <- 100  # Starting price
  
  for (i in 2:n) {
    prices[i, ] <- prices[i-1, ] * (1 + returns[i, ])
  }
  
  colnames(prices) <- paste0("Asset", 1:n_assets)
  return(xts(prices, order.by = dates))
}

#' Generate stationary test data
#' @param n Number of observations
#' @param ar_coef AR coefficient (< 1 for stationarity)
#' @return Stationary time series
generate_stationary_data <- function(n = 100, ar_coef = 0.7) {
  set.seed(123)
  ts(arima.sim(n = n, list(ar = ar_coef)), frequency = 1)
}

#' Generate non-stationary test data
#' @param n Number of observations
#' @return Non-stationary time series (random walk)
generate_nonstationary_data <- function(n = 100) {
  set.seed(456)
  ts(cumsum(rnorm(n)), frequency = 1)
}

#' Generate cointegrated data
#' @param n Number of observations
#' @return List with two cointegrated series
generate_cointegrated_data <- function(n = 200) {
  set.seed(789)
  # Common stochastic trend
  common_trend <- cumsum(rnorm(n))
  
  # Two series with common trend plus stationary components
  x <- common_trend + arima.sim(n = n, list(ar = 0.5))
  y <- 2 * common_trend + arima.sim(n = n, list(ar = 0.3)) + rnorm(n, sd = 0.5)
  
  return(list(x = as.numeric(x), y = as.numeric(y)))
}

# =============================================================================
# Data Loading and Preprocessing Tests
# =============================================================================

test_that("load_packages function works correctly", {
  # Test that the function doesn't error with valid packages
  expect_silent(load_packages(c("stats", "utils")))
  
  # Test that required packages are loaded
  expect_true("stats" %in% loadedNamespaces())
  expect_true("utils" %in% loadedNamespaces())
})

test_that("preprocess_data function handles different return types", {
  test_prices <- generate_test_data(n = 50, n_assets = 1)
  
  # Test log returns
  result_log <- preprocess_data(test_prices, return_type = "log")
  expect_true("log_returns" %in% names(result_log))
  expect_equal(nrow(result_log$log_returns), nrow(test_prices) - 1)
  
  # Test simple returns
  result_simple <- preprocess_data(test_prices, return_type = "simple")
  expect_true("simple_returns" %in% names(result_simple))
  expect_equal(nrow(result_simple$simple_returns), nrow(test_prices) - 1)
  
  # Test both return types
  result_both <- preprocess_data(test_prices, return_type = "both")
  expect_true(all(c("log_returns", "simple_returns") %in% names(result_both)))
})

test_that("preprocess_data handles outlier cleaning", {
  test_prices <- generate_test_data(n = 50, n_assets = 1)
  
  # Add artificial outliers
  test_prices[25] <- test_prices[25] * 10  # Extreme outlier
  
  result_clean <- preprocess_data(test_prices, clean_outliers = TRUE)
  result_no_clean <- preprocess_data(test_prices, clean_outliers = FALSE)
  
  # Cleaned data should have different characteristics
  expect_true(sd(result_clean$log_returns, na.rm = TRUE) < 
              sd(result_no_clean$log_returns, na.rm = TRUE))
})

# =============================================================================
# Stationarity and Unit Root Tests
# =============================================================================

test_that("stationarity_tests correctly identifies stationary data", {
  stationary_data <- generate_stationary_data(n = 100)
  
  result <- stationarity_tests(stationary_data, test_type = "all")
  
  expect_true("adf" %in% names(result))
  expect_true("pp" %in% names(result))
  expect_true("kpss" %in% names(result))
  
  # ADF and PP should reject null (series is stationary)
  expect_lt(result$adf$p.value, 0.1)  # Should be stationary
  expect_lt(result$pp$p.value, 0.1)   # Should be stationary
})

test_that("stationarity_tests correctly identifies non-stationary data", {
  nonstationary_data <- generate_nonstationary_data(n = 100)
  
  result <- stationarity_tests(nonstationary_data, test_type = "all")
  
  # ADF and PP should fail to reject null (series is non-stationary)
  expect_gt(result$adf$p.value, 0.05)  # Should be non-stationary
  expect_gt(result$pp$p.value, 0.05)   # Should be non-stationary
})

test_that("stationarity_tests handles individual test types", {
  test_data <- generate_stationary_data(n = 80)
  
  adf_result <- stationarity_tests(test_data, test_type = "adf")
  expect_true("adf" %in% names(adf_result))
  expect_false("pp" %in% names(adf_result))
  
  pp_result <- stationarity_tests(test_data, test_type = "pp")
  expect_true("pp" %in% names(pp_result))
  expect_false("adf" %in% names(pp_result))
})

# =============================================================================
# Cointegration Tests
# =============================================================================

test_that("cointegration_test detects cointegrated series", {
  coint_data <- generate_cointegrated_data(n = 150)
  
  result <- cointegration_test(coint_data$x, coint_data$y, method = "engle_granger")
  
  expect_true("test_statistic" %in% names(result))
  expect_true("p_value" %in% names(result))
  expect_true("residuals" %in% names(result))
  expect_true("beta" %in% names(result))
  
  # Should detect cointegration (low p-value)
  expect_lt(result$p_value, 0.1)
})

test_that("cointegration_test handles non-cointegrated series", {
  # Generate independent random walks
  set.seed(111)
  x <- cumsum(rnorm(100))
  y <- cumsum(rnorm(100))
  
  result <- cointegration_test(x, y, method = "engle_granger")
  
  # Should not detect cointegration (high p-value)
  expect_gt(result$p_value, 0.1)
})

test_that("cointegration_test supports different methods", {
  coint_data <- generate_cointegrated_data(n = 100)
  
  eg_result <- cointegration_test(coint_data$x, coint_data$y, method = "engle_granger")
  expect_true("beta" %in% names(eg_result))
  
  # Johansen test would require more complex setup
  # jo_result <- cointegration_test(coint_data$x, coint_data$y, method = "johansen")
  # expect_true("test_statistic" %in% names(jo_result))
})

# =============================================================================
# ARIMA Model Tests
# =============================================================================

test_that("fit_arima_model automatic selection works", {
  # Generate AR(2) process for testing
  set.seed(222)
  ar_data <- arima.sim(n = 150, list(ar = c(0.6, -0.2)))
  
  result <- fit_arima_model(ar_data, method = "auto")
  
  expect_true("model" %in% names(result))
  expect_true("order" %in% names(result))
  expect_true("aic" %in% names(result))
  expect_true("bic" %in% names(result))
  expect_true("residuals" %in% names(result))
  expect_true("fitted" %in% names(result))
  expect_true("forecast" %in% names(result))
  expect_true("diagnostics" %in% names(result))
  
  # Check that we get reasonable order
  expect_true(length(result$order) == 3)
  expect_true(all(result$order >= 0))
  
  # Check forecast object
  expect_true(inherits(result$forecast, "forecast"))
})

test_that("fit_arima_model manual selection works", {
  set.seed(333)
  ar_data <- arima.sim(n = 100, list(ar = 0.5))
  
  result <- fit_arima_model(ar_data, method = "manual", max_order = 3)
  
  expect_true("model" %in% names(result))
  expect_true(is.finite(result$aic))
  expect_true(is.finite(result$bic))
  
  # Manual method should find reasonable order
  expect_true(length(result$order) == 3)
})

test_that("fit_arima_model diagnostics are meaningful", {
  set.seed(444)
  ma_data <- arima.sim(n = 120, list(ma = 0.4))
  
  result <- fit_arima_model(ma_data, method = "auto")
  
  expect_true("ljung_box" %in% names(result$diagnostics))
  expect_true("jarque_bera" %in% names(result$diagnostics))
  expect_true("arch_test" %in% names(result$diagnostics))
  
  # Check that diagnostic objects have p-values
  expect_true("p.value" %in% names(result$diagnostics$ljung_box))
  expect_true("p.value" %in% names(result$diagnostics$jarque_bera))
})

# =============================================================================
# GARCH Model Tests
# =============================================================================

test_that("fit_garch_model basic functionality", {
  # Generate returns with volatility clustering
  set.seed(555)
  n <- 200
  returns <- numeric(n)
  sigma <- numeric(n)
  sigma[1] <- 0.02
  
  for (i in 2:n) {
    sigma[i] <- sqrt(0.00001 + 0.05 * returns[i-1]^2 + 0.9 * sigma[i-1]^2)
    returns[i] <- sigma[i] * rnorm(1)
  }
  
  returns_ts <- ts(returns[-1])  # Remove first observation
  
  result <- fit_garch_model(returns_ts, model_type = "sGARCH", distribution = "std")
  
  expect_true("model" %in% names(result))
  expect_true("coefficients" %in% names(result))
  expect_true("volatility" %in% names(result))
  expect_true("standardized_residuals" %in% names(result))
  expect_true("aic" %in% names(result))
  expect_true("bic" %in% names(result))
  expect_true("diagnostics" %in% names(result))
  
  # Check that volatility is positive
  expect_true(all(result$volatility > 0))
  
  # Check that AIC and BIC are finite
  expect_true(is.finite(result$aic))
  expect_true(is.finite(result$bic))
})

test_that("fit_garch_model handles different specifications", {
  set.seed(666)
  returns <- rnorm(150, sd = 0.02)
  
  # Test different model types
  sgarch_result <- fit_garch_model(returns, model_type = "sGARCH")
  expect_true(inherits(sgarch_result$model, "uGARCHfit"))
  
  # Test different distributions
  std_result <- fit_garch_model(returns, distribution = "std")
  norm_result <- fit_garch_model(returns, distribution = "norm")
  
  expect_true(is.finite(std_result$aic))
  expect_true(is.finite(norm_result$aic))
})

# =============================================================================
# VAR Model Tests
# =============================================================================

test_that("fit_var_model works with multiple time series", {
  # Generate VAR(1) process
  set.seed(777)
  n <- 150
  k <- 2  # Two variables
  
  # VAR(1) coefficients
  A1 <- matrix(c(0.6, 0.1, 0.2, 0.7), nrow = k)
  
  # Generate data
  Y <- matrix(0, nrow = n, ncol = k)
  Y[1, ] <- rnorm(k)
  
  for (t in 2:n) {
    Y[t, ] <- A1 %*% Y[t-1, ] + rnorm(k, sd = 0.1)
  }
  
  colnames(Y) <- c("Series1", "Series2")
  Y_ts <- ts(Y)
  
  result <- fit_var_model(Y_ts, lag_max = 3)
  
  expect_true("model" %in% names(result))
  expect_true("lag_selection" %in% names(result))
  expect_true("optimal_lag" %in% names(result))
  expect_true("coefficients" %in% names(result))
  expect_true("granger_causality" %in% names(result))
  expect_true("impulse_response" %in% names(result))
  
  # Check that optimal lag is reasonable
  expect_true(result$optimal_lag >= 1)
  expect_true(result$optimal_lag <= 3)
  
  # Check Granger causality results
  expect_true(length(result$granger_causality) > 0)
})

# =============================================================================
# Regression Model Tests
# =============================================================================

test_that("robust_regression produces valid results", {
  # Generate regression data with heteroskedasticity
  set.seed(888)
  n <- 100
  x <- rnorm(n)
  error <- rnorm(n, sd = abs(x))  # Heteroskedastic errors
  y <- 2 + 3 * x + error
  
  data <- data.frame(y = y, x = x)
  
  result <- robust_regression(y ~ x, data = data, se_type = "HC3")
  
  expect_true("model" %in% names(result))
  expect_true("coefficients" %in% names(result))
  expect_true("robust_se" %in% names(result))
  expect_true("robust_t" %in% names(result))
  expect_true("robust_p" %in% names(result))
  expect_true("diagnostics" %in% names(result))
  
  # Check coefficient estimates are reasonable
  expect_true(abs(result$coefficients[1] - 2) < 1)  # Intercept around 2
  expect_true(abs(result$coefficients[2] - 3) < 1)  # Slope around 3
  
  # Check that robust standard errors are positive
  expect_true(all(result$robust_se > 0))
  
  # Check diagnostic tests
  expect_true("breusch_pagan" %in% names(result$diagnostics))
  expect_true("durbin_watson" %in% names(result$diagnostics))
})

test_that("robust_regression handles different SE types", {
  set.seed(999)
  n <- 80
  x <- rnorm(n)
  y <- 1 + 2 * x + rnorm(n)
  data <- data.frame(y = y, x = x)
  
  hc3_result <- robust_regression(y ~ x, data = data, se_type = "HC3")
  hc1_result <- robust_regression(y ~ x, data = data, se_type = "HC1")
  
  expect_true(all(hc3_result$robust_se > 0))
  expect_true(all(hc1_result$robust_se > 0))
  
  # Different SE types should give different results
  expect_false(identical(hc3_result$robust_se, hc1_result$robust_se))
})

# =============================================================================
# Factor Analysis Tests
# =============================================================================

test_that("factor_analysis_pca works correctly", {
  # Generate factor structure
  set.seed(1111)
  n <- 100
  k <- 5  # 5 assets
  f <- 2  # 2 factors
  
  # Factor loadings
  loadings <- matrix(rnorm(k * f), nrow = k, ncol = f)
  
  # Generate data
  factors <- matrix(rnorm(n * f), nrow = n, ncol = f)
  idiosyncratic <- matrix(rnorm(n * k, sd = 0.5), nrow = n, ncol = k)
  
  returns <- factors %*% t(loadings) + idiosyncratic
  colnames(returns) <- paste0("Asset", 1:k)
  
  result <- factor_analysis_pca(returns, n_factors = 2)
  
  expect_true("loadings" %in% names(result))
  expect_true("scores" %in% names(result))
  expect_true("variance_explained" %in% names(result))
  expect_true("cumulative_variance" %in% names(result))
  
  # Check dimensions
  expect_equal(ncol(result$loadings), 2)  # 2 factors
  expect_equal(nrow(result$loadings), k)  # k assets
  expect_equal(ncol(result$scores), 2)    # 2 factors
  expect_equal(nrow(result$scores), n)    # n observations
  
  # Check that variance explained is reasonable
  expect_true(all(result$variance_explained >= 0))
  expect_true(sum(result$variance_explained) <= 1)
})

# =============================================================================
# Pairs Trading Tests
# =============================================================================

test_that("pairs_trading_signals generates reasonable signals", {
  # Generate cointegrated pair
  coint_data <- generate_cointegrated_data(n = 200)
  price1 <- coint_data$x
  price2 <- coint_data$y
  
  result <- pairs_trading_signals(price1, price2, lookback = 60, 
                                 entry_threshold = 2, exit_threshold = 0.5)
  
  expect_true("spread" %in% names(result))
  expect_true("zscore" %in% names(result))
  expect_true("signals" %in% names(result))
  expect_true("beta" %in% names(result))
  expect_true("entry_threshold" %in% names(result))
  expect_true("exit_threshold" %in% names(result))
  
  # Check signal values
  expect_true(all(result$signals %in% c(-1, 0, 1)))
  
  # Check that spread length matches input
  expect_equal(length(result$spread), length(price1))
  
  # Check that beta is reasonable for cointegrated series
  expect_true(abs(result$beta[length(result$beta)] - 2) < 1)  # Should be around 2
})

test_that("pairs_trading_signals handles different parameters", {
  test_data <- generate_cointegrated_data(n = 120)
  
  # Test different thresholds
  result1 <- pairs_trading_signals(test_data$x, test_data$y, entry_threshold = 1.5)
  result2 <- pairs_trading_signals(test_data$x, test_data$y, entry_threshold = 2.5)
  
  # More conservative threshold should generate fewer signals
  expect_true(sum(abs(result2$signals)) <= sum(abs(result1$signals)))
  
  # Test different lookback periods
  result_short <- pairs_trading_signals(test_data$x, test_data$y, lookback = 30)
  result_long <- pairs_trading_signals(test_data$x, test_data$y, lookback = 90)
  
  expect_equal(length(result_short$signals), length(result_long$signals))
})

# =============================================================================
# Mean Reversion Tests
# =============================================================================

test_that("mean_reversion_model detects mean reversion", {
  # Generate mean-reverting process
  set.seed(1234)
  n <- 200
  kappa <- 0.1  # Mean reversion speed
  theta <- 100  # Long-term mean
  sigma <- 2    # Volatility
  
  prices <- numeric(n)
  prices[1] <- theta
  
  for (i in 2:n) {
    prices[i] <- prices[i-1] + kappa * (theta - prices[i-1]) + rnorm(1, sd = sigma)
  }
  
  result <- mean_reversion_model(prices, lookback = 60, model_type = "ou")
  
  expect_true("model_type" %in% names(result))
  expect_true("parameters" %in% names(result))
  expect_true("half_life" %in% names(result))
  expect_true("signals" %in% names(result))
  
  # Check that half-life is reasonable
  expect_true(result$half_life > 0)
  expect_true(result$half_life < 100)  # Should be much less than sample size
  
  # Check signals
  expect_true(all(result$signals %in% c(-1, 0, 1)))
})

test_that("mean_reversion_model handles different model types", {
  test_prices <- 100 + cumsum(rnorm(150, sd = 0.5))
  
  ou_result <- mean_reversion_model(test_prices, model_type = "ou")
  ar_result <- mean_reversion_model(test_prices, model_type = "ar")
  
  expect_equal(ou_result$model_type, "ou")
  expect_equal(ar_result$model_type, "ar")
  
  expect_true("half_life" %in% names(ou_result))
  expect_true("half_life" %in% names(ar_result))
})

# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

test_that("functions handle missing data appropriately", {
  # Create data with missing values
  test_data <- generate_test_data(n = 50, n_assets = 1)
  test_data[c(10, 25, 40)] <- NA
  
  # Test that functions handle NAs gracefully
  expect_error(preprocess_data(test_data), NA)
  
  # Test with time series containing NAs
  ts_with_na <- ts(c(rnorm(20), NA, rnorm(20)))
  expect_error(stationarity_tests(ts_with_na), NA)
})

test_that("functions handle edge cases", {
  # Test with very short time series
  short_ts <- ts(rnorm(10))
  expect_error(stationarity_tests(short_ts), NA)
  
  # Test with constant series
  constant_ts <- ts(rep(1, 50))
  expect_error(stationarity_tests(constant_ts), NA)
  
  # Test with very volatile series
  volatile_ts <- ts(rnorm(100, sd = 10))
  expect_error(fit_arima_model(volatile_ts), NA)
})

test_that("functions validate input parameters", {
  test_data <- generate_test_data(n = 100, n_assets = 1)
  
  # Test invalid return type
  expect_error(preprocess_data(test_data, return_type = "invalid"))
  
  # Test invalid stationarity test type
  expect_error(stationarity_tests(as.numeric(test_data[,1]), test_type = "invalid"))
})

# =============================================================================
# Integration Tests
# =============================================================================

test_that("complete statistical arbitrage workflow", {
  # Generate synthetic pair of assets
  set.seed(9999)
  test_data <- generate_test_data(n = 250, n_assets = 2)
  
  # Test complete workflow
  # 1. Preprocess data
  processed <- preprocess_data(test_data, return_type = "log")
  expect_true("log_returns" %in% names(processed))
  
  # 2. Test stationarity of returns
  returns1 <- processed$log_returns[, 1]
  stationarity_result <- stationarity_tests(as.numeric(returns1))
  expect_true("adf" %in% names(stationarity_result))
  
  # 3. Test cointegration of price levels
  coint_result <- cointegration_test(test_data[, 1], test_data[, 2])
  expect_true("p_value" %in% names(coint_result))
  
  # 4. Generate pairs trading signals
  pairs_result <- pairs_trading_signals(as.numeric(test_data[, 1]), 
                                       as.numeric(test_data[, 2]), 
                                       lookback = 60)
  expect_true("signals" %in% names(pairs_result))
  
  # Workflow should complete without errors
  expect_true(TRUE)
})

test_that("model comparison and selection workflow", {
  # Generate ARIMA process
  set.seed(7777)
  arima_data <- arima.sim(n = 200, list(ar = c(0.6, -0.2), ma = 0.3))
  
  # Fit different models
  auto_arima <- fit_arima_model(arima_data, method = "auto")
  manual_arima <- fit_arima_model(arima_data, method = "manual", max_order = 3)
  
  # Compare model fits
  expect_true(is.finite(auto_arima$aic))
  expect_true(is.finite(manual_arima$aic))
  
  # Both should be valid ARIMA models
  expect_true(length(auto_arima$order) == 3)
  expect_true(length(manual_arima$order) == 3)
})

# =============================================================================
# Run All Tests
# =============================================================================

# Function to run all tests
run_statistical_tests <- function() {
  cat("Running Statistical Models Unit Tests...\n")
  cat("=====================================\n\n")
  
  # Capture test results
  result <- test_dir(".", reporter = "summary")
  
  cat("\nTest Summary:\n")
  cat("=============\n")
  
  return(result)
}

# Print message when sourced
cat("Statistical Models Test Suite Loaded\n")
cat("====================================\n")
cat("Available functions:\n")
cat("- generate_test_data(): Generate synthetic financial data\n")
cat("- generate_stationary_data(): Generate stationary time series\n") 
cat("- generate_nonstationary_data(): Generate non-stationary time series\n")
cat("- generate_cointegrated_data(): Generate cointegrated series\n")
cat("- run_statistical_tests(): Execute all test cases\n\n")
cat("To run all tests, execute: run_statistical_tests()\n")
cat("To run specific tests, use: test_that() or test_file()\n\n")
