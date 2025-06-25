# =============================================================================
# Statistical Models and Analysis for Statistical Arbitrage
# =============================================================================
# Purpose: Advanced econometric models, time-series analysis, and statistical
#          tests for identifying arbitrage opportunities in financial markets
# Author: Statistical Arbitrage System
# Date: 2025-06-24
# =============================================================================

# Required packages
required_packages <- c(
  "stats", "forecast", "tseries", "lmtest", "quantmod", "urca", "vars",
  "zoo", "xts", "PerformanceAnalytics", "corrplot", "ggplot2", "dplyr",
  "tidyr", "bcp", "strucchange", "dynlm", "car", "sandwich", "MASS",
  "rugarch", "rmgarch", "MTS", "fGarch", "ccgarch", "egcm", "tsDyn",
  "wavelets", "EMD", "pracma", "signal", "changepoint", "bfast"
)

# Function to install and load packages
load_packages <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
    install.packages(new_packages, dependencies = TRUE)
  }
  
  for (pkg in packages) {
    library(pkg, character.only = TRUE)
  }
  cat("All required packages loaded successfully.\n")
}

# Load all required packages
load_packages(required_packages)

# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================

#' Load financial data from various sources
#' @param source Character: "csv", "kdb", "yahoo", "quantmod"
#' @param symbols Character vector: symbols to load
#' @param start_date Date: start date for data
#' @param end_date Date: end date for data
#' @param file_path Character: path for CSV files
#' @return xts object with price data
load_financial_data <- function(source = "csv", symbols, start_date = Sys.Date() - 365, 
                               end_date = Sys.Date(), file_path = NULL) {
  
  if (source == "csv" && !is.null(file_path)) {
    # Load from CSV files
    data_list <- list()
    for (symbol in symbols) {
      file <- file.path(file_path, paste0(symbol, ".csv"))
      if (file.exists(file)) {
        df <- read.csv(file, stringsAsFactors = FALSE)
        df$Date <- as.Date(df$Date)
        data_list[[symbol]] <- xts(df[, -1], order.by = df$Date)
      }
    }
    return(do.call(merge, data_list))
    
  } else if (source == "yahoo" || source == "quantmod") {
    # Load from Yahoo Finance
    data_list <- list()
    for (symbol in symbols) {
      tryCatch({
        getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
        data_list[[symbol]] <- get(symbol)
      }, error = function(e) {
        cat("Error loading", symbol, ":", e$message, "\n")
      })
    }
    return(do.call(merge, data_list))
    
  } else if (source == "kdb") {
    # Load from kdb+ (requires kdb+ R interface)
    # This is a placeholder - actual implementation would require kdb+ connection
    cat("kdb+ interface not implemented in this example\n")
    return(NULL)
  }
  
  stop("Invalid source specified")
}

#' Preprocess financial data for analysis
#' @param data xts object with price data
#' @param return_type Character: "simple", "log", "both"
#' @param clean_outliers Logical: whether to clean outliers
#' @return List with processed data
preprocess_data <- function(data, return_type = "log", clean_outliers = TRUE) {
  
  # Calculate returns
  if (return_type %in% c("simple", "both")) {
    simple_returns <- diff(data) / lag(data, 1)
    simple_returns <- na.omit(simple_returns)
  }
  
  if (return_type %in% c("log", "both")) {
    log_returns <- diff(log(data))
    log_returns <- na.omit(log_returns)
  }
  
  # Clean outliers using Tukey's method
  if (clean_outliers) {
    clean_outliers_func <- function(x) {
      Q1 <- quantile(x, 0.25, na.rm = TRUE)
      Q3 <- quantile(x, 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower <- Q1 - 1.5 * IQR
      upper <- Q3 + 1.5 * IQR
      x[x < lower | x > upper] <- NA
      return(x)
    }
    
    if (exists("log_returns")) {
      log_returns <- apply(log_returns, 2, clean_outliers_func)
      log_returns <- xts(log_returns, order.by = index(log_returns))
    }
    if (exists("simple_returns")) {
      simple_returns <- apply(simple_returns, 2, clean_outliers_func)
      simple_returns <- xts(simple_returns, order.by = index(simple_returns))
    }
  }
  
  result <- list(
    prices = data,
    log_returns = if (exists("log_returns")) log_returns else NULL,
    simple_returns = if (exists("simple_returns")) simple_returns else NULL
  )
  
  return(result)
}

# =============================================================================
# Statistical Tests and Hypothesis Testing
# =============================================================================

#' Comprehensive stationarity testing
#' @param data Numeric vector or time series
#' @param test_type Character: "all", "adf", "kpss", "pp"
#' @return List with test results
stationarity_tests <- function(data, test_type = "all") {
  
  results <- list()
  
  if (test_type %in% c("all", "adf")) {
    # Augmented Dickey-Fuller test
    adf_test <- ur.df(data, type = "trend", selectlags = "AIC")
    results$adf <- list(
      test_stat = adf_test@teststat,
      critical_values = adf_test@cval,
      p_value = if (adf_test@teststat[1] < adf_test@cval[1,1]) "< 0.01" else 
                if (adf_test@teststat[1] < adf_test@cval[1,2]) "< 0.05" else 
                if (adf_test@teststat[1] < adf_test@cval[1,3]) "< 0.10" else "> 0.10",
      stationary = adf_test@teststat[1] < adf_test@cval[1,2]
    )
  }
  
  if (test_type %in% c("all", "kpss")) {
    # KPSS test
    kpss_test <- ur.kpss(data, type = "tau")
    results$kpss <- list(
      test_stat = kpss_test@teststat,
      critical_values = kpss_test@cval,
      stationary = kpss_test@teststat < kpss_test@cval[2] # 5% level
    )
  }
  
  if (test_type %in% c("all", "pp")) {
    # Phillips-Perron test
    pp_test <- ur.pp(data, type = "Z-tau", model = "trend")
    results$pp <- list(
      test_stat = pp_test@teststat,
      critical_values = pp_test@cval,
      stationary = pp_test@teststat < pp_test@cval[2] # 5% level
    )
  }
  
  return(results)
}

#' Cointegration testing for pairs trading
#' @param x First time series
#' @param y Second time series
#' @param method Character: "engle_granger", "johansen"
#' @return List with cointegration test results
cointegration_test <- function(x, y, method = "engle_granger") {
  
  if (method == "engle_granger") {
    # Engle-Granger two-step procedure
    
    # Step 1: Estimate cointegrating relationship
    lm_model <- lm(y ~ x)
    residuals <- residuals(lm_model)
    
    # Step 2: Test residuals for stationarity
    adf_residuals <- ur.df(residuals, type = "none", selectlags = "AIC")
    
    # Enhanced Engle-Granger test
    eg_test <- ca.jo(cbind(x, y), type = "eigen", ecdet = "const", K = 2)
    
    result <- list(
      method = "Engle-Granger",
      cointegrating_vector = coef(lm_model),
      residuals = residuals,
      adf_residuals = adf_residuals@teststat,
      adf_critical = adf_residuals@cval,
      cointegrated = adf_residuals@teststat < adf_residuals@cval[2],
      johansen_test = eg_test,
      r_squared = summary(lm_model)$r.squared
    )
    
  } else if (method == "johansen") {
    # Johansen cointegration test
    data_matrix <- cbind(x, y)
    johansen_test <- ca.jo(data_matrix, type = "eigen", ecdet = "const", K = 2)
    
    result <- list(
      method = "Johansen",
      test_stats = johansen_test@teststat,
      critical_values = johansen_test@cval,
      cointegrating_vectors = johansen_test@V,
      alpha = johansen_test@W,
      cointegrated = any(johansen_test@teststat > johansen_test@cval[,2])
    )
  }
  
  return(result)
}

#' Test for structural breaks
#' @param data Time series data
#' @param method Character: "chow", "cusum", "bcp"
#' @return List with structural break test results
structural_break_test <- function(data, method = "chow") {
  
  results <- list()
  
  if (method == "chow") {
    # Chow test for structural break
    n <- length(data)
    break_point <- floor(n/2)
    
    # Create time index
    time_index <- 1:n
    
    # Full model
    full_model <- lm(data ~ time_index)
    
    # Split models
    model1 <- lm(data[1:break_point] ~ time_index[1:break_point])
    model2 <- lm(data[(break_point+1):n] ~ time_index[(break_point+1):n])
    
    # Chow test statistic
    rss_restricted <- sum(residuals(full_model)^2)
    rss_unrestricted <- sum(residuals(model1)^2) + sum(residuals(model2)^2)
    
    k <- 2  # number of parameters
    chow_stat <- ((rss_restricted - rss_unrestricted) / k) / 
                 (rss_unrestricted / (n - 2*k))
    
    p_value <- 1 - pf(chow_stat, k, n - 2*k)
    
    results$chow <- list(
      statistic = chow_stat,
      p_value = p_value,
      break_detected = p_value < 0.05
    )
  }
  
  if (method == "cusum") {
    # CUSUM test
    model <- lm(data ~ seq_along(data))
    cusum_test <- efp(model, type = "Rec-CUSUM")
    
    results$cusum <- list(
      test = cusum_test,
      break_detected = sctest(cusum_test)$p.value < 0.05
    )
  }
  
  if (method == "bcp") {
    # Bayesian Change Point detection
    bcp_result <- bcp(data, mcmc = 5000, burnin = 500)
    
    results$bcp <- list(
      posterior_probs = bcp_result$posterior.prob,
      change_points = which(bcp_result$posterior.prob > 0.5),
      prob_threshold = 0.5
    )
  }
  
  return(results)
}

# =============================================================================
# Time Series Models
# =============================================================================

#' Fit ARIMA models with automatic order selection
#' @param data Time series data
#' @param method Character: "auto", "manual"
#' @param max_order Integer: maximum order for auto selection
#' @return List with ARIMA model results
fit_arima_model <- function(data, method = "auto", max_order = 5) {
  
  if (method == "auto") {
    # Automatic ARIMA selection
    auto_model <- auto.arima(data, max.p = max_order, max.q = max_order, 
                            max.d = 2, seasonal = FALSE, ic = "aic",
                            stepwise = FALSE, approximation = FALSE)
    
    result <- list(
      model = auto_model,
      order = arimaorder(auto_model),
      aic = AIC(auto_model),
      bic = BIC(auto_model),
      residuals = residuals(auto_model),
      fitted = fitted(auto_model),
      forecast = forecast(auto_model, h = 10)
    )
    
  } else {
    # Manual selection using grid search
    best_aic <- Inf
    best_model <- NULL
    best_order <- c(0, 0, 0)
    
    for (p in 0:max_order) {
      for (d in 0:2) {
        for (q in 0:max_order) {
          tryCatch({
            model <- Arima(data, order = c(p, d, q))
            if (AIC(model) < best_aic) {
              best_aic <- AIC(model)
              best_model <- model
              best_order <- c(p, d, q)
            }
          }, error = function(e) NULL)
        }
      }
    }
    
    result <- list(
      model = best_model,
      order = best_order,
      aic = best_aic,
      bic = BIC(best_model),
      residuals = residuals(best_model),
      fitted = fitted(best_model),
      forecast = forecast(best_model, h = 10)
    )
  }
  
  # Model diagnostics
  result$diagnostics <- list(
    ljung_box = Box.test(result$residuals, lag = 10, type = "Ljung-Box"),
    jarque_bera = jarque.bera.test(result$residuals),
    arch_test = ArchTest(result$residuals, lags = 5)
  )
  
  return(result)
}

#' Fit GARCH models for volatility modeling
#' @param returns Return series
#' @param model_type Character: "sGARCH", "eGARCH", "gjrGARCH"
#' @param distribution Character: "norm", "std", "sstd"
#' @return rugarch model object
fit_garch_model <- function(returns, model_type = "sGARCH", distribution = "std") {
  
  # Specify GARCH model
  spec <- ugarchspec(
    variance.model = list(model = model_type, garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
    distribution.model = distribution
  )
  
  # Fit model
  garch_fit <- ugarchfit(spec, returns)
  
  # Extract results
  result <- list(
    model = garch_fit,
    coefficients = coef(garch_fit),
    volatility = sigma(garch_fit),
    standardized_residuals = residuals(garch_fit, standardize = TRUE),
    log_likelihood = likelihood(garch_fit),
    aic = infocriteria(garch_fit)[1],
    bic = infocriteria(garch_fit)[2]
  )
  
  # Model diagnostics
  result$diagnostics <- list(
    ljung_box_residuals = Box.test(result$standardized_residuals, lag = 10),
    ljung_box_squared = Box.test(result$standardized_residuals^2, lag = 10),
    arch_test = ArchTest(result$standardized_residuals, lags = 5)
  )
  
  return(result)
}

#' Vector Autoregression (VAR) model for multiple time series
#' @param data Matrix or data frame of time series
#' @param lag_max Integer: maximum lag order to consider
#' @return List with VAR model results
fit_var_model <- function(data, lag_max = 5) {
  
  # Lag order selection
  lag_select <- VARselect(data, lag.max = lag_max, type = "const")
  optimal_lag <- lag_select$selection["AIC(n)"]
  
  # Fit VAR model
  var_model <- VAR(data, p = optimal_lag, type = "const")
  
  # Granger causality tests
  granger_results <- list()
  var_names <- colnames(data)
  
  for (i in 1:ncol(data)) {
    for (j in 1:ncol(data)) {
      if (i != j) {
        test_name <- paste(var_names[j], "->", var_names[i])
        granger_results[[test_name]] <- causality(var_model, cause = var_names[j])$Granger
      }
    }
  }
  
  result <- list(
    model = var_model,
    lag_selection = lag_select,
    optimal_lag = optimal_lag,
    coefficients = coef(var_model),
    residuals = residuals(var_model),
    fitted = fitted(var_model),
    granger_causality = granger_results,
    impulse_response = irf(var_model, n.ahead = 10),
    forecast_error_variance = fevd(var_model, n.ahead = 10)
  )
  
  return(result)
}

# =============================================================================
# Regression Models and Factor Analysis
# =============================================================================

#' Linear regression with robust standard errors
#' @param formula Formula object
#' @param data Data frame
#' @param se_type Character: "HC0", "HC1", "HC2", "HC3", "HAC"
#' @return List with regression results
robust_regression <- function(formula, data, se_type = "HC3") {
  
  # Fit OLS model
  ols_model <- lm(formula, data = data)
  
  # Calculate robust standard errors
  if (se_type %in% c("HC0", "HC1", "HC2", "HC3")) {
    robust_se <- sqrt(diag(vcovHC(ols_model, type = se_type)))
  } else if (se_type == "HAC") {
    robust_se <- sqrt(diag(vcovHAC(ols_model)))
  }
  
  # Create robust t-statistics and p-values
  robust_t <- coef(ols_model) / robust_se
  robust_p <- 2 * pt(abs(robust_t), df = df.residual(ols_model), lower.tail = FALSE)
  
  result <- list(
    model = ols_model,
    coefficients = coef(ols_model),
    robust_se = robust_se,
    robust_t = robust_t,
    robust_p = robust_p,
    r_squared = summary(ols_model)$r.squared,
    adj_r_squared = summary(ols_model)$adj.r.squared,
    residuals = residuals(ols_model),
    fitted = fitted(ols_model)
  )
  
  # Diagnostic tests
  result$diagnostics <- list(
    breusch_pagan = bptest(ols_model),
    durbin_watson = dwtest(ols_model),
    jarque_bera = jarque.bera.test(residuals(ols_model)),
    reset_test = resettest(ols_model)
  )
  
  return(result)
}

#' Fama-French factor model
#' @param returns Asset return series
#' @param factors Data frame with factor returns (Mkt.RF, SMB, HML, etc.)
#' @return List with factor model results
fama_french_model <- function(returns, factors) {
  
  # Ensure data alignment
  common_dates <- intersect(index(returns), index(factors))
  returns_aligned <- returns[common_dates]
  factors_aligned <- factors[common_dates]
  
  results <- list()
  
  if (is.vector(returns_aligned)) {
    # Single asset
    model_data <- data.frame(
      excess_return = as.numeric(returns_aligned),
      mkt_rf = as.numeric(factors_aligned$Mkt.RF),
      smb = as.numeric(factors_aligned$SMB),
      hml = as.numeric(factors_aligned$HML)
    )
    
    # Three-factor model
    ff3_model <- lm(excess_return ~ mkt_rf + smb + hml, data = model_data)
    
    results$single_asset <- list(
      model = ff3_model,
      alpha = coef(ff3_model)[1],
      beta_market = coef(ff3_model)[2],
      beta_size = coef(ff3_model)[3],
      beta_value = coef(ff3_model)[4],
      r_squared = summary(ff3_model)$r.squared,
      tracking_error = sd(residuals(ff3_model)),
      information_ratio = coef(ff3_model)[1] / sd(residuals(ff3_model))
    )
    
  } else {
    # Multiple assets
    for (asset in colnames(returns_aligned)) {
      model_data <- data.frame(
        excess_return = as.numeric(returns_aligned[, asset]),
        mkt_rf = as.numeric(factors_aligned$Mkt.RF),
        smb = as.numeric(factors_aligned$SMB),
        hml = as.numeric(factors_aligned$HML)
      )
      
      ff3_model <- lm(excess_return ~ mkt_rf + smb + hml, data = model_data)
      
      results[[asset]] <- list(
        model = ff3_model,
        alpha = coef(ff3_model)[1],
        beta_market = coef(ff3_model)[2],
        beta_size = coef(ff3_model)[3],
        beta_value = coef(ff3_model)[4],
        r_squared = summary(ff3_model)$r.squared,
        tracking_error = sd(residuals(ff3_model)),
        information_ratio = coef(ff3_model)[1] / sd(residuals(ff3_model))
      )
    }
  }
  
  return(results)
}

#' Principal Component Analysis for factor extraction
#' @param data Matrix of return series
#' @param n_factors Integer: number of factors to extract
#' @return List with PCA results
factor_analysis_pca <- function(data, n_factors = 3) {
  
  # Remove NA values
  clean_data <- na.omit(data)
  
  # Perform PCA
  pca_result <- prcomp(clean_data, center = TRUE, scale. = TRUE)
  
  # Extract factor loadings and scores
  loadings <- pca_result$rotation[, 1:n_factors]
  factor_scores <- pca_result$x[, 1:n_factors]
  
  # Calculate explained variance
  explained_var <- (pca_result$sdev^2 / sum(pca_result$sdev^2))[1:n_factors]
  cumulative_var <- cumsum(explained_var)
  
  result <- list(
    pca_object = pca_result,
    loadings = loadings,
    factor_scores = factor_scores,
    explained_variance = explained_var,
    cumulative_variance = cumulative_var,
    eigenvalues = pca_result$sdev^2,
    n_factors = n_factors
  )
  
  return(result)
}

# =============================================================================
# Statistical Arbitrage Specific Models
# =============================================================================

#' Pairs trading signal generation
#' @param price1 First asset price series
#' @param price2 Second asset price series
#' @param lookback Integer: lookback period for calculation
#' @param entry_threshold Numeric: z-score threshold for entry
#' @param exit_threshold Numeric: z-score threshold for exit
#' @return List with pairs trading signals
pairs_trading_signals <- function(price1, price2, lookback = 252, 
                                 entry_threshold = 2, exit_threshold = 0.5) {
  
  # Calculate log price ratio
  log_ratio <- log(price1) - log(price2)
  
  # Calculate rolling mean and standard deviation
  rolling_mean <- rollapply(log_ratio, width = lookback, FUN = mean, align = "right", fill = NA)
  rolling_sd <- rollapply(log_ratio, width = lookback, FUN = sd, align = "right", fill = NA)
  
  # Calculate z-score
  z_score <- (log_ratio - rolling_mean) / rolling_sd
  
  # Generate signals
  signals <- rep(0, length(z_score))
  
  # Long signal: z-score < -entry_threshold (price1 undervalued relative to price2)
  signals[z_score < -entry_threshold] <- 1
  
  # Short signal: z-score > entry_threshold (price1 overvalued relative to price2)
  signals[z_score > entry_threshold] <- -1
  
  # Exit signals: z-score returns to mean
  exit_long <- which(signals == 1 & lead(abs(z_score), 1) < exit_threshold)
  exit_short <- which(signals == -1 & lead(abs(z_score), 1) < exit_threshold)
  
  if (length(exit_long) > 0) signals[exit_long + 1] <- 0
  if (length(exit_short) > 0) signals[exit_short + 1] <- 0
  
  # Cointegration test
  coint_test <- cointegration_test(as.numeric(price1), as.numeric(price2))
  
  result <- list(
    log_ratio = log_ratio,
    rolling_mean = rolling_mean,
    rolling_sd = rolling_sd,
    z_score = z_score,
    signals = signals,
    entry_threshold = entry_threshold,
    exit_threshold = exit_threshold,
    cointegration = coint_test,
    statistics = list(
      mean_reversion_speed = -coef(lm(diff(log_ratio) ~ lag(log_ratio, 1)))[2],
      half_life = -log(2) / coef(lm(diff(log_ratio) ~ lag(log_ratio, 1)))[2],
      correlation = cor(price1, price2, use = "complete.obs")
    )
  )
  
  return(result)
}

#' Mean reversion model for single asset
#' @param prices Price series
#' @param lookback Integer: lookback period
#' @param model_type Character: "ou", "ar1", "linear"
#' @return List with mean reversion model results
mean_reversion_model <- function(prices, lookback = 252, model_type = "ou") {
  
  # Calculate log prices and returns
  log_prices <- log(prices)
  returns <- diff(log_prices)
  
  if (model_type == "ou") {
    # Ornstein-Uhlenbeck process
    # dX = θ(μ - X)dt + σdW
    
    # Estimate parameters using MLE
    X <- as.numeric(log_prices[-1])
    X_lag <- as.numeric(log_prices[-length(log_prices)])
    dt <- 1/252  # daily data
    
    # Linear regression: X[t] = a + b*X[t-1] + ε
    ou_model <- lm(X ~ X_lag)
    a <- coef(ou_model)[1]
    b <- coef(ou_model)[2]
    
    # Parameter conversions
    theta <- -log(b) / dt
    mu <- a / (1 - b)
    sigma <- summary(ou_model)$sigma * sqrt(-2 * log(b) / dt / (1 - b^2))
    
    # Half-life of mean reversion
    half_life <- log(2) / theta
    
    result <- list(
      model_type = "Ornstein-Uhlenbeck",
      theta = theta,
      mu = mu,
      sigma = sigma,
      half_life = half_life,
      regression = ou_model,
      log_likelihood = logLik(ou_model)
    )
    
  } else if (model_type == "ar1") {
    # AR(1) model
    ar1_model <- arima(log_prices, order = c(1, 0, 0))
    
    result <- list(
      model_type = "AR(1)",
      model = ar1_model,
      phi = coef(ar1_model)[1],
      intercept = coef(ar1_model)[2],
      sigma = sqrt(ar1_model$sigma2),
      half_life = -log(2) / log(abs(coef(ar1_model)[1])),
      aic = AIC(ar1_model)
    )
    
  } else if (model_type == "linear") {
    # Linear trend with mean reversion
    time_index <- 1:length(log_prices)
    trend_model <- lm(log_prices ~ time_index)
    detrended <- residuals(trend_model)
    
    # Test detrended series for mean reversion
    mr_test <- ur.df(detrended, type = "none")
    
    result <- list(
      model_type = "Linear Trend + Mean Reversion",
      trend_model = trend_model,
      detrended_series = detrended,
      mean_reversion_test = mr_test,
      trend_coefficient = coef(trend_model)[2],
      mean_reversion_speed = -coef(lm(diff(detrended) ~ lag(detrended, 1)))[2]
    )
  }
  
  return(result)
}

#' Momentum and mean reversion regime identification
#' @param returns Return series
#' @param window Integer: rolling window size
#' @return List with regime identification results
regime_identification <- function(returns, window = 60) {
  
  # Calculate rolling statistics
  rolling_autocorr <- rollapply(returns, width = window, 
                               FUN = function(x) cor(x[-length(x)], x[-1], use = "complete.obs"),
                               align = "right", fill = NA)
  
  rolling_variance <- rollapply(returns, width = window, FUN = var, align = "right", fill = NA)
  
  rolling_skewness <- rollapply(returns, width = window, 
                               FUN = function(x) skewness(x, na.rm = TRUE),
                               align = "right", fill = NA)
  
  rolling_kurtosis <- rollapply(returns, width = window, 
                               FUN = function(x) kurtosis(x, na.rm = TRUE),
                               align = "right", fill = NA)
  
  # Regime classification
  # Mean reversion: negative autocorrelation
  # Momentum: positive autocorrelation
  regimes <- ifelse(rolling_autocorr < -0.1, "Mean Reversion",
                   ifelse(rolling_autocorr > 0.1, "Momentum", "Neutral"))
  
  # Volatility regimes
  vol_quantiles <- quantile(rolling_variance, c(0.33, 0.67), na.rm = TRUE)
  vol_regimes <- ifelse(rolling_variance < vol_quantiles[1], "Low Vol",
                       ifelse(rolling_variance > vol_quantiles[2], "High Vol", "Normal Vol"))
  
  result <- list(
    rolling_autocorr = rolling_autocorr,
    rolling_variance = rolling_variance,
    rolling_skewness = rolling_skewness,
    rolling_kurtosis = rolling_kurtosis,
    regimes = regimes,
    vol_regimes = vol_regimes,
    regime_summary = table(regimes),
    vol_regime_summary = table(vol_regimes)
  )
  
  return(result)
}

# =============================================================================
# Model Validation and Performance Metrics
# =============================================================================

#' Cross-validation for time series models
#' @param data Time series data
#' @param model_func Function that fits the model
#' @param h Integer: forecast horizon
#' @param window Integer: initial window size
#' @return List with cross-validation results
time_series_cv <- function(data, model_func, h = 1, window = 100) {
  
  n <- length(data)
  forecasts <- numeric(n - window)
  actuals <- numeric(n - window)
  
  for (i in 1:(n - window - h + 1)) {
    # Training data
    train_data <- data[i:(i + window - 1)]
    
    # Fit model and forecast
    model <- model_func(train_data)
    forecast_val <- forecast(model, h = h)$mean[h]
    
    # Store results
    forecasts[i] <- forecast_val
    actuals[i] <- data[i + window + h - 1]
  }
  
  # Calculate performance metrics
  errors <- actuals - forecasts
  mae <- mean(abs(errors), na.rm = TRUE)
  rmse <- sqrt(mean(errors^2, na.rm = TRUE))
  mape <- mean(abs(errors / actuals) * 100, na.rm = TRUE)
  
  result <- list(
    forecasts = forecasts,
    actuals = actuals,
    errors = errors,
    mae = mae,
    rmse = rmse,
    mape = mape,
    hit_rate = mean((forecasts > 0 & actuals > 0) | (forecasts < 0 & actuals < 0), na.rm = TRUE)
  )
  
  return(result)
}

#' Model comparison and selection
#' @param models List of fitted models
#' @param data Original data
#' @return Data frame with model comparison metrics
model_comparison <- function(models, data) {
  
  comparison_table <- data.frame(
    Model = character(),
    AIC = numeric(),
    BIC = numeric(),
    Log_Likelihood = numeric(),
    RMSE = numeric(),
    MAE = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:length(models)) {
    model_name <- names(models)[i]
    model <- models[[i]]
    
    # Extract metrics based on model type
    if (inherits(model, "lm")) {
      aic_val <- AIC(model)
      bic_val <- BIC(model)
      loglik_val <- as.numeric(logLik(model))
      residuals_val <- residuals(model)
    } else if (inherits(model, "Arima")) {
      aic_val <- AIC(model)
      bic_val <- BIC(model)
      loglik_val <- model$loglik
      residuals_val <- residuals(model)
    } else {
      # Generic handling
      aic_val <- NA
      bic_val <- NA
      loglik_val <- NA
      residuals_val <- rep(NA, length(data))
    }
    
    rmse_val <- sqrt(mean(residuals_val^2, na.rm = TRUE))
    mae_val <- mean(abs(residuals_val), na.rm = TRUE)
    
    comparison_table <- rbind(comparison_table, data.frame(
      Model = model_name,
      AIC = aic_val,
      BIC = bic_val,
      Log_Likelihood = loglik_val,
      RMSE = rmse_val,
      MAE = mae_val,
      stringsAsFactors = FALSE
    ))
  }
  
  # Rank models
  comparison_table$AIC_Rank <- rank(comparison_table$AIC)
  comparison_table$BIC_Rank <- rank(comparison_table$BIC)
  comparison_table$RMSE_Rank <- rank(comparison_table$RMSE)
  
  return(comparison_table)
}

# =============================================================================
# Utility Functions
# =============================================================================

#' Generate comprehensive model diagnostics report
#' @param model Fitted model object
#' @param data Original data
#' @return List with diagnostic plots and tests
model_diagnostics <- function(model, data) {
  
  # Extract residuals
  if (inherits(model, "lm")) {
    residuals_val <- residuals(model)
    fitted_val <- fitted(model)
  } else if (inherits(model, "Arima")) {
    residuals_val <- residuals(model)
    fitted_val <- fitted(model)
  } else {
    stop("Model type not supported for diagnostics")
  }
  
  # Statistical tests
  normality_test <- shapiro.test(residuals_val)
  autocorr_test <- Box.test(residuals_val, lag = 10, type = "Ljung-Box")
  
  # Plots (would be generated in actual implementation)
  plots_info <- list(
    residuals_vs_fitted = "Residuals vs Fitted Values",
    qq_plot = "Q-Q Plot of Residuals",
    autocorr_plot = "Autocorrelation Function of Residuals",
    histogram = "Histogram of Residuals"
  )
  
  result <- list(
    residuals = residuals_val,
    fitted = fitted_val,
    normality_test = normality_test,
    autocorrelation_test = autocorr_test,
    plots = plots_info,
    summary_stats = list(
      mean_residuals = mean(residuals_val, na.rm = TRUE),
      sd_residuals = sd(residuals_val, na.rm = TRUE),
      skewness_residuals = skewness(residuals_val, na.rm = TRUE),
      kurtosis_residuals = kurtosis(residuals_val, na.rm = TRUE)
    )
  )
  
  return(result)
}

#' Export results to various formats
#' @param results List of analysis results
#' @param output_dir Character: output directory
#' @param format Character: "csv", "rds", "json"
export_results <- function(results, output_dir = ".", format = "csv") {
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  
  if (format == "csv") {
    # Export data frames and matrices as CSV
    for (name in names(results)) {
      item <- results[[name]]
      if (is.data.frame(item) || is.matrix(item)) {
        filename <- file.path(output_dir, paste0(name, "_", timestamp, ".csv"))
        write.csv(item, filename, row.names = TRUE)
      }
    }
  } else if (format == "rds") {
    # Export entire results as RDS
    filename <- file.path(output_dir, paste0("analysis_results_", timestamp, ".rds"))
    saveRDS(results, filename)
  } else if (format == "json") {
    # Export as JSON (simplified)
    filename <- file.path(output_dir, paste0("analysis_results_", timestamp, ".json"))
    # Note: Would need jsonlite package for actual JSON export
    cat("JSON export would require jsonlite package\n")
  }
  
  cat("Results exported to:", output_dir, "\n")
}

# =============================================================================
# Example Usage and Testing
# =============================================================================

#' Run comprehensive statistical analysis pipeline
#' @param data_source Character: data source type
#' @param symbols Character vector: symbols to analyze
#' @return List with complete analysis results
run_statistical_analysis <- function(data_source = "csv", symbols = c("AAPL", "MSFT")) {
  
  cat("Starting comprehensive statistical analysis...\n")
  
  # Load and preprocess data
  cat("Loading data...\n")
  # raw_data <- load_financial_data(data_source, symbols)
  # processed_data <- preprocess_data(raw_data)
  
  # For demonstration, create synthetic data
  set.seed(123)
  dates <- seq(as.Date("2020-01-01"), as.Date("2023-12-31"), by = "day")
  dates <- dates[weekdays(dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
  
  # Generate correlated price series
  n <- length(dates)
  returns1 <- rnorm(n, 0.0005, 0.02)
  returns2 <- 0.7 * returns1 + sqrt(1 - 0.7^2) * rnorm(n, 0.0005, 0.02)
  
  prices1 <- 100 * cumprod(1 + returns1)
  prices2 <- 95 * cumprod(1 + returns2)
  
  price_data <- xts(cbind(AAPL = prices1, MSFT = prices2), order.by = dates)
  return_data <- diff(log(price_data))
  
  results <- list()
  
  # 1. Stationarity tests
  cat("Performing stationarity tests...\n")
  results$stationarity <- list(
    AAPL_returns = stationarity_tests(return_data$AAPL),
    MSFT_returns = stationarity_tests(return_data$MSFT)
  )
  
  # 2. Cointegration analysis
  cat("Testing for cointegration...\n")
  results$cointegration <- cointegration_test(price_data$AAPL, price_data$MSFT)
  
  # 3. Time series modeling
  cat("Fitting time series models...\n")
  results$arima_models <- list(
    AAPL = fit_arima_model(return_data$AAPL),
    MSFT = fit_arima_model(return_data$MSFT)
  )
  
  results$garch_models <- list(
    AAPL = fit_garch_model(return_data$AAPL),
    MSFT = fit_garch_model(return_data$MSFT)
  )
  
  # 4. VAR model
  cat("Fitting VAR model...\n")
  results$var_model <- fit_var_model(return_data)
  
  # 5. Pairs trading analysis
  cat("Analyzing pairs trading opportunities...\n")
  results$pairs_trading <- pairs_trading_signals(price_data$AAPL, price_data$MSFT)
  
  # 6. Mean reversion analysis
  cat("Analyzing mean reversion...\n")
  results$mean_reversion <- list(
    AAPL = mean_reversion_model(price_data$AAPL),
    MSFT = mean_reversion_model(price_data$MSFT)
  )
  
  # 7. Regime identification
  cat("Identifying market regimes...\n")
  results$regimes <- list(
    AAPL = regime_identification(return_data$AAPL),
    MSFT = regime_identification(return_data$MSFT)
  )
  
  cat("Analysis complete!\n")
  return(results)
}

# =============================================================================
# Script Execution
# =============================================================================

# Print header
cat("\n")
cat("=" * 80, "\n")
cat("Statistical Models and Analysis for Statistical Arbitrage\n")
cat("Initialized successfully with", length(required_packages), "packages\n")
cat("=" * 80, "\n")
cat("\n")

# Example: Run analysis if script is executed directly
if (interactive() || !exists(".testing")) {
  cat("To run a complete analysis, use:\n")
  cat("results <- run_statistical_analysis()\n")
  cat("\nAvailable functions:\n")
  cat("- load_financial_data()\n")
  cat("- stationarity_tests()\n")
  cat("- cointegration_test()\n")
  cat("- fit_arima_model()\n")
  cat("- fit_garch_model()\n")
  cat("- pairs_trading_signals()\n")
  cat("- mean_reversion_model()\n")
  cat("- And many more...\n")
}
