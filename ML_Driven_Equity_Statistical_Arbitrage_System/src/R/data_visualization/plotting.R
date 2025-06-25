# =============================================================================
# Data Visualization and Plotting for Statistical Arbitrage
# =============================================================================
# Purpose: High-quality visualizations for exploratory data analysis, model
#          outputs, backtesting results, and research insights
# Author: Statistical Arbitrage System
# Date: 2025-06-24
# =============================================================================

# Required packages
required_packages <- c(
  "ggplot2", "plotly", "dygraphs", "lattice", "corrplot", "pheatmap",
  "ggcorrplot", "GGally", "ggridges", "ggdist", "patchwork", "cowplot",
  "viridis", "RColorBrewer", "scales", "gridExtra", "grid", "gtable",
  "xts", "zoo", "lubridate", "dplyr", "tidyr", "reshape2", "stringr",
  "PerformanceAnalytics", "quantmod", "TTR", "htmlwidgets", "DT",
  "knitr", "kableExtra", "formattable", "sparkline", "networkD3",
  "treemap", "sunburstR", "leaflet", "ggforce", "ggrepel", "gganimate",
  "plotrix", "VennDiagram", "UpSetR", "circlize", "ComplexHeatmap"
)

# Function to install and load packages
load_packages <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
    install.packages(new_packages, dependencies = TRUE)
  }
  
  for (pkg in packages) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
  cat("All visualization packages loaded successfully.\n")
}

# Load all required packages
load_packages(required_packages)

# =============================================================================
# Theme and Color Palette Setup
# =============================================================================

# Custom ggplot2 theme for financial data
theme_financial <- function(base_size = 12, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Grid and background
      panel.grid.major = element_line(color = "grey90", size = 0.5),
      panel.grid.minor = element_line(color = "grey95", size = 0.3),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      
      # Axes
      axis.line = element_line(color = "grey30", size = 0.5),
      axis.text = element_text(color = "grey30", size = rel(0.9)),
      axis.title = element_text(color = "grey30", size = rel(1.1), face = "bold"),
      axis.ticks = element_line(color = "grey30", size = 0.3),
      
      # Titles and labels
      plot.title = element_text(color = "grey20", size = rel(1.4), face = "bold", 
                               hjust = 0, margin = margin(b = 20)),
      plot.subtitle = element_text(color = "grey40", size = rel(1.1), 
                                  hjust = 0, margin = margin(b = 15)),
      plot.caption = element_text(color = "grey50", size = rel(0.8), hjust = 1),
      
      # Legend
      legend.position = "bottom",
      legend.title = element_text(color = "grey30", size = rel(1.0), face = "bold"),
      legend.text = element_text(color = "grey30", size = rel(0.9)),
      legend.background = element_rect(fill = "white", color = "grey80"),
      legend.key = element_rect(fill = "white", color = NA),
      
      # Facets
      strip.background = element_rect(fill = "grey95", color = "grey80"),
      strip.text = element_text(color = "grey30", size = rel(1.0), face = "bold"),
      
      # Margins
      plot.margin = margin(20, 20, 20, 20)
    )
}

# Color palettes for different visualization types
financial_colors <- list(
  # Main colors
  primary = "#2E86C1",      # Blue
  secondary = "#E74C3C",    # Red
  success = "#27AE60",      # Green
  warning = "#F39C12",      # Orange
  info = "#8E44AD",         # Purple
  dark = "#2C3E50",         # Dark grey
  light = "#ECF0F1",        # Light grey
  
  # Price movement colors
  bullish = "#27AE60",      # Green for up moves
  bearish = "#E74C3C",      # Red for down moves
  neutral = "#95A5A6",      # Grey for neutral
  
  # Performance colors
  outperform = "#2ECC71",   # Bright green
  underperform = "#E67E22", # Orange
  benchmark = "#34495E",    # Dark grey
  
  # Risk colors
  low_risk = "#2ECC71",     # Green
  med_risk = "#F39C12",     # Orange  
  high_risk = "#E74C3C",    # Red
  
  # Multi-series palette
  series = c("#2E86C1", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD", 
            "#17A2B8", "#28A745", "#FFC107", "#DC3545", "#6F42C1"),
  
  # Diverging palette for correlation/spread plots
  diverging = c("#E74C3C", "#F8C471", "#F7DC6F", "#ABEBC6", "#27AE60"),
  
  # Sequential palette for heatmaps
  sequential = c("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", 
                "#4292C6", "#2171B5", "#08519C", "#08306B")
)

# Set default ggplot2 theme
theme_set(theme_financial())

# =============================================================================
# Price and Return Visualization Functions
# =============================================================================

#' Plot price time series with volume
#' @param data xts object with OHLCV data or named columns
#' @param symbol Character: symbol name for title
#' @param start_date Date: start date for plotting
#' @param end_date Date: end date for plotting
#' @param interactive Logical: whether to create interactive plot
#' @return ggplot or plotly object
plot_price_series <- function(data, symbol = "Asset", start_date = NULL, end_date = NULL, 
                             interactive = TRUE) {
  
  # Filter data by date range if specified
  if (!is.null(start_date) || !is.null(end_date)) {
    if (is.null(start_date)) start_date <- start(data)
    if (is.null(end_date)) end_date <- end(data)
    data <- data[paste(start_date, end_date, sep = "/")]
  }
  
  # Convert to data frame for ggplot
  df <- data.frame(
    Date = index(data),
    Price = as.numeric(if("Close" %in% colnames(data)) data$Close else data[,1]),
    Volume = if("Volume" %in% colnames(data)) as.numeric(data$Volume) else NULL
  )
  
  # Create price plot
  p1 <- ggplot(df, aes(x = Date, y = Price)) +
    geom_line(color = financial_colors$primary, size = 0.8) +
    labs(
      title = paste("Price Chart:", symbol),
      subtitle = paste("Period:", min(df$Date), "to", max(df$Date)),
      x = "Date",
      y = "Price",
      caption = "Data: Financial Time Series"
    ) +
    scale_x_date(labels = date_format("%Y-%m"), date_breaks = "3 months") +
    scale_y_continuous(labels = dollar_format()) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Add volume subplot if available
  if (!is.null(df$Volume)) {
    p2 <- ggplot(df, aes(x = Date, y = Volume)) +
      geom_col(fill = financial_colors$info, alpha = 0.7) +
      labs(x = "Date", y = "Volume") +
      scale_x_date(labels = date_format("%Y-%m"), date_breaks = "3 months") +
      scale_y_continuous(labels = comma_format()) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Combine plots
    combined_plot <- p1 / p2 + plot_layout(heights = c(3, 1))
  } else {
    combined_plot <- p1
  }
  
  if (interactive) {
    return(ggplotly(combined_plot))
  } else {
    return(combined_plot)
  }
}

#' Plot candlestick chart
#' @param data xts object with OHLCV data
#' @param symbol Character: symbol name
#' @param n_days Integer: number of recent days to plot
#' @param interactive Logical: create interactive plot
#' @return plotly object
plot_candlestick <- function(data, symbol = "Asset", n_days = 252, interactive = TRUE) {
  
  # Take last n_days of data
  if (nrow(data) > n_days) {
    data <- tail(data, n_days)
  }
  
  # Ensure we have OHLC columns
  if (!all(c("Open", "High", "Low", "Close") %in% colnames(data))) {
    stop("Data must contain Open, High, Low, Close columns")
  }
  
  if (interactive) {
    # Create interactive candlestick with plotly
    fig <- plot_ly(
      x = index(data),
      type = "candlestick",
      open = ~data$Open,
      high = ~data$High,
      low = ~data$Low,
      close = ~data$Close,
      increasing = list(line = list(color = financial_colors$bullish)),
      decreasing = list(line = list(color = financial_colors$bearish))
    ) %>%
      layout(
        title = paste("Candlestick Chart:", symbol),
        xaxis = list(title = "Date", rangeslider = list(visible = F)),
        yaxis = list(title = "Price"),
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
      )
    
    return(fig)
  } else {
    # Create static candlestick with ggplot2
    df <- data.frame(
      Date = index(data),
      Open = as.numeric(data$Open),
      High = as.numeric(data$High),
      Low = as.numeric(data$Low),
      Close = as.numeric(data$Close)
    )
    
    df$Direction <- ifelse(df$Close >= df$Open, "Up", "Down")
    
    p <- ggplot(df, aes(x = Date)) +
      geom_segment(aes(y = Low, yend = High), color = "black", size = 0.3) +
      geom_rect(aes(ymin = pmin(Open, Close), ymax = pmax(Open, Close),
                   fill = Direction), alpha = 0.8) +
      scale_fill_manual(values = c("Up" = financial_colors$bullish, 
                                  "Down" = financial_colors$bearish)) +
      labs(
        title = paste("Candlestick Chart:", symbol),
        x = "Date",
        y = "Price",
        fill = "Direction"
      ) +
      scale_y_continuous(labels = dollar_format()) +
      theme(legend.position = "none")
    
    return(p)
  }
}

#' Plot return distributions
#' @param returns xts object or numeric vector of returns
#' @param symbol Character: symbol name
#' @param return_type Character: "simple" or "log"
#' @return ggplot object
plot_return_distribution <- function(returns, symbol = "Asset", return_type = "log") {
  
  # Convert to numeric if xts
  if (is.xts(returns)) {
    returns_vec <- as.numeric(returns)
  } else {
    returns_vec <- returns
  }
  
  # Remove NA values
  returns_vec <- returns_vec[!is.na(returns_vec)]
  
  # Calculate statistics
  mean_ret <- mean(returns_vec)
  sd_ret <- sd(returns_vec)
  skew_ret <- skewness(returns_vec)
  kurt_ret <- kurtosis(returns_vec)
  
  # Create data frame
  df <- data.frame(Returns = returns_vec)
  
  # Create the plot
  p <- ggplot(df, aes(x = Returns)) +
    geom_histogram(aes(y = ..density..), bins = 50, fill = financial_colors$primary, 
                   alpha = 0.7, color = "white") +
    geom_density(color = financial_colors$secondary, size = 1.2) +
    geom_vline(xintercept = mean_ret, color = financial_colors$warning, 
               linetype = "dashed", size = 1) +
    geom_vline(xintercept = 0, color = "black", linetype = "solid", size = 0.5) +
    labs(
      title = paste("Return Distribution:", symbol),
      subtitle = sprintf("Mean: %.4f, SD: %.4f, Skew: %.2f, Kurt: %.2f", 
                        mean_ret, sd_ret, skew_ret, kurt_ret),
      x = paste(str_to_title(return_type), "Returns"),
      y = "Density",
      caption = "Vertical dashed line = mean return"
    ) +
    scale_x_continuous(labels = percent_format(accuracy = 0.1))
  
  return(p)
}

#' Plot rolling statistics
#' @param data xts object with price or return data
#' @param window Integer: rolling window size
#' @param stats Character vector: statistics to plot
#' @param symbol Character: symbol name
#' @return ggplot object
plot_rolling_stats <- function(data, window = 60, 
                              stats = c("mean", "volatility", "sharpe"), 
                              symbol = "Asset") {
  
  # Calculate rolling statistics
  rolling_data <- data.frame(Date = index(data))
  
  if ("mean" %in% stats) {
    rolling_data$Mean <- as.numeric(rollapply(data, width = window, FUN = mean, 
                                            align = "right", fill = NA))
  }
  
  if ("volatility" %in% stats) {
    rolling_data$Volatility <- as.numeric(rollapply(data, width = window, FUN = sd, 
                                                   align = "right", fill = NA)) * sqrt(252)
  }
  
  if ("sharpe" %in% stats) {
    rolling_mean <- rollapply(data, width = window, FUN = mean, align = "right", fill = NA)
    rolling_sd <- rollapply(data, width = window, FUN = sd, align = "right", fill = NA)
    rolling_data$Sharpe <- as.numeric((rolling_mean / rolling_sd) * sqrt(252))
  }
  
  if ("skewness" %in% stats) {
    rolling_data$Skewness <- as.numeric(rollapply(data, width = window, 
                                                 FUN = function(x) skewness(x, na.rm = TRUE), 
                                                 align = "right", fill = NA))
  }
  
  if ("kurtosis" %in% stats) {
    rolling_data$Kurtosis <- as.numeric(rollapply(data, width = window, 
                                                 FUN = function(x) kurtosis(x, na.rm = TRUE), 
                                                 align = "right", fill = NA))
  }
  
  # Reshape for plotting
  plot_data <- gather(rolling_data, key = "Statistic", value = "Value", -Date)
  plot_data <- plot_data[!is.na(plot_data$Value), ]
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = Date, y = Value, color = Statistic)) +
    geom_line(size = 0.8) +
    facet_wrap(~Statistic, scales = "free_y", ncol = 2) +
    scale_color_manual(values = financial_colors$series) +
    labs(
      title = paste("Rolling Statistics:", symbol),
      subtitle = paste("Window:", window, "days"),
      x = "Date",
      y = "Value",
      color = "Statistic"
    ) +
    theme(legend.position = "none")
  
  return(p)
}

# =============================================================================
# Correlation and Relationship Analysis
# =============================================================================

#' Plot correlation matrix heatmap
#' @param data Matrix or data frame of returns
#' @param method Character: correlation method
#' @param interactive Logical: create interactive heatmap
#' @return ggplot or plotly object
plot_correlation_matrix <- function(data, method = "pearson", interactive = FALSE) {
  
  # Calculate correlation matrix
  cor_matrix <- cor(data, use = "complete.obs", method = method)
  
  if (interactive) {
    # Interactive heatmap with plotly
    fig <- plot_ly(
      x = colnames(cor_matrix),
      y = rownames(cor_matrix),
      z = cor_matrix,
      type = "heatmap",
      colorscale = list(
        c(0, financial_colors$bearish),
        c(0.5, "white"),
        c(1, financial_colors$bullish)
      ),
      zmid = 0,
      text = round(cor_matrix, 3),
      texttemplate = "%{text}",
      textfont = list(size = 10)
    ) %>%
      layout(
        title = paste("Correlation Matrix (", str_to_title(method), ")", sep = ""),
        xaxis = list(title = ""),
        yaxis = list(title = "")
      )
    
    return(fig)
  } else {
    # Static heatmap with ggplot2
    melted_cor <- melt(cor_matrix)
    
    p <- ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
      geom_tile(color = "white", size = 0.5) +
      geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
      scale_fill_gradient2(
        low = financial_colors$bearish,
        mid = "white",
        high = financial_colors$bullish,
        midpoint = 0,
        limit = c(-1, 1),
        space = "Lab",
        name = "Correlation"
      ) +
      labs(
        title = paste("Correlation Matrix (", str_to_title(method), ")", sep = ""),
        x = "", y = ""
      ) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
      ) +
      coord_fixed()
    
    return(p)
  }
}

#' Plot scatter plot matrix
#' @param data Data frame of returns or prices
#' @param sample_size Integer: number of observations to sample
#' @return ggplot object
plot_scatter_matrix <- function(data, sample_size = 1000) {
  
  # Sample data if too large
  if (nrow(data) > sample_size) {
    data <- data[sample(nrow(data), sample_size), ]
  }
  
  # Create scatter plot matrix
  p <- ggpairs(
    data,
    lower = list(continuous = wrap("points", alpha = 0.5, size = 0.8, 
                                  color = financial_colors$primary)),
    upper = list(continuous = wrap("cor", size = 4, color = financial_colors$dark)),
    diag = list(continuous = wrap("densityDiag", fill = financial_colors$primary, 
                                 alpha = 0.7))
  ) +
    theme_minimal() +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 8)
    )
  
  return(p)
}

#' Plot rolling correlation between two assets
#' @param x First time series
#' @param y Second time series  
#' @param window Integer: rolling window size
#' @param symbol1 Character: name of first asset
#' @param symbol2 Character: name of second asset
#' @return ggplot object
plot_rolling_correlation <- function(x, y, window = 60, symbol1 = "Asset1", symbol2 = "Asset2") {
  
  # Align data
  combined_data <- merge(x, y, all = FALSE)
  
  # Calculate rolling correlation
  rolling_cor <- rollapply(combined_data, width = window, 
                          FUN = function(data) cor(data[,1], data[,2], use = "complete.obs"),
                          align = "right", fill = NA, by.column = FALSE)
  
  # Create data frame for plotting
  df <- data.frame(
    Date = index(rolling_cor),
    Correlation = as.numeric(rolling_cor)
  )
  
  # Remove NA values
  df <- df[!is.na(df$Correlation), ]
  
  # Create the plot
  p <- ggplot(df, aes(x = Date, y = Correlation)) +
    geom_line(color = financial_colors$primary, size = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.7) +
    geom_hline(yintercept = c(-0.5, 0.5), linetype = "dotted", color = "grey60") +
    labs(
      title = paste("Rolling Correlation:", symbol1, "vs", symbol2),
      subtitle = paste("Window:", window, "days"),
      x = "Date",
      y = "Correlation",
      caption = "Horizontal lines at 0, ±0.5"
    ) +
    scale_y_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.25)) +
    theme(panel.grid.minor.y = element_blank())
  
  return(p)
}

# =============================================================================
# Statistical Model Visualization
# =============================================================================

#' Plot pairs trading analysis
#' @param prices1 Price series for asset 1
#' @param prices2 Price series for asset 2
#' @param signals Trading signals from pairs_trading_signals()
#' @param symbol1 Character: name of asset 1
#' @param symbol2 Character: name of asset 2
#' @return List of ggplot objects
plot_pairs_analysis <- function(prices1, prices2, signals, symbol1 = "Asset1", symbol2 = "Asset2") {
  
  # Prepare data
  df <- data.frame(
    Date = index(prices1),
    Price1 = as.numeric(prices1),
    Price2 = as.numeric(prices2),
    LogRatio = as.numeric(signals$log_ratio),
    ZScore = as.numeric(signals$z_score),
    Signal = signals$signals,
    RollingMean = as.numeric(signals$rolling_mean),
    UpperBound = as.numeric(signals$rolling_mean + signals$entry_threshold * signals$rolling_sd),
    LowerBound = as.numeric(signals$rolling_mean - signals$entry_threshold * signals$rolling_sd)
  )
  
  # Remove NA values
  df <- df[complete.cases(df), ]
  
  # Plot 1: Price series
  p1 <- df %>%
    select(Date, Price1, Price2) %>%
    gather(key = "Asset", value = "Price", -Date) %>%
    mutate(Asset = ifelse(Asset == "Price1", symbol1, symbol2)) %>%
    ggplot(aes(x = Date, y = Price, color = Asset)) +
    geom_line(size = 0.8) +
    scale_color_manual(values = c(financial_colors$primary, financial_colors$secondary)) +
    labs(
      title = "Price Series",
      x = "Date",
      y = "Price",
      color = "Asset"
    ) +
    scale_y_continuous(labels = dollar_format())
  
  # Plot 2: Log ratio and signals
  p2 <- ggplot(df, aes(x = Date)) +
    geom_ribbon(aes(ymin = LowerBound, ymax = UpperBound), 
                fill = "grey80", alpha = 0.5) +
    geom_line(aes(y = LogRatio), color = financial_colors$dark, size = 0.8) +
    geom_line(aes(y = RollingMean), color = financial_colors$warning, 
              linetype = "dashed", size = 0.6) +
    geom_point(data = df[df$Signal == 1, ], aes(y = LogRatio), 
               color = financial_colors$bullish, size = 2, shape = 24) +
    geom_point(data = df[df$Signal == -1, ], aes(y = LogRatio), 
               color = financial_colors$bearish, size = 2, shape = 25) +
    labs(
      title = "Log Price Ratio and Trading Signals",
      subtitle = "Triangles indicate entry signals",
      x = "Date",
      y = "Log Ratio",
      caption = "Grey band = entry thresholds, dashed line = rolling mean"
    )
  
  # Plot 3: Z-score
  p3 <- ggplot(df, aes(x = Date, y = ZScore)) +
    geom_line(color = financial_colors$primary, size = 0.8) +
    geom_hline(yintercept = c(-signals$entry_threshold, signals$entry_threshold), 
               linetype = "dashed", color = financial_colors$bearish) +
    geom_hline(yintercept = c(-signals$exit_threshold, signals$exit_threshold), 
               linetype = "dotted", color = financial_colors$warning) +
    geom_hline(yintercept = 0, color = "black", alpha = 0.7) +
    geom_point(data = df[df$Signal != 0, ], aes(color = factor(Signal)), size = 1.5) +
    scale_color_manual(values = c("-1" = financial_colors$bearish, 
                                 "1" = financial_colors$bullish),
                      name = "Signal", labels = c("Short", "Long")) +
    labs(
      title = "Z-Score and Entry/Exit Levels",
      x = "Date",
      y = "Z-Score",
      caption = "Dashed lines = entry levels, dotted lines = exit levels"
    )
  
  # Combine plots
  combined_plot <- (p1 / p2 / p3) + plot_layout(heights = c(1, 1, 1))
  
  return(list(
    combined = combined_plot,
    prices = p1,
    log_ratio = p2,
    z_score = p3
  ))
}

#' Plot ARIMA model diagnostics
#' @param arima_result Result from fit_arima_model()
#' @param symbol Character: asset symbol
#' @return ggplot object
plot_arima_diagnostics <- function(arima_result, symbol = "Asset") {
  
  model <- arima_result$model
  residuals_data <- as.numeric(arima_result$residuals)
  fitted_data <- as.numeric(arima_result$fitted)
  
  # Create diagnostic plots
  df <- data.frame(
    Time = 1:length(residuals_data),
    Residuals = residuals_data,
    Fitted = fitted_data,
    StandardizedResiduals = residuals_data / sd(residuals_data, na.rm = TRUE)
  )
  
  # Plot 1: Residuals vs Time
  p1 <- ggplot(df, aes(x = Time, y = Residuals)) +
    geom_line(color = financial_colors$primary, alpha = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(title = "Residuals vs Time", x = "Time", y = "Residuals")
  
  # Plot 2: ACF of residuals
  acf_data <- acf(residuals_data, plot = FALSE, na.action = na.pass)
  acf_df <- data.frame(
    Lag = acf_data$lag[-1],
    ACF = acf_data$acf[-1]
  )
  
  p2 <- ggplot(acf_df, aes(x = Lag, y = ACF)) +
    geom_col(fill = financial_colors$primary, alpha = 0.7) +
    geom_hline(yintercept = c(-1.96/sqrt(length(residuals_data)), 
                             1.96/sqrt(length(residuals_data))), 
               linetype = "dashed", color = financial_colors$bearish) +
    labs(title = "ACF of Residuals", x = "Lag", y = "Autocorrelation")
  
  # Plot 3: Q-Q plot
  p3 <- ggplot(df, aes(sample = StandardizedResiduals)) +
    geom_qq(color = financial_colors$primary, alpha = 0.7) +
    geom_qq_line(color = financial_colors$secondary, linetype = "dashed") +
    labs(title = "Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles")
  
  # Plot 4: Histogram of residuals
  p4 <- ggplot(df, aes(x = StandardizedResiduals)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = financial_colors$primary, 
                   alpha = 0.7, color = "white") +
    geom_density(color = financial_colors$secondary, size = 1) +
    stat_function(fun = dnorm, args = list(mean = 0, sd = 1), 
                 color = "black", linetype = "dashed") +
    labs(title = "Residual Distribution", x = "Standardized Residuals", y = "Density")
  
  # Combine plots
  combined_plot <- (p1 + p2) / (p3 + p4) +
    plot_annotation(
      title = paste("ARIMA Model Diagnostics:", symbol),
      subtitle = paste("Order:", paste(arimaorder(model), collapse = ", "))
    )
  
  return(combined_plot)
}

#' Plot GARCH model results
#' @param garch_result Result from fit_garch_model()
#' @param symbol Character: asset symbol
#' @return ggplot object
plot_garch_analysis <- function(garch_result, symbol = "Asset") {
  
  # Extract data
  dates <- as.Date(names(garch_result$volatility))
  volatility <- as.numeric(garch_result$volatility)
  std_residuals <- as.numeric(garch_result$standardized_residuals)
  
  df <- data.frame(
    Date = dates,
    Volatility = volatility,
    StdResiduals = std_residuals
  )
  
  # Plot 1: Conditional volatility
  p1 <- ggplot(df, aes(x = Date, y = Volatility)) +
    geom_line(color = financial_colors$primary, size = 0.8) +
    labs(
      title = "Conditional Volatility",
      x = "Date",
      y = "Volatility"
    ) +
    scale_y_continuous(labels = percent_format())
  
  # Plot 2: Standardized residuals
  p2 <- ggplot(df, aes(x = Date, y = StdResiduals)) +
    geom_line(color = financial_colors$secondary, alpha = 0.7) +
    geom_hline(yintercept = c(-2, 2), linetype = "dashed", 
               color = financial_colors$warning) +
    labs(
      title = "Standardized Residuals",
      x = "Date",
      y = "Standardized Residuals"
    )
  
  # Plot 3: ACF of squared standardized residuals
  acf_data <- acf(std_residuals^2, plot = FALSE, na.action = na.pass)
  acf_df <- data.frame(
    Lag = acf_data$lag[-1],
    ACF = acf_data$acf[-1]
  )
  
  p3 <- ggplot(acf_df, aes(x = Lag, y = ACF)) +
    geom_col(fill = financial_colors$info, alpha = 0.7) +
    geom_hline(yintercept = c(-1.96/sqrt(length(std_residuals)), 
                             1.96/sqrt(length(std_residuals))), 
               linetype = "dashed", color = financial_colors$bearish) +
    labs(title = "ACF of Squared Residuals", x = "Lag", y = "Autocorrelation")
  
  # Plot 4: Q-Q plot of standardized residuals
  p4 <- ggplot(df, aes(sample = StdResiduals)) +
    geom_qq(color = financial_colors$primary, alpha = 0.7) +
    geom_qq_line(color = financial_colors$secondary, linetype = "dashed") +
    labs(title = "Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles")
  
  # Combine plots
  combined_plot <- (p1 + p2) / (p3 + p4) +
    plot_annotation(
      title = paste("GARCH Model Analysis:", symbol),
      subtitle = paste("AIC:", round(garch_result$aic, 2), 
                      "| BIC:", round(garch_result$bic, 2))
    )
  
  return(combined_plot)
}

# =============================================================================
# Performance and Backtesting Visualization
# =============================================================================

#' Plot portfolio performance
#' @param returns xts object with portfolio returns
#' @param benchmark xts object with benchmark returns (optional)
#' @param portfolio_name Character: portfolio name
#' @param benchmark_name Character: benchmark name
#' @return ggplot object
plot_portfolio_performance <- function(returns, benchmark = NULL, 
                                     portfolio_name = "Portfolio", 
                                     benchmark_name = "Benchmark") {
  
  # Calculate cumulative returns
  cum_returns <- cumprod(1 + returns)
  
  # Prepare data
  df <- data.frame(
    Date = index(cum_returns),
    Portfolio = as.numeric(cum_returns)
  )
  
  if (!is.null(benchmark)) {
    # Align benchmark with portfolio
    benchmark_aligned <- benchmark[index(returns)]
    cum_benchmark <- cumprod(1 + benchmark_aligned)
    df$Benchmark <- as.numeric(cum_benchmark)
  }
  
  # Reshape for plotting
  plot_data <- gather(df, key = "Series", value = "CumReturn", -Date)
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = Date, y = CumReturn, color = Series)) +
    geom_line(size = 0.8) +
    scale_color_manual(values = c("Portfolio" = financial_colors$primary,
                                 "Benchmark" = financial_colors$secondary)) +
    labs(
      title = "Cumulative Performance",
      subtitle = paste("Portfolio:", portfolio_name),
      x = "Date",
      y = "Cumulative Return",
      color = "Series"
    ) +
    scale_y_continuous(labels = function(x) paste0(round((x-1)*100, 1), "%"))
  
  return(p)
}

#' Plot drawdown analysis
#' @param returns xts object with returns
#' @param portfolio_name Character: portfolio name
#' @return ggplot object
plot_drawdown_analysis <- function(returns, portfolio_name = "Portfolio") {
  
  # Calculate drawdowns
  cum_returns <- cumprod(1 + returns)
  peak <- cummax(cum_returns)
  drawdown <- (cum_returns - peak) / peak
  
  # Find major drawdown periods
  dd_threshold <- -0.05  # 5% drawdown threshold
  major_dd <- which(drawdown <= dd_threshold)
  
  # Prepare data
  df <- data.frame(
    Date = index(drawdown),
    Drawdown = as.numeric(drawdown),
    CumReturn = as.numeric(cum_returns)
  )
  
  # Plot 1: Cumulative returns with drawdown shading
  p1 <- ggplot(df, aes(x = Date)) +
    geom_line(aes(y = CumReturn), color = financial_colors$primary, size = 0.8) +
    geom_ribbon(data = df[df$Drawdown < 0, ], 
                aes(ymin = CumReturn * (1 + Drawdown), ymax = CumReturn),
                fill = financial_colors$bearish, alpha = 0.3) +
    labs(
      title = "Cumulative Returns with Drawdowns",
      x = "Date",
      y = "Cumulative Return"
    ) +
    scale_y_continuous(labels = function(x) paste0(round((x-1)*100, 1), "%"))
  
  # Plot 2: Drawdown time series
  p2 <- ggplot(df, aes(x = Date, y = Drawdown)) +
    geom_area(fill = financial_colors$bearish, alpha = 0.7) +
    geom_line(color = financial_colors$dark, size = 0.5) +
    geom_hline(yintercept = 0, color = "black") +
    labs(
      title = "Drawdown Time Series",
      x = "Date",
      y = "Drawdown"
    ) +
    scale_y_continuous(labels = percent_format())
  
  # Combine plots
  combined_plot <- p1 / p2 +
    plot_annotation(
      title = paste("Drawdown Analysis:", portfolio_name),
      subtitle = paste("Maximum Drawdown:", 
                      round(min(drawdown, na.rm = TRUE) * 100, 2), "%")
    )
  
  return(combined_plot)
}

#' Plot risk-return scatter
#' @param returns_matrix Matrix of returns for multiple assets/strategies
#' @param risk_free_rate Numeric: risk-free rate for Sharpe ratio
#' @return ggplot object
plot_risk_return_scatter <- function(returns_matrix, risk_free_rate = 0.02/252) {
  
  # Calculate risk-return metrics
  metrics <- data.frame(
    Asset = colnames(returns_matrix),
    Return = apply(returns_matrix, 2, function(x) mean(x, na.rm = TRUE) * 252),
    Risk = apply(returns_matrix, 2, function(x) sd(x, na.rm = TRUE) * sqrt(252)),
    stringsAsFactors = FALSE
  )
  
  metrics$Sharpe <- (metrics$Return - risk_free_rate) / metrics$Risk
  
  # Create the plot
  p <- ggplot(metrics, aes(x = Risk, y = Return)) +
    geom_point(aes(color = Sharpe, size = abs(Sharpe)), alpha = 0.8) +
    geom_text_repel(aes(label = Asset), size = 3, 
                   box.padding = 0.3, point.padding = 0.3) +
    scale_color_gradient2(
      low = financial_colors$bearish,
      mid = "white",
      high = financial_colors$bullish,
      midpoint = 1,
      name = "Sharpe\nRatio"
    ) +
    scale_size_continuous(range = c(2, 8), guide = "none") +
    labs(
      title = "Risk-Return Analysis",
      subtitle = "Bubble size indicates Sharpe ratio magnitude",
      x = "Annualized Risk (Volatility)",
      y = "Annualized Return",
      caption = paste("Risk-free rate:", percent(risk_free_rate * 252, accuracy = 0.1))
    ) +
    scale_x_continuous(labels = percent_format()) +
    scale_y_continuous(labels = percent_format()) +
    theme(legend.position = "right")
  
  return(p)
}

# =============================================================================
# Interactive Dashboard Functions
# =============================================================================

#' Create interactive time series dashboard
#' @param data xts object with multiple time series
#' @param title Character: dashboard title
#' @return dygraph object
create_time_series_dashboard <- function(data, title = "Financial Time Series Dashboard") {
  
  # Create interactive plot with dygraphs
  dygraph(data, main = title) %>%
    dyOptions(
      colors = financial_colors$series[1:ncol(data)],
      strokeWidth = 2,
      drawGrid = TRUE,
      gridLineColor = "lightgray",
      axisLineColor = "gray",
      fillGraph = FALSE
    ) %>%
    dyAxis("y", label = "Value", valueFormatter = 'function(d){return d.toFixed(2)}') %>%
    dyAxis("x", label = "Date") %>%
    dyHighlight(
      highlightCircleSize = 5,
      highlightSeriesBackgroundAlpha = 0.2,
      hideOnMouseOut = FALSE
    ) %>%
    dyRangeSelector(height = 40, strokeColor = "") %>%
    dyCrosshair(direction = "vertical") %>%
    dyLegend(show = "onmouseover", hideOnMouseOut = TRUE, width = 400)
}

#' Create performance summary table
#' @param returns_matrix Matrix of returns
#' @param risk_free_rate Numeric: risk-free rate
#' @return formattable object
create_performance_table <- function(returns_matrix, risk_free_rate = 0.02/252) {
  
  # Calculate performance metrics
  performance_table <- data.frame(
    Asset = colnames(returns_matrix),
    `Annual Return` = apply(returns_matrix, 2, function(x) mean(x, na.rm = TRUE) * 252),
    `Annual Vol` = apply(returns_matrix, 2, function(x) sd(x, na.rm = TRUE) * sqrt(252)),
    `Sharpe Ratio` = apply(returns_matrix, 2, function(x) {
      (mean(x, na.rm = TRUE) - risk_free_rate) / sd(x, na.rm = TRUE) * sqrt(252)
    }),
    `Max Drawdown` = apply(returns_matrix, 2, function(x) {
      cum_ret <- cumprod(1 + x)
      min((cum_ret - cummax(cum_ret)) / cummax(cum_ret), na.rm = TRUE)
    }),
    `Skewness` = apply(returns_matrix, 2, function(x) skewness(x, na.rm = TRUE)),
    `Kurtosis` = apply(returns_matrix, 2, function(x) kurtosis(x, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
  
  # Format the table
  formatted_table <- formattable(
    performance_table,
    list(
      `Annual Return` = color_tile("white", financial_colors$bullish),
      `Annual Vol` = color_tile(financial_colors$bullish, "white"),
      `Sharpe Ratio` = formatter("span",
        style = x ~ style(color = ifelse(x > 1, financial_colors$bullish, 
                                       ifelse(x > 0, "black", financial_colors$bearish)))
      ),
      `Max Drawdown` = color_tile(financial_colors$bearish, "white"),
      `Skewness` = formatter("span",
        style = x ~ style(color = ifelse(abs(x) > 0.5, financial_colors$warning, "black"))
      ),
      `Kurtosis` = formatter("span",
        style = x ~ style(color = ifelse(x > 3, financial_colors$warning, "black"))
      )
    ),
    digits = 3
  )
  
  return(formatted_table)
}

# =============================================================================
# Report Generation Functions
# =============================================================================

#' Generate comprehensive analysis report
#' @param analysis_results List of analysis results
#' @param output_file Character: output file path
#' @param format Character: "html" or "pdf"
generate_analysis_report <- function(analysis_results, output_file = "analysis_report.html", 
                                   format = "html") {
  
  cat("Generating comprehensive analysis report...\n")
  
  # This function would generate a comprehensive report
  # For now, we'll provide the framework
  
  report_content <- list(
    summary = "Analysis Summary",
    plots = list(),
    tables = list(),
    diagnostics = list()
  )
  
  # Add plots to report
  if ("pairs_trading" %in% names(analysis_results)) {
    cat("Adding pairs trading analysis...\n")
    # Add pairs trading plots
  }
  
  if ("performance" %in% names(analysis_results)) {
    cat("Adding performance analysis...\n")
    # Add performance plots
  }
  
  if ("risk_analysis" %in% names(analysis_results)) {
    cat("Adding risk analysis...\n")
    # Add risk plots
  }
  
  cat("Report generated:", output_file, "\n")
  return(report_content)
}

# =============================================================================
# Utility and Helper Functions
# =============================================================================

#' Save plot with consistent formatting
#' @param plot ggplot object
#' @param filename Character: file name
#' @param width Numeric: plot width in inches
#' @param height Numeric: plot height in inches
#' @param dpi Numeric: resolution
save_financial_plot <- function(plot, filename, width = 12, height = 8, dpi = 300) {
  
  ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
  
  cat("Plot saved:", filename, "\n")
}

#' Create plot grid with consistent styling
#' @param plot_list List of ggplot objects
#' @param ncol Integer: number of columns
#' @param title Character: overall title
#' @return ggplot object
create_plot_grid <- function(plot_list, ncol = 2, title = NULL) {
  
  # Combine plots using patchwork
  if (length(plot_list) == 1) {
    combined_plot <- plot_list[[1]]
  } else {
    combined_plot <- wrap_plots(plot_list, ncol = ncol)
  }
  
  if (!is.null(title)) {
    combined_plot <- combined_plot + plot_annotation(title = title)
  }
  
  return(combined_plot)
}

# =============================================================================
# Example Usage and Demo Functions
# =============================================================================

#' Demonstrate visualization capabilities
#' @param generate_sample_data Logical: whether to generate sample data
#' @return List of example plots
demo_visualizations <- function(generate_sample_data = TRUE) {
  
  cat("Demonstrating visualization capabilities...\n")
  
  if (generate_sample_data) {
    # Generate sample data
    set.seed(123)
    dates <- seq(as.Date("2020-01-01"), as.Date("2023-12-31"), by = "day")
    dates <- dates[weekdays(dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
    
    n <- length(dates)
    returns1 <- rnorm(n, 0.0005, 0.02)
    returns2 <- 0.7 * returns1 + sqrt(1 - 0.7^2) * rnorm(n, 0.0005, 0.02)
    
    prices1 <- 100 * cumprod(1 + returns1)
    prices2 <- 95 * cumprod(1 + returns2)
    
    sample_data <- list(
      prices = xts(cbind(AAPL = prices1, MSFT = prices2), order.by = dates),
      returns = xts(cbind(AAPL = returns1, MSFT = returns2), order.by = dates)
    )
  }
  
  demo_plots <- list()
  
  # Price series plot
  demo_plots$price_series <- plot_price_series(
    sample_data$prices[, 1], 
    symbol = "AAPL", 
    interactive = FALSE
  )
  
  # Return distribution
  demo_plots$return_dist <- plot_return_distribution(
    sample_data$returns[, 1], 
    symbol = "AAPL"
  )
  
  # Correlation matrix
  demo_plots$correlation <- plot_correlation_matrix(
    sample_data$returns, 
    interactive = FALSE
  )
  
  # Rolling correlation
  demo_plots$rolling_corr <- plot_rolling_correlation(
    sample_data$prices[, 1], 
    sample_data$prices[, 2],
    symbol1 = "AAPL", 
    symbol2 = "MSFT"
  )
  
  cat("Demo visualizations created successfully!\n")
  return(demo_plots)
}

# =============================================================================
# Script Initialization
# =============================================================================

# Print header
cat("\n")
cat(rep("=", 80), "\n")
cat("Data Visualization and Plotting for Statistical Arbitrage\n")
cat("Initialized successfully with", length(required_packages), "packages\n")
cat(rep("=", 80), "\n")
cat("\n")

# Set up global options
options(
  digits = 4,
  scipen = 999,
  warn = -1  # Suppress warnings for cleaner output
)

# Example usage message
if (interactive() || !exists(".testing")) {
  cat("Visualization system ready! Available functions:\n")
  cat("- plot_price_series()\n")
  cat("- plot_candlestick()\n") 
  cat("- plot_return_distribution()\n")
  cat("- plot_correlation_matrix()\n")
  cat("- plot_pairs_analysis()\n")
  cat("- plot_portfolio_performance()\n")
  cat("- create_time_series_dashboard()\n")
  cat("- demo_visualizations()\n")
  cat("\nTo run demos: demo_plots <- demo_visualizations()\n")
}
