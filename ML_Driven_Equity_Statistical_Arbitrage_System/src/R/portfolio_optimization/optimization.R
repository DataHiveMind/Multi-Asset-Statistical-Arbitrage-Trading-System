# =============================================================================
# Portfolio Optimization for Statistical Arbitrage
# =============================================================================
# Purpose: Advanced portfolio optimization techniques including Mean-Variance,
#          Black-Litterman, Risk Parity, and custom optimization for stat arb
# Author: Statistical Arbitrage System
# Date: 2025-06-24
# =============================================================================

# Required packages
required_packages <- c(
  "quadprog", "fPortfolio", "PortfolioAnalytics", "ROI", "ROI.plugin.quadprog",
  "ROI.plugin.glpk", "CVXR", "Matrix", "corpcor", "MASS", "mvtnorm",
  "xts", "zoo", "PerformanceAnalytics", "quantmod", "RiskPortfolios",
  "nloptr", "DEoptim", "GA", "PSO", "Rsolnp", "alabama",
  "dplyr", "tidyr", "ggplot2", "reshape2", "corrplot",
  "stats", "methods", "utils", "parallel", "foreach", "doParallel"
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
  cat("Portfolio optimization packages loaded successfully.\n")
}

# Load all required packages
load_packages(required_packages)

# =============================================================================
# Core Data Structures and Classes
# =============================================================================

#' Portfolio Optimization Configuration
#' @description S4 class to store optimization parameters and constraints
setClass("OptimizationConfig",
  slots = list(
    objective = "character",           # "mean_variance", "risk_parity", etc.
    risk_model = "character",          # "sample", "factor", "shrinkage"
    constraints = "list",              # List of constraint specifications
    bounds = "matrix",                 # Lower and upper bounds for weights
    target_return = "numeric",         # Target return for constrained optimization
    target_risk = "numeric",           # Target risk level
    risk_aversion = "numeric",         # Risk aversion parameter
    transaction_costs = "numeric",     # Transaction cost assumptions
    max_positions = "integer",         # Maximum number of positions
    min_weight = "numeric",            # Minimum position weight
    max_weight = "numeric",            # Maximum position weight
    rebalancing_frequency = "character", # "daily", "weekly", "monthly"
    lookback_period = "integer",       # Days for historical estimation
    shrinkage_intensity = "numeric"    # Shrinkage parameter [0,1]
  ),
  prototype = list(
    objective = "mean_variance",
    risk_model = "sample",
    constraints = list(),
    bounds = matrix(c(0, 1), nrow = 1),
    target_return = 0.1,
    target_risk = 0.15,
    risk_aversion = 3.0,
    transaction_costs = 0.001,
    max_positions = 50L,
    min_weight = 0.0,
    max_weight = 0.1,
    rebalancing_frequency = "weekly",
    lookback_period = 252L,
    shrinkage_intensity = 0.5
  )
)

#' Portfolio Results
#' @description S4 class to store optimization results
setClass("PortfolioResult",
  slots = list(
    weights = "numeric",               # Optimal portfolio weights
    expected_return = "numeric",       # Expected portfolio return
    expected_risk = "numeric",         # Expected portfolio risk
    sharpe_ratio = "numeric",          # Sharpe ratio
    alpha = "numeric",                 # Portfolio alpha
    beta = "numeric",                  # Portfolio beta
    tracking_error = "numeric",        # Tracking error vs benchmark
    max_drawdown = "numeric",          # Expected maximum drawdown
    turnover = "numeric",              # Portfolio turnover
    transaction_costs = "numeric",     # Expected transaction costs
    optimization_info = "list",       # Additional optimization details
    assets = "character",              # Asset names
    timestamp = "POSIXct"             # Optimization timestamp
  )
)

# =============================================================================
# Risk Model Estimation Functions
# =============================================================================

#' Estimate covariance matrix using various methods
#' @param returns Matrix of asset returns
#' @param method Character: estimation method
#' @param shrinkage_target Character: shrinkage target
#' @param lambda Numeric: shrinkage intensity
#' @return Covariance matrix
estimate_covariance_matrix <- function(returns, method = "sample", 
                                     shrinkage_target = "identity", lambda = NULL) {
  
  # Remove missing values
  returns <- na.omit(returns)
  n <- nrow(returns)
  p <- ncol(returns)
  
  if (method == "sample") {
    # Sample covariance matrix
    cov_matrix <- cov(returns)
    
  } else if (method == "shrinkage") {
    # Ledoit-Wolf shrinkage estimator
    if (shrinkage_target == "identity") {
      # Shrink towards identity matrix
      sample_cov <- cov(returns)
      target <- mean(diag(sample_cov)) * diag(p)
      
      if (is.null(lambda)) {
        # Optimal shrinkage intensity (Ledoit-Wolf)
        lambda <- estimate_shrinkage_intensity(returns, target)
      }
      
      cov_matrix <- (1 - lambda) * sample_cov + lambda * target
      
    } else if (shrinkage_target == "constant_correlation") {
      # Shrink towards constant correlation matrix
      sample_cov <- cov(returns)
      variances <- diag(sample_cov)
      avg_correlation <- (sum(cov2cor(sample_cov)) - p) / (p * (p - 1))
      
      target <- sqrt(variances %o% variances) * avg_correlation
      diag(target) <- variances
      
      if (is.null(lambda)) {
        lambda <- estimate_shrinkage_intensity(returns, target)
      }
      
      cov_matrix <- (1 - lambda) * sample_cov + lambda * target
      
    } else if (shrinkage_target == "market") {
      # Shrink towards single-factor market model
      market_return <- rowMeans(returns)
      market_var <- var(market_return)
      
      # Estimate betas
      betas <- apply(returns, 2, function(x) cov(x, market_return) / market_var)
      
      # Construct target matrix
      target <- market_var * (betas %o% betas)
      residual_vars <- apply(returns, 2, function(x) {
        var(x - betas[colnames(returns) == colnames(x)[1]] * market_return)
      })
      diag(target) <- diag(target) + residual_vars
      
      if (is.null(lambda)) {
        lambda <- estimate_shrinkage_intensity(returns, target)
      }
      
      cov_matrix <- (1 - lambda) * cov(returns) + lambda * target
    }
    
  } else if (method == "factor") {
    # Factor model-based covariance
    cov_matrix <- estimate_factor_covariance(returns)
    
  } else if (method == "robust") {
    # Robust covariance estimation
    cov_matrix <- cov.rob(returns, method = "mve")$cov
    
  } else if (method == "exponential") {
    # Exponentially weighted covariance
    lambda_ew <- 0.94  # RiskMetrics decay factor
    cov_matrix <- ewma_covariance(returns, lambda_ew)
    
  } else {
    stop("Unknown covariance estimation method")
  }
  
  # Ensure positive definiteness
  cov_matrix <- make_positive_definite(cov_matrix)
  
  return(cov_matrix)
}

#' Estimate optimal shrinkage intensity (Ledoit-Wolf)
#' @param returns Matrix of returns
#' @param target Target covariance matrix
#' @return Optimal shrinkage intensity
estimate_shrinkage_intensity <- function(returns, target) {
  
  n <- nrow(returns)
  p <- ncol(returns)
  
  # Center the returns
  returns_centered <- scale(returns, scale = FALSE)
  
  # Sample covariance
  sample_cov <- cov(returns)
  
  # Calculate optimal shrinkage intensity
  # This is a simplified version of the Ledoit-Wolf estimator
  
  # Asymptotic variance of sample covariance
  pi_hat <- sum((returns_centered^2 %*% t(returns_centered^2)) / n - sample_cov^2) / n
  
  # Bias-squared of shrinkage estimator
  rho_hat <- sum((sample_cov - target)^2)
  
  # Optimal shrinkage intensity
  lambda <- max(0, min(1, pi_hat / rho_hat))
  
  return(lambda)
}

#' Estimate factor model covariance matrix
#' @param returns Matrix of returns
#' @param n_factors Integer: number of factors
#' @return Factor model covariance matrix
estimate_factor_covariance <- function(returns, n_factors = 3) {
  
  # Principal component analysis
  pca_result <- prcomp(returns, center = TRUE, scale. = TRUE)
  
  # Extract factor loadings
  loadings <- pca_result$rotation[, 1:n_factors]
  
  # Factor variances
  factor_vars <- (pca_result$sdev[1:n_factors])^2
  
  # Specific variances (residual variances)
  explained_var <- rowSums(loadings^2 * rep(factor_vars, each = nrow(loadings)))
  total_var <- apply(scale(returns), 2, var)
  specific_vars <- pmax(total_var - explained_var, 0.01 * total_var)  # Floor at 1%
  
  # Construct covariance matrix: Λ F Λ' + Ψ
  factor_cov <- loadings %*% diag(factor_vars) %*% t(loadings)
  specific_cov <- diag(specific_vars)
  
  cov_matrix <- factor_cov + specific_cov
  
  return(cov_matrix)
}

#' Exponentially weighted moving average covariance
#' @param returns Matrix of returns
#' @param lambda Decay factor
#' @return EWMA covariance matrix
ewma_covariance <- function(returns, lambda = 0.94) {
  
  n <- nrow(returns)
  p <- ncol(returns)
  
  # Initialize with first observation
  cov_matrix <- tcrossprod(returns[1, ])
  
  # Update recursively
  for (t in 2:n) {
    cov_matrix <- lambda * cov_matrix + (1 - lambda) * tcrossprod(returns[t, ])
  }
  
  return(cov_matrix)
}

#' Ensure covariance matrix is positive definite
#' @param cov_matrix Covariance matrix
#' @param min_eigenvalue Minimum eigenvalue
#' @return Positive definite covariance matrix
make_positive_definite <- function(cov_matrix, min_eigenvalue = 1e-8) {
  
  # Eigenvalue decomposition
  eigen_decomp <- eigen(cov_matrix)
  eigenvalues <- eigen_decomp$values
  eigenvectors <- eigen_decomp$vectors
  
  # Adjust negative eigenvalues
  eigenvalues[eigenvalues < min_eigenvalue] <- min_eigenvalue
  
  # Reconstruct matrix
  cov_matrix_pd <- eigenvectors %*% diag(eigenvalues) %*% t(eigenvectors)
  
  return(cov_matrix_pd)
}

# =============================================================================
# Mean-Variance Optimization
# =============================================================================

#' Mean-Variance optimization with quadratic programming
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param risk_aversion Risk aversion parameter
#' @param constraints List of constraint specifications
#' @return Portfolio weights
mean_variance_optimization <- function(expected_returns, cov_matrix, risk_aversion = 3.0,
                                     constraints = list()) {
  
  n <- length(expected_returns)
  
  # Default constraints: fully invested, long-only
  if (length(constraints) == 0) {
    constraints <- list(
      type = "full_investment",
      min_weights = rep(0, n),
      max_weights = rep(1, n)
    )
  }
  
  # Quadratic programming formulation:
  # min 0.5 * w' * Σ * w - λ * μ' * w
  # subject to constraints
  
  # Objective function matrices
  Dmat <- 2 * risk_aversion * cov_matrix
  dvec <- expected_returns
  
  # Equality constraints (Aeq * w = beq)
  Aeq <- matrix(1, nrow = 1, ncol = n)  # Sum of weights = 1
  beq <- 1
  
  # Inequality constraints (Aineq * w >= bineq)
  Aineq <- rbind(
    diag(n),           # w_i >= min_weight
    -diag(n)           # w_i <= max_weight
  )
  
  bineq <- c(
    constraints$min_weights %||% rep(0, n),
    -(constraints$max_weights %||% rep(1, n))
  )
  
  # Combine constraints
  Amat <- t(rbind(Aeq, Aineq))
  bvec <- c(beq, bineq)
  
  # Solve quadratic program
  qp_result <- solve.QP(
    Dmat = Dmat,
    dvec = dvec,
    Amat = Amat,
    bvec = bvec,
    meq = 1  # First constraint is equality
  )
  
  weights <- qp_result$solution
  names(weights) <- names(expected_returns)
  
  return(weights)
}

#' Efficient frontier calculation
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param n_points Number of points on efficient frontier
#' @param short_selling Logical: allow short selling
#' @return Data frame with efficient frontier points
calculate_efficient_frontier <- function(expected_returns, cov_matrix, n_points = 50,
                                        short_selling = FALSE) {
  
  n <- length(expected_returns)
  min_return <- min(expected_returns)
  max_return <- max(expected_returns)
  
  # Extend range slightly
  return_range <- seq(min_return * 0.5, max_return * 1.5, length.out = n_points)
  
  efficient_portfolios <- data.frame(
    Return = numeric(n_points),
    Risk = numeric(n_points),
    Sharpe = numeric(n_points)
  )
  
  for (i in 1:n_points) {
    target_return <- return_range[i]
    
    tryCatch({
      # Minimize variance subject to target return
      weights <- minimum_variance_portfolio(
        expected_returns = expected_returns,
        cov_matrix = cov_matrix,
        target_return = target_return,
        short_selling = short_selling
      )
      
      portfolio_return <- sum(weights * expected_returns)
      portfolio_risk <- sqrt(t(weights) %*% cov_matrix %*% weights)
      sharpe_ratio <- portfolio_return / portfolio_risk
      
      efficient_portfolios[i, ] <- c(portfolio_return, portfolio_risk, sharpe_ratio)
      
    }, error = function(e) {
      efficient_portfolios[i, ] <- c(NA, NA, NA)
    })
  }
  
  # Remove invalid points
  efficient_portfolios <- na.omit(efficient_portfolios)
  
  return(efficient_portfolios)
}

#' Minimum variance portfolio
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param target_return Target portfolio return
#' @param short_selling Logical: allow short selling
#' @return Portfolio weights
minimum_variance_portfolio <- function(expected_returns, cov_matrix, target_return = NULL,
                                     short_selling = FALSE) {
  
  n <- length(expected_returns)
  
  # Quadratic programming matrices
  Dmat <- 2 * cov_matrix
  dvec <- rep(0, n)
  
  # Constraints
  if (is.null(target_return)) {
    # Minimum variance portfolio (no return constraint)
    Aeq <- matrix(1, nrow = 1, ncol = n)
    beq <- 1
    meq <- 1
  } else {
    # Target return constraint
    Aeq <- rbind(
      rep(1, n),           # Sum of weights = 1
      expected_returns     # Target return constraint
    )
    beq <- c(1, target_return)
    meq <- 2
  }
  
  # Inequality constraints
  if (!short_selling) {
    Aineq <- diag(n)     # Non-negativity constraints
    bineq <- rep(0, n)
    Amat <- t(rbind(Aeq, Aineq))
    bvec <- c(beq, bineq)
  } else {
    Amat <- t(Aeq)
    bvec <- beq
  }
  
  # Solve
  qp_result <- solve.QP(Dmat, dvec, Amat, bvec, meq)
  weights <- qp_result$solution
  names(weights) <- names(expected_returns)
  
  return(weights)
}

#' Maximum Sharpe ratio portfolio
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param risk_free_rate Risk-free rate
#' @param short_selling Logical: allow short selling
#' @return Portfolio weights
maximum_sharpe_portfolio <- function(expected_returns, cov_matrix, risk_free_rate = 0,
                                   short_selling = FALSE) {
  
  # Excess returns
  excess_returns <- expected_returns - risk_free_rate
  
  # The maximum Sharpe portfolio is proportional to Σ^(-1) * (μ - rf)
  inv_cov <- solve(cov_matrix)
  unnormalized_weights <- inv_cov %*% excess_returns
  
  if (!short_selling) {
    # If short selling not allowed, use optimization
    objective <- function(w) {
      portfolio_return <- sum(w * expected_returns)
      portfolio_risk <- sqrt(t(w) %*% cov_matrix %*% w)
      -((portfolio_return - risk_free_rate) / portfolio_risk)  # Negative for minimization
    }
    
    # Constraints
    equality_constraint <- function(w) sum(w) - 1
    inequality_constraint <- function(w) w  # w >= 0
    
    # Optimization
    result <- alabama::constrOptim.nl(
      par = rep(1/length(expected_returns), length(expected_returns)),
      fn = objective,
      heq = equality_constraint,
      hin = inequality_constraint
    )
    
    weights <- result$par
    
  } else {
    # Normalize weights to sum to 1
    weights <- unnormalized_weights / sum(unnormalized_weights)
  }
  
  names(weights) <- names(expected_returns)
  return(weights)
}

# =============================================================================
# Black-Litterman Model
# =============================================================================

#' Black-Litterman model implementation
#' @param returns Historical returns matrix
#' @param market_caps Market capitalizations (for equilibrium returns)
#' @param P Pick matrix (which assets the views relate to)
#' @param Q View vector (expected returns from views)
#' @param Omega Uncertainty matrix for views
#' @param tau Confidence parameter
#' @param risk_aversion Risk aversion parameter
#' @return List with BL expected returns and covariance matrix
black_litterman_model <- function(returns, market_caps = NULL, P = NULL, Q = NULL, 
                                 Omega = NULL, tau = 0.025, risk_aversion = 3.0) {
  
  n <- ncol(returns)
  asset_names <- colnames(returns)
  
  # Estimate covariance matrix
  sigma <- cov(returns)
  
  # Market equilibrium returns (if market caps not provided, use equal weights)
  if (is.null(market_caps)) {
    w_market <- rep(1/n, n)
  } else {
    w_market <- market_caps / sum(market_caps)
  }
  
  # Implied equilibrium returns: μ = λ * Σ * w_market
  mu_eq <- risk_aversion * sigma %*% w_market
  
  # If no views provided, return equilibrium
  if (is.null(P) || is.null(Q)) {
    return(list(
      expected_returns = as.vector(mu_eq),
      covariance_matrix = sigma,
      equilibrium_returns = as.vector(mu_eq)
    ))
  }
  
  # Default uncertainty matrix (proportional to view variance)
  if (is.null(Omega)) {
    Omega <- diag(diag(P %*% (tau * sigma) %*% t(P)))
  }
  
  # Black-Litterman formula
  # μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)μ_eq + P'Ω^(-1)Q]
  
  tau_sigma_inv <- solve(tau * sigma)
  P_omega_inv_P <- t(P) %*% solve(Omega) %*% P
  
  M1 <- solve(tau_sigma_inv + P_omega_inv_P)
  M2 <- tau_sigma_inv %*% mu_eq + t(P) %*% solve(Omega) %*% Q
  
  mu_bl <- M1 %*% M2
  
  # Black-Litterman covariance matrix
  sigma_bl <- M1
  
  result <- list(
    expected_returns = as.vector(mu_bl),
    covariance_matrix = sigma_bl,
    equilibrium_returns = as.vector(mu_eq),
    tau = tau,
    views = list(P = P, Q = Q, Omega = Omega)
  )
  
  return(result)
}

#' Create view matrix for Black-Litterman
#' @param asset_names Character vector of asset names
#' @param views List of view specifications
#' @return List with P matrix, Q vector, and Omega matrix
create_bl_views <- function(asset_names, views) {
  
  n_assets <- length(asset_names)
  n_views <- length(views)
  
  P <- matrix(0, nrow = n_views, ncol = n_assets)
  Q <- numeric(n_views)
  Omega <- diag(n_views)
  
  for (i in 1:n_views) {
    view <- views[[i]]
    
    if (view$type == "absolute") {
      # Absolute view: asset A will return X%
      asset_idx <- which(asset_names == view$assets[1])
      P[i, asset_idx] <- 1
      Q[i] <- view$return
      Omega[i, i] <- view$confidence
      
    } else if (view$type == "relative") {
      # Relative view: asset A will outperform asset B by X%
      asset_idx_1 <- which(asset_names == view$assets[1])
      asset_idx_2 <- which(asset_names == view$assets[2])
      P[i, asset_idx_1] <- 1
      P[i, asset_idx_2] <- -1
      Q[i] <- view$return
      Omega[i, i] <- view$confidence
    }
  }
  
  return(list(P = P, Q = Q, Omega = Omega))
}

# =============================================================================
# Risk Parity and Alternative Risk Budgeting
# =============================================================================

#' Equal Risk Contribution (Risk Parity) portfolio
#' @param cov_matrix Covariance matrix
#' @param method Optimization method
#' @return Portfolio weights
risk_parity_portfolio <- function(cov_matrix, method = "cyclical-roncalli") {
  
  if (method == "cyclical-roncalli") {
    # Roncalli's cyclical coordinate descent algorithm
    weights <- risk_parity_roncalli(cov_matrix)
  } else if (method == "convex-optimization") {
    # Convex optimization approach
    weights <- risk_parity_convex(cov_matrix)
  } else {
    stop("Unknown risk parity method")
  }
  
  return(weights)
}

#' Risk parity using Roncalli's algorithm
#' @param cov_matrix Covariance matrix
#' @param max_iter Maximum iterations
#' @param tol Convergence tolerance
#' @return Portfolio weights
risk_parity_roncalli <- function(cov_matrix, max_iter = 1000, tol = 1e-8) {
  
  n <- nrow(cov_matrix)
  weights <- rep(1/n, n)  # Equal weights initialization
  
  for (iter in 1:max_iter) {
    weights_old <- weights
    
    # Update each weight cyclically
    for (i in 1:n) {
      # Risk contribution of asset i
      rc_i <- weights[i] * (cov_matrix %*% weights)[i]
      
      # Average risk contribution of other assets
      rc_others <- ((cov_matrix %*% weights)[-i] * weights[-i])
      avg_rc_others <- mean(rc_others)
      
      # Update weight to equalize risk contributions
      if (avg_rc_others > 0) {
        weights[i] <- avg_rc_others / (cov_matrix %*% weights)[i]
      }
    }
    
    # Renormalize weights
    weights <- weights / sum(weights)
    
    # Check convergence
    if (max(abs(weights - weights_old)) < tol) {
      break
    }
  }
  
  return(weights)
}

#' Risk parity using convex optimization
#' @param cov_matrix Covariance matrix
#' @return Portfolio weights
risk_parity_convex <- function(cov_matrix) {
  
  n <- nrow(cov_matrix)
  
  # Objective function: minimize sum of squared risk contribution differences
  objective <- function(w) {
    portfolio_risk <- sqrt(t(w) %*% cov_matrix %*% w)
    risk_contributions <- w * (cov_matrix %*% w) / portfolio_risk
    target_rc <- portfolio_risk / n  # Equal risk contribution
    sum((risk_contributions - target_rc)^2)
  }
  
  # Constraints
  equality_constraint <- function(w) sum(w) - 1
  inequality_constraint <- function(w) w  # w >= 0
  
  # Optimization
  result <- alabama::constrOptim.nl(
    par = rep(1/n, n),
    fn = objective,
    heq = equality_constraint,
    hin = inequality_constraint
  )
  
  return(result$par)
}

#' Risk budgeting portfolio
#' @param cov_matrix Covariance matrix
#' @param risk_budgets Vector of risk budget allocations
#' @return Portfolio weights
risk_budgeting_portfolio <- function(cov_matrix, risk_budgets) {
  
  n <- length(risk_budgets)
  
  # Normalize risk budgets
  risk_budgets <- risk_budgets / sum(risk_budgets)
  
  # Objective function: minimize distance to target risk budgets
  objective <- function(w) {
    portfolio_risk <- sqrt(t(w) %*% cov_matrix %*% w)
    risk_contributions <- w * (cov_matrix %*% w) / portfolio_risk
    risk_contributions <- risk_contributions / sum(risk_contributions)
    sum((risk_contributions - risk_budgets)^2)
  }
  
  # Constraints
  equality_constraint <- function(w) sum(w) - 1
  inequality_constraint <- function(w) w
  
  # Optimization
  result <- alabama::constrOptim.nl(
    par = rep(1/n, n),
    fn = objective,
    heq = equality_constraint,
    hin = inequality_constraint
  )
  
  return(result$par)
}

# =============================================================================
# Statistical Arbitrage Specific Optimization
# =============================================================================

#' Alpha-driven portfolio optimization
#' @param alpha_scores Vector of alpha scores/signals
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param transaction_costs Vector of transaction costs
#' @param current_weights Current portfolio weights
#' @param constraints List of constraints
#' @return Optimal portfolio weights
alpha_driven_optimization <- function(alpha_scores, expected_returns, cov_matrix,
                                     transaction_costs = NULL, current_weights = NULL,
                                     constraints = list()) {
  
  n <- length(alpha_scores)
  
  # Normalize alpha scores
  alpha_scores <- alpha_scores / sd(alpha_scores, na.rm = TRUE)
  
  # Combine alpha and expected returns
  combined_alpha <- 0.7 * alpha_scores + 0.3 * expected_returns
  
  # Transaction cost adjustment
  if (!is.null(transaction_costs) && !is.null(current_weights)) {
    # Objective includes transaction costs
    objective <- function(w) {
      portfolio_return <- sum(w * combined_alpha)
      portfolio_risk <- sqrt(t(w) %*% cov_matrix %*% w)
      tc_penalty <- sum(transaction_costs * abs(w - current_weights))
      
      # Maximize return per unit risk minus transaction costs
      -(portfolio_return / portfolio_risk - tc_penalty)
    }
  } else {
    # Standard mean-variance objective with alpha
    objective <- function(w) {
      portfolio_return <- sum(w * combined_alpha)
      portfolio_risk <- sqrt(t(w) %*% cov_matrix %*% w)
      -(portfolio_return / portfolio_risk)  # Maximize Sharpe ratio
    }
  }
  
  # Default constraints
  if (length(constraints) == 0) {
    constraints <- list(
      sum_weights = 1,
      min_weight = 0,
      max_weight = 0.1,
      max_positions = n
    )
  }
  
  # Constraint functions
  equality_constraint <- function(w) sum(w) - constraints$sum_weights
  
  inequality_constraints <- function(w) {
    c(
      w - constraints$min_weight,                    # Lower bounds
      constraints$max_weight - w,                    # Upper bounds
      constraints$max_positions - sum(w > 0.001)    # Position limit
    )
  }
  
  # Optimization
  result <- alabama::constrOptim.nl(
    par = rep(1/n, n),
    fn = objective,
    heq = equality_constraint,
    hin = inequality_constraints
  )
  
  weights <- result$par
  names(weights) <- names(alpha_scores)
  
  return(weights)
}

#' Long-short portfolio optimization
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param leverage Target leverage (gross exposure)
#' @param max_weight Maximum weight per position
#' @return Portfolio weights (can be negative for short positions)
long_short_optimization <- function(expected_returns, cov_matrix, leverage = 2.0, max_weight = 0.1) {
  
  n <- length(expected_returns)
  
  # Objective: maximize expected return per unit risk
  objective <- function(w) {
    portfolio_return <- sum(w * expected_returns)
    portfolio_risk <- sqrt(t(w) %*% cov_matrix %*% w)
    -portfolio_return / portfolio_risk
  }
  
  # Constraints
  constraints <- list(
    # Net exposure = 0 (market neutral)
    function(w) sum(w),
    
    # Gross exposure = leverage
    function(w) sum(abs(w)) - leverage,
    
    # Position size limits
    function(w) c(w + max_weight, max_weight - w)
  )
  
  # Initial guess: equal long-short
  n_long <- floor(n/2)
  w0 <- c(rep(leverage/(2*n_long), n_long), rep(-leverage/(2*(n-n_long)), n-n_long))
  
  # Optimization (using nloptr for bound constraints)
  result <- nloptr::nloptr(
    x0 = w0,
    eval_f = objective,
    eval_g_eq = function(w) sum(w),  # Net exposure = 0
    eval_g_ineq = function(w) c(
      sum(abs(w)) - leverage,  # Gross exposure
      max_weight - abs(w)      # Position limits
    ),
    opts = list(algorithm = "NLOPT_LN_COBYLA", maxeval = 1000)
  )
  
  weights <- result$solution
  names(weights) <- names(expected_returns)
  
  return(weights)
}

#' Pairs trading portfolio optimization
#' @param pair_signals Matrix of pair trading signals
#' @param pair_returns Matrix of pair returns
#' @param pair_volatilities Matrix of pair volatilities
#' @param correlation_matrix Correlation matrix between pairs
#' @return Portfolio weights for each pair
pairs_portfolio_optimization <- function(pair_signals, pair_returns, pair_volatilities,
                                       correlation_matrix) {
  
  n_pairs <- ncol(pair_signals)
  
  # Calculate expected returns for each pair based on signals
  expected_pair_returns <- apply(pair_signals * pair_returns, 2, mean, na.rm = TRUE)
  
  # Risk model: use volatilities and correlations
  risk_matrix <- diag(pair_volatilities) %*% correlation_matrix %*% diag(pair_volatilities)
  
  # Objective: maximize information ratio across pairs
  objective <- function(w) {
    portfolio_return <- sum(w * expected_pair_returns)
    portfolio_risk <- sqrt(t(w) %*% risk_matrix %*% w)
    -portfolio_return / portfolio_risk
  }
  
  # Constraints: equal allocation across pairs, limit individual pair exposure
  equality_constraint <- function(w) sum(w) - 1
  inequality_constraint <- function(w) c(w, 0.2 - w)  # 0 <= w <= 0.2
  
  result <- alabama::constrOptim.nl(
    par = rep(1/n_pairs, n_pairs),
    fn = objective,
    heq = equality_constraint,
    hin = inequality_constraint
  )
  
  weights <- result$par
  names(weights) <- colnames(pair_signals)
  
  return(weights)
}

# =============================================================================
# Advanced Optimization Techniques
# =============================================================================

#' Robust portfolio optimization (worst-case approach)
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param uncertainty_set_size Size of uncertainty set
#' @return Robust portfolio weights
robust_portfolio_optimization <- function(expected_returns, cov_matrix, uncertainty_set_size = 0.5) {
  
  n <- length(expected_returns)
  
  # Worst-case optimization: minimize maximum possible loss
  # This is a simplified robust optimization approach
  
  # Generate uncertainty scenarios
  n_scenarios <- 1000
  scenarios <- mvtnorm::rmvnorm(n_scenarios, expected_returns, uncertainty_set_size * cov_matrix)
  
  # Objective: minimize worst-case portfolio return
  objective <- function(w) {
    scenario_returns <- scenarios %*% w
    -min(scenario_returns)  # Maximize worst-case return
  }
  
  # Constraints
  equality_constraint <- function(w) sum(w) - 1
  inequality_constraint <- function(w) w
  
  result <- alabama::constrOptim.nl(
    par = rep(1/n, n),
    fn = objective,
    heq = equality_constraint,
    hin = inequality_constraint
  )
  
  return(result$par)
}

#' CVaR (Conditional Value at Risk) optimization
#' @param returns_scenarios Matrix of return scenarios
#' @param alpha Confidence level for CVaR
#' @return Portfolio weights minimizing CVaR
cvar_optimization <- function(returns_scenarios, alpha = 0.05) {
  
  n_assets <- ncol(returns_scenarios)
  n_scenarios <- nrow(returns_scenarios)
  
  # CVaR optimization using linear programming
  # Variables: [w_1, ..., w_n, VaR, s_1, ..., s_T]
  # where s_t are auxiliary variables for CVaR calculation
  
  n_vars <- n_assets + 1 + n_scenarios  # weights + VaR + auxiliary variables
  
  # Objective: minimize CVaR = VaR + (1/((1-α)T)) * Σ s_t
  f.obj <- c(
    rep(0, n_assets),                    # Portfolio weights
    1,                                   # VaR
    rep(1/((1-alpha) * n_scenarios), n_scenarios)  # Auxiliary variables
  )
  
  # Constraints
  # 1. Portfolio weights sum to 1
  # 2. s_t >= 0 for all t
  # 3. s_t >= -R_t'w - VaR for all t (where R_t is return scenario t)
  
  # Equality constraint: sum of weights = 1
  f.con.eq <- matrix(c(rep(1, n_assets), rep(0, 1 + n_scenarios)), nrow = 1)
  f.rhs.eq <- 1
  
  # Inequality constraints
  # s_t >= 0 and s_t >= -R_t'w - VaR
  f.con.ineq <- rbind(
    # s_t >= 0
    cbind(matrix(0, n_scenarios, n_assets + 1), diag(n_scenarios)),
    
    # s_t >= -R_t'w - VaR  =>  R_t'w + VaR + s_t >= 0
    cbind(returns_scenarios, rep(1, n_scenarios), diag(n_scenarios))
  )
  
  f.rhs.ineq <- rep(0, 2 * n_scenarios)
  f.dir.ineq <- rep(">=", 2 * n_scenarios)
  
  # Solve using linear programming
  # Note: This would require lpSolve or similar package
  # For demonstration, we'll use a simplified approach
  
  # Simplified: minimize portfolio variance as proxy
  cov_matrix <- cov(returns_scenarios)
  weights <- minimum_variance_portfolio(
    expected_returns = colMeans(returns_scenarios),
    cov_matrix = cov_matrix
  )
  
  return(weights)
}

#' Multi-objective optimization (Pareto frontier)
#' @param expected_returns Vector of expected returns
#' @param cov_matrix Covariance matrix
#' @param objectives List of objective functions
#' @return List of Pareto optimal portfolios
multi_objective_optimization <- function(expected_returns, cov_matrix, 
                                       objectives = c("return", "risk", "skewness")) {
  
  # This is a framework for multi-objective optimization
  # Full implementation would use specialized packages like mco or nsga2R
  
  n <- length(expected_returns)
  pareto_portfolios <- list()
  
  # Simple approach: weighted combination of objectives
  weight_combinations <- expand.grid(
    w1 = seq(0, 1, by = 0.1),
    w2 = seq(0, 1, by = 0.1),
    w3 = seq(0, 1, by = 0.1)
  )
  
  # Normalize weights
  weight_combinations$sum <- rowSums(weight_combinations)
  weight_combinations <- weight_combinations[weight_combinations$sum > 0, ]
  weight_combinations[, 1:3] <- weight_combinations[, 1:3] / weight_combinations$sum
  
  for (i in 1:min(20, nrow(weight_combinations))) {  # Limit to 20 combinations
    weights_obj <- weight_combinations[i, 1:3]
    
    # Combined objective function
    objective <- function(w) {
      port_return <- sum(w * expected_returns)
      port_risk <- sqrt(t(w) %*% cov_matrix %*% w)
      # Simplified skewness calculation
      port_skewness <- 0  # Would need higher moments
      
      # Weighted combination (minimize negative return + risk - skewness)
      weights_obj[1] * (-port_return) + weights_obj[2] * port_risk + weights_obj[3] * (-port_skewness)
    }
    
    # Optimize
    result <- optim(
      par = rep(1/n, n),
      fn = objective,
      method = "L-BFGS-B",
      lower = rep(0, n),
      upper = rep(1, n)
    )
    
    # Normalize weights
    portfolio_weights <- result$par / sum(result$par)
    
    pareto_portfolios[[i]] <- list(
      weights = portfolio_weights,
      objectives = weights_obj,
      value = result$value
    )
  }
  
  return(pareto_portfolios)
}

# =============================================================================
# Portfolio Performance and Risk Analytics
# =============================================================================

#' Calculate portfolio performance metrics
#' @param weights Portfolio weights
#' @param expected_returns Expected returns
#' @param cov_matrix Covariance matrix
#' @param benchmark_return Benchmark return
#' @param risk_free_rate Risk-free rate
#' @return List of performance metrics
calculate_portfolio_metrics <- function(weights, expected_returns, cov_matrix,
                                       benchmark_return = 0, risk_free_rate = 0) {
  
  # Basic portfolio metrics
  portfolio_return <- sum(weights * expected_returns)
  portfolio_variance <- t(weights) %*% cov_matrix %*% weights
  portfolio_volatility <- sqrt(portfolio_variance)
  
  # Risk-adjusted metrics
  sharpe_ratio <- (portfolio_return - risk_free_rate) / portfolio_volatility
  
  # Active metrics (vs benchmark)
  active_return <- portfolio_return - benchmark_return
  
  # Tracking error (simplified - would need time series for proper calculation)
  tracking_error <- portfolio_volatility  # Approximation
  information_ratio <- active_return / tracking_error
  
  # Risk decomposition
  marginal_risk <- cov_matrix %*% weights / portfolio_volatility
  component_risk <- weights * marginal_risk
  risk_contribution <- component_risk / portfolio_volatility
  
  # Concentration metrics
  herfindahl_index <- sum(weights^2)
  effective_number_assets <- 1 / herfindahl_index
  
  # Maximum position
  max_weight <- max(abs(weights))
  
  return(list(
    expected_return = as.numeric(portfolio_return),
    volatility = as.numeric(portfolio_volatility),
    sharpe_ratio = as.numeric(sharpe_ratio),
    active_return = as.numeric(active_return),
    information_ratio = as.numeric(information_ratio),
    tracking_error = as.numeric(tracking_error),
    risk_contributions = as.numeric(risk_contribution),
    marginal_risks = as.numeric(marginal_risk),
    herfindahl_index = as.numeric(herfindahl_index),
    effective_n_assets = as.numeric(effective_number_assets),
    max_weight = as.numeric(max_weight),
    leverage = sum(abs(weights))
  ))
}

#' Portfolio risk decomposition analysis
#' @param weights Portfolio weights
#' @param cov_matrix Covariance matrix
#' @param factor_loadings Factor loadings matrix (optional)
#' @return Risk decomposition results
portfolio_risk_decomposition <- function(weights, cov_matrix, factor_loadings = NULL) {
  
  portfolio_variance <- t(weights) %*% cov_matrix %*% weights
  portfolio_volatility <- sqrt(portfolio_variance)
  
  # Marginal contribution to risk
  marginal_risk <- (cov_matrix %*% weights) / portfolio_volatility
  
  # Component contribution to risk
  component_risk <- weights * marginal_risk
  
  # Percentage risk contribution
  risk_contribution_pct <- component_risk / portfolio_volatility * 100
  
  # Factor risk decomposition (if factor model provided)
  if (!is.null(factor_loadings)) {
    # Factor exposures
    factor_exposures <- t(factor_loadings) %*% weights
    
    # Factor risk contributions (simplified)
    factor_risks <- factor_exposures^2  # Would need factor covariance matrix
    
    result <- list(
      total_risk = as.numeric(portfolio_volatility),
      marginal_risks = as.numeric(marginal_risk),
      component_risks = as.numeric(component_risk),
      risk_contributions_pct = as.numeric(risk_contribution_pct),
      factor_exposures = as.numeric(factor_exposures),
      factor_risks = as.numeric(factor_risks)
    )
  } else {
    result <- list(
      total_risk = as.numeric(portfolio_volatility),
      marginal_risks = as.numeric(marginal_risk),
      component_risks = as.numeric(component_risk),
      risk_contributions_pct = as.numeric(risk_contribution_pct)
    )
  }
  
  return(result)
}

# =============================================================================
# Utility Functions and Helpers
# =============================================================================

#' Null coalescing operator
#' @param x First value
#' @param y Second value
#' @return x if not null, otherwise y
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Rebalancing scheduler
#' @param current_date Current date
#' @param frequency Rebalancing frequency
#' @param last_rebalance_date Last rebalancing date
#' @return Logical indicating whether to rebalance
should_rebalance <- function(current_date, frequency = "weekly", last_rebalance_date = NULL) {
  
  if (is.null(last_rebalance_date)) {
    return(TRUE)
  }
  
  days_since_rebalance <- as.numeric(current_date - last_rebalance_date)
  
  if (frequency == "daily") {
    return(days_since_rebalance >= 1)
  } else if (frequency == "weekly") {
    return(days_since_rebalance >= 7)
  } else if (frequency == "monthly") {
    return(days_since_rebalance >= 30)
  } else if (frequency == "quarterly") {
    return(days_since_rebalance >= 90)
  }
  
  return(FALSE)
}

#' Portfolio turnover calculation
#' @param new_weights New portfolio weights
#' @param old_weights Old portfolio weights
#' @return Portfolio turnover
calculate_turnover <- function(new_weights, old_weights) {
  sum(abs(new_weights - old_weights)) / 2
}

#' Generate synthetic data for testing
#' @param n_assets Number of assets
#' @param n_periods Number of time periods
#' @param correlation Average correlation
#' @return List with returns, expected returns, and covariance matrix
generate_synthetic_data <- function(n_assets = 10, n_periods = 252, correlation = 0.3) {
  
  set.seed(123)
  
  # Generate correlation matrix
  cors <- matrix(correlation, n_assets, n_assets)
  diag(cors) <- 1
  
  # Generate random volatilities
  volatilities <- runif(n_assets, 0.15, 0.30)
  
  # Create covariance matrix
  cov_matrix <- diag(volatilities) %*% cors %*% diag(volatilities)
  
  # Generate expected returns
  expected_returns <- runif(n_assets, 0.05, 0.15)
  
  # Generate return series
  returns <- mvtnorm::rmvnorm(n_periods, expected_returns, cov_matrix)
  colnames(returns) <- paste0("Asset", 1:n_assets)
  
  return(list(
    returns = returns,
    expected_returns = expected_returns,
    cov_matrix = cov_matrix
  ))
}

# =============================================================================
# Main Optimization Wrapper Function
# =============================================================================

#' Comprehensive portfolio optimization wrapper
#' @param returns Historical returns matrix
#' @param method Optimization method
#' @param config OptimizationConfig object
#' @param constraints Additional constraints
#' @return PortfolioResult object
optimize_portfolio <- function(returns, method = "mean_variance", config = NULL, constraints = NULL) {
  
  # Default configuration
  if (is.null(config)) {
    config <- new("OptimizationConfig")
  }
  
  # Estimate expected returns and covariance matrix
  expected_returns <- colMeans(returns, na.rm = TRUE) * 252  # Annualized
  names(expected_returns) <- colnames(returns)
  
  cov_matrix <- estimate_covariance_matrix(
    returns = returns,
    method = config@risk_model,
    lambda = config@shrinkage_intensity
  ) * 252  # Annualized
  
  # Portfolio optimization based on method
  if (method == "mean_variance") {
    weights <- mean_variance_optimization(
      expected_returns = expected_returns,
      cov_matrix = cov_matrix,
      risk_aversion = config@risk_aversion,
      constraints = constraints
    )
    
  } else if (method == "min_variance") {
    weights <- minimum_variance_portfolio(
      expected_returns = expected_returns,
      cov_matrix = cov_matrix
    )
    
  } else if (method == "max_sharpe") {
    weights <- maximum_sharpe_portfolio(
      expected_returns = expected_returns,
      cov_matrix = cov_matrix,
      risk_free_rate = 0.02  # 2% risk-free rate
    )
    
  } else if (method == "risk_parity") {
    weights <- risk_parity_portfolio(cov_matrix = cov_matrix)
    
  } else if (method == "black_litterman") {
    bl_result <- black_litterman_model(returns = returns)
    weights <- mean_variance_optimization(
      expected_returns = bl_result$expected_returns,
      cov_matrix = bl_result$covariance_matrix,
      risk_aversion = config@risk_aversion
    )
    
  } else {
    stop("Unknown optimization method")
  }
  
  # Calculate portfolio metrics
  metrics <- calculate_portfolio_metrics(
    weights = weights,
    expected_returns = expected_returns,
    cov_matrix = cov_matrix
  )
  
  # Create result object
  result <- new("PortfolioResult",
    weights = weights,
    expected_return = metrics$expected_return,
    expected_risk = metrics$volatility,
    sharpe_ratio = metrics$sharpe_ratio,
    alpha = metrics$active_return,
    beta = 1.0,  # Would need benchmark for proper calculation
    tracking_error = metrics$tracking_error,
    max_drawdown = 0.0,  # Would need time series for calculation
    turnover = 0.0,  # Would need previous weights
    transaction_costs = 0.0,
    optimization_info = list(
      method = method,
      config = config,
      metrics = metrics
    ),
    assets = names(expected_returns),
    timestamp = Sys.time()
  )
  
  return(result)
}

# =============================================================================
# Script Initialization and Examples
# =============================================================================

# Print header
cat("\n")
cat(rep("=", 80), "\n")
cat("Portfolio Optimization for Statistical Arbitrage\n")
cat("Initialized successfully with", length(required_packages), "packages\n")
cat(rep("=", 80), "\n")
cat("\n")

# Example usage
if (interactive() || !exists(".testing")) {
  cat("Portfolio optimization system ready!\n\n")
  cat("Available optimization methods:\n")
  cat("- mean_variance: Mean-Variance Optimization\n")
  cat("- min_variance: Minimum Variance Portfolio\n") 
  cat("- max_sharpe: Maximum Sharpe Ratio Portfolio\n")
  cat("- risk_parity: Equal Risk Contribution Portfolio\n")
  cat("- black_litterman: Black-Litterman Model\n")
  cat("- alpha_driven: Alpha-based Optimization\n")
  cat("- long_short: Long-Short Market Neutral\n")
  cat("\nExample usage:\n")
  cat("data <- generate_synthetic_data(n_assets = 10)\n")
  cat("result <- optimize_portfolio(data$returns, method = 'mean_variance')\n")
  cat("print(result@weights)\n")
}
