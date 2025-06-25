#!/usr/bin/env Rscript

# =============================================================================
# R Test Runner Script
# =============================================================================
# Purpose: Execute all R statistical model tests with proper reporting
# Usage: Rscript run_r_tests.R [options]
# =============================================================================

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Default options
options <- list(
  verbose = FALSE,
  coverage = FALSE,
  output_format = "text",
  test_filter = NULL,
  install_deps = FALSE
)

# Parse arguments
for (arg in args) {
  if (arg == "--verbose" || arg == "-v") {
    options$verbose <- TRUE
  } else if (arg == "--coverage" || arg == "-c") {
    options$coverage <- TRUE
  } else if (arg == "--html") {
    options$output_format <- "html"
  } else if (arg == "--junit") {
    options$output_format <- "junit"
  } else if (arg == "--install-deps" || arg == "-i") {
    options$install_deps <- TRUE
  } else if (startsWith(arg, "--filter=")) {
    options$test_filter <- sub("^--filter=", "", arg)
  } else if (arg == "--help" || arg == "-h") {
    cat("R Test Runner for Statistical Models\n")
    cat("====================================\n\n")
    cat("Usage: Rscript run_r_tests.R [options]\n\n")
    cat("Options:\n")
    cat("  -h, --help         Show this help message\n")
    cat("  -v, --verbose      Run tests with verbose output\n")
    cat("  -c, --coverage     Generate code coverage report\n")
    cat("  -i, --install-deps Install missing dependencies\n")
    cat("  --html             Generate HTML test report\n")
    cat("  --junit            Generate JUnit XML report\n")
    cat("  --filter=PATTERN   Run only tests matching pattern\n\n")
    cat("Examples:\n")
    cat("  Rscript run_r_tests.R --verbose\n")
    cat("  Rscript run_r_tests.R --coverage --html\n")
    cat("  Rscript run_r_tests.R --filter='ARIMA'\n\n")
    quit(status = 0)
  }
}

# Function to install required packages
install_dependencies <- function() {
  cat("Installing R package dependencies...\n")
  
  # Core testing packages
  testing_packages <- c("testthat", "covr", "devtools")
  
  # Statistical analysis packages
  stat_packages <- c(
    "stats", "forecast", "tseries", "lmtest", "quantmod", "urca", "vars",
    "zoo", "xts", "PerformanceAnalytics", "corrplot", "ggplot2", "dplyr",
    "tidyr", "bcp", "strucchange", "dynlm", "car", "sandwich", "MASS",
    "rugarch", "rmgarch", "MTS", "fGarch", "ccgarch", "egcm", "tsDyn"
  )
  
  all_packages <- c(testing_packages, stat_packages)
  
  # Check which packages are missing
  installed_packages <- rownames(installed.packages())
  missing_packages <- all_packages[!all_packages %in% installed_packages]
  
  if (length(missing_packages) > 0) {
    cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
    
    # Try to install from CRAN
    tryCatch({
      install.packages(missing_packages, dependencies = TRUE, repos = "https://cloud.r-project.org/")
    }, error = function(e) {
      cat("Error installing packages:", e$message, "\n")
      cat("Some packages may need to be installed manually\n")
    })
  } else {
    cat("All required packages are already installed\n")
  }
}

# Function to run tests with different reporters
run_tests <- function(verbose = FALSE, output_format = "text", test_filter = NULL) {
  library(testthat)
  
  # Set working directory to test directory
  test_dir <- file.path("src", "tests", "R_tests")
  if (!dir.exists(test_dir)) {
    cat("Error: Test directory not found:", test_dir, "\n")
    quit(status = 1)
  }
  
  original_wd <- getwd()
  setwd(test_dir)
  
  tryCatch({
    # Choose reporter based on output format
    if (output_format == "html") {
      reporter <- "html"
      output_file <- "test_results.html"
    } else if (output_format == "junit") {
      reporter <- "junit"
      output_file <- "test_results.xml"
    } else if (verbose) {
      reporter <- "progress"
      output_file <- NULL
    } else {
      reporter <- "summary"
      output_file <- NULL
    }
    
    cat("Running R Statistical Model Tests\n")
    cat("=================================\n")
    cat("Test directory:", getwd(), "\n")
    cat("Reporter:", reporter, "\n")
    if (!is.null(test_filter)) {
      cat("Filter:", test_filter, "\n")
    }
    cat("\n")
    
    # Run tests
    if (is.null(test_filter)) {
      # Run all tests
      if (is.null(output_file)) {
        result <- test_file("test_stat_models.R", reporter = reporter)
      } else {
        result <- test_file("test_stat_models.R", reporter = reporter, output_file = output_file)
      }
    } else {
      # Run filtered tests
      cat("Note: Test filtering not fully implemented. Running all tests.\n\n")
      if (is.null(output_file)) {
        result <- test_file("test_stat_models.R", reporter = reporter)
      } else {
        result <- test_file("test_stat_models.R", reporter = reporter, output_file = output_file)
      }
    }
    
    # Print summary
    if (exists("result") && !is.null(result)) {
      cat("\nTest Results Summary:\n")
      cat("====================\n")
      if (is.list(result) && "results" %in% names(result)) {
        passed <- sum(sapply(result$results, function(x) x$passed))
        failed <- sum(sapply(result$results, function(x) x$failed))
        cat("Passed:", passed, "\n")
        cat("Failed:", failed, "\n")
      }
      
      if (!is.null(output_file) && file.exists(output_file)) {
        cat("Results saved to:", output_file, "\n")
      }
    }
    
    return(result)
    
  }, error = function(e) {
    cat("Error running tests:", e$message, "\n")
    return(NULL)
  }, finally = {
    setwd(original_wd)
  })
}

# Function to generate coverage report
generate_coverage <- function() {
  cat("Generating code coverage report...\n")
  
  tryCatch({
    library(covr)
    
    # Calculate coverage for the statistical models
    cov <- file_coverage(
      source_files = "../../R/statistical_analysis/stat_models.R",
      test_files = "test_stat_models.R"
    )
    
    # Print coverage summary
    cat("\nCode Coverage Summary:\n")
    cat("=====================\n")
    print(cov)
    
    # Generate HTML coverage report
    report_file <- "coverage_report.html"
    tryCatch({
      covr::report(cov, file = report_file)
      cat("Coverage report saved to:", report_file, "\n")
    }, error = function(e) {
      cat("Could not generate HTML coverage report:", e$message, "\n")
    })
    
    return(cov)
    
  }, error = function(e) {
    cat("Error generating coverage report:", e$message, "\n")
    cat("Make sure the 'covr' package is installed\n")
    return(NULL)
  })
}

# Main execution
main <- function() {
  cat("R Statistical Models Test Runner\n")
  cat("===============================\n\n")
  
  # Install dependencies if requested
  if (options$install_deps) {
    install_dependencies()
    cat("\n")
  }
  
  # Check if testthat is available
  if (!requireNamespace("testthat", quietly = TRUE)) {
    cat("Error: testthat package is required but not installed\n")
    cat("Install it with: install.packages('testthat')\n")
    cat("Or run with --install-deps flag\n")
    quit(status = 1)
  }
  
  # Run tests
  start_time <- Sys.time()
  result <- run_tests(
    verbose = options$verbose,
    output_format = options$output_format,
    test_filter = options$test_filter
  )
  end_time <- Sys.time()
  
  cat("\nTest execution time:", 
      round(as.numeric(difftime(end_time, start_time, units = "secs")), 2), 
      "seconds\n")
  
  # Generate coverage if requested
  if (options$coverage) {
    cat("\n")
    coverage_result <- generate_coverage()
  }
  
  # Determine exit status
  if (is.null(result)) {
    exit_status <- 1
  } else if (is.list(result) && "results" %in% names(result)) {
    failed_tests <- sum(sapply(result$results, function(x) x$failed))
    exit_status <- if (failed_tests > 0) 1 else 0
  } else {
    exit_status <- 0
  }
  
  cat("\nTest run completed with exit status:", exit_status, "\n")
  quit(status = exit_status)
}

# Execute main function
main()
