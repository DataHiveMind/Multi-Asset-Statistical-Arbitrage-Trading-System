% =============================================================================
% Advanced Quantitative Methods for Statistical Arbitrage
% =============================================================================
% Purpose: Implements advanced quantitative methods leveraging MATLAB's
%          optimized numerical capabilities for statistical arbitrage systems
% Author: Statistical Arbitrage System
% Date: 2025-06-24
% Required Toolboxes: Financial, Statistics and ML, Optimization, Signal Processing
% =============================================================================

%% Initialize and Check Toolboxes
function init_quant_methods()
    fprintf('\n');
    fprintf(repmat('=', 1, 80)); fprintf('\n');
    fprintf('Advanced Quantitative Methods for Statistical Arbitrage\n');
    fprintf('MATLAB Implementation - Initialized\n');
    fprintf(repmat('=', 1, 80)); fprintf('\n');
    
    % Check required toolboxes
    required_toolboxes = {
        'Financial Toolbox', 'finance';
        'Statistics and Machine Learning Toolbox', 'stats';
        'Optimization Toolbox', 'optim';
        'Signal Processing Toolbox', 'signal';
        'Econometrics Toolbox', 'econ';
        'Parallel Computing Toolbox', 'parallel'
    };
    
    fprintf('\nChecking required toolboxes:\n');
    for i = 1:size(required_toolboxes, 1)
        if license('test', required_toolboxes{i, 2})
            fprintf('✓ %s: Available\n', required_toolboxes{i, 1});
        else
            fprintf('✗ %s: Not available\n', required_toolboxes{i, 1});
        end
    end
    fprintf('\n');
end

% =============================================================================
% Option Pricing Models
% =============================================================================

%% Black-Scholes-Merton Model with Greeks
function [price, greeks] = black_scholes_greeks(S, K, r, T, sigma, q, option_type)
    % Advanced Black-Scholes pricing with all Greeks
    % Inputs:
    %   S - Current stock price
    %   K - Strike price
    %   r - Risk-free rate
    %   T - Time to expiration
    %   sigma - Volatility
    %   q - Dividend yield
    %   option_type - 'call' or 'put'
    
    % Calculate d1 and d2
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T));
    d2 = d1 - sigma*sqrt(T);
    
    % Standard normal CDF and PDF
    N_d1 = normcdf(d1);
    N_d2 = normcdf(d2);
    n_d1 = normpdf(d1);
    
    if strcmpi(option_type, 'call')
        % Call option price
        price = S*exp(-q*T)*N_d1 - K*exp(-r*T)*N_d2;
        
        % Greeks for call
        delta = exp(-q*T) * N_d1;
        theta = (-S*exp(-q*T)*n_d1*sigma/(2*sqrt(T)) - ...
                r*K*exp(-r*T)*N_d2 + q*S*exp(-q*T)*N_d1) / 365;
    else
        % Put option price
        price = K*exp(-r*T)*normcdf(-d2) - S*exp(-q*T)*normcdf(-d1);
        
        % Greeks for put
        delta = exp(-q*T) * (N_d1 - 1);
        theta = (-S*exp(-q*T)*n_d1*sigma/(2*sqrt(T)) + ...
                r*K*exp(-r*T)*normcdf(-d2) - q*S*exp(-q*T)*normcdf(-d1)) / 365;
    end
    
    % Common Greeks
    gamma = exp(-q*T) * n_d1 / (S * sigma * sqrt(T));
    vega = S * exp(-q*T) * n_d1 * sqrt(T) / 100;
    rho = K * T * exp(-r*T) * N_d2 / 100;
    if strcmpi(option_type, 'put')
        rho = -K * T * exp(-r*T) * normcdf(-d2) / 100;
    end
    
    greeks = struct('delta', delta, 'gamma', gamma, 'theta', theta, ...
                   'vega', vega, 'rho', rho);
end

%% American Option Pricing using Binomial Trees
function [price, early_exercise_boundary] = american_option_binomial(S, K, r, T, sigma, q, option_type, n_steps)
    % American option pricing using Cox-Ross-Rubinstein binomial model
    if nargin < 8, n_steps = 100; end
    
    dt = T / n_steps;
    u = exp(sigma * sqrt(dt));
    d = 1 / u;
    p = (exp((r-q)*dt) - d) / (u - d);
    
    % Initialize asset price tree
    S_tree = zeros(n_steps+1, n_steps+1);
    for i = 0:n_steps
        for j = 0:i
            S_tree(i+1, j+1) = S * u^j * d^(i-j);
        end
    end
    
    % Initialize option value tree
    V_tree = zeros(n_steps+1, n_steps+1);
    early_exercise_boundary = zeros(n_steps+1, 1);
    
    % Terminal condition
    for j = 0:n_steps
        if strcmpi(option_type, 'call')
            V_tree(n_steps+1, j+1) = max(S_tree(n_steps+1, j+1) - K, 0);
        else
            V_tree(n_steps+1, j+1) = max(K - S_tree(n_steps+1, j+1), 0);
        end
    end
    
    % Backward induction
    for i = n_steps-1:-1:0
        for j = 0:i
            % European value
            european_value = exp(-r*dt) * (p*V_tree(i+2, j+2) + (1-p)*V_tree(i+2, j+1));
            
            % Intrinsic value
            if strcmpi(option_type, 'call')
                intrinsic_value = max(S_tree(i+1, j+1) - K, 0);
            else
                intrinsic_value = max(K - S_tree(i+1, j+1), 0);
            end
            
            % American value (max of European and intrinsic)
            V_tree(i+1, j+1) = max(european_value, intrinsic_value);
            
            % Record early exercise boundary
            if intrinsic_value > european_value
                early_exercise_boundary(i+1) = S_tree(i+1, j+1);
            end
        end
    end
    
    price = V_tree(1, 1);
end

%% Heston Stochastic Volatility Model
function [call_price, put_price] = heston_option_pricing(S, K, r, T, v0, kappa, theta, sigma_v, rho, q)
    % Heston model option pricing using characteristic function
    if nargin < 10, q = 0; end
    
    % Model parameters
    params = [kappa, theta, sigma_v, rho, v0];
    
    % Characteristic function for Heston model
    char_func = @(u, params) heston_characteristic_function(u, S, r, q, T, params);
    
    % Call price using Fourier transform
    call_price = heston_fourier_inversion(S, K, r, q, T, params, 'call');
    put_price = heston_fourier_inversion(S, K, r, q, T, params, 'put');
end

function phi = heston_characteristic_function(u, S, r, q, T, params)
    % Heston characteristic function
    kappa = params(1);
    theta = params(2);
    sigma_v = params(3);
    rho = params(4);
    v0 = params(5);
    
    % Complex number calculations
    xi = kappa - rho * sigma_v * 1i * u;
    d = sqrt(xi.^2 + sigma_v^2 * (1i*u + u.^2));
    
    A1 = (1i*u + u.^2) * sinh(0.5*d*T);
    A2 = d * cosh(0.5*d*T) + xi * sinh(0.5*d*T);
    
    A = (1i*u + u.^2) * kappa * theta * T / sigma_v^2;
    B = 2 * kappa * theta / sigma_v^2 * log(2*d ./ (d*cosh(0.5*d*T) + xi*sinh(0.5*d*T)));
    C = v0 * A1 ./ A2;
    
    phi = exp(1i * u * (log(S) + (r-q)*T) + A + B - C);
end

function price = heston_fourier_inversion(S, K, r, q, T, params, option_type)
    % Fourier inversion for option pricing
    
    % Integration parameters
    alpha = 1.5;  % Damping parameter
    eta = 0.25;   % Grid spacing
    n = 2^12;     % Number of points
    
    % Simpson's rule weights
    lambda = 2*pi/(n*eta);
    b = n*lambda/2;
    
    % Integration points
    u = eta * (0:n-1)';
    
    % Characteristic function values
    if strcmpi(option_type, 'call')
        psi = exp(-1i*u*log(K)) .* heston_characteristic_function(u-(alpha+1)*1i, S, r, q, T, params) ...
              ./ (alpha^2 + alpha - u.^2 + 1i*(2*alpha+1)*u);
    else
        psi = exp(-1i*u*log(K)) .* heston_characteristic_function(u-(alpha+1)*1i, S, r, q, T, params) ...
              ./ (alpha^2 + alpha - u.^2 + 1i*(2*alpha+1)*u);
    end
    
    % Numerical integration
    integral = sum(real(psi .* exp(-1i*u*b) .* eta));
    
    if strcmpi(option_type, 'call')
        price = S*exp(-q*T) - sqrt(S*K)*exp(-r*T) * integral / pi;
    else
        price = sqrt(S*K)*exp(-r*T) * integral / pi - S*exp(-q*T) + K*exp(-r*T);
    end
end

% =============================================================================
% Numerical Methods for PDEs
% =============================================================================

%% Finite Difference Method for Option Pricing
function [option_grid, S_grid, t_grid] = finite_difference_option(S_max, K, r, T, sigma, option_type, n_S, n_t)
    % Finite difference method for European option pricing
    if nargin < 7, n_S = 100; end
    if nargin < 8, n_t = 100; end
    
    % Grid setup
    dS = S_max / n_S;
    dt = T / n_t;
    S_grid = (0:dS:S_max)';
    t_grid = (0:dt:T);
    
    % Initialize option value grid
    option_grid = zeros(n_S+1, n_t+1);
    
    % Boundary conditions
    if strcmpi(option_type, 'call')
        % Call option boundaries
        option_grid(:, end) = max(S_grid - K, 0);  % Terminal condition
        option_grid(1, :) = 0;                     % S = 0
        option_grid(end, :) = S_max - K*exp(-r*(T-t_grid));  % S = S_max
    else
        % Put option boundaries
        option_grid(:, end) = max(K - S_grid, 0);  % Terminal condition
        option_grid(1, :) = K*exp(-r*(T-t_grid)); % S = 0
        option_grid(end, :) = 0;                   % S = S_max
    end
    
    % Finite difference coefficients
    alpha = zeros(n_S-1, 1);
    beta = zeros(n_S-1, 1);
    gamma = zeros(n_S-1, 1);
    
    for i = 2:n_S
        S_i = S_grid(i);
        alpha(i-1) = 0.5*dt*(sigma^2*i^2 - r*i);
        beta(i-1) = 1 - dt*(sigma^2*i^2 + r);
        gamma(i-1) = 0.5*dt*(sigma^2*i^2 + r*i);
    end
    
    % Backward time stepping (implicit method)
    for j = n_t:-1:1
        % Tridiagonal system Ax = b
        A = diag(beta) + diag(alpha(2:end), -1) + diag(gamma(1:end-1), 1);
        b = option_grid(2:n_S, j+1);
        
        % Boundary adjustments
        b(1) = b(1) - alpha(1)*option_grid(1, j);
        b(end) = b(end) - gamma(end)*option_grid(end, j);
        
        % Solve tridiagonal system
        option_grid(2:n_S, j) = A \ b;
    end
end

%% Monte Carlo Methods for Complex Derivatives
function [price, std_error, paths] = monte_carlo_option(S, K, r, T, sigma, q, option_type, n_sims, n_steps, control_variate)
    % Monte Carlo simulation for option pricing with variance reduction
    if nargin < 9, n_steps = 252; end
    if nargin < 10, control_variate = false; end
    
    dt = T / n_steps;
    
    % Generate random paths
    Z = randn(n_sims, n_steps);
    
    % Initialize price paths
    S_paths = zeros(n_sims, n_steps+1);
    S_paths(:, 1) = S;
    
    % Simulate paths using geometric Brownian motion
    for i = 1:n_steps
        S_paths(:, i+1) = S_paths(:, i) .* exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z(:, i));
    end
    
    % Calculate payoffs
    if strcmpi(option_type, 'call')
        payoffs = max(S_paths(:, end) - K, 0);
    elseif strcmpi(option_type, 'put')
        payoffs = max(K - S_paths(:, end), 0);
    elseif strcmpi(option_type, 'asian_call')
        avg_prices = mean(S_paths(:, 2:end), 2);
        payoffs = max(avg_prices - K, 0);
    elseif strcmpi(option_type, 'asian_put')
        avg_prices = mean(S_paths(:, 2:end), 2);
        payoffs = max(K - avg_prices, 0);
    elseif strcmpi(option_type, 'barrier_up_out_call')
        barrier = 1.2 * S;  % 20% above current price
        knocked_out = any(S_paths >= barrier, 2);
        payoffs = max(S_paths(:, end) - K, 0);
        payoffs(knocked_out) = 0;
    end
    
    % Control variate (use Black-Scholes as control)
    if control_variate && (strcmpi(option_type, 'call') || strcmpi(option_type, 'put'))
        [bs_price, ~] = black_scholes_greeks(S, K, r, T, sigma, q, option_type);
        
        % Calculate control variate payoffs
        cv_payoffs = max(S_paths(:, end) - K, 0);
        if strcmpi(option_type, 'put')
            cv_payoffs = max(K - S_paths(:, end), 0);
        end
        
        % Optimal control variate coefficient
        beta_star = -cov(payoffs, cv_payoffs) / var(cv_payoffs);
        
        % Adjusted payoffs
        adjusted_payoffs = payoffs + beta_star * (cv_payoffs - bs_price);
        payoffs = adjusted_payoffs;
    end
    
    % Discount to present value
    discounted_payoffs = exp(-r*T) * payoffs;
    
    % Calculate price and standard error
    price = mean(discounted_payoffs);
    std_error = std(discounted_payoffs) / sqrt(n_sims);
    
    paths = S_paths;
end

% =============================================================================
% Advanced Signal Processing for Financial Time Series
% =============================================================================

%% Kalman Filter for State Space Models
function [filtered_states, predicted_states, log_likelihood] = kalman_filter_finance(observations, F, H, Q, R, x0, P0)
    % Kalman filter implementation for financial time series
    % F: state transition matrix
    % H: observation matrix  
    % Q: process noise covariance
    % R: observation noise covariance
    % x0: initial state
    % P0: initial state covariance
    
    [n_obs, T] = size(observations);
    n_states = size(F, 1);
    
    % Initialize storage
    filtered_states = zeros(n_states, T);
    predicted_states = zeros(n_states, T);
    filtered_covariances = zeros(n_states, n_states, T);
    predicted_covariances = zeros(n_states, n_states, T);
    
    % Initial conditions
    x_pred = x0;
    P_pred = P0;
    log_likelihood = 0;
    
    for t = 1:T
        % Prediction step
        if t > 1
            x_pred = F * filtered_states(:, t-1);
            P_pred = F * filtered_covariances(:, :, t-1) * F' + Q;
        end
        
        predicted_states(:, t) = x_pred;
        predicted_covariances(:, :, t) = P_pred;
        
        % Update step
        y = observations(:, t);
        y_pred = H * x_pred;
        innovation = y - y_pred;
        
        S = H * P_pred * H' + R;  % Innovation covariance
        K = P_pred * H' / S;      % Kalman gain
        
        % Filtered estimates
        filtered_states(:, t) = x_pred + K * innovation;
        filtered_covariances(:, :, t) = (eye(n_states) - K * H) * P_pred;
        
        % Log-likelihood update
        log_likelihood = log_likelihood - 0.5 * (log(det(S)) + innovation' / S * innovation);
    end
end

%% Wavelet Analysis for Multi-Scale Decomposition
function [coeffs, frequencies, cone_of_influence] = continuous_wavelet_transform(signal, dt, scales, wavelet_type)
    % Continuous wavelet transform for financial time series analysis
    if nargin < 4, wavelet_type = 'morlet'; end
    if nargin < 3, scales = 2.^(1:0.25:7); end
    
    N = length(signal);
    
    % Frequency vector
    frequencies = 1./(scales * dt);
    
    % FFT of signal
    signal_fft = fft(signal - mean(signal));
    
    % Initialize coefficient matrix
    coeffs = zeros(length(scales), N);
    
    for i = 1:length(scales)
        scale = scales(i);
        
        % Create wavelet in frequency domain
        if strcmpi(wavelet_type, 'morlet')
            % Morlet wavelet
            omega0 = 6;  % Central frequency
            k = (1:N) - (N+1)/2;
            k = k * 2 * pi / N;
            
            wavelet_fft = pi^(-1/4) * sqrt(2*scale) * exp(1i*omega0*scale*k) .* exp(-0.5*(scale*k).^2);
        end
        
        % Convolution in frequency domain
        conv_fft = signal_fft .* conj(wavelet_fft);
        coeffs(i, :) = ifft(conv_fft);
    end
    
    % Cone of influence (edge effects)
    cone_of_influence = sqrt(2) * scales;
end

%% Empirical Mode Decomposition (EMD)
function [imfs, residue] = empirical_mode_decomposition(signal, max_imfs)
    % Empirical Mode Decomposition for non-linear, non-stationary signals
    if nargin < 2, max_imfs = 10; end
    
    imfs = [];
    residue = signal(:);
    
    for imf_count = 1:max_imfs
        [imf, residue] = extract_imf(residue);
        
        if isempty(imf)
            break;
        end
        
        imfs = [imfs, imf];
        
        % Stop if residue is monotonic
        if is_monotonic(residue)
            break;
        end
    end
end

function [imf, residue] = extract_imf(signal)
    % Extract single IMF using sifting process
    h = signal;
    max_iterations = 100;
    tolerance = 0.2;
    
    for iter = 1:max_iterations
        % Find local maxima and minima
        [max_idx, min_idx] = find_extrema(h);
        
        if length(max_idx) < 2 || length(min_idx) < 2
            imf = [];
            residue = signal;
            return;
        end
        
        % Interpolate envelopes
        t = 1:length(h);
        upper_envelope = interp1(max_idx, h(max_idx), t, 'spline', 'extrap');
        lower_envelope = interp1(min_idx, h(min_idx), t, 'spline', 'extrap');
        
        % Calculate mean
        mean_envelope = (upper_envelope + lower_envelope) / 2;
        
        % New component
        h_new = h - mean_envelope(:);
        
        % Check stopping criterion
        sd = sum((h - h_new).^2) / sum(h.^2);
        
        if sd < tolerance
            imf = h_new;
            residue = signal - imf;
            return;
        end
        
        h = h_new;
    end
    
    imf = h;
    residue = signal - imf;
end

function [max_idx, min_idx] = find_extrema(signal)
    % Find local extrema
    max_idx = find(diff(sign(diff(signal))) < 0) + 1;
    min_idx = find(diff(sign(diff(signal))) > 0) + 1;
    
    % Add endpoints if they are extrema
    if signal(1) > signal(2)
        max_idx = [1; max_idx];
    elseif signal(1) < signal(2)
        min_idx = [1; min_idx];
    end
    
    if signal(end) > signal(end-1)
        max_idx = [max_idx; length(signal)];
    elseif signal(end) < signal(end-1)
        min_idx = [min_idx; length(signal)];
    end
end

function is_mono = is_monotonic(signal)
    % Check if signal is monotonic
    diff_signal = diff(signal);
    is_mono = all(diff_signal >= 0) || all(diff_signal <= 0);
end

% =============================================================================
% Advanced Optimization Methods
% =============================================================================

%% Genetic Algorithm for Portfolio Optimization
function [optimal_weights, optimal_fitness, convergence] = genetic_algorithm_portfolio(expected_returns, cov_matrix, constraints)
    % Genetic algorithm for portfolio optimization with complex constraints
    
    n_assets = length(expected_returns);
    
    % GA options
    options = optimoptions('ga', ...
        'PopulationSize', 100, ...
        'MaxGenerations', 200, ...
        'FunctionTolerance', 1e-6, ...
        'Display', 'iter');
    
    % Objective function (negative Sharpe ratio for minimization)
    objective = @(w) -portfolio_sharpe_ratio(w, expected_returns, cov_matrix);
    
    % Constraints
    if nargin < 3
        constraints = struct();
    end
    
    % Bounds
    lb = zeros(n_assets, 1);
    ub = ones(n_assets, 1);
    
    % Linear constraints (sum of weights = 1)
    Aeq = ones(1, n_assets);
    beq = 1;
    
    % Nonlinear constraints
    nonlcon = @(w) portfolio_constraints(w, constraints);
    
    % Run genetic algorithm
    [optimal_weights, optimal_fitness, ~, output] = ga(objective, n_assets, ...
        [], [], Aeq, beq, lb, ub, nonlcon, options);
    
    optimal_fitness = -optimal_fitness;  % Convert back to positive Sharpe ratio
    convergence = output;
end

function sharpe = portfolio_sharpe_ratio(weights, expected_returns, cov_matrix)
    % Calculate portfolio Sharpe ratio
    portfolio_return = weights' * expected_returns;
    portfolio_risk = sqrt(weights' * cov_matrix * weights);
    sharpe = portfolio_return / portfolio_risk;
end

function [c, ceq] = portfolio_constraints(weights, constraints)
    % Nonlinear constraints for portfolio optimization
    c = [];    % Inequality constraints
    ceq = [];  % Equality constraints
    
    % Maximum concentration constraint
    if isfield(constraints, 'max_weight')
        c = [c; weights - constraints.max_weight];
    end
    
    % Sector constraints
    if isfield(constraints, 'sector_weights') && isfield(constraints, 'sector_limits')
        sector_exposure = constraints.sector_weights' * weights;
        c = [c; sector_exposure - constraints.sector_limits(:, 2)];  % Upper limits
        c = [c; constraints.sector_limits(:, 1) - sector_exposure];  % Lower limits
    end
    
    % Turnover constraint
    if isfield(constraints, 'current_weights') && isfield(constraints, 'max_turnover')
        turnover = sum(abs(weights - constraints.current_weights)) / 2;
        c = [c; turnover - constraints.max_turnover];
    end
end

%% Particle Swarm Optimization for Signal Processing
function [optimal_params, optimal_fitness, swarm_history] = pso_signal_optimization(signal, parameter_bounds, objective_function)
    % Particle Swarm Optimization for signal processing parameter tuning
    
    % PSO parameters
    n_particles = 50;
    n_iterations = 100;
    w = 0.7;      % Inertia weight
    c1 = 1.5;     % Cognitive coefficient
    c2 = 1.5;     % Social coefficient
    
    n_params = size(parameter_bounds, 1);
    
    % Initialize particles
    particles = zeros(n_particles, n_params);
    velocities = zeros(n_particles, n_params);
    personal_best = zeros(n_particles, n_params);
    personal_best_fitness = inf(n_particles, 1);
    global_best = zeros(1, n_params);
    global_best_fitness = inf;
    
    % Random initialization
    for i = 1:n_particles
        for j = 1:n_params
            particles(i, j) = parameter_bounds(j, 1) + ...
                rand * (parameter_bounds(j, 2) - parameter_bounds(j, 1));
        end
        
        % Evaluate initial fitness
        fitness = objective_function(signal, particles(i, :));
        personal_best(i, :) = particles(i, :);
        personal_best_fitness(i) = fitness;
        
        if fitness < global_best_fitness
            global_best = particles(i, :);
            global_best_fitness = fitness;
        end
    end
    
    % PSO iterations
    swarm_history = zeros(n_iterations, 1);
    
    for iter = 1:n_iterations
        for i = 1:n_particles
            % Update velocity
            r1 = rand(1, n_params);
            r2 = rand(1, n_params);
            
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * r1 .* (personal_best(i, :) - particles(i, :)) + ...
                c2 * r2 .* (global_best - particles(i, :));
            
            % Update position
            particles(i, :) = particles(i, :) + velocities(i, :);
            
            % Apply bounds
            for j = 1:n_params
                particles(i, j) = max(parameter_bounds(j, 1), ...
                    min(parameter_bounds(j, 2), particles(i, j)));
            end
            
            % Evaluate fitness
            fitness = objective_function(signal, particles(i, :));
            
            % Update personal best
            if fitness < personal_best_fitness(i)
                personal_best(i, :) = particles(i, :);
                personal_best_fitness(i) = fitness;
                
                % Update global best
                if fitness < global_best_fitness
                    global_best = particles(i, :);
                    global_best_fitness = fitness;
                end
            end
        end
        
        swarm_history(iter) = global_best_fitness;
    end
    
    optimal_params = global_best;
    optimal_fitness = global_best_fitness;
end

% =============================================================================
% Statistical Arbitrage Specific Methods
% =============================================================================

%% Pairs Trading Signal Generation
function [signals, statistics] = pairs_trading_signals(price_series1, price_series2, lookback_window, entry_threshold, exit_threshold)
    % Generate pairs trading signals using cointegration and mean reversion
    if nargin < 3, lookback_window = 60; end
    if nargin < 4, entry_threshold = 2.0; end
    if nargin < 5, exit_threshold = 0.5; end
    
    n_obs = length(price_series1);
    signals = zeros(n_obs, 1);
    spread = zeros(n_obs, 1);
    zscore = zeros(n_obs, 1);
    
    for t = lookback_window:n_obs
        % Extract rolling window
        window_idx = (t-lookback_window+1):t;
        P1_window = price_series1(window_idx);
        P2_window = price_series2(window_idx);
        
        % Cointegration regression: P1 = alpha + beta * P2 + error
        X = [ones(lookback_window, 1), P2_window];
        coeffs = X \ P1_window;
        
        % Current spread
        current_spread = price_series1(t) - coeffs(1) - coeffs(2) * price_series2(t);
        spread(t) = current_spread;
        
        % Z-score of spread
        spread_window = spread(window_idx);
        spread_mean = mean(spread_window(spread_window ~= 0));
        spread_std = std(spread_window(spread_window ~= 0));
        
        if spread_std > 0
            zscore(t) = (current_spread - spread_mean) / spread_std;
        end
        
        % Generate signals
        if abs(zscore(t)) > entry_threshold
            if zscore(t) > 0
                signals(t) = -1;  % Short signal (spread too high)
            else
                signals(t) = 1;   % Long signal (spread too low)
            end
        elseif abs(zscore(t)) < exit_threshold
            signals(t) = 0;       % Exit signal
        else
            signals(t) = signals(t-1);  % Hold previous position
        end
    end
    
    % Calculate statistics
    statistics = struct();
    statistics.spread = spread;
    statistics.zscore = zscore;
    statistics.mean_reversion_half_life = calculate_half_life(spread);
    statistics.cointegration_pvalue = perform_cointegration_test(price_series1, price_series2);
end

function half_life = calculate_half_life(spread)
    % Calculate mean reversion half-life using AR(1) model
    spread_clean = spread(spread ~= 0);
    
    if length(spread_clean) < 10
        half_life = NaN;
        return;
    end
    
    % AR(1): spread(t) = a + b * spread(t-1) + error
    y = spread_clean(2:end);
    x = spread_clean(1:end-1);
    
    X = [ones(length(x), 1), x];
    coeffs = X \ y;
    
    b = coeffs(2);
    
    if b < 1 && b > 0
        half_life = -log(2) / log(b);
    else
        half_life = NaN;
    end
end

function p_value = perform_cointegration_test(series1, series2)
    % Simplified Engle-Granger cointegration test
    
    % Step 1: Cointegration regression
    X = [ones(length(series2), 1), series2];
    coeffs = X \ series1;
    residuals = series1 - X * coeffs;
    
    % Step 2: ADF test on residuals
    p_value = adf_test(residuals);
end

function p_value = adf_test(series)
    % Simplified Augmented Dickey-Fuller test
    % This is a basic implementation; for production use, consider econometrics toolbox
    
    n = length(series);
    y = series(2:end);
    x = series(1:end-1);
    
    % Delta y = alpha + beta * y(t-1) + error
    X = [ones(n-1, 1), x];
    coeffs = X \ y;
    
    beta = coeffs(2);
    
    % Calculate t-statistic
    residuals = y - X * coeffs;
    se_beta = sqrt(sum(residuals.^2) / (n-3)) / sqrt(sum((x - mean(x)).^2));
    t_stat = beta / se_beta;
    
    % Critical values (approximate)
    if t_stat < -3.43
        p_value = 0.01;
    elseif t_stat < -2.86
        p_value = 0.05;
    elseif t_stat < -2.57
        p_value = 0.10;
    else
        p_value = 0.15;
    end
end

%% Machine Learning Based Alpha Signals
function [alpha_signals, model_performance] = ml_alpha_signals(features, returns, model_type, training_ratio)
    % Generate alpha signals using machine learning models
    if nargin < 3, model_type = 'ensemble'; end
    if nargin < 4, training_ratio = 0.7; end
    
    % Data preparation
    [n_obs, n_features] = size(features);
    n_train = floor(n_obs * training_ratio);
    
    % Split data
    X_train = features(1:n_train, :);
    y_train = returns(1:n_train);
    X_test = features(n_train+1:end, :);
    y_test = returns(n_train+1:end);
    
    % Feature scaling
    feature_means = mean(X_train);
    feature_stds = std(X_train);
    X_train_scaled = (X_train - feature_means) ./ feature_stds;
    X_test_scaled = (X_test - feature_means) ./ feature_stds;
    
    if strcmpi(model_type, 'ensemble')
        % Ensemble of models
        alpha_signals = zeros(size(returns));
        
        % Random Forest
        rf_model = TreeBagger(100, X_train_scaled, y_train, 'Method', 'regression');
        rf_predictions = predict(rf_model, X_test_scaled);
        
        % Support Vector Regression
        svr_model = fitrsvm(X_train_scaled, y_train, 'KernelFunction', 'gaussian');
        svr_predictions = predict(svr_model, X_test_scaled);
        
        % Neural Network (if Deep Learning Toolbox available)
        if license('test', 'neural_network_toolbox')
            net = fitnet([10, 5]);  % Hidden layers
            net = train(net, X_train_scaled', y_train');
            nn_predictions = net(X_test_scaled')';
        else
            nn_predictions = zeros(size(svr_predictions));
        end
        
        % Ensemble prediction (equal weights)
        ensemble_predictions = (rf_predictions + svr_predictions + nn_predictions) / 3;
        alpha_signals(n_train+1:end) = ensemble_predictions;
        
    elseif strcmpi(model_type, 'ridge')
        % Ridge regression with cross-validation
        lambda_values = logspace(-4, 2, 50);
        cv_errors = zeros(length(lambda_values), 1);
        
        k_folds = 5;
        cv_indices = crossvalind('Kfold', n_train, k_folds);
        
        for i = 1:length(lambda_values)
            lambda = lambda_values(i);
            fold_errors = zeros(k_folds, 1);
            
            for k = 1:k_folds
                train_idx = cv_indices ~= k;
                val_idx = cv_indices == k;
                
                % Ridge regression
                ridge_coeffs = (X_train_scaled(train_idx, :)' * X_train_scaled(train_idx, :) + ...
                               lambda * eye(n_features)) \ ...
                               (X_train_scaled(train_idx, :)' * y_train(train_idx));
                
                % Validation predictions
                val_pred = X_train_scaled(val_idx, :) * ridge_coeffs;
                fold_errors(k) = mean((y_train(val_idx) - val_pred).^2);
            end
            
            cv_errors(i) = mean(fold_errors);
        end
        
        % Select optimal lambda
        [~, optimal_idx] = min(cv_errors);
        optimal_lambda = lambda_values(optimal_idx);
        
        % Final model
        ridge_coeffs = (X_train_scaled' * X_train_scaled + optimal_lambda * eye(n_features)) \ ...
                      (X_train_scaled' * y_train);
        
        % Predictions
        alpha_signals = zeros(size(returns));
        alpha_signals(n_train+1:end) = X_test_scaled * ridge_coeffs;
    end
    
    % Model performance metrics
    test_predictions = alpha_signals(n_train+1:end);
    mse = mean((y_test - test_predictions).^2);
    mae = mean(abs(y_test - test_predictions));
    correlation = corr(y_test, test_predictions);
    
    % Information coefficient (rank correlation)
    ic = corr(y_test, test_predictions, 'type', 'Spearman');
    
    model_performance = struct();
    model_performance.mse = mse;
    model_performance.mae = mae;
    model_performance.correlation = correlation;
    model_performance.information_coefficient = ic;
    model_performance.optimal_lambda = optimal_lambda;
end

% =============================================================================
% Risk Management and Position Sizing
% =============================================================================

%% Kelly Criterion for Optimal Position Sizing
function [optimal_fraction, expected_growth, kelly_stats] = kelly_criterion(win_probability, win_loss_ratio, max_fraction)
    % Calculate optimal position size using Kelly Criterion
    if nargin < 3, max_fraction = 0.25; end  % Cap at 25%
    
    % Kelly formula: f* = p - q/b
    % where p = win probability, q = loss probability, b = win/loss ratio
    q = 1 - win_probability;
    
    optimal_fraction = win_probability - q / win_loss_ratio;
    
    % Apply maximum fraction constraint
    optimal_fraction = min(optimal_fraction, max_fraction);
    optimal_fraction = max(optimal_fraction, 0);  % No negative positions
    
    % Expected growth rate
    if optimal_fraction > 0
        expected_growth = win_probability * log(1 + optimal_fraction * win_loss_ratio) + ...
                         q * log(1 - optimal_fraction);
    else
        expected_growth = 0;
    end
    
    % Additional statistics
    kelly_stats = struct();
    kelly_stats.unconstrained_fraction = win_probability - q / win_loss_ratio;
    kelly_stats.win_probability = win_probability;
    kelly_stats.loss_probability = q;
    kelly_stats.win_loss_ratio = win_loss_ratio;
    kelly_stats.expected_return = win_probability * win_loss_ratio - q;
end

%% Value at Risk (VaR) Calculations
function [var_estimates, expected_shortfall] = calculate_var(returns, confidence_levels, method)
    % Calculate Value at Risk using various methods
    if nargin < 2, confidence_levels = [0.95, 0.99]; end
    if nargin < 3, method = 'historical'; end
    
    returns = returns(:);  % Ensure column vector
    n_obs = length(returns);
    n_levels = length(confidence_levels);
    
    var_estimates = zeros(n_levels, 1);
    expected_shortfall = zeros(n_levels, 1);
    
    if strcmpi(method, 'historical')
        % Historical simulation
        sorted_returns = sort(returns);
        
        for i = 1:n_levels
            alpha = 1 - confidence_levels(i);
            var_idx = ceil(alpha * n_obs);
            var_estimates(i) = -sorted_returns(var_idx);
            
            % Expected Shortfall (Conditional VaR)
            tail_returns = sorted_returns(1:var_idx);
            expected_shortfall(i) = -mean(tail_returns);
        end
        
    elseif strcmpi(method, 'parametric')
        % Parametric (normal distribution)
        mu = mean(returns);
        sigma = std(returns);
        
        for i = 1:n_levels
            alpha = 1 - confidence_levels(i);
            var_estimates(i) = -(mu + norminv(alpha) * sigma);
            
            % Expected Shortfall for normal distribution
            expected_shortfall(i) = -(mu - sigma * normpdf(norminv(alpha)) / alpha);
        end
        
    elseif strcmpi(method, 'cornish_fisher')
        % Cornish-Fisher expansion (accounts for skewness and kurtosis)
        mu = mean(returns);
        sigma = std(returns);
        skewness = skewness(returns);
        kurt = kurtosis(returns) - 3;  % Excess kurtosis
        
        for i = 1:n_levels
            alpha = 1 - confidence_levels(i);
            z_alpha = norminv(alpha);
            
            % Cornish-Fisher quantile
            cf_quantile = z_alpha + ...
                (z_alpha^2 - 1) * skewness / 6 + ...
                (z_alpha^3 - 3*z_alpha) * kurt / 24 - ...
                (2*z_alpha^3 - 5*z_alpha) * skewness^2 / 36;
            
            var_estimates(i) = -(mu + cf_quantile * sigma);
            
            % Approximate Expected Shortfall
            expected_shortfall(i) = var_estimates(i) * 1.2;  % Rough approximation
        end
    end
end

% =============================================================================
% Utility Functions and Testing
% =============================================================================

%% Generate Synthetic Market Data
function [prices, returns, volatility] = generate_synthetic_data(n_assets, n_periods, correlation_matrix)
    % Generate synthetic market data for testing
    if nargin < 3
        correlation_matrix = 0.3 * ones(n_assets) + 0.7 * eye(n_assets);
    end
    
    % Parameters
    dt = 1/252;  % Daily data
    mu = 0.08 + 0.04 * randn(n_assets, 1);  % Expected returns
    sigma = 0.15 + 0.1 * abs(randn(n_assets, 1));  % Volatilities
    
    % Generate correlated random returns
    L = chol(correlation_matrix, 'lower');  % Cholesky decomposition
    Z = randn(n_periods, n_assets);
    correlated_Z = Z * L';
    
    % Generate returns using geometric Brownian motion
    returns = repmat(mu', n_periods, 1) * dt + ...
              repmat(sigma', n_periods, 1) .* sqrt(dt) .* correlated_Z;
    
    % Generate price paths
    prices = zeros(n_periods+1, n_assets);
    prices(1, :) = 100;  % Initial price
    
    for t = 1:n_periods
        prices(t+1, :) = prices(t, :) .* exp(returns(t, :));
    end
    
    % Calculate realized volatility
    volatility = std(returns) * sqrt(252);
end

%% Performance Testing Suite
function run_performance_tests()
    % Comprehensive performance testing of quantitative methods
    
    fprintf('Running Performance Tests for Quantitative Methods...\n\n');
    
    % Test 1: Option Pricing Speed
    fprintf('Test 1: Option Pricing Performance\n');
    S = 100; K = 100; r = 0.05; T = 0.25; sigma = 0.2; q = 0.02;
    
    tic;
    for i = 1:1000
        [price, greeks] = black_scholes_greeks(S, K, r, T, sigma, q, 'call');
    end
    bs_time = toc;
    fprintf('Black-Scholes (1000 calculations): %.4f seconds\n', bs_time);
    
    tic;
    [am_price, ~] = american_option_binomial(S, K, r, T, sigma, q, 'call', 100);
    binomial_time = toc;
    fprintf('American Binomial (100 steps): %.4f seconds\n', binomial_time);
    
    % Test 2: Monte Carlo Performance
    fprintf('\nTest 2: Monte Carlo Performance\n');
    tic;
    [mc_price, ~, ~] = monte_carlo_option(S, K, r, T, sigma, q, 'call', 100000, 252);
    mc_time = toc;
    fprintf('Monte Carlo (100k paths, 252 steps): %.4f seconds\n', mc_time);
    
    % Test 3: Signal Processing Performance
    fprintf('\nTest 3: Signal Processing Performance\n');
    signal_length = 10000;
    test_signal = cumsum(randn(signal_length, 1));
    
    tic;
    [imfs, residue] = empirical_mode_decomposition(test_signal, 5);
    emd_time = toc;
    fprintf('EMD (10k points, 5 IMFs): %.4f seconds\n', emd_time);
    
    % Test 4: Optimization Performance
    fprintf('\nTest 4: Portfolio Optimization Performance\n');
    n_assets = 50;
    expected_returns = 0.08 + 0.04 * randn(n_assets, 1);
    correlation_matrix = 0.3 * ones(n_assets) + 0.7 * eye(n_assets);
    volatilities = 0.15 + 0.1 * abs(randn(n_assets, 1));
    cov_matrix = diag(volatilities) * correlation_matrix * diag(volatilities);
    
    tic;
    [opt_weights, ~, ~] = genetic_algorithm_portfolio(expected_returns, cov_matrix);
    ga_time = toc;
    fprintf('Genetic Algorithm Portfolio (50 assets): %.4f seconds\n', ga_time);
    
    fprintf('\nPerformance tests completed.\n');
end

% =============================================================================
% Main Initialization
% =============================================================================

% Initialize the quantitative methods module
init_quant_methods();

% Print available functions
fprintf('Available Quantitative Methods:\n\n');
fprintf('Option Pricing:\n');
fprintf('  - black_scholes_greeks: Black-Scholes with all Greeks\n');
fprintf('  - american_option_binomial: American options using binomial trees\n');
fprintf('  - heston_option_pricing: Heston stochastic volatility model\n');
fprintf('  - finite_difference_option: PDE finite difference method\n');
fprintf('  - monte_carlo_option: Monte Carlo with variance reduction\n\n');

fprintf('Signal Processing:\n');
fprintf('  - kalman_filter_finance: Kalman filter for state space models\n');
fprintf('  - continuous_wavelet_transform: Multi-scale wavelet analysis\n');
fprintf('  - empirical_mode_decomposition: EMD for non-stationary signals\n\n');

fprintf('Optimization:\n');
fprintf('  - genetic_algorithm_portfolio: GA for portfolio optimization\n');
fprintf('  - pso_signal_optimization: PSO for signal processing\n\n');

fprintf('Statistical Arbitrage:\n');
fprintf('  - pairs_trading_signals: Cointegration-based pairs trading\n');
fprintf('  - ml_alpha_signals: Machine learning alpha generation\n\n');

fprintf('Risk Management:\n');
fprintf('  - kelly_criterion: Optimal position sizing\n');
fprintf('  - calculate_var: Value at Risk estimation\n\n');

fprintf('Utilities:\n');
fprintf('  - generate_synthetic_data: Synthetic market data generation\n');
fprintf('  - run_performance_tests: Performance benchmarking\n\n');

fprintf('Quantitative Methods module ready for use!\n');
fprintf('Example: [price, greeks] = black_scholes_greeks(100, 100, 0.05, 0.25, 0.2, 0, ''call'')\n\n');