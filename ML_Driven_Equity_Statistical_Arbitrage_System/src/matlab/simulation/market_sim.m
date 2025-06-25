% =============================================================================
% Market Simulation for Statistical Arbitrage Testing
% =============================================================================
% Purpose: Comprehensive market simulation framework for testing strategies
%          under various hypothetical market conditions and stress scenarios
% Author: Statistical Arbitrage System
% Date: 2025-06-24
% Required: Statistics and ML Toolbox, Financial Toolbox, Parallel Computing Toolbox
% =============================================================================

%% Initialize Market Simulation Framework
function init_market_simulation()
    fprintf('\n');
    fprintf(repmat('=', 1, 80)); fprintf('\n');
    fprintf('Market Simulation Framework for Statistical Arbitrage\n');
    fprintf('MATLAB Implementation - Initialized\n');
    fprintf(repmat('=', 1, 80)); fprintf('\n');
    
    % Check required toolboxes
    required_toolboxes = {
        'Statistics and Machine Learning Toolbox', 'stats';
        'Financial Toolbox', 'finance';
        'Parallel Computing Toolbox', 'parallel';
        'Optimization Toolbox', 'optim'
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
% Market Simulation Configuration
% =============================================================================

%% Market Simulation Parameters Structure
function params = create_simulation_params()
    % Create default simulation parameters structure
    
    params = struct();
    
    % Time parameters
    params.T = 1.0;              % Total simulation time (years)
    params.dt = 1/252;           % Time step (daily)
    params.n_steps = round(params.T / params.dt);
    params.n_assets = 10;        % Number of assets
    params.n_simulations = 1000; % Number of Monte Carlo paths
    
    % Market regime parameters
    params.regime_model = 'markov_switching';  % 'constant', 'markov_switching', 'stochastic_vol'
    params.n_regimes = 3;        % Number of market regimes
    
    % Asset dynamics
    params.price_model = 'gbm';  % 'gbm', 'jump_diffusion', 'heston', 'merton_jump'
    params.correlation_model = 'constant';  % 'constant', 'dcc', 'stochastic'
    
    % Economic scenarios
    params.stress_scenarios = {'normal', 'crisis', 'bubble', 'recession'};
    params.include_microstructure = true;
    params.include_liquidity_effects = true;
    
    % Random seed for reproducibility
    params.random_seed = 12345;
end

% =============================================================================
% Stochastic Process Models
% =============================================================================

%% Geometric Brownian Motion with Regime Switching
function [prices, returns, regimes] = simulate_regime_switching_gbm(params)
    % Simulate asset prices using regime-switching GBM
    
    rng(params.random_seed);
    
    n_steps = params.n_steps;
    n_assets = params.n_assets;
    n_sims = params.n_simulations;
    dt = params.dt;
    
    % Define regime parameters
    regimes_params = define_market_regimes(params.n_regimes);
    
    % Initialize storage
    prices = zeros(n_steps + 1, n_assets, n_sims);
    returns = zeros(n_steps, n_assets, n_sims);
    regimes = zeros(n_steps, n_sims);
    
    % Markov chain transition matrix for regimes
    P_regime = create_regime_transition_matrix(params.n_regimes);
    
    for sim = 1:n_sims
        % Initialize
        prices(1, :, sim) = 100;  % Initial price
        current_regime = 1;       % Start in regime 1
        
        for t = 1:n_steps
            % Regime transition
            if t > 1
                current_regime = simulate_regime_transition(current_regime, P_regime);
            end
            regimes(t, sim) = current_regime;
            
            % Get current regime parameters
            mu = regimes_params.drift(current_regime, :);
            sigma = regimes_params.volatility(current_regime, :);
            corr_matrix = regimes_params.correlation{current_regime};
            
            % Generate correlated innovations
            Z = mvnrnd(zeros(1, n_assets), corr_matrix);
            
            % GBM step
            drift_term = (mu - 0.5 * sigma.^2) * dt;
            diffusion_term = sigma .* sqrt(dt) .* Z;
            
            returns(t, :, sim) = drift_term + diffusion_term;
            prices(t+1, :, sim) = prices(t, :, sim) .* exp(returns(t, :, sim));
        end
    end
end

%% Jump-Diffusion Model (Merton Model)
function [prices, returns, jumps] = simulate_jump_diffusion(params)
    % Simulate asset prices using Merton jump-diffusion model
    
    rng(params.random_seed);
    
    n_steps = params.n_steps;
    n_assets = params.n_assets;
    n_sims = params.n_simulations;
    dt = params.dt;
    
    % Jump parameters
    lambda = 0.1;           % Jump intensity (jumps per year)
    mu_j = -0.05;          % Mean jump size
    sigma_j = 0.15;        % Jump volatility
    
    % Diffusion parameters
    mu = 0.08 * ones(1, n_assets);           % Drift
    sigma = 0.2 * ones(1, n_assets);         % Volatility
    correlation_matrix = 0.3 * ones(n_assets) + 0.7 * eye(n_assets);
    
    % Initialize storage
    prices = zeros(n_steps + 1, n_assets, n_sims);
    returns = zeros(n_steps, n_assets, n_sims);
    jumps = cell(n_sims, 1);
    
    for sim = 1:n_sims
        prices(1, :, sim) = 100;  % Initial price
        sim_jumps = [];
        
        for t = 1:n_steps
            % Diffusion component
            Z = mvnrnd(zeros(1, n_assets), correlation_matrix);
            diffusion_return = (mu - 0.5 * sigma.^2) * dt + sigma .* sqrt(dt) .* Z;
            
            % Jump component
            N_jumps = poissrnd(lambda * dt, 1, n_assets);  % Number of jumps
            jump_return = zeros(1, n_assets);
            
            for asset = 1:n_assets
                if N_jumps(asset) > 0
                    jump_sizes = normrnd(mu_j, sigma_j, 1, N_jumps(asset));
                    jump_return(asset) = sum(jump_sizes);
                    
                    % Record jump information
                    for jump = 1:N_jumps(asset)
                        sim_jumps = [sim_jumps; t, asset, jump_sizes(jump)];
                    end
                end
            end
            
            % Total return
            total_return = diffusion_return + jump_return;
            returns(t, :, sim) = total_return;
            prices(t+1, :, sim) = prices(t, :, sim) .* exp(total_return);
        end
        
        jumps{sim} = sim_jumps;
    end
end

%% Heston Stochastic Volatility Model
function [prices, returns, volatilities] = simulate_heston_model(params)
    % Simulate asset prices using Heston stochastic volatility model
    
    rng(params.random_seed);
    
    n_steps = params.n_steps;
    n_assets = params.n_assets;
    n_sims = params.n_simulations;
    dt = params.dt;
    
    % Heston parameters (per asset)
    mu = 0.08 * ones(1, n_assets);      % Drift
    kappa = 2.0 * ones(1, n_assets);    % Mean reversion speed
    theta = 0.04 * ones(1, n_assets);   % Long-term variance
    xi = 0.3 * ones(1, n_assets);       % Volatility of volatility
    rho = -0.7 * ones(1, n_assets);     % Correlation between price and vol
    
    % Initialize storage
    prices = zeros(n_steps + 1, n_assets, n_sims);
    returns = zeros(n_steps, n_assets, n_sims);
    volatilities = zeros(n_steps + 1, n_assets, n_sims);
    
    for sim = 1:n_sims
        prices(1, :, sim) = 100;
        volatilities(1, :, sim) = sqrt(theta);  % Initial volatility
        
        for t = 1:n_steps
            % Current volatility
            v_t = volatilities(t, :, sim).^2;
            
            % Generate correlated Brownian motions
            Z = randn(2, n_assets);
            W1 = Z(1, :);  % For price process
            W2 = rho .* W1 + sqrt(1 - rho.^2) .* Z(2, :);  % For volatility process
            
            % Update variance using full truncation scheme
            v_next = v_t + kappa .* (theta - max(v_t, 0)) * dt + ...
                     xi .* sqrt(max(v_t, 0) * dt) .* W2;
            v_next = max(v_next, 0);  % Ensure non-negative variance
            
            volatilities(t+1, :, sim) = sqrt(v_next);
            
            % Update price
            price_return = (mu - 0.5 * v_t) * dt + sqrt(v_t * dt) .* W1;
            returns(t, :, sim) = price_return;
            prices(t+1, :, sim) = prices(t, :, sim) .* exp(price_return);
        end
    end
end

%% Dynamic Conditional Correlation (DCC) Model
function [returns, correlation_matrices] = simulate_dcc_model(params)
    % Simulate returns with time-varying correlations using DCC model
    
    rng(params.random_seed);
    
    n_steps = params.n_steps;
    n_assets = params.n_assets;
    n_sims = params.n_simulations;
    
    % DCC parameters
    alpha = 0.05;  % Short-term correlation persistence
    beta = 0.90;   % Long-term correlation persistence
    
    % Initialize storage
    returns = zeros(n_steps, n_assets, n_sims);
    correlation_matrices = zeros(n_assets, n_assets, n_steps, n_sims);
    
    for sim = 1:n_sims
        % Initial correlation matrix
        R_bar = 0.3 * ones(n_assets) + 0.7 * eye(n_assets);  % Unconditional correlation
        Q_t = R_bar;  % Initial Q matrix
        
        for t = 1:n_steps
            % Generate standardized innovations
            z_t = mvnrnd(zeros(1, n_assets), eye(n_assets));
            
            if t > 1
                % Update Q matrix
                z_prev = returns(t-1, :, sim) ./ sqrt(diag(cov(returns(1:t-1, :, sim)))');
                Q_t = (1 - alpha - beta) * R_bar + alpha * (z_prev' * z_prev) + beta * Q_t;
            end
            
            % Convert Q to correlation matrix
            Q_diag_inv_sqrt = diag(1 ./ sqrt(diag(Q_t)));
            R_t = Q_diag_inv_sqrt * Q_t * Q_diag_inv_sqrt;
            
            % Ensure positive definiteness
            [V, D] = eig(R_t);
            D = diag(max(diag(D), 1e-8));
            R_t = V * D * V';
            
            correlation_matrices(:, :, t, sim) = R_t;
            
            % Generate correlated returns
            volatilities = 0.15 + 0.1 * abs(randn(1, n_assets));  % Time-varying volatility
            returns(t, :, sim) = (mvnrnd(zeros(1, n_assets), R_t) .* volatilities);
        end
    end
end

% =============================================================================
% Market Regime Definition and Transition
% =============================================================================

%% Define Market Regimes
function regimes_params = define_market_regimes(n_regimes)
    % Define parameters for different market regimes
    
    regimes_params = struct();
    
    if n_regimes == 2
        % Bull and Bear markets
        regimes_params.names = {'Bull', 'Bear'};
        regimes_params.drift = [0.12; -0.05] * ones(1, 10);           % Annual drift
        regimes_params.volatility = [0.15; 0.35] * ones(1, 10);       % Annual volatility
        
        % Correlation matrices
        regimes_params.correlation{1} = 0.2 * ones(10) + 0.8 * eye(10);  % Bull: lower correlation
        regimes_params.correlation{2} = 0.6 * ones(10) + 0.4 * eye(10);  % Bear: higher correlation
        
    elseif n_regimes == 3
        % Normal, Crisis, Recovery
        regimes_params.names = {'Normal', 'Crisis', 'Recovery'};
        regimes_params.drift = [0.08; -0.15; 0.20] * ones(1, 10);
        regimes_params.volatility = [0.18; 0.45; 0.25] * ones(1, 10);
        
        regimes_params.correlation{1} = 0.3 * ones(10) + 0.7 * eye(10);  % Normal
        regimes_params.correlation{2} = 0.8 * ones(10) + 0.2 * eye(10);  % Crisis: very high correlation
        regimes_params.correlation{3} = 0.4 * ones(10) + 0.6 * eye(10);  % Recovery
        
    elseif n_regimes == 4
        % Bull, Normal, Bear, Crisis
        regimes_params.names = {'Bull', 'Normal', 'Bear', 'Crisis'};
        regimes_params.drift = [0.15; 0.08; -0.05; -0.25] * ones(1, 10);
        regimes_params.volatility = [0.12; 0.18; 0.30; 0.50] * ones(1, 10);
        
        regimes_params.correlation{1} = 0.2 * ones(10) + 0.8 * eye(10);  % Bull
        regimes_params.correlation{2} = 0.3 * ones(10) + 0.7 * eye(10);  % Normal
        regimes_params.correlation{3} = 0.5 * ones(10) + 0.5 * eye(10);  % Bear
        regimes_params.correlation{4} = 0.9 * ones(10) + 0.1 * eye(10);  % Crisis
    end
end

%% Create Regime Transition Matrix
function P = create_regime_transition_matrix(n_regimes)
    % Create Markov chain transition matrix for regime switching
    
    if n_regimes == 2
        % Bull/Bear with moderate persistence
        P = [0.95, 0.05;    % Bull -> {Bull, Bear}
             0.10, 0.90];   % Bear -> {Bull, Bear}
        
    elseif n_regimes == 3
        % Normal/Crisis/Recovery with realistic transitions
        P = [0.90, 0.08, 0.02;   % Normal -> {Normal, Crisis, Recovery}
             0.05, 0.80, 0.15;   % Crisis -> {Normal, Crisis, Recovery}
             0.30, 0.05, 0.65];  % Recovery -> {Normal, Crisis, Recovery}
        
    elseif n_regimes == 4
        % Bull/Normal/Bear/Crisis
        P = [0.85, 0.10, 0.04, 0.01;   % Bull
             0.15, 0.75, 0.08, 0.02;   % Normal
             0.05, 0.20, 0.70, 0.05;   % Bear
             0.02, 0.08, 0.20, 0.70];  % Crisis
    else
        % Default: equal transition probabilities with self-persistence
        P = 0.1 * ones(n_regimes) + 0.8 * eye(n_regimes);
        P = P ./ sum(P, 2);  % Normalize rows
    end
end

%% Simulate Regime Transition
function next_regime = simulate_regime_transition(current_regime, P)
    % Simulate next regime based on transition probabilities
    
    transition_probs = P(current_regime, :);
    cumulative_probs = cumsum(transition_probs);
    
    u = rand();
    next_regime = find(cumulative_probs >= u, 1, 'first');
end

% =============================================================================
% Microstructure and Liquidity Effects
% =============================================================================

%% Add Microstructure Noise
function [noisy_prices, microstructure_effects] = add_microstructure_noise(clean_prices, params)
    % Add realistic microstructure effects to simulated prices
    
    [n_steps, n_assets, n_sims] = size(clean_prices);
    noisy_prices = clean_prices;
    
    % Microstructure parameters
    bid_ask_spread = 0.001 * ones(1, n_assets);  % 10 bps spread
    tick_size = 0.01 * ones(1, n_assets);        % 1 cent tick size
    noise_volatility = 0.0005 * ones(1, n_assets); % Microstructure noise
    
    microstructure_effects = struct();
    microstructure_effects.bid_ask_impacts = zeros(n_steps, n_assets, n_sims);
    microstructure_effects.rounding_effects = zeros(n_steps, n_assets, n_sims);
    microstructure_effects.noise = zeros(n_steps, n_assets, n_sims);
    
    for sim = 1:n_sims
        for t = 1:n_steps
            for asset = 1:n_assets
                % Bid-ask bounce effect
                if rand() > 0.5
                    ba_effect = bid_ask_spread(asset) / 2;  % Hit ask
                else
                    ba_effect = -bid_ask_spread(asset) / 2; % Hit bid
                end
                microstructure_effects.bid_ask_impacts(t, asset, sim) = ba_effect;
                
                % Tick size rounding
                raw_price = noisy_prices(t, asset, sim);
                rounded_price = round(raw_price / tick_size(asset)) * tick_size(asset);
                rounding_effect = rounded_price - raw_price;
                microstructure_effects.rounding_effects(t, asset, sim) = rounding_effect;
                
                % Random microstructure noise
                noise = normrnd(0, noise_volatility(asset));
                microstructure_effects.noise(t, asset, sim) = noise;
                
                % Apply all effects
                noisy_prices(t, asset, sim) = rounded_price + ba_effect + noise;
            end
        end
    end
end

%% Simulate Liquidity Effects
function [adjusted_prices, liquidity_impacts] = simulate_liquidity_effects(prices, volume_profile, params)
    % Simulate liquidity effects on price impact
    
    [n_steps, n_assets, n_sims] = size(prices);
    adjusted_prices = prices;
    
    % Liquidity parameters
    market_impact_coeff = 0.1 * ones(1, n_assets);  % Price impact coefficient
    temporary_impact_decay = 0.5;                    % Decay rate for temporary impact
    
    liquidity_impacts = struct();
    liquidity_impacts.permanent_impact = zeros(n_steps, n_assets, n_sims);
    liquidity_impacts.temporary_impact = zeros(n_steps, n_assets, n_sims);
    liquidity_impacts.volume = zeros(n_steps, n_assets, n_sims);
    
    for sim = 1:n_sims
        temp_impact_memory = zeros(1, n_assets);  % Temporary impact memory
        
        for t = 1:n_steps
            % Generate trading volume (correlated with volatility)
            if t > 1
                volatility = abs(log(prices(t, :, sim) ./ prices(t-1, :, sim)));
                volume = volume_profile .* (1 + 2 * volatility) .* lognrnd(0, 0.5, 1, n_assets);
            else
                volume = volume_profile .* lognrnd(0, 0.5, 1, n_assets);
            end
            
            liquidity_impacts.volume(t, :, sim) = volume;
            
            % Calculate price impact
            permanent_impact = market_impact_coeff .* sqrt(volume) .* sign(randn(1, n_assets));
            temporary_impact = 2 * permanent_impact;  % Temporary impact is larger
            
            % Apply temporary impact decay
            temp_impact_memory = temp_impact_memory * temporary_impact_decay + temporary_impact;
            
            liquidity_impacts.permanent_impact(t, :, sim) = permanent_impact;
            liquidity_impacts.temporary_impact(t, :, sim) = temp_impact_memory;
            
            % Adjust prices
            total_impact = permanent_impact + temp_impact_memory;
            adjusted_prices(t, :, sim) = prices(t, :, sim) .* (1 + total_impact / 10000);  % Convert to price units
        end
    end
end

% =============================================================================
% Stress Testing Scenarios
% =============================================================================

%% Generate Stress Test Scenarios
function scenarios = generate_stress_scenarios(base_params)
    % Generate various stress test scenarios
    
    scenarios = struct();
    
    % 1. Market Crash Scenario
    crash_params = base_params;
    crash_params.name = 'Market Crash';
    crash_params.regime_override = struct();
    crash_params.regime_override.active = true;
    crash_params.regime_override.start_time = 0.3;  % Crash starts 30% into simulation
    crash_params.regime_override.duration = 0.1;    % Lasts 10% of simulation
    crash_params.regime_override.drift = -0.5 * ones(1, base_params.n_assets);
    crash_params.regime_override.volatility = 0.8 * ones(1, base_params.n_assets);
    crash_params.regime_override.correlation = 0.95 * ones(base_params.n_assets) + 0.05 * eye(base_params.n_assets);
    scenarios.market_crash = crash_params;
    
    % 2. Flash Crash Scenario
    flash_params = base_params;
    flash_params.name = 'Flash Crash';
    flash_params.flash_crash = struct();
    flash_params.flash_crash.active = true;
    flash_params.flash_crash.time = 0.5;           % Middle of simulation
    flash_params.flash_crash.magnitude = -0.15;    % 15% instant drop
    flash_params.flash_crash.recovery_time = 0.01; % 1% of simulation for recovery
    scenarios.flash_crash = flash_params;
    
    % 3. High Volatility Regime
    high_vol_params = base_params;
    high_vol_params.name = 'High Volatility';
    high_vol_params.volatility_multiplier = 2.5;   % 2.5x normal volatility
    scenarios.high_volatility = high_vol_params;
    
    % 4. Correlation Breakdown
    corr_breakdown_params = base_params;
    corr_breakdown_params.name = 'Correlation Breakdown';
    corr_breakdown_params.correlation_shock = struct();
    corr_breakdown_params.correlation_shock.active = true;
    corr_breakdown_params.correlation_shock.start_time = 0.4;
    corr_breakdown_params.correlation_shock.new_correlation = 0.05 * ones(base_params.n_assets) + 0.95 * eye(base_params.n_assets);
    scenarios.correlation_breakdown = corr_breakdown_params;
    
    % 5. Liquidity Crisis
    liquidity_crisis_params = base_params;
    liquidity_crisis_params.name = 'Liquidity Crisis';
    liquidity_crisis_params.liquidity_crisis = struct();
    liquidity_crisis_params.liquidity_crisis.active = true;
    liquidity_crisis_params.liquidity_crisis.impact_multiplier = 5.0;  % 5x normal market impact
    liquidity_crisis_params.liquidity_crisis.spread_multiplier = 3.0;  % 3x normal spreads
    scenarios.liquidity_crisis = liquidity_crisis_params;
    
    % 6. Interest Rate Shock
    rate_shock_params = base_params;
    rate_shock_params.name = 'Interest Rate Shock';
    rate_shock_params.rate_shock = struct();
    rate_shock_params.rate_shock.active = true;
    rate_shock_params.rate_shock.magnitude = 0.02;  % 200 bps shock
    rate_shock_params.rate_shock.time = 0.25;       % Early in simulation
    scenarios.interest_rate_shock = rate_shock_params;
end

%% Apply Stress Scenario Effects
function [stressed_prices, scenario_effects] = apply_stress_scenario(base_prices, scenario_params)
    % Apply stress scenario effects to base simulation
    
    [n_steps, n_assets, n_sims] = size(base_prices);
    stressed_prices = base_prices;
    scenario_effects = struct();
    
    time_grid = linspace(0, scenario_params.T, n_steps);
    
    % Apply regime override
    if isfield(scenario_params, 'regime_override') && scenario_params.regime_override.active
        start_idx = find(time_grid >= scenario_params.regime_override.start_time, 1);
        end_idx = find(time_grid >= scenario_params.regime_override.start_time + scenario_params.regime_override.duration, 1);
        
        if isempty(end_idx), end_idx = n_steps; end
        
        % Apply regime-specific returns during override period
        for sim = 1:n_sims
            for t = start_idx:end_idx
                if t > 1
                    regime_return = scenario_params.regime_override.drift * scenario_params.dt + ...
                                   scenario_params.regime_override.volatility .* sqrt(scenario_params.dt) .* randn(1, n_assets);
                    
                    stressed_prices(t, :, sim) = stressed_prices(t-1, :, sim) .* exp(regime_return);
                end
            end
        end
        
        scenario_effects.regime_override_period = [start_idx, end_idx];
    end
    
    % Apply flash crash
    if isfield(scenario_params, 'flash_crash') && scenario_params.flash_crash.active
        crash_idx = find(time_grid >= scenario_params.flash_crash.time, 1);
        recovery_idx = find(time_grid >= scenario_params.flash_crash.time + scenario_params.flash_crash.recovery_time, 1);
        
        if isempty(recovery_idx), recovery_idx = min(crash_idx + 5, n_steps); end
        
        for sim = 1:n_sims
            % Instant crash
            crash_multiplier = 1 + scenario_params.flash_crash.magnitude;
            stressed_prices(crash_idx:end, :, sim) = stressed_prices(crash_idx:end, :, sim) * crash_multiplier;
            
            % Gradual recovery
            recovery_steps = recovery_idx - crash_idx;
            if recovery_steps > 0
                recovery_per_step = -scenario_params.flash_crash.magnitude / recovery_steps;
                for t = crash_idx:recovery_idx-1
                    recovery_factor = 1 + recovery_per_step;
                    stressed_prices(t+1:end, :, sim) = stressed_prices(t+1:end, :, sim) * recovery_factor;
                end
            end
        end
        
        scenario_effects.flash_crash_time = crash_idx;
        scenario_effects.recovery_time = recovery_idx;
    end
    
    % Apply volatility multiplier
    if isfield(scenario_params, 'volatility_multiplier')
        vol_mult = scenario_params.volatility_multiplier;
        
        for sim = 1:n_sims
            returns = diff(log(stressed_prices(:, :, sim)));
            
            % Scale returns by volatility multiplier
            mean_returns = mean(returns);
            centered_returns = returns - mean_returns;
            scaled_returns = centered_returns * sqrt(vol_mult) + mean_returns;
            
            # Reconstruct prices
            log_prices = cumsum([log(stressed_prices(1, :, sim)); scaled_returns]);
            stressed_prices(:, :, sim) = exp(log_prices);
        end
        
        scenario_effects.volatility_multiplier = vol_mult;
    end
end

% =============================================================================
% Strategy Integration and Backtesting
% =============================================================================

%% Pairs Trading Strategy for Simulation Testing
function [strategy_results, trade_log] = test_pairs_strategy_simulation(prices, strategy_params)
    % Test pairs trading strategy on simulated data
    
    [n_steps, n_assets, n_sims] = size(prices);
    
    % Strategy parameters
    if nargin < 2
        strategy_params = struct();
        strategy_params.lookback = 60;
        strategy_params.entry_threshold = 2.0;
        strategy_params.exit_threshold = 0.5;
        strategy_params.stop_loss = 5.0;
        strategy_params.position_size = 0.1;
    end
    
    strategy_results = struct();
    strategy_results.returns = zeros(n_steps-1, n_sims);
    strategy_results.positions = zeros(n_steps, n_assets, n_sims);
    strategy_results.cumulative_pnl = zeros(n_steps, n_sims);
    
    trade_log = cell(n_sims, 1);
    
    for sim = 1:n_sims
        sim_prices = squeeze(prices(:, :, sim));
        sim_trade_log = [];
        
        % Simple pairs: test all adjacent pairs
        for pair = 1:n_assets-1
            asset1_idx = pair;
            asset2_idx = pair + 1;
            
            [pair_signals, ~] = pairs_trading_signals_simulation(sim_prices(:, [asset1_idx, asset2_idx]), strategy_params);
            
            # Apply signals to strategy
            for t = strategy_params.lookback:n_steps-1
                signal = pair_signals(t);
                
                if signal ~= 0
                    # Enter position
                    position_value = strategy_params.position_size * sim_prices(t, asset1_idx);
                    
                    if signal > 0  % Long pair
                        strategy_results.positions(t, asset1_idx, sim) = strategy_results.positions(t, asset1_idx, sim) + position_value / sim_prices(t, asset1_idx);
                        strategy_results.positions(t, asset2_idx, sim) = strategy_results.positions(t, asset2_idx, sim) - position_value / sim_prices(t, asset2_idx);
                    else  % Short pair
                        strategy_results.positions(t, asset1_idx, sim) = strategy_results.positions(t, asset1_idx, sim) - position_value / sim_prices(t, asset1_idx);
                        strategy_results.positions(t, asset2_idx, sim) = strategy_results.positions(t, asset2_idx, sim) + position_value / sim_prices(t, asset2_idx);
                    end
                    
                    # Record trade
                    sim_trade_log = [sim_trade_log; t, asset1_idx, asset2_idx, signal, position_value];
                end
            end
        end
        
        # Calculate returns
        for t = 2:n_steps
            price_change = sim_prices(t, :) - sim_prices(t-1, :);
            position_pnl = sum(strategy_results.positions(t-1, :, sim) .* price_change);
            strategy_results.returns(t-1, sim) = position_pnl;
            strategy_results.cumulative_pnl(t, sim) = strategy_results.cumulative_pnl(t-1, sim) + position_pnl;
        end
        
        trade_log{sim} = sim_trade_log;
    end
end

function [signals, spread_stats] = pairs_trading_signals_simulation(pair_prices, params)
    % Generate pairs trading signals for simulation testing
    
    n_obs = size(pair_prices, 1);
    signals = zeros(n_obs, 1);
    spread_stats = struct();
    
    spread = zeros(n_obs, 1);
    zscore = zeros(n_obs, 1);
    
    for t = params.lookback:n_obs
        # Rolling window cointegration
        window_data = pair_prices(t-params.lookback+1:t, :);
        
        # Simple spread calculation
        X = [ones(params.lookback, 1), window_data(:, 2)];
        beta = X \ window_data(:, 1);
        
        current_spread = pair_prices(t, 1) - beta(1) - beta(2) * pair_prices(t, 2);
        spread(t) = current_spread;
        
        # Z-score
        window_spreads = spread(t-params.lookback+1:t);
        spread_mean = mean(window_spreads(window_spreads ~= 0));
        spread_std = std(window_spreads(window_spreads ~= 0));
        
        if spread_std > 0
            zscore(t) = (current_spread - spread_mean) / spread_std;
        end
        
        # Generate signals
        if abs(zscore(t)) > params.entry_threshold
            signals(t) = sign(-zscore(t));  # Contrarian signal
        elseif abs(zscore(t)) < params.exit_threshold
            signals(t) = 0;
        elseif t > 1
            signals(t) = signals(t-1);
        end
    end
    
    spread_stats.spread = spread;
    spread_stats.zscore = zscore;
end

% =============================================================================
% Monte Carlo Analysis and Statistics
% =============================================================================

%% Monte Carlo Performance Analysis
function mc_results = monte_carlo_analysis(simulation_results, confidence_levels)
    % Comprehensive Monte Carlo analysis of simulation results
    
    if nargin < 2
        confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95];
    end
    
    mc_results = struct();
    
    # Extract key metrics across simulations
    if isfield(simulation_results, 'cumulative_pnl')
        final_pnl = simulation_results.cumulative_pnl(end, :);
        
        mc_results.final_pnl = struct();
        mc_results.final_pnl.mean = mean(final_pnl);
        mc_results.final_pnl.std = std(final_pnl);
        mc_results.final_pnl.percentiles = prctile(final_pnl, confidence_levels * 100);
        mc_results.final_pnl.var_95 = prctile(final_pnl, 5);
        mc_results.final_pnl.var_99 = prctile(final_pnl, 1);
    end
    
    if isfield(simulation_results, 'returns')
        returns = simulation_results.returns;
        
        # Sharpe ratio analysis
        sharpe_ratios = mean(returns) ./ std(returns) * sqrt(252);
        mc_results.sharpe_ratio = struct();
        mc_results.sharpe_ratio.mean = mean(sharpe_ratios);
        mc_results.sharpe_ratio.std = std(sharpe_ratios);
        mc_results.sharpe_ratio.percentiles = prctile(sharpe_ratios, confidence_levels * 100);
        
        # Maximum drawdown analysis
        max_drawdowns = zeros(1, size(returns, 2));
        for sim = 1:size(returns, 2)
            cumulative = cumsum(returns(:, sim));
            running_max = cummax(cumulative);
            drawdowns = (cumulative - running_max) ./ max(running_max, 1);
            max_drawdowns(sim) = min(drawdowns);
        end
        
        mc_results.max_drawdown = struct();
        mc_results.max_drawdown.mean = mean(max_drawdowns);
        mc_results.max_drawdown.std = std(max_drawdowns);
        mc_results.max_drawdown.percentiles = prctile(max_drawdowns, confidence_levels * 100);
        
        # Hit ratio (percentage of profitable periods)
        hit_ratios = mean(returns > 0, 1);
        mc_results.hit_ratio = struct();
        mc_results.hit_ratio.mean = mean(hit_ratios);
        mc_results.hit_ratio.std = std(hit_ratios);
        mc_results.hit_ratio.percentiles = prctile(hit_ratios, confidence_levels * 100);
    end
    
    # Risk metrics
    mc_results.risk_metrics = calculate_risk_metrics_simulation(simulation_results);
end

function risk_metrics = calculate_risk_metrics_simulation(simulation_results)
    % Calculate comprehensive risk metrics from simulation results
    
    risk_metrics = struct();
    
    if isfield(simulation_results, 'returns')
        returns = simulation_results.returns;
        n_sims = size(returns, 2);
        
        # VaR and ES
        var_95 = zeros(1, n_sims);
        var_99 = zeros(1, n_sims);
        es_95 = zeros(1, n_sims);
        es_99 = zeros(1, n_sims);
        
        for sim = 1:n_sims
            sim_returns = returns(:, sim);
            sorted_returns = sort(sim_returns);
            
            var_95(sim) = -sorted_returns(ceil(0.05 * length(sorted_returns)));
            var_99(sim) = -sorted_returns(ceil(0.01 * length(sorted_returns)));
            
            es_95(sim) = -mean(sorted_returns(1:ceil(0.05 * length(sorted_returns))));
            es_99(sim) = -mean(sorted_returns(1:ceil(0.01 * length(sorted_returns))));
        end
        
        risk_metrics.var_95 = struct('mean', mean(var_95), 'std', std(var_95));
        risk_metrics.var_99 = struct('mean', mean(var_99), 'std', std(var_99));
        risk_metrics.es_95 = struct('mean', mean(es_95), 'std', std(es_95));
        risk_metrics.es_99 = struct('mean', mean(es_99), 'std', std(es_99));
        
        # Tail risk metrics
        tail_ratios = es_95 ./ var_95;
        risk_metrics.tail_ratio = struct('mean', mean(tail_ratios), 'std', std(tail_ratios));
        
        # Volatility of volatility
        vol_of_vol = std(std(returns));
        risk_metrics.volatility_of_volatility = vol_of_vol;
    end
end

% =============================================================================
% Visualization and Reporting
% =============================================================================

%% Generate Simulation Report
function generate_simulation_report(simulation_results, scenario_name, output_path)
    % Generate comprehensive simulation report
    
    if nargin < 3
        output_path = 'simulation_report.html';
    end
    
    # Create report structure
    report = struct();
    report.scenario_name = scenario_name;
    report.generation_time = datestr(now);
    report.simulation_results = simulation_results;
    
    # Calculate summary statistics
    if isfield(simulation_results, 'prices')
        prices = simulation_results.prices;
        [n_steps, n_assets, n_sims] = size(prices);
        
        # Price statistics
        final_prices = squeeze(prices(end, :, :));
        initial_prices = squeeze(prices(1, :, :));
        total_returns = (final_prices ./ initial_prices - 1) * 100;
        
        report.price_statistics = struct();
        report.price_statistics.mean_return = mean(total_returns, 2);
        report.price_statistics.volatility = std(total_returns, 0, 2);
        report.price_statistics.min_return = min(total_returns, [], 2);
        report.price_statistics.max_return = max(total_returns, [], 2);
    end
    
    # Strategy performance
    if isfield(simulation_results, 'strategy_results')
        strategy_perf = simulation_results.strategy_results;
        
        if isfield(strategy_perf, 'cumulative_pnl')
            final_pnl = strategy_perf.cumulative_pnl(end, :);
            
            report.strategy_performance = struct();
            report.strategy_performance.mean_pnl = mean(final_pnl);
            report.strategy_performance.std_pnl = std(final_pnl);
            report.strategy_performance.win_rate = mean(final_pnl > 0) * 100;
            report.strategy_performance.profit_factor = sum(final_pnl(final_pnl > 0)) / abs(sum(final_pnl(final_pnl < 0)));
        end
    end
    
    # Save report
    save([output_path, '.mat'], 'report');
    
    # Generate plots
    generate_simulation_plots(simulation_results, scenario_name);
    
    fprintf('Simulation report generated: %s\n', output_path);
    fprintf('Scenario: %s\n', scenario_name);
    fprintf('Generation time: %s\n', report.generation_time);
end

function generate_simulation_plots(simulation_results, scenario_name)
    % Generate key visualization plots for simulation results
    
    figure('Name', ['Simulation Results: ', scenario_name], 'Position', [100, 100, 1200, 800]);
    
    # Plot 1: Price paths
    if isfield(simulation_results, 'prices')
        subplot(2, 3, 1);
        prices = simulation_results.prices;
        
        # Plot first 10 simulations for first 3 assets
        for sim = 1:min(10, size(prices, 3))
            plot(squeeze(prices(:, 1, sim)), 'b-', 'LineWidth', 0.5, 'Color', [0.5, 0.5, 1, 0.3]);
            hold on;
        end
        
        # Plot mean path
        mean_path = mean(prices(:, 1, :), 3);
        plot(mean_path, 'r-', 'LineWidth', 2);
        
        title('Asset Price Paths (Asset 1)');
        xlabel('Time Steps');
        ylabel('Price');
        legend('Individual Paths', 'Mean Path', 'Location', 'best');
        grid on;
    end
    
    # Plot 2: Return distribution
    if isfield(simulation_results, 'strategy_results') && isfield(simulation_results.strategy_results, 'returns')
        subplot(2, 3, 2);
        returns = simulation_results.strategy_results.returns;
        final_returns = returns(end, :);
        
        histogram(final_returns, 30, 'Normalization', 'probability');
        title('Final Strategy Returns Distribution');
        xlabel('Returns');
        ylabel('Probability');
        grid on;
        
        # Add statistics
        mean_ret = mean(final_returns);
        std_ret = std(final_returns);
        text(0.05, 0.95, sprintf('Mean: %.4f\nStd: %.4f', mean_ret, std_ret), ...
             'Units', 'normalized', 'VerticalAlignment', 'top');
    end
    
    # Plot 3: Cumulative P&L
    if isfield(simulation_results, 'strategy_results') && isfield(simulation_results.strategy_results, 'cumulative_pnl')
        subplot(2, 3, 3);
        cum_pnl = simulation_results.strategy_results.cumulative_pnl;
        
        # Plot percentiles
        pnl_percentiles = prctile(cum_pnl, [5, 25, 50, 75, 95], 2);
        
        plot(pnl_percentiles(:, 3), 'r-', 'LineWidth', 2); hold on;  # Median
        plot(pnl_percentiles(:, 1), 'b--', 'LineWidth', 1);         # 5th percentile
        plot(pnl_percentiles(:, 5), 'b--', 'LineWidth', 1);         # 95th percentile
        fill([1:size(pnl_percentiles, 1), fliplr(1:size(pnl_percentiles, 1))], ...
             [pnl_percentiles(:, 2)', fliplr(pnl_percentiles(:, 4)')], ...
             'g', 'FaceAlpha', 0.3, 'EdgeColor', 'none');           # IQR
        
        title('Strategy Cumulative P&L');
        xlabel('Time Steps');
        ylabel('Cumulative P&L');
        legend('Median', '5th-95th Percentile', '', '25th-75th Percentile', 'Location', 'best');
        grid on;
    end
    
    # Plot 4: Drawdown analysis
    if isfield(simulation_results, 'strategy_results') && isfield(simulation_results.strategy_results, 'cumulative_pnl')
        subplot(2, 3, 4);
        cum_pnl = simulation_results.strategy_results.cumulative_pnl;
        
        # Calculate drawdowns for each simulation
        drawdowns = zeros(size(cum_pnl));
        for sim = 1:size(cum_pnl, 2)
            running_max = cummax(cum_pnl(:, sim));
            drawdowns(:, sim) = (cum_pnl(:, sim) - running_max) ./ max(running_max, 1);
        end
        
        # Plot worst drawdown paths
        [~, worst_sims] = sort(min(drawdowns));
        for i = 1:min(5, length(worst_sims))
            plot(drawdowns(:, worst_sims(i)) * 100, 'LineWidth', 1); hold on;
        end
        
        title('Worst Drawdown Paths');
        xlabel('Time Steps');
        ylabel('Drawdown (%)');
        grid on;
    end
    
    # Plot 5: Risk metrics
    subplot(2, 3, 5);
    if isfield(simulation_results, 'strategy_results') && isfield(simulation_results.strategy_results, 'returns')
        returns = simulation_results.strategy_results.returns;
        
        # Calculate rolling Sharpe ratio
        window = 60;  # 60-day rolling window
        rolling_sharpe = zeros(size(returns, 1) - window + 1, size(returns, 2));
        
        for sim = 1:size(returns, 2)
            for t = window:size(returns, 1)
                window_returns = returns(t-window+1:t, sim);
                rolling_sharpe(t-window+1, sim) = mean(window_returns) / std(window_returns) * sqrt(252);
            end
        end
        
        # Plot Sharpe ratio distribution
        plot(mean(rolling_sharpe, 2), 'LineWidth', 2);
        title('Rolling Sharpe Ratio (60-day)');
        xlabel('Time Steps');
        ylabel('Sharpe Ratio');
        grid on;
    end
    
    # Plot 6: Performance summary
    subplot(2, 3, 6);
    if isfield(simulation_results, 'strategy_results')
        # Create performance metrics table
        metrics_text = {
            'Performance Summary';
            '==================';
            '';
        };
        
        if isfield(simulation_results.strategy_results, 'cumulative_pnl')
            final_pnl = simulation_results.strategy_results.cumulative_pnl(end, :);
            metrics_text{end+1} = sprintf('Mean Final P&L: %.2f', mean(final_pnl));
            metrics_text{end+1} = sprintf('Std Final P&L: %.2f', std(final_pnl));
            metrics_text{end+1} = sprintf('Win Rate: %.1f%%', mean(final_pnl > 0) * 100);
            metrics_text{end+1} = sprintf('95%% VaR: %.2f', prctile(final_pnl, 5));
        end
        
        if isfield(simulation_results.strategy_results, 'returns')
            returns = simulation_results.strategy_results.returns;
            overall_sharpe = mean(mean(returns)) / std(mean(returns, 2)) * sqrt(252);
            metrics_text{end+1} = sprintf('Sharpe Ratio: %.2f', overall_sharpe);
        end
        
        # Display as text
        text(0.1, 0.9, metrics_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
             'FontSize', 10, 'FontName', 'Courier');
        axis off;
    end
    
    # Save plot
    saveas(gcf, ['simulation_plots_', scenario_name, '.png']);
end

% =============================================================================
% Main Execution and Testing Functions
% =============================================================================

%% Run Complete Market Simulation Suite
function run_market_simulation_suite()
    % Run comprehensive market simulation testing suite
    
    fprintf('Running Market Simulation Suite...\n\n');
    
    # Create base parameters
    base_params = create_simulation_params();
    base_params.n_simulations = 100;  # Reduced for testing
    base_params.T = 0.5;             # 6 months
    
    # Generate stress scenarios
    scenarios = generate_stress_scenarios(base_params);
    scenario_names = fieldnames(scenarios);
    
    all_results = struct();
    
    for i = 1:length(scenario_names)
        scenario_name = scenario_names{i};
        scenario_params = scenarios.(scenario_name);
        
        fprintf('Running scenario: %s\n', scenario_params.name);
        
        # Run simulation
        tic;
        if strcmp(scenario_params.price_model, 'gbm')
            [prices, returns, regimes] = simulate_regime_switching_gbm(scenario_params);
        elseif strcmp(scenario_params.price_model, 'jump_diffusion')
            [prices, returns, jumps] = simulate_jump_diffusion(scenario_params);
        elseif strcmp(scenario_params.price_model, 'heston')
            [prices, returns, volatilities] = simulate_heston_model(scenario_params);
        end
        
        # Apply stress effects if needed
        if isfield(scenario_params, 'regime_override') || isfield(scenario_params, 'flash_crash')
            [prices, scenario_effects] = apply_stress_scenario(prices, scenario_params);
        end
        
        # Test strategy
        [strategy_results, trade_log] = test_pairs_strategy_simulation(prices);
        
        sim_time = toc;
        
        # Store results
        all_results.(scenario_name) = struct();
        all_results.(scenario_name).prices = prices;
        all_results.(scenario_name).returns = returns;
        all_results.(scenario_name).strategy_results = strategy_results;
        all_results.(scenario_name).simulation_time = sim_time;
        
        if exist('regimes', 'var')
            all_results.(scenario_name).regimes = regimes;
        end
        
        # Monte Carlo analysis
        mc_results = monte_carlo_analysis(strategy_results);
        all_results.(scenario_name).monte_carlo = mc_results;
        
        # Generate report
        generate_simulation_report(all_results.(scenario_name), scenario_params.name);
        
        fprintf('  Completed in %.2f seconds\n', sim_time);
        fprintf('  Mean strategy return: %.4f\n', mc_results.final_pnl.mean);
        fprintf('  Strategy Sharpe ratio: %.2f\n', mc_results.sharpe_ratio.mean);
        fprintf('  Max drawdown: %.2f%%\n', abs(mc_results.max_drawdown.mean) * 100);
        fprintf('\n');
    end
    
    # Summary comparison
    fprintf('Scenario Comparison Summary:\n');
    fprintf('============================\n');
    for i = 1:length(scenario_names)
        scenario_name = scenario_names{i};
        mc_results = all_results.(scenario_name).monte_carlo;
        
        fprintf('%-20s | Return: %8.4f | Sharpe: %6.2f | MaxDD: %6.1f%%\n', ...
                scenario_name, mc_results.final_pnl.mean, mc_results.sharpe_ratio.mean, ...
                abs(mc_results.max_drawdown.mean) * 100);
    end
    
    # Save complete results
    save('complete_simulation_results.mat', 'all_results', 'base_params');
    
    fprintf('\nMarket simulation suite completed!\n');
    fprintf('Results saved to: complete_simulation_results.mat\n');
end

% =============================================================================
% Module Initialization
% =============================================================================

# Initialize the market simulation module
init_market_simulation();

# Print available functions
fprintf('Available Market Simulation Functions:\n\n');
fprintf('Core Simulation Models:\n');
fprintf('  - simulate_regime_switching_gbm: Regime-switching geometric Brownian motion\n');
fprintf('  - simulate_jump_diffusion: Merton jump-diffusion model\n');
fprintf('  - simulate_heston_model: Heston stochastic volatility model\n');
fprintf('  - simulate_dcc_model: Dynamic conditional correlation model\n\n');

fprintf('Market Effects:\n');
fprintf('  - add_microstructure_noise: Realistic microstructure effects\n');
fprintf('  - simulate_liquidity_effects: Market impact and liquidity modeling\n\n');

fprintf('Stress Testing:\n');
fprintf('  - generate_stress_scenarios: Various market stress scenarios\n');
fprintf('  - apply_stress_scenario: Apply stress effects to simulations\n\n');

fprintf('Strategy Testing:\n');
fprintf('  - test_pairs_strategy_simulation: Test pairs trading on simulated data\n');
fprintf('  - monte_carlo_analysis: Comprehensive Monte Carlo analysis\n\n');

fprintf('Analysis and Reporting:\n');
fprintf('  - generate_simulation_report: Comprehensive simulation report\n');
fprintf('  - generate_simulation_plots: Visualization of results\n');
fprintf('  - run_market_simulation_suite: Complete testing suite\n\n');

fprintf('Market Simulation Framework ready for use!\n');
fprintf('Example: run_market_simulation_suite()\n\n');