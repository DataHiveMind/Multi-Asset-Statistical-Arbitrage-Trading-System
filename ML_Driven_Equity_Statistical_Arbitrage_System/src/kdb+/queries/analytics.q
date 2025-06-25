/ =============================================================================
/ Analytics Query Library for Statistical Arbitrage System
/ =============================================================================
/ Purpose: Comprehensive library of Q queries for time-series analytics on
/          trade and quote data, optimized for statistical arbitrage strategies
/ Author: Statistical Arbitrage System
/ Date: 2025-06-25
/ Version: 1.0
/ =============================================================================

/ Load required schema and utilities
\l ../schema/trades.q
\l ../schema/quotes.q

/ =============================================================================
/ Core Time-Series Analytics Functions
/ =============================================================================

/ VWAP (Volume Weighted Average Price) calculations
calc_vwap:{[tbl;start_time;end_time;syms]
  / Calculate VWAP for given symbols and time range
  data:$[`trades_enhanced~tbl;
    select from trades_enhanced where time within (start_time;end_time), sym in syms;
    select from quotes_enhanced where time within (start_time;end_time), sym in syms];
  
  if[`trades_enhanced~tbl;
    select vwap:sum[price*size]%sum size by sym from data where not null price, size>0
  ;
    / For quotes, use mid price and average of bid/ask sizes
    select vwap:sum[mid*(bidsize+asksize)]%sum bidsize+asksize by sym from data 
      where not null bid, not null ask, bidsize>0, asksize>0
  ]
  };

/ TWAP (Time Weighted Average Price) calculations
calc_twap:{[tbl;start_time;end_time;syms;interval]
  / Calculate TWAP with specified time intervals
  if[null interval; interval:00:01:00.000];  / Default 1 minute
  
  data:$[`trades_enhanced~tbl;
    select time, sym, price from trades_enhanced 
      where time within (start_time;end_time), sym in syms;
    select time, sym, mid as price from quotes_enhanced 
      where time within (start_time;end_time), sym in syms];
  
  / Group by time buckets
  data:update bucket:interval xbar time from data;
  
  / Calculate TWAP per bucket
  twap_buckets:select twap:avg price by sym, bucket from data;
  
  / Overall TWAP
  select twap:avg twap by sym from twap_buckets
  };

/ Volume analysis functions
calc_volume_profile:{[start_time;end_time;syms;price_buckets]
  / Calculate volume at price levels
  if[null price_buckets; price_buckets:100];
  
  trades:select from trades_enhanced 
    where time within (start_time;end_time), sym in syms, size>0;
  
  / Calculate price buckets for each symbol
  price_stats:select min_price:min price, max_price:max price, 
                    price_range:max[price]-min price by sym from trades;
  
  / Create volume profile
  volume_profile:();
  symbols:exec distinct sym from trades;
  
  {[s;trades;stats;buckets]
    sym_trades:select from trades where sym=s;
    if[count sym_trades;
      min_p:stats[s;`min_price];
      max_p:stats[s;`max_price];
      bucket_size:(max_p-min_p)%buckets;
      
      if[bucket_size>0;
        sym_trades:update price_bucket:min_p+bucket_size*floor (price-min_p)%bucket_size 
          from sym_trades;
        
        profile:select total_volume:sum size, avg_price:avg price, trade_count:count i 
          by sym, price_bucket from sym_trades;
        
        volume_profile,:profile
      ]
    ]
  }[;trades;price_stats;price_buckets] each symbols;
  
  volume_profile
  };

/ Spread analysis
calc_spread_analytics:{[start_time;end_time;syms;interval]
  / Comprehensive bid-ask spread analysis
  if[null interval; interval:00:01:00.000];
  
  quotes:select from quotes_enhanced 
    where time within (start_time;end_time), sym in syms,
          not null bid, not null ask, bid>0, ask>0;
  
  quotes:update bucket:interval xbar time from quotes;
  
  / Calculate spread metrics
  spread_stats:select 
    avg_spread:avg spread,
    avg_spread_bps:avg spread_bps,
    min_spread:min spread,
    max_spread:max spread,
    spread_volatility:dev spread,
    avg_bid_size:avg bidsize,
    avg_ask_size:avg asksize,
    avg_imbalance:avg imbalance,
    quote_count:count i
  by sym, bucket from quotes;
  
  / Summary statistics
  select 
    overall_avg_spread:avg avg_spread,
    overall_avg_spread_bps:avg avg_spread_bps,
    spread_stability:dev avg_spread,
    avg_quote_frequency:avg quote_count,
    avg_market_imbalance:avg avg_imbalance
  by sym from spread_stats
  };

/ =============================================================================
/ Advanced Statistical Analytics
/ =============================================================================

/ Rolling statistics calculation
calc_rolling_stats:{[tbl;sym;start_time;end_time;window;metrics]
  / Calculate rolling statistics with specified window
  if[null window; window:20];  / Default 20 periods
  if[null metrics; metrics:`avg`dev`min`max];
  
  data:$[`trades_enhanced~tbl;
    select time, price, size from trades_enhanced 
      where sym=sym, time within (start_time;end_time);
    select time, mid as price, bidsize+asksize as size from quotes_enhanced 
      where sym=sym, time within (start_time;end_time)];
  
  / Apply rolling calculations
  if[`avg in metrics; data:update mavg_price:mavg[window;price] from data];
  if[`dev in metrics; data:update mdev_price:mdev[window;price] from data];
  if[`min in metrics; data:update mmin_price:mmin[window;price] from data];
  if[`max in metrics; data:update mmax_price:mmax[window;price] from data];
  
  / Add derived metrics
  if[(`avg in metrics) and (`dev in metrics);
    data:update bollinger_upper:mavg_price+2*mdev_price,
                bollinger_lower:mavg_price-2*mdev_price from data];
  
  data
  };

/ Correlation analysis
calc_correlation_matrix:{[syms;start_time;end_time;interval;method]
  / Calculate correlation matrix between symbols
  if[null interval; interval:00:01:00.000];
  if[null method; method:`price];  / `price or `returns
  
  / Get data for all symbols
  data:select time, sym, mid from quotes_enhanced 
    where time within (start_time;end_time), sym in syms,
          not null mid;
  
  / Create time buckets
  data:update bucket:interval xbar time from data;
  
  / Pivot to get symbol columns
  price_matrix:exec syms#(sym!mid) by bucket from data;
  
  / Calculate returns if requested
  if[`returns~method;
    price_matrix:1_price_matrix%prev price_matrix;
    price_matrix:update (syms):(syms){log x%y}prior (syms) from price_matrix;
  ];
  
  / Calculate correlation matrix
  corr_matrix:cor flip value flip price_matrix;
  
  / Format as table
  `sym1`sym2`correlation xcols 
    raze {[s1;s2;corr] ([]sym1:s1;sym2:s2;correlation:corr s1 s2)}[;;corr_matrix] 
      each/: (syms;syms)
  };

/ Volatility calculations
calc_volatility_metrics:{[sym;start_time;end_time;intervals]
  / Calculate volatility across multiple timeframes
  if[null intervals; intervals:00:01:00.000 00:05:00.000 00:15:00.000 01:00:00.000];
  
  quotes:select time, mid from quotes_enhanced 
    where sym=sym, time within (start_time;end_time), not null mid;
  
  volatility_metrics:();
  
  {[interval;quotes;sym]
    / Group by interval
    bucketed:update bucket:interval xbar time from quotes;
    
    / Calculate returns
    returns:select bucket, ret:log mid%prev mid from bucketed;
    returns:select from returns where not null ret, not 0w=ret;
    
    / Calculate volatility metrics
    vol_stats:select 
      interval:interval,
      realized_vol:dev ret,
      annualized_vol:dev[ret]*sqrt 252*24*3600%`long$interval,
      skewness:(.5*sum (ret-avg ret) xexp 3)%((count ret)*dev[ret] xexp 3),
      kurtosis:(sum (ret-avg ret) xexp 4)%((count ret)*dev[ret] xexp 4),
      var_95:0.05 mavg ret,
      max_return:max ret,
      min_return:min ret
    from ([]dummy:1) where count returns;
    
    volatility_metrics,:update sym:sym from vol_stats;
  }[;quotes;sym] each intervals;
  
  volatility_metrics
  };

/ =============================================================================
/ Market Microstructure Analytics
/ =============================================================================

/ Order flow analysis
calc_order_flow:{[start_time;end_time;syms;interval]
  / Analyze order flow patterns
  if[null interval; interval:00:01:00.000];
  
  / Get trades with side information
  trades:select from trades_enhanced 
    where time within (start_time;end_time), sym in syms, not null side;
  
  trades:update bucket:interval xbar time from trades;
  
  / Calculate order flow metrics
  flow_stats:select 
    buy_volume:sum size where side=`buy,
    sell_volume:sum size where side=`sell,
    buy_trades:count i where side=`buy,
    sell_trades:count i where side=`sell,
    net_flow:sum[size where side=`buy] - sum[size where side=`sell],
    order_imbalance:(sum[size where side=`buy] - sum[size where side=`sell])
                   %(sum[size where side=`buy] + sum[size where side=`sell]),
    avg_trade_size:avg size,
    aggressive_ratio:count[i where side=`buy]%count i
  by sym, bucket from trades;
  
  / Add flow direction indicators
  flow_stats:update 
    flow_direction:?[net_flow>0;`buying;`selling],
    imbalance_strength:?[abs order_imbalance > 0.2;`strong;
                        abs order_imbalance > 0.1;`moderate;`weak]
  from flow_stats;
  
  flow_stats
  };

/ Market impact analysis
calc_market_impact:{[start_time;end_time;syms;trade_size_buckets]
  / Analyze price impact of different trade sizes
  if[null trade_size_buckets; trade_size_buckets:100 500 1000 5000 10000];
  
  / Get trades with price impact data
  trades:select from trades_enhanced 
    where time within (start_time;end_time), sym in syms, 
          not null impact, not null size;
  
  / Categorize trades by size
  size_categories:(`small`medium`large`xlarge`jumbo)!trade_size_buckets;
  
  trades:update 
    size_category:?[size <= trade_size_buckets[0];`small;
                   size <= trade_size_buckets[1];`medium;
                   size <= trade_size_buckets[2];`large;
                   size <= trade_size_buckets[3];`xlarge;`jumbo]
  from trades;
  
  / Calculate impact statistics
  impact_stats:select 
    avg_impact:avg impact,
    median_impact:med impact,
    impact_volatility:dev impact,
    max_impact:max impact,
    impact_per_share:avg impact%size,
    trade_count:count i,
    total_volume:sum size
  by sym, size_category from trades;
  
  / Add impact efficiency metrics
  impact_stats:update 
    impact_efficiency:avg_impact%avg sqrt total_volume%trade_count,
    impact_consistency:avg_impact%impact_volatility
  from impact_stats;
  
  impact_stats
  };

/ Liquidity analysis
calc_liquidity_metrics:{[start_time;end_time;syms;interval]
  / Comprehensive liquidity analysis
  if[null interval; interval:00:05:00.000];
  
  / Get quotes data
  quotes:select from quotes_enhanced 
    where time within (start_time;end_time), sym in syms,
          not null bid, not null ask, bidsize>0, asksize>0;
  
  quotes:update bucket:interval xbar time from quotes;
  
  / Calculate liquidity metrics
  liquidity_stats:select 
    avg_spread_bps:avg spread_bps,
    avg_depth:avg bidsize+asksize,
    depth_at_best:avg (bidsize*bid + asksize*ask)%(bidsize+asksize),
    quote_count:count i,
    effective_spread:avg 2*abs mid - (bid+ask)%2,
    price_improvement:avg ?[mid between (bid;ask); (ask-bid)%2; 0],
    market_efficiency:1 - avg spread_bps%100,
    liquidity_score:avg liquidity_score
  by sym, bucket from quotes;
  
  / Add liquidity categories
  liquidity_stats:update 
    liquidity_tier:?[avg_spread_bps < 5;`excellent;
                    avg_spread_bps < 15;`good;
                    avg_spread_bps < 30;`fair;`poor],
    depth_category:?[avg_depth > 10000;`deep;
                    avg_depth > 5000;`moderate;`shallow]
  from liquidity_stats;
  
  liquidity_stats
  };

/ =============================================================================
/ Statistical Arbitrage Specific Analytics
/ =============================================================================

/ Pairs analysis
calc_pairs_metrics:{[sym1;sym2;start_time;end_time;interval]
  / Calculate statistical arbitrage metrics for pairs
  if[null interval; interval:00:01:00.000];
  
  / Get price data for both symbols
  prices1:select time, price:mid from quotes_enhanced 
    where sym=sym1, time within (start_time;end_time), not null mid;
  prices2:select time, price:mid from quotes_enhanced 
    where sym=sym2, time within (start_time;end_time), not null mid;
  
  / Align timestamps
  prices:select time, price1:price from prices1 lj 
         `time xkey select time, price2:price from prices2;
  prices:select from prices where not null price1, not null price2;
  
  / Add time buckets
  prices:update bucket:interval xbar time from prices;
  
  / Calculate spread and ratio
  prices:update 
    spread:price1-price2,
    ratio:price1%price2,
    log_ratio:log price1%price2
  from prices;
  
  / Statistical tests
  spread_stats:select 
    correlation:cor[price1;price2],
    spread_mean:avg spread,
    spread_std:dev spread,
    ratio_mean:avg ratio,
    ratio_std:dev ratio,
    cointegration_stat:avg spread, / Simplified - would need proper Engle-Granger test
    half_life:calc_half_life spread,
    hurst_exponent:calc_hurst_exponent spread
  from ([]dummy:1) where count prices;
  
  / Add trading signals
  prices:update 
    z_score:(spread - spread_stats[`spread_mean]) % spread_stats[`spread_std],
    signal:?[abs((spread - spread_stats[`spread_mean]) % spread_stats[`spread_std]) > 2;
            ?[spread > spread_stats[`spread_mean];`short;`long];`hold]
  from prices;
  
  / Return both statistics and signals
  (`statistics`signals)!(spread_stats;prices)
  };

/ Alpha signal analysis
calc_alpha_signals:{[syms;start_time;end_time;factors]
  / Calculate alpha signals based on multiple factors
  if[null factors; factors:`momentum`mean_reversion`volatility];
  
  signals:();
  
  {[sym;start_time;end_time;factors]
    / Get price and volume data
    data:select time, mid, bidsize+asksize as volume from quotes_enhanced 
      where sym=sym, time within (start_time;end_time), not null mid;
    
    / Calculate returns
    data:update return:log mid%prev mid from data;
    data:select from data where not null return, not 0w=return;
    
    alpha_signals:([]time:data`time; sym:sym);
    
    / Momentum signals
    if[`momentum in factors;
      data:update 
        momentum_5:mavg[5;return],
        momentum_20:mavg[20;return],
        momentum_signal:?[mavg[5;return] > mavg[20;return];1;-1]
      from data;
      alpha_signals:alpha_signals lj `time xkey select time, momentum_signal from data;
    ];
    
    / Mean reversion signals
    if[`mean_reversion in factors;
      data:update 
        price_deviation:(mid - mavg[20;mid]) % mdev[20;mid],
        reversion_signal:?[abs((mid - mavg[20;mid]) % mdev[20;mid]) > 2;
                          ?[mid > mavg[20;mid];-1;1];0]
      from data;
      alpha_signals:alpha_signals lj `time xkey select time, reversion_signal from data;
    ];
    
    / Volatility signals
    if[`volatility in factors;
      data:update 
        vol_regime:?[mdev[20;return] > 1.5*mavg[60;mdev[20;return]];`high;
                    mdev[20;return] < 0.5*mavg[60;mdev[20;return]];`low;`normal],
        vol_signal:?[mdev[20;return] > 1.5*mavg[60;mdev[20;return]];-0.5;
                    mdev[20;return] < 0.5*mavg[60;mdev[20;return]];0.5;0]
      from data;
      alpha_signals:alpha_signals lj `time xkey select time, vol_signal from data;
    ];
    
    / Combine signals
    signal_cols:cols[alpha_signals] except `time`sym;
    alpha_signals:update 
      combined_signal:sum flip signal_cols,
      signal_strength:abs sum flip signal_cols
    from alpha_signals;
    
    signals,:alpha_signals;
  }[;start_time;end_time;factors] each syms;
  
  signals
  };

/ Performance attribution
calc_performance_attribution:{[portfolio_trades;start_time;end_time]
  / Analyze performance attribution for trading strategies
  trades:select from portfolio_trades 
    where time within (start_time;end_time), not null pnl;
  
  / Basic performance metrics
  perf_summary:select 
    total_pnl:sum pnl,
    total_trades:count i,
    win_rate:avg pnl>0,
    avg_win:avg pnl where pnl>0,
    avg_loss:avg pnl where pnl<0,
    profit_factor:sum[pnl where pnl>0] % abs sum[pnl where pnl<0],
    sharpe_ratio:avg[pnl] % dev pnl,
    max_drawdown:min mins pnl
  from ([]dummy:1) where count trades;
  
  / Attribution by symbol
  symbol_attribution:select 
    pnl:sum pnl,
    trade_count:count i,
    win_rate:avg pnl>0,
    contribution:sum[pnl] % perf_summary[`total_pnl]
  by sym from trades;
  
  / Attribution by strategy
  strategy_attribution:select 
    pnl:sum pnl,
    trade_count:count i,
    win_rate:avg pnl>0,
    contribution:sum[pnl] % perf_summary[`total_pnl]
  by strategy from trades where not null strategy;
  
  / Attribution by time period
  trades:update hour:time.hh from trades;
  hourly_attribution:select 
    pnl:sum pnl,
    trade_count:count i,
    win_rate:avg pnl>0
  by hour from trades;
  
  (`summary`by_symbol`by_strategy`by_hour)!
    (perf_summary;symbol_attribution;strategy_attribution;hourly_attribution)
  };

/ =============================================================================
/ Helper Functions for Complex Calculations
/ =============================================================================

/ Calculate half-life for mean reversion
calc_half_life:{[series]
  / Simple AR(1) regression to estimate half-life
  if[count[series] < 10; :0n];
  
  y:1_ series;
  x:neg1_ series;
  
  / Linear regression: y = a + b*x
  n:count x;
  sum_x:sum x;
  sum_y:sum y;
  sum_xy:sum x*y;
  sum_x2:sum x*x;
  
  b:(n*sum_xy - sum_x*sum_y) % (n*sum_x2 - sum_x*sum_x);
  
  / Half-life calculation
  if[(b > 0) and (b < 1); neg log[0.5] % log b; 0n]
  };

/ Calculate Hurst exponent
calc_hurst_exponent:{[series]
  / Simplified Hurst exponent calculation
  if[count[series] < 50; :0n];
  
  n:count series;
  lags:2 4 8 16 32;
  rs_stats:();
  
  {[lag;series]
    chunks:0N lag#series;
    chunk_stats:{[chunk]
      if[lag <= count chunk;
        chunk_mean:avg chunk;
        deviations:chunk - chunk_mean;
        cum_deviations:sums deviations;
        range_val:max[cum_deviations] - min cum_deviations;
        std_val:dev chunk;
        rs_ratio:range_val % std_val;
        rs_ratio
      ; 0n]
    } each chunks;
    
    rs_stats,:avg chunk_stats where not null chunk_stats;
  }[;series] each lags;
  
  / Calculate Hurst exponent from R/S statistics
  if[all not null rs_stats;
    log_lags:log lags;
    log_rs:log rs_stats;
    
    / Linear regression
    n:count log_lags;
    sum_x:sum log_lags;
    sum_y:sum log_rs;
    sum_xy:sum log_lags*log_rs;
    sum_x2:sum log_lags*log_lags;
    
    hurst:(n*sum_xy - sum_x*sum_y) % (n*sum_x2 - sum_x*sum_x);
    hurst
  ; 0n]
  };

/ =============================================================================
/ Optimized Query Templates
/ =============================================================================

/ Fast OHLC calculation
calc_ohlc:{[tbl;start_time;end_time;syms;interval]
  / Optimized OHLC calculation using appropriate source
  if[null interval; interval:00:01:00.000];
  
  data:$[`trades_enhanced~tbl;
    select time, sym, price from trades_enhanced 
      where time within (start_time;end_time), sym in syms;
    select time, sym, mid as price from quotes_enhanced 
      where time within (start_time;end_time), sym in syms];
  
  / Add time buckets
  data:update bucket:interval xbar time from data;
  
  / Calculate OHLC
  ohlc:select 
    open:first price,
    high:max price,
    low:min price,
    close:last price,
    volume:count i
  by sym, bucket from data;
  
  ohlc
  };

/ Fast last quote lookup
get_last_quotes:{[syms;as_of_time]
  / Optimized last quote retrieval
  if[null as_of_time; as_of_time:.z.p];
  
  select last bid, last ask, last mid, last spread_bps, last time 
    by sym from quotes_enhanced 
    where sym in syms, time <= as_of_time
  };

/ Fast trade lookup with aggregation
get_trade_summary:{[syms;start_time;end_time;group_by]
  / Optimized trade summary with flexible grouping
  if[null group_by; group_by:`sym];
  
  trades:select from trades_enhanced 
    where sym in syms, time within (start_time;end_time);
  
  / Dynamic grouping
  if[`sym~group_by;
    select 
      trade_count:count i,
      total_volume:sum size,
      total_value:sum value,
      avg_price:avg price,
      vwap:sum[price*size]%sum size
    by sym from trades
  ; `hour~group_by;
    select 
      trade_count:count i,
      total_volume:sum size,
      total_value:sum value,
      avg_price:avg price,
      vwap:sum[price*size]%sum size
    by sym, hour:time.hh from trades
  ; / Add more grouping options as needed
    select 
      trade_count:count i,
      total_volume:sum size,
      total_value:sum value
    by sym from trades
  ]
  };

/ =============================================================================
/ Batch Processing and Performance Optimization
/ =============================================================================

/ Parallel processing for large datasets
calc_metrics_parallel:{[syms;start_time;end_time;metrics;batch_size]
  / Process symbols in parallel batches
  if[null batch_size; batch_size:10];
  if[null metrics; metrics:`vwap`volatility`liquidity];
  
  batches:0N batch_size#syms;
  results:();
  
  {[batch;start_time;end_time;metrics]
    batch_results:();
    
    if[`vwap in metrics;
      vwap_results:calc_vwap[`quotes_enhanced;start_time;end_time;batch];
      batch_results,:vwap_results;
    ];
    
    if[`volatility in metrics;
      vol_results:raze {calc_volatility_metrics[x;start_time;end_time;enlist 00:15:00.000]} each batch;
      batch_results,:vol_results;
    ];
    
    if[`liquidity in metrics;
      liq_results:calc_liquidity_metrics[start_time;end_time;batch;00:05:00.000];
      batch_results,:liq_results;
    ];
    
    batch_results
  }[;start_time;end_time;metrics] each batches;
  
  raze results
  };

/ Memory-efficient processing for historical analysis
process_historical_data:{[start_date;end_date;syms;analysis_func;chunk_days]
  / Process historical data in date chunks to manage memory
  if[null chunk_days; chunk_days:7];
  
  date_chunks:start_date + chunk_days * til ceiling (end_date-start_date) % chunk_days;
  results:();
  
  {[chunk_start;chunk_days;end_date;syms;analysis_func]
    chunk_end:min[chunk_start + chunk_days; end_date];
    
    / Convert dates to timestamps
    start_time:`timestamp$chunk_start;
    end_time:`timestamp$chunk_end + 1;
    
    / Run analysis for this chunk
    chunk_result:analysis_func[start_time;end_time;syms];
    results,:chunk_result;
    
    / Optional: garbage collection for large datasets
    .Q.gc[];
  }[;chunk_days;end_date;syms;analysis_func] each date_chunks;
  
  results
  };

/ =============================================================================
/ Specialized Analytics for Risk Management
/ =============================================================================

/ Real-time risk monitoring
calc_realtime_risk:{[portfolio_positions;current_prices]
  / Calculate real-time portfolio risk metrics
  portfolio:portfolio_positions lj `sym xkey current_prices;
  
  / Position-level risk
  portfolio:update 
    market_value:position * current_price,
    unrealized_pnl:(current_price - avg_cost) * position
  from portfolio;
  
  / Portfolio-level risk
  portfolio_summary:select 
    total_market_value:sum market_value,
    total_unrealized_pnl:sum unrealized_pnl,
    gross_exposure:sum abs market_value,
    net_exposure:sum market_value,
    long_exposure:sum market_value where position > 0,
    short_exposure:sum market_value where position < 0,
    position_count:count i
  from ([]dummy:1) where count portfolio;
  
  / Risk metrics
  portfolio_summary:update 
    leverage:gross_exposure % abs net_exposure,
    market_neutral_ratio:abs[long_exposure + short_exposure] % gross_exposure
  from portfolio_summary;
  
  (`positions`summary)!(portfolio;portfolio_summary)
  };

/ Stress testing analytics
calc_stress_scenarios:{[portfolio_positions;shock_scenarios]
  / Apply various shock scenarios to portfolio
  stress_results:();
  
  {[scenario_name;shocks;positions]
    / Apply shocks to current positions
    shocked_portfolio:positions lj `sym xkey shocks;
    
    shocked_portfolio:update 
      shocked_price:current_price * (1 + price_shock),
      shocked_pnl:(current_price * (1 + price_shock) - avg_cost) * position
    from shocked_portfolio;
    
    / Calculate scenario impact
    scenario_impact:select 
      scenario:scenario_name,
      total_pnl:sum shocked_pnl,
      pnl_change:sum[shocked_pnl] - sum (current_price - avg_cost) * position,
      worst_position:min shocked_pnl,
      best_position:max shocked_pnl
    from ([]dummy:1) where count shocked_portfolio;
    
    stress_results,:scenario_impact;
  }[;shock_scenarios] each key shock_scenarios;
  
  stress_results
  };

/ =============================================================================
/ Initialization and Documentation
/ =============================================================================

/ Display available analytics functions
show_analytics_functions:{[]
  -1 "=== Analytics Query Library ===";
  -1 "";
  -1 "Time-Series Analytics:";
  -1 "  calc_vwap[tbl;start;end;syms]           - VWAP calculation";
  -1 "  calc_twap[tbl;start;end;syms;interval]  - TWAP calculation";
  -1 "  calc_volume_profile[start;end;syms;buckets] - Volume at price";
  -1 "  calc_spread_analytics[start;end;syms;interval] - Spread analysis";
  -1 "";
  -1 "Statistical Analytics:";
  -1 "  calc_rolling_stats[tbl;sym;start;end;window;metrics] - Rolling statistics";
  -1 "  calc_correlation_matrix[syms;start;end;interval;method] - Correlation analysis";
  -1 "  calc_volatility_metrics[sym;start;end;intervals] - Volatility analysis";
  -1 "";
  -1 "Market Microstructure:";
  -1 "  calc_order_flow[start;end;syms;interval] - Order flow analysis";
  -1 "  calc_market_impact[start;end;syms;buckets] - Market impact analysis";
  -1 "  calc_liquidity_metrics[start;end;syms;interval] - Liquidity analysis";
  -1 "";
  -1 "Statistical Arbitrage:";
  -1 "  calc_pairs_metrics[sym1;sym2;start;end;interval] - Pairs analysis";
  -1 "  calc_alpha_signals[syms;start;end;factors] - Alpha signal generation";
  -1 "  calc_performance_attribution[trades;start;end] - Performance attribution";
  -1 "";
  -1 "Optimized Queries:";
  -1 "  calc_ohlc[tbl;start;end;syms;interval]  - Fast OHLC calculation";
  -1 "  get_last_quotes[syms;as_of_time]        - Last quote lookup";
  -1 "  get_trade_summary[syms;start;end;group] - Trade summary";
  -1 "";
  -1 "Risk Management:";
  -1 "  calc_realtime_risk[positions;prices]    - Real-time risk monitoring";
  -1 "  calc_stress_scenarios[positions;shocks] - Stress testing";
  -1 "";
  };

/ Performance benchmarking
benchmark_analytics:{[]
  / Run performance benchmarks on key analytics functions
  test_syms:`AAPL`MSFT`GOOGL;
  start_time:2025.06.25D09:30:00.000000000;
  end_time:2025.06.25D16:00:00.000000000;
  
  -1 "Running analytics performance benchmarks...";
  
  / VWAP benchmark
  start:.z.p;
  vwap_result:calc_vwap[`trades_enhanced;start_time;end_time;test_syms];
  vwap_time:(`long$.z.p - start) % 1000000;
  -1 "VWAP calculation: ",string[vwap_time]," ms";
  
  / Correlation benchmark
  start:.z.p;
  corr_result:calc_correlation_matrix[test_syms;start_time;end_time;00:01:00.000;`price];
  corr_time:(`long$.z.p - start) % 1000000;
  -1 "Correlation matrix: ",string[corr_time]," ms";
  
  / Liquidity benchmark
  start:.z.p;
  liq_result:calc_liquidity_metrics[start_time;end_time;test_syms;00:05:00.000];
  liq_time:(`long$.z.p - start) % 1000000;
  -1 "Liquidity metrics: ",string[liq_time]," ms";
  
  -1 "Benchmark completed.";
  };

/ Initialize analytics module
-1 "Analytics Query Library Loaded Successfully";
-1 "Type 'show_analytics_functions[]' for function documentation";
-1 "Type 'benchmark_analytics[]' to run performance benchmarks";
-1 "";