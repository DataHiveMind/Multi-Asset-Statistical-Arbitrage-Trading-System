/ =============================================================================
/ Quotes Table Schema for Statistical Arbitrage System
/ =============================================================================
/ Purpose: Defines the schema for the quotes table in kdb+, optimized for
/          real-time bid/ask data processing and efficient querying
/ Author: Statistical Arbitrage System
/ Date: 2025-06-24
/ Version: 1.0
/ =============================================================================

/ Load required utilities and dependencies
\l util.q
\l sym.q

/ =============================================================================
/ Core Quotes Table Schema Definition
/ =============================================================================

/ Basic quotes table schema
quotes:([]
  time:`timestamp$();           / Quote timestamp (nanosecond precision)
  sym:`symbol$();               / Trading symbol/instrument
  bid:`real$();                 / Best bid price
  ask:`real$();                 / Best ask price
  bidsize:`long$();             / Best bid size/quantity
  asksize:`long$();             / Best ask size/quantity
  exchange:`symbol$();          / Exchange/venue identifier
  quotecond:`symbol$();         / Quote condition codes
  seq:`long$();                 / Sequence number for ordering
  src:`symbol$()                / Data source identifier
  );

/ Enhanced quotes table with additional market microstructure fields
quotes_enhanced:([]
  time:`timestamp$();           / Quote timestamp (nanosecond precision)
  date:`date$();                / Quote date (for partitioning)
  sym:`symbol$();               / Trading symbol/instrument
  
  / Level 1 market data
  bid:`real$();                 / Best bid price
  ask:`real$();                 / Best ask price
  bidsize:`long$();             / Best bid size/quantity
  asksize:`long$();             / Best ask size/quantity
  
  / Derived market data
  mid:`real$();                 / Mid price (bid+ask)/2
  spread:`real$();              / Bid-ask spread (ask-bid)
  spread_bps:`real$();          / Spread in basis points
  spread_pct:`real$();          / Spread as percentage of mid
  
  / Market depth and liquidity
  total_bid_qty:`long$();       / Total quantity at bid across all levels
  total_ask_qty:`long$();       / Total quantity at ask across all levels
  depth_ratio:`real$();         / Bid depth / ask depth ratio
  imbalance:`real$();           / Order imbalance metric
  
  / Market microstructure
  tick_direction:`symbol$();    / Tick direction: `up`down`same`unknown
  quote_type:`symbol$();        / Quote type: `normal`opening`closing`halt
  market_state:`symbol$();      / Market state: `open`close`halt`auction
  
  / Exchange and venue data
  exchange:`symbol$();          / Primary exchange identifier
  ecn:`symbol$();               / ECN/venue identifier
  quotecond:`symbol$();         / Quote condition codes
  seq:`long$();                 / Sequence number for ordering
  src:`symbol$();               / Data source identifier
  
  / Statistical arbitrage specific fields
  fair_value:`real$();          / Theoretical fair value
  fair_value_source:`symbol$(); / Fair value calculation source
  model_price:`real$();         / Model-derived price
  signal_strength:`real$();     / Alpha signal strength
  volatility:`real$();          / Realized volatility estimate
  beta:`real$();                / Beta relative to market
  
  / Technical indicators (real-time)
  sma_20:`real$();              / 20-period simple moving average
  ema_12:`real$();              / 12-period exponential moving average
  ema_26:`real$();              / 26-period exponential moving average
  macd:`real$();                / MACD indicator
  rsi:`real$();                 / Relative strength index
  bb_upper:`real$();            / Bollinger band upper
  bb_lower:`real$();            / Bollinger band lower
  atr:`real$();                 / Average true range
  
  / Risk and monitoring
  price_change:`real$();        / Price change from previous quote
  price_change_pct:`real$();    / Price change percentage
  volatility_regime:`symbol$(); / Volatility regime: `low`normal`high`extreme
  liquidity_score:`real$();     / Liquidity quality score
  trade_opportunity:`symbol$(); / Trading opportunity flag
  
  / Quality and validation
  data_quality:`symbol$();      / Data quality flag: `good`suspect`bad
  validation_flag:`symbol$();   / Validation status
  quote_age:`long$();           / Age of quote in nanoseconds
  staleness_flag:`boolean$();   / Quote staleness indicator
  
  / Audit trail
  created_time:`timestamp$();   / Record creation timestamp
  updated_time:`timestamp$();   / Last update timestamp
  version:`long$();             / Record version for updates
  status:`symbol$()             / Record status: `active`stale`invalid
  );

/ Level II quotes table for full market depth
quotes_l2:([]
  time:`timestamp$();           / Quote timestamp
  sym:`symbol$();               / Trading symbol
  side:`symbol$();              / Quote side: `bid`ask
  price:`real$();               / Price level
  size:`long$();                / Size at price level
  level:`int$();                / Price level (1=best, 2=second best, etc.)
  exchange:`symbol$();          / Exchange identifier
  mm:`symbol$();                / Market maker identifier
  seq:`long$();                 / Sequence number
  action:`symbol$();            / Action: `add`update`delete
  src:`symbol$()                / Data source
  );

/ =============================================================================
/ Table Attributes and Optimization
/ =============================================================================

/ Set table attributes for optimal performance
quotes_attrs:{[t]
  / Sort by time and symbol for efficient queries
  `s#`time`sym xcols t
  };

/ Apply sorted attribute to time column for binary search
update `s#time from `quotes;
update `s#time from `quotes_enhanced;
update `s#time from `quotes_l2;

/ Apply grouped attribute to symbol column for efficient symbol-based queries
update `g#sym from `quotes;
update `g#sym from `quotes_enhanced;
update `g#sym from `quotes_l2;

/ Apply parted attribute to exchange if many exchanges
/ update `p#exchange from `quotes_enhanced;

/ =============================================================================
/ Real-time Data Processing Functions
/ =============================================================================

/ Real-time quote update with derived calculations
update_quote_realtime:{[quote_data]
  / Validate input
  if[null quote_data`time; quote_data[`time]:.z.p];
  if[any null quote_data`bid`ask`bidsize`asksize; '"Invalid quote data"];
  
  / Calculate derived fields
  quote_data[`mid]:(quote_data[`bid] + quote_data[`ask]) % 2;
  quote_data[`spread]:quote_data[`ask] - quote_data[`bid];
  quote_data[`spread_bps]:10000 * quote_data[`spread] % quote_data[`mid];
  quote_data[`spread_pct]:100 * quote_data[`spread] % quote_data[`mid];
  
  / Calculate imbalance
  total_qty:quote_data[`bidsize] + quote_data[`asksize];
  if[total_qty > 0; 
    quote_data[`imbalance]:(quote_data[`bidsize] - quote_data[`asksize]) % total_qty
  ];
  
  / Determine tick direction
  prev_mid:exec last mid from quotes_enhanced where sym=quote_data`sym;
  if[not null prev_mid;
    if[quote_data[`mid] > prev_mid; quote_data[`tick_direction]:`up];
    if[quote_data[`mid] < prev_mid; quote_data[`tick_direction]:`down];
    if[quote_data[`mid] = prev_mid; quote_data[`tick_direction]:`same]
  ];
  
  / Add timestamp and status
  quote_data[`created_time]:.z.p;
  quote_data[`version]:1;
  quote_data[`status]:`active;
  
  quote_data
  };

/ Bulk quote processing for historical data
process_quote_batch:{[quotes_batch]
  / Process each quote in batch
  processed:{update_quote_realtime x} each quotes_batch;
  
  / Calculate technical indicators for batch
  processed:calculate_technical_indicators processed;
  
  processed
  };

/ Calculate technical indicators
calculate_technical_indicators:{[quotes_data]
  / This would integrate with technical analysis functions
  / For now, placeholder calculations
  
  / Simple moving average (would use rolling window)
  quotes_data[`sma_20]:quotes_data[`mid];  / Placeholder
  
  / EMA calculations (would use proper EMA formula)
  quotes_data[`ema_12]:quotes_data[`mid];  / Placeholder
  quotes_data[`ema_26]:quotes_data[`mid];  / Placeholder
  
  / MACD
  quotes_data[`macd]:quotes_data[`ema_12] - quotes_data[`ema_26];
  
  quotes_data
  };

/ =============================================================================
/ Quote Validation and Quality Control
/ =============================================================================

/ Comprehensive quote validation
validate_quote:{[quote]
  / Check required fields
  required_fields:`time`sym`bid`ask`bidsize`asksize;
  if[any null quote required_fields; '"Missing required fields"];
  
  / Check price validity
  if[any 0 >= quote`bid`ask; '"Invalid prices"];
  if[quote[`ask] <= quote[`bid]; '"Ask must be greater than bid"];
  
  / Check size validity
  if[any 0 >= quote`bidsize`asksize; '"Invalid sizes"];
  
  / Check spread reasonableness (max 5% of mid price)
  mid:(quote[`bid] + quote[`ask]) % 2;
  spread:quote[`ask] - quote[`bid];
  if[spread > 0.05 * mid; '"Spread too wide"];
  
  / Check timestamp validity
  if[quote[`time] > .z.p + 00:00:01; '"Future timestamp"];
  
  / Mark data quality
  quote[`data_quality]:`good;
  quote[`validation_flag]:`passed;
  
  quote
  };

/ Detect stale quotes
detect_stale_quotes:{[symbol;max_age_ms]
  current_time:.z.p;
  cutoff:current_time - `timespan$max_age_ms * 1000000;
  
  stale:select from quotes_enhanced 
    where sym=symbol, time < cutoff, status=`active;
  
  / Mark as stale
  update status:`stale, updated_time:current_time 
    from `quotes_enhanced 
    where sym=symbol, time < cutoff, status=`active;
  
  count stale
  };

/ Quote quality scoring
calculate_quality_score:{[quote]
  score:100; / Start with perfect score
  
  / Penalize wide spreads
  spread_pct:100 * quote[`spread] % quote[`mid];
  if[spread_pct > 1; score-:10 * spread_pct];
  
  / Penalize small sizes
  if[quote[`bidsize] < 100; score-:10];
  if[quote[`asksize] < 100; score-:10];
  
  / Penalize age
  age_ms:(`long$.z.p - quote[`time]) % 1000000;
  if[age_ms > 1000; score-:age_ms % 100]; / 1 point per 100ms
  
  / Ensure score bounds
  max[0; min[100; score]]
  };

/ =============================================================================
/ Market Data Analysis Functions
/ =============================================================================

/ Best bid/offer (BBO) extraction
get_bbo:{[symbol;timestamp]
  select last bid, last ask, last bidsize, last asksize, last mid 
    from quotes_enhanced 
    where sym=symbol, time<=timestamp
  };

/ Market depth analysis
analyze_market_depth:{[symbol;start_time;end_time]
  quotes:select from quotes_enhanced 
    where sym=symbol, time within (start_time;end_time);
  
  select 
    avg_bid:avg bid,
    avg_ask:avg ask,
    avg_spread:avg spread,
    avg_spread_bps:avg spread_bps,
    avg_imbalance:avg imbalance,
    liquidity_score:avg liquidity_score,
    quote_count:count i,
    avg_bid_size:avg bidsize,
    avg_ask_size:avg asksize
  from quotes
  };

/ Tick analysis
analyze_tick_data:{[symbol;start_time;end_time]
  quotes:select from quotes_enhanced 
    where sym=symbol, time within (start_time;end_time);
  
  / Count tick directions
  tick_summary:select count i by tick_direction from quotes;
  
  / Price movement analysis
  price_moves:select from quotes where tick_direction in `up`down;
  
  select 
    total_quotes:count quotes,
    up_ticks:tick_summary[`up;`x],
    down_ticks:tick_summary[`down;`x],
    same_ticks:tick_summary[`same;`x],
    avg_price_change:avg abs price_change,
    max_price_change:max abs price_change,
    volatility:dev mid
  from ([]dummy:1)
  };

/ Quote frequency analysis
analyze_quote_frequency:{[symbol;start_time;end_time]
  quotes:select time, mid from quotes_enhanced 
    where sym=symbol, time within (start_time;end_time);
  
  / Calculate inter-quote times
  quotes:update inter_quote_time:time - prev time from quotes;
  
  select 
    quote_count:count i,
    avg_frequency:avg inter_quote_time,
    min_frequency:min inter_quote_time,
    max_frequency:max inter_quote_time,
    frequency_std:dev inter_quote_time
  from quotes where not null inter_quote_time
  };

/ =============================================================================
/ Query Optimization Functions
/ =============================================================================

/ Get quotes by symbol and time range
get_quotes_by_symbol_time:{[symbol;start_time;end_time]
  select from quotes_enhanced 
    where sym=symbol, time within (start_time;end_time)
  };

/ Get quotes by multiple symbols
get_quotes_by_symbols:{[symbols;start_time;end_time]
  select from quotes_enhanced 
    where sym in symbols, time within (start_time;end_time)
  };

/ Get latest quote for symbol
get_latest_quote:{[symbol]
  select from quotes_enhanced 
    where sym=symbol, time=(max;time) fby sym
  };

/ Get quotes around specific time
get_quotes_around_time:{[symbol;target_time;window_ms]
  start_time:target_time - `timespan$window_ms * 1000000;
  end_time:target_time + `timespan$window_ms * 1000000;
  
  select from quotes_enhanced 
    where sym=symbol, time within (start_time;end_time)
  };

/ Get Level II quotes
get_l2_quotes:{[symbol;start_time;end_time;max_levels]
  if[null max_levels; max_levels:10];
  
  select from quotes_l2 
    where sym=symbol, time within (start_time;end_time), level <= max_levels
  };

/ =============================================================================
/ Real-time Monitoring and Alerts
/ =============================================================================

/ Real-time quote monitoring dashboard
monitor_quotes:{[]
  / Get latest quotes for all symbols
  latest:select last time, last bid, last ask, last mid, last spread_bps, 
           last imbalance, last data_quality by sym 
         from quotes_enhanced;
  
  / Add derived metrics
  latest:update 
    age_ms:(`long$.z.p - time) % 1000000,
    freshness:?[(`long$.z.p - time) % 1000000 < 1000; `fresh; `stale]
  from latest;
  
  `sym`bid`ask`mid`spread_bps`imbalance`freshness xcols latest
  };

/ Quote alert system
check_quote_alerts:{[symbol]
  / Get recent quotes
  recent:select from quotes_enhanced 
    where sym=symbol, time > .z.p - 00:01;  / Last minute
  
  alerts:();
  
  / Wide spread alert
  if[count recent;
    max_spread:max recent`spread_bps;
    if[max_spread > 100;  / > 100 bps
      alerts,:enlist `wide_spread`time`.z.p`sym`symbol`spread_bps`max_spread]
  ];
  
  / Quote frequency alert (too few quotes)
  quote_count:count recent;
  if[quote_count < 10;  / Less than 10 quotes per minute
    alerts,:enlist `low_frequency`time`.z.p`sym`symbol`count`quote_count
  ];
  
  / Stale quote alert
  if[count recent;
    latest_time:max recent`time;
    age_ms:(`long$.z.p - latest_time) % 1000000;
    if[age_ms > 5000;  / > 5 seconds old
      alerts,:enlist `stale_quote`time`.z.p`sym`symbol`age_ms`age_ms]
  ];
  
  / Price volatility alert
  if[count recent;
    price_std:dev recent`mid;
    avg_price:avg recent`mid;
    volatility_pct:100 * price_std % avg_price;
    if[volatility_pct > 5;  / > 5% volatility
      alerts,:enlist `high_volatility`time`.z.p`sym`symbol`volatility_pct`volatility_pct]
  ];
  
  alerts
  };

/ Market quality monitoring
monitor_market_quality:{[]
  / Analyze market quality across all symbols
  recent:select from quotes_enhanced where time > .z.p - 00:05;  / Last 5 minutes
  
  quality_stats:select 
    quote_count:count i,
    avg_spread_bps:avg spread_bps,
    avg_imbalance:avg abs imbalance,
    avg_quality_score:avg liquidity_score,
    stale_quotes:sum status=`stale,
    bad_quality:sum data_quality=`bad
  by sym from recent;
  
  / Add quality grade
  quality_stats:update 
    quality_grade:?[avg_spread_bps < 10; `excellent;
                   avg_spread_bps < 25; `good;
                   avg_spread_bps < 50; `fair; `poor]
  from quality_stats;
  
  `sym`quality_grade`quote_count`avg_spread_bps`avg_imbalance xcols quality_stats
  };

/ =============================================================================
/ Data Management and Maintenance
/ =============================================================================

/ Insert quote with validation and processing
insert_quote:{[quote_data]
  / Validate quote
  validated:validate_quote quote_data;
  
  / Process for real-time updates
  processed:update_quote_realtime validated;
  
  / Insert into table
  `quotes_enhanced insert processed;
  
  / Return success
  1
  };

/ Bulk insert quotes
bulk_insert_quotes:{[quotes_data;batch_size]
  if[batch_size<=0; batch_size:1000];
  
  n:count quotes_data;
  batches:0N batch_size#quotes_data;
  
  inserted:0;
  {[batch] 
    processed:process_quote_batch batch;
    `quotes_enhanced insert processed;
    inserted+:count processed
  } each batches;
  
  inserted
  };

/ Clean up old quotes
cleanup_old_quotes:{[retention_days]
  cutoff:.z.d - retention_days;
  old_count:exec count i from quotes_enhanced where date < cutoff;
  delete from `quotes_enhanced where date < cutoff;
  delete from `quotes_l2 where date < cutoff;
  old_count
  };

/ Compress historical quotes
compress_historical_quotes:{[compress_date]
  / Archive old quote data
  old_data:select from quotes_enhanced where date < compress_date;
  
  / Save to compressed storage
  (` sv `compressed_quotes,compress_date) set old_data;
  
  / Remove from main table
  delete from `quotes_enhanced where date < compress_date;
  
  count old_data
  };

/ Rebuild indexes
rebuild_indexes:{[]
  / Reapply table attributes
  update `s#time from `quotes_enhanced;
  update `g#sym from `quotes_enhanced;
  update `s#time from `quotes_l2;
  update `g#sym from `quotes_l2;
  
  1b
  };

/ =============================================================================
/ Performance Analytics
/ =============================================================================

/ Quote latency analysis
analyze_quote_latency:{[symbol;start_time;end_time]
  quotes:select time, created_time from quotes_enhanced 
    where sym=symbol, time within (start_time;end_time), 
          not null created_time;
  
  / Calculate processing latency
  quotes:update latency_ns:`long$created_time - time from quotes;
  
  select 
    quote_count:count i,
    avg_latency_ns:avg latency_ns,
    median_latency_ns:med latency_ns,
    p95_latency_ns:0.95 mavg latency_ns,
    p99_latency_ns:0.99 mavg latency_ns,
    max_latency_ns:max latency_ns
  from quotes
  };

/ Table performance statistics
get_quotes_table_stats:{[]
  stats:`table`rows`columns`memory_mb!(
    `quotes_enhanced;
    count quotes_enhanced;
    count cols quotes_enhanced;
    sum .Q.s each quotes_enhanced);
  
  / Add L2 stats
  l2_stats:`table`rows`columns`memory_mb!(
    `quotes_l2;
    count quotes_l2;
    count cols quotes_l2;
    sum .Q.s each quotes_l2);
  
  (stats;l2_stats)
  };

/ =============================================================================
/ Export and Integration
/ =============================================================================

/ Export quotes to CSV
export_quotes_csv:{[filename;symbol;start_time;end_time]
  quotes:get_quotes_by_symbol_time[symbol;start_time;end_time];
  (hsym `$filename) 0: csv 0: quotes;
  count quotes
  };

/ Export to JSON for API
export_quotes_json:{[symbol;start_time;end_time]
  quotes:get_quotes_by_symbol_time[symbol;start_time;end_time];
  .j.j quotes
  };

/ Real-time quote feed for external systems
setup_quote_feed:{[symbols;callback_func]
  / Set up real-time quote distribution
  / This would integrate with publish/subscribe mechanisms
  };

/ =============================================================================
/ Initialization and Setup
/ =============================================================================

/ Initialize quotes schema
init_quotes_schema:{[]
  / Create tables if they don't exist
  if[not `quotes in tables[]; quotes::quotes];
  if[not `quotes_enhanced in tables[]; quotes_enhanced::quotes_enhanced];
  if[not `quotes_l2 in tables[]; quotes_l2::quotes_l2];
  
  / Set up attributes
  quotes_attrs[`quotes];
  quotes_attrs[`quotes_enhanced];
  quotes_attrs[`quotes_l2];
  
  / Initialize derived calculations
  / setup_real_time_calculations[];
  
  1b  / Success
  };

/ Schema information
quotes_schema_info:{[]
  info:([]
    table:`quotes`quotes_enhanced`quotes_l2;
    columns:(count cols quotes;count cols quotes_enhanced;count cols quotes_l2);
    rows:(count quotes;count quotes_enhanced;count quotes_l2);
    partitioned:(0b;0b;0b);  / Update based on actual setup
    indexed:(1b;1b;1b)       / All have indexes
    );
  info
  };

/ =============================================================================
/ Documentation and Help
/ =============================================================================

/ Display schema documentation
show_quotes_schema:{[]
  -1 "=== Quotes Table Schema ===";
  -1 "";
  -1 "Tables:";
  -1 "  quotes         - Basic quotes table (BBO)";
  -1 "  quotes_enhanced - Enhanced table with microstructure and analytics";
  -1 "  quotes_l2      - Level II market depth data";
  -1 "";
  -1 "Key Functions:";
  -1 "  insert_quote[quote_data]           - Insert single quote";
  -1 "  bulk_insert_quotes[data;batch]     - Bulk insert with batching";
  -1 "  get_quotes_by_symbol_time[sym;s;e] - Query by symbol and time";
  -1 "  get_latest_quote[sym]              - Get latest quote";
  -1 "  monitor_quotes[]                   - Real-time monitoring";
  -1 "  check_quote_alerts[sym]            - Alert system";
  -1 "";
  -1 "Analysis:";
  -1 "  analyze_market_depth[sym;s;e]      - Market depth analysis";
  -1 "  analyze_tick_data[sym;s;e]         - Tick-by-tick analysis";
  -1 "  monitor_market_quality[]           - Market quality monitoring";
  -1 "";
  -1 "Maintenance:";
  -1 "  cleanup_old_quotes[days]           - Remove old data";
  -1 "  detect_stale_quotes[sym;age]       - Find stale quotes";
  -1 "  get_quotes_table_stats[]           - Table statistics";
  -1 "";
  };

/ Display table schemas
show_quotes_tables:{[]
  -1 "=== Quote Tables Structure ===";
  -1 "";
  -1 "Basic Quotes Table:";
  show meta quotes;
  -1 "";
  -1 "Enhanced Quotes Table:";
  show meta quotes_enhanced;
  -1 "";
  -1 "Level II Quotes Table:";
  show meta quotes_l2;
  };

/ =============================================================================
/ Module Initialization
/ =============================================================================

/ Initialize the schema on load
init_quotes_schema[];

/ Display startup message
-1 "Quotes Schema Module Loaded Successfully";
-1 "Type 'show_quotes_schema[]' for documentation";
-1 "Type 'show_quotes_tables[]' to see table structures";
-1 "";