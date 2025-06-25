/ =============================================================================
/ Trades Table Schema for Statistical Arbitrage System
/ =============================================================================
/ Purpose: Defines the schema for the trades table in kdb+, optimized for
/          high-frequency trading data storage and efficient querying
/ Author: Statistical Arbitrage System
/ Date: 2025-06-24
/ Version: 1.0
/ =============================================================================

/ Load required utilities and dependencies
\l util.q
\l sym.q

/ =============================================================================
/ Core Trades Table Schema Definition
/ =============================================================================

/ Primary trades table schema
trades:([]
  time:`timestamp$();           / Trade timestamp (nanosecond precision)
  sym:`symbol$();               / Trading symbol/instrument
  price:`real$();               / Trade price
  size:`long$();                / Trade size/quantity
  side:`symbol$();              / Trade side: `buy`sell`unknown
  exchange:`symbol$();          / Exchange/venue identifier
  tradeid:`symbol$();           / Unique trade identifier
  condition:`symbol$();         / Trade condition codes
  seq:`long$();                 / Sequence number for ordering
  src:`symbol$()                / Data source identifier
  );

/ Enhanced trades table with additional statistical arbitrage fields
trades_enhanced:([]
  time:`timestamp$();           / Trade timestamp (nanosecond precision)
  date:`date$();                / Trade date (for partitioning)
  sym:`symbol$();               / Trading symbol/instrument
  price:`real$();               / Trade price
  size:`long$();                / Trade size/quantity
  value:`real$();               / Trade value (price * size)
  side:`symbol$();              / Trade side: `buy`sell`unknown
  exchange:`symbol$();          / Exchange/venue identifier
  tradeid:`symbol$();           / Unique trade identifier
  condition:`symbol$();         / Trade condition codes
  seq:`long$();                 / Sequence number for ordering
  src:`symbol$();               / Data source identifier
  
  / Market microstructure fields
  bid:`real$();                 / Best bid at trade time
  ask:`real$();                 / Best ask at trade time
  bidsize:`long$();             / Best bid size
  asksize:`long$();             / Best ask size
  spread:`real$();              / Bid-ask spread
  midprice:`real$();            / Mid price
  
  / Statistical arbitrage specific fields
  vwap:`real$();                / Volume weighted average price
  twap:`real$();                / Time weighted average price
  participation:`real$();        / Participation rate
  impact:`real$();              / Price impact (basis points)
  urgency:`symbol$();           / Urgency flag: `low`medium`high`critical
  strategy:`symbol$();          / Strategy identifier
  portfolio:`symbol$();         / Portfolio identifier
  trader:`symbol$();            / Trader identifier
  
  / Technical indicators (computed real-time)
  sma_5:`real$();               / 5-period simple moving average
  sma_20:`real$();              / 20-period simple moving average
  ema_12:`real$();              / 12-period exponential moving average
  ema_26:`real$();              / 26-period exponential moving average
  rsi:`real$();                 / Relative strength index
  bollinger_upper:`real$();     / Bollinger band upper
  bollinger_lower:`real$();     / Bollinger band lower
  
  / Risk and compliance fields
  risk_limit:`real$();          / Position risk limit
  pnl:`real$();                 / Realized P&L for this trade
  commission:`real$();          / Commission/fees
  tax:`real$();                 / Tax implications
  compliance_flag:`symbol$();   / Compliance status: `ok`warning`violation
  
  / Audit trail
  created_time:`timestamp$();   / Record creation timestamp
  updated_time:`timestamp$();   / Last update timestamp
  version:`long$();             / Record version for updates
  status:`symbol$()             / Record status: `active`cancelled`corrected
  );

/ =============================================================================
/ Table Attributes and Optimization
/ =============================================================================

/ Set table attributes for optimal performance
trades_attrs:{[t]
  / Sort by time and symbol for efficient queries
  `s#`time`sym xcols t
  };

/ Apply sorted attribute to time column (assumes data comes in time order)
/ This enables binary search for time-based queries
update `s#time from `trades;
update `s#time from `trades_enhanced;

/ Apply grouped attribute to symbol column for efficient symbol-based queries
/ This groups all records for each symbol together
update `g#sym from `trades;
update `g#sym from `trades_enhanced;

/ Apply parted attribute to exchange column if applicable
/ update `p#exchange from `trades;

/ =============================================================================
/ Partitioning Strategy
/ =============================================================================

/ Partitioning function for date-based partitioning
/ This enables efficient queries by date range and parallel processing
partition_trades:{[t;dt]
  / Partition table by date
  .Q.dpft[`:.;dt;`sym;t]
  };

/ Monthly partitioning for historical data
partition_trades_monthly:{[t;month]
  / Partition by year-month for longer retention
  dt:month;
  .Q.dpft[`:.;dt;`sym;t]
  };

/ Splayed table definition for very large datasets
/ This stores each column in a separate file for memory efficiency
splay_trades:{[t;path]
  / Create splayed table
  (` sv path,`trades) set .Q.en[`:.]t;
  };

/ =============================================================================
/ Index Definitions
/ =============================================================================

/ Define indexes for frequently queried columns
create_indexes:{[]
  / Time-based index (already sorted)
  / Symbol-based index (already grouped)
  
  / Exchange index for venue analysis
  .Q.ind[`exchange;trades_enhanced];
  
  / Strategy index for strategy performance analysis
  .Q.ind[`strategy;trades_enhanced];
  
  / Portfolio index for portfolio analysis
  .Q.ind[`portfolio;trades_enhanced];
  
  / Side index for buy/sell analysis
  .Q.ind[`side;trades_enhanced];
  
  / Composite indexes for common query patterns
  .Q.ind[`sym`exchange;trades_enhanced];
  .Q.ind[`strategy`sym;trades_enhanced];
  .Q.ind[`portfolio`sym;trades_enhanced];
  };

/ =============================================================================
/ Data Validation and Constraints
/ =============================================================================

/ Validation functions for data integrity
validate_trade:{[t]
  / Check required fields are not null
  if[any null t`time`sym`price`size; '"Missing required fields"];
  
  / Check price is positive
  if[any 0>=t`price; '"Invalid price"];
  
  / Check size is positive
  if[any 0>=t`size; '"Invalid size"];
  
  / Check side is valid
  if[not all t[`side] in `buy`sell`unknown; '"Invalid side"];
  
  / Check timestamp is valid
  if[any null t`time; '"Invalid timestamp"];
  
  / Return validated table
  t
  };

/ Data type enforcement
enforce_types:{[t]
  / Ensure correct data types
  update time:`timestamp$time from t;
  update sym:`symbol$sym from t;
  update price:`real$price from t;
  update size:`long$size from t;
  update side:`symbol$side from t;
  update exchange:`symbol$exchange from t
  };

/ =============================================================================
/ Trade Processing Functions
/ =============================================================================

/ Insert trade with validation
insert_trade:{[t]
  / Validate trade data
  validated:validate_trade t;
  
  / Enforce types
  typed:enforce_types validated;
  
  / Add audit fields
  enriched:update created_time:.z.p, version:1, status:`active from typed;
  
  / Insert into table
  `trades_enhanced insert enriched;
  
  / Return number of records inserted
  count enriched
  };

/ Bulk insert with batching
bulk_insert_trades:{[trades_data;batch_size]
  if[batch_size<=0; batch_size:1000];
  
  n:count trades_data;
  batches:0N batch_size#trades_data;
  
  inserted:0;
  {[batch] inserted+:insert_trade batch} each batches;
  
  inserted
  };

/ Update existing trade
update_trade:{[tradeid;updates]
  / Find existing record
  existing:select from trades_enhanced where tradeid=tradeid;
  
  if[0=count existing; '"Trade not found"];
  
  / Update fields
  updated:existing,'updates;
  updated:update updated_time:.z.p, version:version+1 from updated;
  
  / Replace in table
  delete from `trades_enhanced where tradeid=tradeid;
  `trades_enhanced insert updated;
  
  count updated
  };

/ Cancel/correct trade
cancel_trade:{[tradeid;reason]
  update status:`cancelled, updated_time:.z.p, version:version+1 
    from `trades_enhanced where tradeid=tradeid;
  };

/ =============================================================================
/ Query Optimization Functions
/ =============================================================================

/ Optimized query functions for common patterns

/ Get trades by symbol and time range
get_trades_by_symbol_time:{[symbol;start_time;end_time]
  select from trades_enhanced 
    where sym=symbol, time within (start_time;end_time)
  };

/ Get trades by multiple symbols
get_trades_by_symbols:{[symbols;start_time;end_time]
  select from trades_enhanced 
    where sym in symbols, time within (start_time;end_time)
  };

/ Get trades by exchange
get_trades_by_exchange:{[exch;start_time;end_time]
  select from trades_enhanced 
    where exchange=exch, time within (start_time;end_time)
  };

/ Get trades by strategy
get_trades_by_strategy:{[strat;start_time;end_time]
  select from trades_enhanced 
    where strategy=strat, time within (start_time;end_time)
  };

/ Get recent trades (last N minutes)
get_recent_trades:{[symbol;minutes]
  cutoff:.z.p - 00:00:00.000000000 + `timespan$minutes*60000000000;
  select from trades_enhanced 
    where sym=symbol, time>=cutoff
  };

/ =============================================================================
/ Aggregation and Analytics Functions
/ =============================================================================

/ VWAP calculation
calc_vwap:{[symbol;start_time;end_time]
  trades:get_trades_by_symbol_time[symbol;start_time;end_time];
  exec sum[price*size]%sum size from trades
  };

/ TWAP calculation
calc_twap:{[symbol;start_time;end_time]
  trades:get_trades_by_symbol_time[symbol;start_time;end_time];
  exec avg price from trades
  };

/ Volume analysis
calc_volume_stats:{[symbol;start_time;end_time]
  trades:get_trades_by_symbol_time[symbol;start_time;end_time];
  select 
    total_volume:sum size,
    total_value:sum value,
    avg_trade_size:avg size,
    trade_count:count i,
    vwap:sum[price*size]%sum size,
    twap:avg price,
    min_price:min price,
    max_price:max price,
    price_std:dev price
  from trades
  };

/ Trade impact analysis
calc_trade_impact:{[symbol;start_time;end_time]
  trades:get_trades_by_symbol_time[symbol;start_time;end_time];
  select 
    avg_impact:avg impact,
    max_impact:max impact,
    impact_volatility:dev impact,
    high_impact_trades:sum impact>10,  / >10 bps impact
    total_impact_cost:sum abs impact * value
  from trades where not null impact
  };

/ =============================================================================
/ Real-time Monitoring Functions
/ =============================================================================

/ Real-time trade monitoring
monitor_trades:{[]
  / Get trades from last 5 minutes
  recent:get_recent_trades[;5] each exec distinct sym from trades_enhanced;
  
  / Calculate key metrics
  stats:calc_volume_stats[;.z.p-00:05;.z.p] each exec distinct sym from trades_enhanced;
  
  / Return monitoring dashboard
  `sym`volume`value`vwap`trades xcols 
    select sym, total_volume, total_value, vwap, trade_count from stats
  };

/ Alert system for unusual activity
check_trade_alerts:{[symbol]
  / Get recent activity
  recent:get_recent_trades[symbol;10];  / Last 10 minutes
  
  / Define alert conditions
  alerts:();
  
  / Volume spike alert
  recent_volume:exec sum size from recent;
  avg_volume:exec avg sum size by 10 xbar time.minute from recent;
  if[recent_volume > 3 * avg avg_volume; 
    alerts,:enlist `volume_spike`time`.z.p`sym`symbol`value`recent_volume];
  
  / Price movement alert
  prices:exec price from recent;
  if[count prices;
    price_change:abs (last prices) - first prices;
    avg_price:avg prices;
    if[price_change > 0.02 * avg_price;  / 2% price move
      alerts,:enlist `price_move`time`.z.p`sym`symbol`change`price_change]];
  
  / Return alerts
  alerts
  };

/ =============================================================================
/ Data Maintenance Functions
/ =============================================================================

/ Cleanup old data (retention management)
cleanup_old_trades:{[retention_days]
  cutoff:.z.d - retention_days;
  old_count:exec count i from trades_enhanced where date < cutoff;
  delete from `trades_enhanced where date < cutoff;
  old_count
  };

/ Compress historical data
compress_historical:{[compress_date]
  / Move older data to compressed storage
  old_data:select from trades_enhanced where date < compress_date;
  
  / Save to compressed file
  (` sv `compressed,compress_date) set old_data;
  
  / Remove from main table
  delete from `trades_enhanced where date < compress_date;
  
  count old_data
  };

/ Rebalance partitions
rebalance_partitions:{[]
  / Analyze partition sizes
  partition_sizes:system "ls -l hdb";
  
  / Identify large partitions
  / Rebalance if needed
  / This would be implementation specific
  };

/ =============================================================================
/ Export and Integration Functions
/ =============================================================================

/ Export to CSV
export_trades_csv:{[filename;start_time;end_time]
  trades:select from trades_enhanced where time within (start_time;end_time);
  (hsym `$filename) 0: csv 0: trades;
  count trades
  };

/ Export to JSON for API consumption
export_trades_json:{[start_time;end_time]
  trades:select from trades_enhanced where time within (start_time;end_time);
  .j.j trades
  };

/ Integration with external systems
send_to_risk_system:{[trades_data]
  / Send trade data to risk management system
  / Implementation would depend on specific risk system API
  };

/ =============================================================================
/ Performance Monitoring
/ =============================================================================

/ Table statistics
get_table_stats:{[]
  stats:`table`rows`columns`memory_mb!(
    `trades_enhanced;
    count trades_enhanced;
    count cols trades_enhanced;
    sum .Q.s each trades_enhanced);
  stats
  };

/ Query performance monitoring
monitor_query_performance:{[query_func;params]
  start:.z.p;
  result:query_func . params;
  end_time:.z.p;
  duration:end_time - start;
  
  `query`duration`rows`start_time`end_time!(
    query_func;`timespan$duration;count result;start;end_time)
  };

/ =============================================================================
/ Initialization and Setup
/ =============================================================================

/ Initialize trades schema
init_trades_schema:{[]
  / Create tables if they don't exist
  if[not `trades in tables[]; trades::trades];
  if[not `trades_enhanced in tables[]; trades_enhanced::trades_enhanced];
  
  / Set up attributes
  trades_attrs[`trades];
  trades_attrs[`trades_enhanced];
  
  / Create indexes
  create_indexes[];
  
  / Set up partitioning (if in production)
  / partition_trades[`trades_enhanced;.z.d];
  
  1b  / Success
  };

/ Schema information
schema_info:{[]
  info:([]
    table:`trades`trades_enhanced;
    columns:(count cols trades;count cols trades_enhanced);
    rows:(count trades;count trades_enhanced);
    partitioned:(0b;0b);  / Update based on actual setup
    indexed:(1b;1b)       / Both have indexes
    );
  info
  };

/ =============================================================================
/ Documentation and Help
/ =============================================================================

/ Display schema documentation
show_schema:{[]
  -1 "=== Trades Table Schema ===";
  -1 "";
  -1 "Tables:";
  -1 "  trades         - Basic trades table";
  -1 "  trades_enhanced - Enhanced table with microstructure and stat arb fields";
  -1 "";
  -1 "Key Functions:";
  -1 "  insert_trade[trade_data]           - Insert single trade";
  -1 "  bulk_insert_trades[data;batch]     - Bulk insert with batching";
  -1 "  get_trades_by_symbol_time[sym;s;e] - Query by symbol and time";
  -1 "  calc_vwap[sym;start;end]           - Calculate VWAP";
  -1 "  monitor_trades[]                   - Real-time monitoring";
  -1 "  check_trade_alerts[sym]            - Alert system";
  -1 "";
  -1 "Maintenance:";
  -1 "  cleanup_old_trades[days]           - Remove old data";
  -1 "  get_table_stats[]                  - Table statistics";
  -1 "";
  };

/ Display table schemas
show_tables:{[]
  -1 "=== Trade Tables Structure ===";
  -1 "";
  -1 "Basic Trades Table:";
  show meta trades;
  -1 "";
  -1 "Enhanced Trades Table:";
  show meta trades_enhanced;
  };

/ =============================================================================
/ Module Initialization
/ =============================================================================

/ Initialize the schema on load
init_trades_schema[];

/ Display startup message
-1 "Trades Schema Module Loaded Successfully";
-1 "Type 'show_schema[]' for documentation";
-1 "Type 'show_tables[]' to see table structures";
-1 "";