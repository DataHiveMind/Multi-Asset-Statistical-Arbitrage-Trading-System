/ Real-time Market Data Feed Handler for kdb+
/ Handles multiple data sources: IPC, WebSocket, FIX, TCP/UDP feeds
/ Optimized for low-latency ingestion and storage

/ Load required schemas and analytics
\l ../schema/trades.q
\l ../schema/quotes.q
\l ../queries/analytics.q

/ Global configuration
feedConfig:`host`port`protocol`buffer_size`batch_size`timeout_ms`reconnect_delay!(
    `localhost;5010;`ipc;10000;100;1000;5000);

/ Feed connection status tracking
feedStatus:`connections`last_heartbeat`message_count`error_count!(
    ()!();.z.p;0;0);

/ Message parsing configuration
msgTypes:`trade`quote`orderbook`heartbeat`error!();
msgHandlers:`trade`quote`orderbook`heartbeat`error!(
    parseTrade;parseQuote;parseOrderBook;parseHeartbeat;parseError);

/ Buffer management for batching
tradeBuffer:();
quoteBuffer:();
bufferLock:0b;

/ Performance monitoring
perfStats:`messages_per_sec`avg_latency_us`max_latency_us`buffer_utilization!(
    0f;0f;0f;0f);

/ Initialize feed handler
initFeedHandler:{[]
    -1"[",string[.z.t],"] Initializing feed handler...";
    
    / Set up signal handlers
    .z.exit:{cleanup[];exit 0};
    .z.pc:{[h] handleDisconnect h};
    .z.po:{[h] handleConnect h};
    
    / Initialize performance monitoring
    .z.ts:{updatePerformanceStats[]};
    system"t 1000"; / Update stats every second
    
    / Set up error handling
    .z.pg:{[x] -1"[",string[.z.t],"] Error: ",x; feedStatus[`error_count]+:1};
    
    -1"[",string[.z.t],"] Feed handler initialized successfully";
 };

/ Connect to data feed
connectFeed:{[config]
    host:config`host;
    port:config`port;
    protocol:config`protocol;
    
    -1"[",string[.z.t],"] Connecting to ",string[host],":",string[port]," via ",string[protocol];
    
    handle:$[protocol=`ipc;
        @[hopen;(host;port);{-1"Connection failed: ",x; 0Ni}];
        protocol=`websocket;
        @[connectWebSocket;(host;port);{-1"WebSocket connection failed: ",x; 0Ni}];
        protocol=`tcp;
        @[connectTCP;(host;port);{-1"TCP connection failed: ",x; 0Ni}];
        -1"Unsupported protocol: ",string[protocol]; 0Ni];
    
    if[not null handle;
        feedStatus[`connections;host,port]:handle;
        -1"[",string[.z.t],"] Connected successfully to ",string[host],":",string[port];
        / Start heartbeat monitoring
        startHeartbeat[handle];
    ];
    
    handle
 };

/ WebSocket connection handler
connectWebSocket:{[host;port]
    / Implementation for WebSocket connection
    / This would typically use .ws.open or similar WebSocket library
    -1"[",string[.z.t],"] WebSocket connection not implemented - placeholder";
    0Ni
 };

/ TCP connection handler
connectTCP:{[host;port]
    / Implementation for raw TCP connection
    handle:hopen `$":tcp://",string[host],":",string[port];
    handle
 };

/ Handle new connections
handleConnect:{[h]
    -1"[",string[.z.t],"] New connection established: ",string[h];
    feedStatus[`connections;h]:h;
 };

/ Handle disconnections
handleDisconnect:{[h]
    -1"[",string[.z.t],"] Connection lost: ",string[h];
    feedStatus[`connections]:feedStatus[`connections] _ h;
    / Attempt reconnection
    reconnectFeed[h];
 };

/ Reconnection logic
reconnectFeed:{[h]
    -1"[",string[.z.t],"] Attempting to reconnect...";
    system"sleep ",string[feedConfig`reconnect_delay];
    / Reconnect using stored configuration
    connectFeed[feedConfig];
 };

/ Heartbeat management
startHeartbeat:{[h]
    / Send periodic heartbeat to maintain connection
    .z.ts:{sendHeartbeat[h]};
 };

sendHeartbeat:{[h]
    if[h in key feedStatus`connections;
        @[h;"heartbeat";{-1"Heartbeat failed: ",x}];
        feedStatus[`last_heartbeat]:.z.p;
    ];
 };

/ Main message router
routeMessage:{[msg]
    msgType:msg`type;
    if[msgType in key msgHandlers;
        handler:msgHandlers msgType;
        @[handler;msg;{-1"Message handling error: ",x}];
        feedStatus[`message_count]+:1;
    ];
 };

/ Trade message parser
parseTrade:{[msg]
    startTime:.z.p;
    
    / Extract trade data from message
    trade:parseTradeData msg;
    
    / Validate trade data
    if[validateTradeData trade;
        / Add to buffer or insert directly
        if[feedConfig[`batch_size]>1;
            addToTradeBuffer trade;
        ;
            insertTrade trade;
        ];
    ];
    
    / Update latency statistics
    latency:`long$startTime-.z.p;
    updateLatencyStats[latency];
 };

/ Quote message parser
parseQuote:{[msg]
    startTime:.z.p;
    
    / Extract quote data from message
    quote:parseQuoteData msg;
    
    / Validate quote data
    if[validateQuoteData quote;
        / Add to buffer or insert directly
        if[feedConfig[`batch_size]>1;
            addToQuoteBuffer quote;
        ;
            insertQuote quote;
        ];
    ];
    
    / Update latency statistics
    latency:`long$startTime-.z.p;
    updateLatencyStats[latency];
 };

/ Order book parser (Level II data)
parseOrderBook:{[msg]
    startTime:.z.p;
    
    / Parse order book snapshot or delta
    orderBook:parseOrderBookData msg;
    
    / Process each level
    {[level]
        if[level`side=`bid;
            quote:(`time`sym`bid`bidsize`ask`asksize`exchange`condition)!
                  (level`time;level`sym;level`price;level`size;0n;0;level`exchange;`);
        ];
        if[level`side=`ask;
            quote:(`time`sym`bid`bidsize`ask`asksize`exchange`condition)!
                  (level`time;level`sym;0n;0;level`price;level`size;level`exchange;`);
        ];
        
        if[validateQuoteData quote;
            addToQuoteBuffer quote;
        ];
    } each orderBook;
    
    / Trigger buffer flush if needed
    if[count[quoteBuffer]>=feedConfig`batch_size;
        flushQuoteBuffer[];
    ];
    
    latency:`long$startTime-.z.p;
    updateLatencyStats[latency];
 };

/ Heartbeat handler
parseHeartbeat:{[msg]
    feedStatus[`last_heartbeat]:.z.p;
    / Optional: respond to heartbeat
    if[`respond in key msg;
        if[msg`respond;
            / Send heartbeat response
            h:msg`handle;
            @[h;"heartbeat_ack";{-1"Heartbeat response failed: ",x}];
        ];
    ];
 };

/ Error handler
parseError:{[msg]
    -1"[",string[.z.t],"] Feed error: ",msg`error;
    feedStatus[`error_count]+:1;
 };

/ Data parsing functions
parseTradeData:{[msg]
    / Parse trade message into standardized format
    trade:()!();
    
    / Common FIX-style or JSON parsing
    if[`fix in key msg;
        trade:parseFixTrade msg`fix;
    ];
    if[`json in key msg;
        trade:parseJsonTrade msg`json;
    ];
    if[`binary in key msg;
        trade:parseBinaryTrade msg`binary;
    ];
    
    / Ensure required fields
    if[not `time in key trade; trade[`time]:.z.p];
    if[not `sym in key trade; trade[`sym]:`UNKNOWN];
    
    trade
 };

parseQuoteData:{[msg]
    / Parse quote message into standardized format
    quote:()!();
    
    / Common parsing based on message format
    if[`fix in key msg;
        quote:parseFixQuote msg`fix;
    ];
    if[`json in key msg;
        quote:parseJsonQuote msg`json;
    ];
    if[`binary in key msg;
        quote:parseBinaryQuote msg`binary;
    ];
    
    / Ensure required fields
    if[not `time in key quote; quote[`time]:.z.p];
    if[not `sym in key quote; quote[`sym]:`UNKNOWN];
    
    quote
 };

parseOrderBookData:{[msg]
    / Parse order book data (Level II)
    orderBook:();
    
    / Parse based on message format
    if[`snapshot in key msg;
        orderBook:parseOrderBookSnapshot msg`snapshot;
    ];
    if[`delta in key msg;
        orderBook:parseOrderBookDelta msg`delta;
    ];
    
    orderBook
 };

/ Format-specific parsers
parseFixTrade:{[fixMsg]
    / Parse FIX protocol trade message
    fields:parseFixFields fixMsg;
    trade:(`time`sym`price`size`side`exchange`condition)!(
        parseFixTime fields 52; / TransactTime
        `$fields 55; / Symbol
        "F"$fields 31; / LastPx
        "J"$fields 32; / LastShares
        parseFixSide fields 54; / Side
        `$fields 30; / LastMkt
        `$fields 29 / LastCapacity
    );
    trade
 };

parseJsonTrade:{[jsonMsg]
    / Parse JSON trade message
    data:.j.k jsonMsg;
    trade:(`time`sym`price`size`side`exchange`condition)!(
        "P"$data`timestamp;
        `$data`symbol;
        data`price;
        data`quantity;
        `$data`side;
        `$data`exchange;
        `$data`condition
    );
    trade
 };

parseBinaryTrade:{[binaryMsg]
    / Parse binary trade message (custom format)
    / This would depend on the specific binary protocol
    trade:()!();
    trade
 };

parseFixQuote:{[fixMsg]
    / Parse FIX protocol quote message
    fields:parseFixFields fixMsg;
    quote:(`time`sym`bid`bidsize`ask`asksize`exchange`condition)!(
        parseFixTime fields 52;
        `$fields 55;
        "F"$fields 132; / BidPx
        "J"$fields 134; / BidSize
        "F"$fields 133; / OfferPx
        "J"$fields 135; / OfferSize
        `$fields 30;
        `
    );
    quote
 };

parseJsonQuote:{[jsonMsg]
    / Parse JSON quote message
    data:.j.k jsonMsg;
    quote:(`time`sym`bid`bidsize`ask`asksize`exchange`condition)!(
        "P"$data`timestamp;
        `$data`symbol;
        data`bid_price;
        data`bid_size;
        data`ask_price;
        data`ask_size;
        `$data`exchange;
        `$data`condition
    );
    quote
 };

parseBinaryQuote:{[binaryMsg]
    / Parse binary quote message
    quote:()!();
    quote
 };

/ Helper functions for FIX parsing
parseFixFields:{[fixMsg]
    / Parse FIX message into field dictionary
    fields:()!();
    tokens:"|" vs fixMsg;
    {[token]
        if["=" in token;
            parts:"=" vs token;
            fields[parts 0]:parts 1;
        ];
    } each tokens;
    fields
 };

parseFixTime:{[timeStr]
    / Parse FIX timestamp format
    "P"$timeStr
 };

parseFixSide:{[sideStr]
    / Parse FIX side field
    $[sideStr="1"; `buy; sideStr="2"; `sell; `unknown]
 };

/ Data validation
validateTradeData:{[trade]
    / Validate trade data completeness and sanity
    checks:(
        `time in key trade;
        `sym in key trade;
        `price in key trade;
        `size in key trade;
        trade[`price]>0;
        trade[`size]>0
    );
    all checks
 };

validateQuoteData:{[quote]
    / Validate quote data completeness and sanity
    checks:(
        `time in key quote;
        `sym in key quote;
        (`bid in key quote) or (`ask in key quote);
        (null quote`bid) or (quote[`bid]>0);
        (null quote`ask) or (quote[`ask]>0);
        (null quote`bid) or (null quote`ask) or (quote[`ask]>=quote`bid)
    );
    all checks
 };

/ Buffer management
addToTradeBuffer:{[trade]
    if[not bufferLock;
        bufferLock:1b;
        tradeBuffer,:enlist trade;
        bufferLock:0b;
        
        / Check if buffer is full
        if[count[tradeBuffer]>=feedConfig`batch_size;
            flushTradeBuffer[];
        ];
    ];
 };

addToQuoteBuffer:{[quote]
    if[not bufferLock;
        bufferLock:1b;
        quoteBuffer,:enlist quote;
        bufferLock:0b;
        
        / Check if buffer is full
        if[count[quoteBuffer]>=feedConfig`batch_size;
            flushQuoteBuffer[];
        ];
    ];
 };

flushTradeBuffer:{[]
    if[(count tradeBuffer)>0;
        bufferLock:1b;
        trades:tradeBuffer;
        tradeBuffer:();
        bufferLock:0b;
        
        / Bulk insert trades
        insertTrades trades;
        -1"[",string[.z.t],"] Flushed ",string[count trades]," trades";
    ];
 };

flushQuoteBuffer:{[]
    if[(count quoteBuffer)>0;
        bufferLock:1b;
        quotes:quoteBuffer;
        quoteBuffer:();
        bufferLock:0b;
        
        / Bulk insert quotes
        insertQuotes quotes;
        -1"[",string[.z.t],"] Flushed ",string[count quotes]," quotes";
    ];
 };

/ Database insertion functions
insertTrade:{[trade]
    / Insert single trade
    `.trades insert (trade`time;trade`sym;trade`price;trade`size;trade`side;trade`exchange;trade`condition);
 };

insertTrades:{[trades]
    / Bulk insert trades
    data:flip (`time`sym`price`size`side`exchange`condition)!(
        trades[;`time];trades[;`sym];trades[;`price];trades[;`size];
        trades[;`side];trades[;`exchange];trades[;`condition]
    );
    `.trades insert data;
 };

insertQuote:{[quote]
    / Insert single quote
    `.quotes insert (quote`time;quote`sym;quote`bid;quote`bidsize;quote`ask;quote`asksize;quote`exchange;quote`condition);
 };

insertQuotes:{[quotes]
    / Bulk insert quotes
    data:flip (`time`sym`bid`bidsize`ask`asksize`exchange`condition)!(
        quotes[;`time];quotes[;`sym];quotes[;`bid];quotes[;`bidsize];
        quotes[;`ask];quotes[;`asksize];quotes[;`exchange];quotes[;`condition]
    );
    `.quotes insert data;
 };

/ Performance monitoring
updateLatencyStats:{[latency]
    perfStats[`max_latency_us]:max perfStats[`max_latency_us],latency;
    / Simple moving average for avg_latency
    perfStats[`avg_latency_us]:0.9*perfStats[`avg_latency_us]+0.1*latency;
 };

updatePerformanceStats:{[]
    currentTime:.z.p;
    static lastTime:.z.p;
    static lastCount:0;
    
    timeDiff:`long$currentTime-lastTime;
    countDiff:feedStatus[`message_count]-lastCount;
    
    if[timeDiff>0;
        perfStats[`messages_per_sec]:1000000000f*countDiff%timeDiff;
        perfStats[`buffer_utilization]:100f*(count[tradeBuffer]+count[quoteBuffer])%2*feedConfig`batch_size;
    ];
    
    lastTime:currentTime;
    lastCount:feedStatus`message_count;
    
    / Log performance stats every 10 seconds
    if[0=`long$.z.t mod 10000;
        -1"[",string[.z.t],"] Performance: ",
          string[perfStats`messages_per_sec]," msg/s, ",
          string[perfStats`avg_latency_us]," μs avg latency, ",
          string[perfStats`buffer_utilization],"% buffer utilization";
    ];
 };

/ Main message processing loop
processMessages:{[]
    -1"[",string[.z.t],"] Starting message processing loop...";
    
    / Main event loop
    while[1b;
        / Process incoming messages from all connections
        {[h]
            / Non-blocking read from connection
            msg:@[h;();{`error!`msg!(`connection_error;x)}];
            if[not `error~first key msg;
                routeMessage msg;
            ];
        } each value feedStatus`connections;
        
        / Small sleep to prevent busy waiting
        system"sleep 1";
    ];
 };

/ Shutdown and cleanup
cleanup:{[]
    -1"[",string[.z.t],"] Shutting down feed handler...";
    
    / Flush remaining buffers
    flushTradeBuffer[];
    flushQuoteBuffer[];
    
    / Close connections
    {[h] hclose h} each value feedStatus`connections;
    
    / Save final statistics
    -1"[",string[.z.t],"] Final stats: ",
      string[feedStatus`message_count]," messages processed, ",
      string[feedStatus`error_count]," errors";
 };

/ Status reporting
getStatus:{[]
    status:`feed_status`performance`buffer_status!(
        feedStatus;
        perfStats;
        (`trade_buffer_size`quote_buffer_size)!(count tradeBuffer;count quoteBuffer)
    );
    status
 };

/ Configuration management
updateConfig:{[newConfig]
    feedConfig:feedConfig,newConfig;
    -1"[",string[.z.t],"] Configuration updated: ",string[newConfig];
 };

/ Example usage and testing functions
testFeedHandler:{[]
    -1"[",string[.z.t],"] Testing feed handler...";
    
    / Simulate some test messages
    testTrade:(`type`time`sym`price`size`side`exchange`condition)!(
        `trade;.z.p;`AAPL;150.25;100;`buy;`NYSE;`
    );
    
    testQuote:(`type`time`sym`bid`bidsize`ask`asksize`exchange`condition)!(
        `quote;.z.p;`AAPL;150.20;500;150.25;300;`NYSE;`
    );
    
    routeMessage testTrade;
    routeMessage testQuote;
    
    -1"[",string[.z.t],"] Test completed";
 };

/ Start the feed handler
startFeedHandler:{[]
    initFeedHandler[];
    connectFeed[feedConfig];
    processMessages[];
 };

/ Initialize on load
-1"[",string[.z.t],"] Data feed handler loaded. Use startFeedHandler[] to begin processing.";

/ Example startup sequence:
/ startFeedHandler[];