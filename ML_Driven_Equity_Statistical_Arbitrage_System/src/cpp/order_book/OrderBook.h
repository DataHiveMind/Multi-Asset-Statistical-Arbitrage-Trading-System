#pragma once

#include <map>
#include <unordered_map>
#include <list>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <functional>

namespace StatArb {

/**
 * @brief Order side enumeration
 */
enum class OrderSide {
    BUY,
    SELL
};

/**
 * @brief Order type enumeration
 */
enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT
};

/**
 * @brief Order status enumeration
 */
enum class OrderStatus {
    PENDING,
    PARTIAL_FILL,
    FILLED,
    CANCELLED,
    REJECTED
};

/**
 * @brief Order structure representing a single order
 */
struct Order {
    uint64_t orderId;
    std::string symbol;
    OrderSide side;
    OrderType type;
    double price;
    uint64_t quantity;
    uint64_t filledQuantity;
    OrderStatus status;
    std::chrono::nanoseconds timestamp;
    std::string clientId;
    
    // Constructor
    Order(uint64_t id, const std::string& sym, OrderSide s, OrderType t, 
          double p, uint64_t q, const std::string& client = "")
        : orderId(id), symbol(sym), side(s), type(t), price(p), quantity(q),
          filledQuantity(0), status(OrderStatus::PENDING), 
          timestamp(std::chrono::high_resolution_clock::now().time_since_epoch()),
          clientId(client) {}
    
    // Remaining quantity
    uint64_t getRemainingQuantity() const {
        return quantity - filledQuantity;
    }
    
    // Check if order is fully filled
    bool isFullyFilled() const {
        return filledQuantity >= quantity;
    }
    
    // Fill order (partial or full)
    uint64_t fill(uint64_t fillQuantity) {
        uint64_t actualFill = std::min(fillQuantity, getRemainingQuantity());
        filledQuantity += actualFill;
        
        if (isFullyFilled()) {
            status = OrderStatus::FILLED;
        } else {
            status = OrderStatus::PARTIAL_FILL;
        }
        
        return actualFill;
    }
};

/**
 * @brief Trade execution result
 */
struct Trade {
    uint64_t tradeId;
    uint64_t buyOrderId;
    uint64_t sellOrderId;
    std::string symbol;
    double price;
    uint64_t quantity;
    std::chrono::nanoseconds timestamp;
    
    Trade(uint64_t id, uint64_t buyId, uint64_t sellId, const std::string& sym,
          double p, uint64_t q)
        : tradeId(id), buyOrderId(buyId), sellOrderId(sellId), symbol(sym),
          price(p), quantity(q),
          timestamp(std::chrono::high_resolution_clock::now().time_since_epoch()) {}
};

/**
 * @brief Book level representing orders at a specific price
 */
class BookLevel {
public:
    using OrderList = std::list<std::shared_ptr<Order>>;
    using OrderIterator = OrderList::iterator;
    
private:
    double price_;
    uint64_t totalQuantity_;
    OrderList orders_;
    std::unordered_map<uint64_t, OrderIterator> orderMap_;
    
public:
    explicit BookLevel(double price) : price_(price), totalQuantity_(0) {}
    
    // Add order to this level
    void addOrder(std::shared_ptr<Order> order);
    
    // Remove order from this level
    bool removeOrder(uint64_t orderId);
    
    // Modify order quantity
    bool modifyOrder(uint64_t orderId, uint64_t newQuantity);
    
    // Get total quantity at this level
    uint64_t getTotalQuantity() const { return totalQuantity_; }
    
    // Get price of this level
    double getPrice() const { return price_; }
    
    // Get number of orders at this level
    size_t getOrderCount() const { return orders_.size(); }
    
    // Check if level is empty
    bool isEmpty() const { return orders_.empty(); }
    
    // Get orders (read-only access)
    const OrderList& getOrders() const { return orders_; }
    
    // Get first order in queue (FIFO)
    std::shared_ptr<Order> getFirstOrder() const {
        return orders_.empty() ? nullptr : orders_.front();
    }
    
    // Execute against this level (for matching)
    std::vector<Trade> executeAgainst(std::shared_ptr<Order> incomingOrder, 
                                     uint64_t& tradeIdCounter);
};

/**
 * @brief Market depth information
 */
struct MarketDepth {
    std::vector<std::pair<double, uint64_t>> bids;  // price, quantity
    std::vector<std::pair<double, uint64_t>> asks;  // price, quantity
    double bidPrice;
    double askPrice;
    uint64_t bidQuantity;
    uint64_t askQuantity;
    double spread;
    double midPrice;
    
    MarketDepth() : bidPrice(0.0), askPrice(0.0), bidQuantity(0), 
                   askQuantity(0), spread(0.0), midPrice(0.0) {}
};

/**
 * @brief Order book statistics
 */
struct OrderBookStats {
    uint64_t totalOrders;
    uint64_t totalTrades;
    uint64_t totalVolume;
    double totalValue;
    double vwap;  // Volume Weighted Average Price
    uint64_t bidLevels;
    uint64_t askLevels;
    std::chrono::nanoseconds lastUpdateTime;
    
    OrderBookStats() : totalOrders(0), totalTrades(0), totalVolume(0),
                      totalValue(0.0), vwap(0.0), bidLevels(0), askLevels(0),
                      lastUpdateTime(std::chrono::nanoseconds::zero()) {}
};

/**
 * @brief Callback function types for order book events
 */
using OrderCallback = std::function<void(const Order&)>;
using TradeCallback = std::function<void(const Trade&)>;
using DepthCallback = std::function<void(const MarketDepth&)>;

/**
 * @brief High-performance order book implementation
 * 
 * This class provides a thread-safe, high-performance order book suitable for
 * statistical arbitrage and algorithmic trading applications. It supports:
 * - Fast order insertion, modification, and cancellation
 * - Efficient price-time priority matching
 * - Market depth queries
 * - Real-time statistics
 * - Event callbacks for order and trade updates
 */
class OrderBook {
public:
    using BidMap = std::map<double, std::unique_ptr<BookLevel>, std::greater<double>>;
    using AskMap = std::map<double, std::unique_ptr<BookLevel>>;
    
private:
    // Core data structures
    std::string symbol_;
    BidMap bidLevels_;     // Bids sorted by price (highest first)
    AskMap askLevels_;     // Asks sorted by price (lowest first)
    
    // Order tracking
    std::unordered_map<uint64_t, std::shared_ptr<Order>> orders_;
    
    // Thread safety
    mutable std::shared_mutex bookMutex_;
    
    // Statistics and state
    std::atomic<uint64_t> orderIdCounter_;
    std::atomic<uint64_t> tradeIdCounter_;
    OrderBookStats stats_;
    
    // Event callbacks
    OrderCallback orderCallback_;
    TradeCallback tradeCallback_;
    DepthCallback depthCallback_;
    
    // Performance optimization
    mutable std::atomic<double> cachedBestBid_;
    mutable std::atomic<double> cachedBestAsk_;
    mutable std::atomic<bool> cacheValid_;
    
    // Internal helper methods
    void updateCache();
    void invalidateCache();
    std::vector<Trade> matchOrder(std::shared_ptr<Order> order);
    void updateStats(const Trade& trade);
    void notifyOrderUpdate(const Order& order);
    void notifyTradeUpdate(const Trade& trade);
    void notifyDepthUpdate();
    
public:
    /**
     * @brief Constructor
     * @param symbol The symbol for this order book
     */
    explicit OrderBook(const std::string& symbol);
    
    /**
     * @brief Destructor
     */
    ~OrderBook() = default;
    
    // Disable copy constructor and assignment operator
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;
    
    // Enable move constructor and assignment operator
    OrderBook(OrderBook&&) = default;
    OrderBook& operator=(OrderBook&&) = default;
    
    /**
     * @brief Add a new order to the book
     * @param side Order side (BUY or SELL)
     * @param type Order type
     * @param price Order price (ignored for market orders)
     * @param quantity Order quantity
     * @param clientId Optional client identifier
     * @return Order ID of the added order
     */
    uint64_t addOrder(OrderSide side, OrderType type, double price, 
                     uint64_t quantity, const std::string& clientId = "");
    
    /**
     * @brief Add a pre-constructed order to the book
     * @param order Shared pointer to the order
     * @return Vector of trades generated by this order
     */
    std::vector<Trade> addOrder(std::shared_ptr<Order> order);
    
    /**
     * @brief Cancel an existing order
     * @param orderId ID of the order to cancel
     * @return True if order was successfully cancelled
     */
    bool cancelOrder(uint64_t orderId);
    
    /**
     * @brief Modify an existing order's quantity
     * @param orderId ID of the order to modify
     * @param newQuantity New quantity for the order
     * @return True if order was successfully modified
     */
    bool modifyOrder(uint64_t orderId, uint64_t newQuantity);
    
    /**
     * @brief Modify an existing order's price and quantity
     * @param orderId ID of the order to modify
     * @param newPrice New price for the order
     * @param newQuantity New quantity for the order
     * @return True if order was successfully modified
     */
    bool modifyOrder(uint64_t orderId, double newPrice, uint64_t newQuantity);
    
    /**
     * @brief Get the best bid price
     * @return Best bid price, or 0.0 if no bids exist
     */
    double getBestBid() const;
    
    /**
     * @brief Get the best ask price
     * @return Best ask price, or 0.0 if no asks exist
     */
    double getBestAsk() const;
    
    /**
     * @brief Get the best bid quantity
     * @return Best bid quantity, or 0 if no bids exist
     */
    uint64_t getBestBidQuantity() const;
    
    /**
     * @brief Get the best ask quantity
     * @return Best ask quantity, or 0 if no asks exist
     */
    uint64_t getBestAskQuantity() const;
    
    /**
     * @brief Get the current spread
     * @return Spread (ask - bid), or 0.0 if no complete market
     */
    double getSpread() const;
    
    /**
     * @brief Get the mid price
     * @return Mid price ((bid + ask) / 2), or 0.0 if no complete market
     */
    double getMidPrice() const;
    
    /**
     * @brief Get market depth information
     * @param levels Number of levels to include (0 = all levels)
     * @return MarketDepth structure with bid/ask information
     */
    MarketDepth getDepth(size_t levels = 10) const;
    
    /**
     * @brief Get order book statistics
     * @return OrderBookStats structure with current statistics
     */
    OrderBookStats getStats() const;
    
    /**
     * @brief Get an order by ID
     * @param orderId Order ID to lookup
     * @return Shared pointer to order, or nullptr if not found
     */
    std::shared_ptr<Order> getOrder(uint64_t orderId) const;
    
    /**
     * @brief Get all orders for a specific client
     * @param clientId Client identifier
     * @return Vector of orders for the client
     */
    std::vector<std::shared_ptr<Order>> getOrdersByClient(const std::string& clientId) const;
    
    /**
     * @brief Check if the order book is empty
     * @return True if no orders exist in the book
     */
    bool isEmpty() const;
    
    /**
     * @brief Get the number of orders in the book
     * @return Total number of orders
     */
    size_t getOrderCount() const;
    
    /**
     * @brief Get the number of bid levels
     * @return Number of price levels on the bid side
     */
    size_t getBidLevelCount() const;
    
    /**
     * @brief Get the number of ask levels
     * @return Number of price levels on the ask side
     */
    size_t getAskLevelCount() const;
    
    /**
     * @brief Clear all orders from the book
     */
    void clear();
    
    /**
     * @brief Get the symbol for this order book
     * @return Symbol string
     */
    const std::string& getSymbol() const { return symbol_; }
    
    /**
     * @brief Set callback for order events
     * @param callback Function to call on order updates
     */
    void setOrderCallback(OrderCallback callback) {
        std::unique_lock<std::shared_mutex> lock(bookMutex_);
        orderCallback_ = std::move(callback);
    }
    
    /**
     * @brief Set callback for trade events
     * @param callback Function to call on trade executions
     */
    void setTradeCallback(TradeCallback callback) {
        std::unique_lock<std::shared_mutex> lock(bookMutex_);
        tradeCallback_ = std::move(callback);
    }
    
    /**
     * @brief Set callback for depth updates
     * @param callback Function to call on depth changes
     */
    void setDepthCallback(DepthCallback callback) {
        std::unique_lock<std::shared_mutex> lock(bookMutex_);
        depthCallback_ = std::move(callback);
    }
    
    /**
     * @brief Get volume at a specific price level
     * @param price Price level to query
     * @param side Side to query (BUY for bids, SELL for asks)
     * @return Total quantity at the price level
     */
    uint64_t getVolumeAtPrice(double price, OrderSide side) const;
    
    /**
     * @brief Get volume above/below a specific price
     * @param price Reference price
     * @param side Side to query
     * @param above If true, get volume above price; if false, get volume below
     * @return Total quantity above/below the specified price
     */
    uint64_t getVolumeAroundPrice(double price, OrderSide side, bool above) const;
    
    /**
     * @brief Calculate volume-weighted average price (VWAP)
     * @param side Side to calculate VWAP for
     * @param levels Number of levels to include (0 = all levels)
     * @return VWAP for the specified side and levels
     */
    double calculateVWAP(OrderSide side, size_t levels = 0) const;
    
    /**
     * @brief Simulate market impact of a hypothetical order
     * @param side Order side
     * @param quantity Order quantity
     * @return Estimated average execution price
     */
    double simulateMarketImpact(OrderSide side, uint64_t quantity) const;
    
    /**
     * @brief Get time of last update
     * @return Timestamp of last book modification
     */
    std::chrono::nanoseconds getLastUpdateTime() const {
        std::shared_lock<std::shared_mutex> lock(bookMutex_);
        return stats_.lastUpdateTime;
    }
    
    /**
     * @brief Debug function to print book state
     * @param levels Number of levels to print (0 = all)
     */
    void printBook(size_t levels = 5) const;
    
    /**
     * @brief Validate book integrity (for debugging)
     * @return True if book state is consistent
     */
    bool validateBook() const;
};

/**
 * @brief Order book manager for multiple symbols
 */
class OrderBookManager {
private:
    std::unordered_map<std::string, std::unique_ptr<OrderBook>> books_;
    mutable std::shared_mutex managerMutex_;
    
public:
    /**
     * @brief Get or create order book for a symbol
     * @param symbol Symbol to get book for
     * @return Reference to the order book
     */
    OrderBook& getBook(const std::string& symbol);
    
    /**
     * @brief Check if a book exists for a symbol
     * @param symbol Symbol to check
     * @return True if book exists
     */
    bool hasBook(const std::string& symbol) const;
    
    /**
     * @brief Remove a book for a symbol
     * @param symbol Symbol to remove
     * @return True if book was removed
     */
    bool removeBook(const std::string& symbol);
    
    /**
     * @brief Get all symbols with active books
     * @return Vector of symbol strings
     */
    std::vector<std::string> getSymbols() const;
    
    /**
     * @brief Get total number of orders across all books
     * @return Total order count
     */
    size_t getTotalOrderCount() const;
    
    /**
     * @brief Clear all books
     */
    void clear();
};

} // namespace StatArb