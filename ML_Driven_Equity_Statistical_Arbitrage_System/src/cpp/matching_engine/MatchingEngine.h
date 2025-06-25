#pragma once

#include "../order_book/OrderBook.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <chrono>

namespace StatArb {

/**
 * @brief Order processing result enumeration
 */
enum class OrderResult {
    ACCEPTED,           // Order accepted and processed
    REJECTED,           // Order rejected
    PARTIALLY_FILLED,   // Order partially executed
    FULLY_FILLED,       // Order completely executed
    CANCELLED,          // Order cancelled
    MODIFIED,           // Order modified
    EXPIRED             // Order expired
};

/**
 * @brief Order processing statistics
 */
struct ProcessingStats {
    uint64_t ordersProcessed;
    uint64_t ordersAccepted;
    uint64_t ordersRejected;
    uint64_t tradesGenerated;
    uint64_t totalVolume;
    double totalValue;
    double vwap;  // Volume Weighted Average Price
    std::chrono::nanoseconds avgProcessingTime;
    std::chrono::nanoseconds maxProcessingTime;
    std::chrono::nanoseconds minProcessingTime;
    
    ProcessingStats() : ordersProcessed(0), ordersAccepted(0), ordersRejected(0),
                       tradesGenerated(0), totalVolume(0), totalValue(0.0), vwap(0.0),
                       avgProcessingTime(std::chrono::nanoseconds::zero()),
                       maxProcessingTime(std::chrono::nanoseconds::zero()),
                       minProcessingTime(std::chrono::nanoseconds::max()) {}
};

/**
 * @brief Order validation result
 */
struct ValidationResult {
    bool isValid;
    std::string errorMessage;
    
    ValidationResult() : isValid(true) {}
    ValidationResult(bool valid, const std::string& error = "") 
        : isValid(valid), errorMessage(error) {}
};

/**
 * @brief Risk check result
 */
struct RiskCheckResult {
    bool passed;
    std::string riskReason;
    double exposureLimit;
    double currentExposure;
    
    RiskCheckResult() : passed(true), exposureLimit(0.0), currentExposure(0.0) {}
    RiskCheckResult(bool pass, const std::string& reason = "") 
        : passed(pass), riskReason(reason), exposureLimit(0.0), currentExposure(0.0) {}
};

/**
 * @brief Order processing context
 */
struct OrderContext {
    std::shared_ptr<Order> order;
    std::chrono::nanoseconds receiveTime;
    std::chrono::nanoseconds processStartTime;
    std::chrono::nanoseconds processEndTime;
    OrderResult result;
    std::vector<Trade> generatedTrades;
    ValidationResult validation;
    RiskCheckResult riskCheck;
    
    OrderContext(std::shared_ptr<Order> ord) 
        : order(ord), 
          receiveTime(std::chrono::high_resolution_clock::now().time_since_epoch()),
          result(OrderResult::ACCEPTED) {}
};

/**
 * @brief Event callback function types
 */
using OrderProcessedCallback = std::function<void(const OrderContext&)>;
using TradeExecutedCallback = std::function<void(const Trade&, const std::string& symbol)>;
using OrderRejectedCallback = std::function<void(const Order&, const std::string& reason)>;
using RiskBreachCallback = std::function<void(const Order&, const RiskCheckResult&)>;
using EngineStatusCallback = std::function<void(const std::string& status, bool isHealthy)>;

/**
 * @brief Order validation interface
 */
class IOrderValidator {
public:
    virtual ~IOrderValidator() = default;
    virtual ValidationResult validate(const Order& order) const = 0;
};

/**
 * @brief Risk management interface
 */
class IRiskManager {
public:
    virtual ~IRiskManager() = default;
    virtual RiskCheckResult checkRisk(const Order& order, const std::string& symbol) const = 0;
    virtual void updatePosition(const Trade& trade, const std::string& symbol) = 0;
    virtual double getCurrentExposure(const std::string& symbol, const std::string& clientId = "") const = 0;
};

/**
 * @brief Basic order validator implementation
 */
class BasicOrderValidator : public IOrderValidator {
private:
    double minPrice_;
    double maxPrice_;
    uint64_t minQuantity_;
    uint64_t maxQuantity_;
    
public:
    BasicOrderValidator(double minPrice = 0.01, double maxPrice = 100000.0,
                       uint64_t minQty = 1, uint64_t maxQty = 1000000)
        : minPrice_(minPrice), maxPrice_(maxPrice), 
          minQuantity_(minQty), maxQuantity_(maxQty) {}
    
    ValidationResult validate(const Order& order) const override;
};

/**
 * @brief Basic risk manager implementation
 */
class BasicRiskManager : public IRiskManager {
private:
    struct ClientLimits {
        double maxPosition;
        double maxOrderValue;
        uint64_t maxOrderQuantity;
        double dailyVolumeLimit;
        
        ClientLimits() : maxPosition(1000000.0), maxOrderValue(100000.0),
                        maxOrderQuantity(10000), dailyVolumeLimit(10000000.0) {}
    };
    
    std::unordered_map<std::string, ClientLimits> clientLimits_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> positions_; // symbol -> client -> position
    std::unordered_map<std::string, std::unordered_map<std::string, double>> dailyVolume_; // symbol -> client -> volume
    mutable std::shared_mutex riskMutex_;
    
public:
    BasicRiskManager() = default;
    
    RiskCheckResult checkRisk(const Order& order, const std::string& symbol) const override;
    void updatePosition(const Trade& trade, const std::string& symbol) override;
    double getCurrentExposure(const std::string& symbol, const std::string& clientId = "") const override;
    
    // Configuration methods
    void setClientLimits(const std::string& clientId, double maxPos, double maxOrderVal, 
                        uint64_t maxOrderQty, double dailyVolLimit);
    void resetDailyVolume();
};

/**
 * @brief High-performance matching engine
 * 
 * This class provides a complete order processing and matching engine suitable for
 * statistical arbitrage and algorithmic trading. Features include:
 * - Order validation and risk management
 * - Multi-threaded order processing
 * - Event-driven architecture with callbacks
 * - Comprehensive performance monitoring
 * - Configurable matching logic
 * - Support for multiple symbols
 */
class MatchingEngine {
public:
    /**
     * @brief Matching engine configuration
     */
    struct Config {
        size_t workerThreads;           // Number of processing threads
        size_t maxQueueSize;            // Maximum order queue size
        bool enableRiskChecks;          // Enable risk management
        bool enableValidation;          // Enable order validation
        bool enablePerformanceStats;    // Enable performance monitoring
        std::chrono::milliseconds orderTimeout; // Order timeout duration
        bool rejectOnQueueFull;         // Reject orders when queue is full
        size_t maxOrdersPerSecond;      // Rate limiting
        
        Config() : workerThreads(4), maxQueueSize(10000), enableRiskChecks(true),
                  enableValidation(true), enablePerformanceStats(true),
                  orderTimeout(std::chrono::milliseconds(5000)), 
                  rejectOnQueueFull(true), maxOrdersPerSecond(10000) {}
    };

private:
    // Core components
    std::unique_ptr<OrderBookManager> bookManager_;
    std::unique_ptr<IOrderValidator> validator_;
    std::unique_ptr<IRiskManager> riskManager_;
    
    // Configuration and state
    Config config_;
    std::atomic<bool> isRunning_;
    std::atomic<bool> isPaused_;
    
    // Order processing queue
    std::queue<std::shared_ptr<OrderContext>> orderQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    
    // Worker threads
    std::vector<std::thread> workerThreads_;
    
    // Statistics and monitoring
    ProcessingStats stats_;
    mutable std::mutex statsMutex_;
    std::chrono::steady_clock::time_point startTime_;
    
    // Rate limiting
    std::atomic<uint64_t> ordersThisSecond_;
    std::chrono::steady_clock::time_point lastSecond_;
    std::mutex rateLimitMutex_;
    
    // Event callbacks
    OrderProcessedCallback orderProcessedCallback_;
    TradeExecutedCallback tradeExecutedCallback_;
    OrderRejectedCallback orderRejectedCallback_;
    RiskBreachCallback riskBreachCallback_;
    EngineStatusCallback engineStatusCallback_;
    mutable std::shared_mutex callbackMutex_;
    
    // Internal methods
    void workerThreadFunction();
    void processOrderInternal(std::shared_ptr<OrderContext> context);
    bool checkRateLimit();
    void updateStatistics(const OrderContext& context);
    void notifyOrderProcessed(const OrderContext& context);
    void notifyTradeExecuted(const Trade& trade, const std::string& symbol);
    void notifyOrderRejected(const Order& order, const std::string& reason);
    void notifyRiskBreach(const Order& order, const RiskCheckResult& result);
    void notifyEngineStatus(const std::string& status, bool isHealthy);
    
public:
    /**
     * @brief Constructor
     * @param config Engine configuration
     */
    explicit MatchingEngine(const Config& config = Config());
    
    /**
     * @brief Destructor
     */
    ~MatchingEngine();
    
    // Disable copy constructor and assignment
    MatchingEngine(const MatchingEngine&) = delete;
    MatchingEngine& operator=(const MatchingEngine&) = delete;
    
    // Enable move constructor and assignment
    MatchingEngine(MatchingEngine&&) = default;
    MatchingEngine& operator=(MatchingEngine&&) = default;
    
    /**
     * @brief Start the matching engine
     * @return True if started successfully
     */
    bool start();
    
    /**
     * @brief Stop the matching engine
     * @param graceful If true, process remaining orders before stopping
     */
    void stop(bool graceful = true);
    
    /**
     * @brief Pause the matching engine
     */
    void pause();
    
    /**
     * @brief Resume the matching engine
     */
    void resume();
    
    /**
     * @brief Check if the engine is running
     * @return True if running
     */
    bool isRunning() const { return isRunning_.load(); }
    
    /**
     * @brief Check if the engine is paused
     * @return True if paused
     */
    bool isPaused() const { return isPaused_.load(); }
    
    /**
     * @brief Process a new order
     * @param order Order to process
     * @return Order ID if accepted, 0 if rejected
     */
    uint64_t processOrder(std::shared_ptr<Order> order);
    
    /**
     * @brief Process a new order (convenience method)
     * @param symbol Symbol for the order
     * @param side Buy or sell
     * @param type Order type
     * @param price Order price
     * @param quantity Order quantity
     * @param clientId Client identifier
     * @return Order ID if accepted, 0 if rejected
     */
    uint64_t processOrder(const std::string& symbol, OrderSide side, OrderType type,
                         double price, uint64_t quantity, const std::string& clientId = "");
    
    /**
     * @brief Cancel an existing order
     * @param symbol Symbol of the order
     * @param orderId ID of the order to cancel
     * @return True if successfully cancelled
     */
    bool cancelOrder(const std::string& symbol, uint64_t orderId);
    
    /**
     * @brief Modify an existing order
     * @param symbol Symbol of the order
     * @param orderId ID of the order to modify
     * @param newQuantity New quantity for the order
     * @return True if successfully modified
     */
    bool modifyOrder(const std::string& symbol, uint64_t orderId, uint64_t newQuantity);
    
    /**
     * @brief Modify an existing order (price and quantity)
     * @param symbol Symbol of the order
     * @param orderId ID of the order to modify
     * @param newPrice New price for the order
     * @param newQuantity New quantity for the order
     * @return True if successfully modified
     */
    bool modifyOrder(const std::string& symbol, uint64_t orderId, 
                    double newPrice, uint64_t newQuantity);
    
    /**
     * @brief Get order book for a symbol
     * @param symbol Symbol to get book for
     * @return Reference to order book
     */
    OrderBook& getOrderBook(const std::string& symbol);
    
    /**
     * @brief Get processing statistics
     * @return Current processing statistics
     */
    ProcessingStats getStats() const;
    
    /**
     * @brief Reset processing statistics
     */
    void resetStats();
    
    /**
     * @brief Get current queue size
     * @return Number of orders in processing queue
     */
    size_t getQueueSize() const;
    
    /**
     * @brief Get engine configuration
     * @return Current configuration
     */
    const Config& getConfig() const { return config_; }
    
    /**
     * @brief Update engine configuration (only some settings can be changed while running)
     * @param config New configuration
     * @return True if configuration was updated
     */
    bool updateConfig(const Config& config);
    
    /**
     * @brief Set order validator
     * @param validator Custom order validator
     */
    void setOrderValidator(std::unique_ptr<IOrderValidator> validator);
    
    /**
     * @brief Set risk manager
     * @param riskManager Custom risk manager
     */
    void setRiskManager(std::unique_ptr<IRiskManager> riskManager);
    
    /**
     * @brief Set order processed callback
     * @param callback Function to call when order is processed
     */
    void setOrderProcessedCallback(OrderProcessedCallback callback);
    
    /**
     * @brief Set trade executed callback
     * @param callback Function to call when trade is executed
     */
    void setTradeExecutedCallback(TradeExecutedCallback callback);
    
    /**
     * @brief Set order rejected callback
     * @param callback Function to call when order is rejected
     */
    void setOrderRejectedCallback(OrderRejectedCallback callback);
    
    /**
     * @brief Set risk breach callback
     * @param callback Function to call when risk limits are breached
     */
    void setRiskBreachCallback(RiskBreachCallback callback);
    
    /**
     * @brief Set engine status callback
     * @param callback Function to call on engine status changes
     */
    void setEngineStatusCallback(EngineStatusCallback callback);
    
    /**
     * @brief Get symbols with active order books
     * @return Vector of symbol strings
     */
    std::vector<std::string> getActiveSymbols() const;
    
    /**
     * @brief Get total number of orders across all symbols
     * @return Total order count
     */
    size_t getTotalOrderCount() const;
    
    /**
     * @brief Check engine health
     * @return True if engine is healthy
     */
    bool isHealthy() const;
    
    /**
     * @brief Get engine uptime
     * @return Duration since engine start
     */
    std::chrono::duration<double> getUptime() const;
    
    /**
     * @brief Force garbage collection of empty order books
     */
    void cleanupEmptyBooks();
    
    /**
     * @brief Get detailed engine status
     * @return Status information as string
     */
    std::string getDetailedStatus() const;
    
    /**
     * @brief Dump engine state for debugging
     * @param includeOrderBooks Include order book details
     * @return Debug information
     */
    std::string dumpState(bool includeOrderBooks = false) const;
};

/**
 * @brief Matching engine factory for creating preconfigured engines
 */
class MatchingEngineFactory {
public:
    /**
     * @brief Create a high-frequency trading engine
     * @return Configured matching engine for HFT
     */
    static std::unique_ptr<MatchingEngine> createHFTEngine();
    
    /**
     * @brief Create a statistical arbitrage engine
     * @return Configured matching engine for stat arb
     */
    static std::unique_ptr<MatchingEngine> createStatArbEngine();
    
    /**
     * @brief Create a market making engine
     * @return Configured matching engine for market making
     */
    static std::unique_ptr<MatchingEngine> createMarketMakingEngine();
    
    /**
     * @brief Create a basic testing engine
     * @return Simple engine for testing/development
     */
    static std::unique_ptr<MatchingEngine> createTestEngine();
};

} // namespace StatArb
