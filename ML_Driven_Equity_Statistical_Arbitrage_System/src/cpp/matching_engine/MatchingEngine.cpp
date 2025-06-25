#include "MatchingEngine.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>

namespace StatArb {

// ============================================================================
// BasicOrderValidator Implementation
// ============================================================================

ValidationResult BasicOrderValidator::validate(const Order& order) const {
    // Check quantity bounds
    if (order.quantity < minQuantity_) {
        return ValidationResult(false, "Order quantity below minimum: " + std::to_string(order.quantity));
    }
    
    if (order.quantity > maxQuantity_) {
        return ValidationResult(false, "Order quantity exceeds maximum: " + std::to_string(order.quantity));
    }
    
    // Check price bounds (for limit orders)
    if (order.type == OrderType::LIMIT || order.type == OrderType::STOP_LIMIT) {
        if (order.price < minPrice_) {
            return ValidationResult(false, "Order price below minimum: " + std::to_string(order.price));
        }
        
        if (order.price > maxPrice_) {
            return ValidationResult(false, "Order price exceeds maximum: " + std::to_string(order.price));
        }
    }
    
    // Check symbol
    if (order.symbol.empty()) {
        return ValidationResult(false, "Empty symbol");
    }
    
    // Check for reasonable price (no negative prices)
    if (order.price < 0.0) {
        return ValidationResult(false, "Negative price not allowed");
    }
    
    return ValidationResult(true);
}

// ============================================================================
// BasicRiskManager Implementation
// ============================================================================

RiskCheckResult BasicRiskManager::checkRisk(const Order& order, const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(riskMutex_);
    
    // Get client limits
    auto limitsIt = clientLimits_.find(order.clientId);
    ClientLimits limits;
    if (limitsIt != clientLimits_.end()) {
        limits = limitsIt->second;
    }
    
    // Check order value limit
    double orderValue = order.price * order.quantity;
    if (orderValue > limits.maxOrderValue) {
        return RiskCheckResult(false, "Order value exceeds limit: " + std::to_string(orderValue) + 
                              " > " + std::to_string(limits.maxOrderValue));
    }
    
    // Check order quantity limit
    if (order.quantity > limits.maxOrderQuantity) {
        return RiskCheckResult(false, "Order quantity exceeds limit: " + std::to_string(order.quantity) + 
                              " > " + std::to_string(limits.maxOrderQuantity));
    }
    
    // Check position limits
    auto symbolPosIt = positions_.find(symbol);
    double currentPosition = 0.0;
    if (symbolPosIt != positions_.end()) {
        auto clientPosIt = symbolPosIt->second.find(order.clientId);
        if (clientPosIt != symbolPosIt->second.end()) {
            currentPosition = clientPosIt->second;
        }
    }
    
    // Calculate potential new position
    double positionChange = (order.side == OrderSide::BUY) ? 
                           static_cast<double>(order.quantity) : 
                           -static_cast<double>(order.quantity);
    double newPosition = currentPosition + positionChange;
    
    if (std::abs(newPosition) > limits.maxPosition) {
        RiskCheckResult result(false, "Position limit would be exceeded");
        result.currentExposure = currentPosition;
        result.exposureLimit = limits.maxPosition;
        return result;
    }
    
    // Check daily volume limits
    auto symbolVolIt = dailyVolume_.find(symbol);
    double currentDailyVolume = 0.0;
    if (symbolVolIt != dailyVolume_.end()) {
        auto clientVolIt = symbolVolIt->second.find(order.clientId);
        if (clientVolIt != symbolVolIt->second.end()) {
            currentDailyVolume = clientVolIt->second;
        }
    }
    
    if (currentDailyVolume + orderValue > limits.dailyVolumeLimit) {
        return RiskCheckResult(false, "Daily volume limit would be exceeded: " + 
                              std::to_string(currentDailyVolume + orderValue) + 
                              " > " + std::to_string(limits.dailyVolumeLimit));
    }
    
    return RiskCheckResult(true);
}

void BasicRiskManager::updatePosition(const Trade& trade, const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(riskMutex_);
    
    // Update positions for both buy and sell orders
    // Note: We need to get the client IDs from the original orders
    // For now, we'll update based on the trade direction
    
    double tradeValue = trade.price * trade.quantity;
    
    // Update daily volume (simplified - in reality we'd track by client)
    dailyVolume_[symbol][""] += tradeValue;  // Global volume tracking
}

double BasicRiskManager::getCurrentExposure(const std::string& symbol, const std::string& clientId) const {
    std::shared_lock<std::shared_mutex> lock(riskMutex_);
    
    auto symbolIt = positions_.find(symbol);
    if (symbolIt == positions_.end()) {
        return 0.0;
    }
    
    if (clientId.empty()) {
        // Return total exposure for symbol
        double totalExposure = 0.0;
        for (const auto& [client, position] : symbolIt->second) {
            totalExposure += std::abs(position);
        }
        return totalExposure;
    }
    
    auto clientIt = symbolIt->second.find(clientId);
    return (clientIt != symbolIt->second.end()) ? std::abs(clientIt->second) : 0.0;
}

void BasicRiskManager::setClientLimits(const std::string& clientId, double maxPos, 
                                      double maxOrderVal, uint64_t maxOrderQty, double dailyVolLimit) {
    std::unique_lock<std::shared_mutex> lock(riskMutex_);
    
    ClientLimits& limits = clientLimits_[clientId];
    limits.maxPosition = maxPos;
    limits.maxOrderValue = maxOrderVal;
    limits.maxOrderQuantity = maxOrderQty;
    limits.dailyVolumeLimit = dailyVolLimit;
}

void BasicRiskManager::resetDailyVolume() {
    std::unique_lock<std::shared_mutex> lock(riskMutex_);
    dailyVolume_.clear();
}

// ============================================================================
// MatchingEngine Implementation
// ============================================================================

MatchingEngine::MatchingEngine(const Config& config)
    : config_(config), isRunning_(false), isPaused_(false), 
      ordersThisSecond_(0), lastSecond_(std::chrono::steady_clock::now()) {
    
    // Initialize components
    bookManager_ = std::make_unique<OrderBookManager>();
    validator_ = std::make_unique<BasicOrderValidator>();
    riskManager_ = std::make_unique<BasicRiskManager>();
    
    startTime_ = std::chrono::steady_clock::now();
}

MatchingEngine::~MatchingEngine() {
    if (isRunning_.load()) {
        stop(false);  // Force stop
    }
}

bool MatchingEngine::start() {
    if (isRunning_.load()) {
        return false;  // Already running
    }
    
    isRunning_.store(true);
    isPaused_.store(false);
    
    // Start worker threads
    workerThreads_.reserve(config_.workerThreads);
    for (size_t i = 0; i < config_.workerThreads; ++i) {
        workerThreads_.emplace_back(&MatchingEngine::workerThreadFunction, this);
    }
    
    notifyEngineStatus("Engine started with " + std::to_string(config_.workerThreads) + " worker threads", true);
    return true;
}

void MatchingEngine::stop(bool graceful) {
    if (!isRunning_.load()) {
        return;  // Already stopped
    }
    
    isRunning_.store(false);
    
    if (graceful) {
        // Wait for queue to empty
        std::unique_lock<std::mutex> lock(queueMutex_);
        while (!orderQueue_.empty()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            lock.lock();
        }
    }
    
    // Notify all worker threads to wake up
    queueCondition_.notify_all();
    
    // Wait for all worker threads to finish
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    workerThreads_.clear();
    
    notifyEngineStatus("Engine stopped", false);
}

void MatchingEngine::pause() {
    isPaused_.store(true);
    notifyEngineStatus("Engine paused", true);
}

void MatchingEngine::resume() {
    isPaused_.store(false);
    queueCondition_.notify_all();
    notifyEngineStatus("Engine resumed", true);
}

uint64_t MatchingEngine::processOrder(std::shared_ptr<Order> order) {
    if (!order) {
        return 0;
    }
    
    if (!isRunning_.load()) {
        notifyOrderRejected(*order, "Engine not running");
        return 0;
    }
    
    // Check rate limiting
    if (!checkRateLimit()) {
        notifyOrderRejected(*order, "Rate limit exceeded");
        return 0;
    }
    
    // Check queue capacity
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (orderQueue_.size() >= config_.maxQueueSize) {
            if (config_.rejectOnQueueFull) {
                notifyOrderRejected(*order, "Order queue full");
                return 0;
            } else {
                // Drop oldest order
                orderQueue_.pop();
            }
        }
        
        // Add to queue
        auto context = std::make_shared<OrderContext>(order);
        orderQueue_.push(context);
    }
    
    // Notify worker threads
    queueCondition_.notify_one();
    
    return order->orderId;
}

uint64_t MatchingEngine::processOrder(const std::string& symbol, OrderSide side, OrderType type,
                                    double price, uint64_t quantity, const std::string& clientId) {
    // Generate unique order ID
    static std::atomic<uint64_t> orderIdCounter{1};
    uint64_t orderId = orderIdCounter.fetch_add(1);
    
    auto order = std::make_shared<Order>(orderId, symbol, side, type, price, quantity, clientId);
    return processOrder(order);
}

bool MatchingEngine::cancelOrder(const std::string& symbol, uint64_t orderId) {
    try {
        auto& book = bookManager_->getBook(symbol);
        return book.cancelOrder(orderId);
    } catch (const std::exception& e) {
        return false;
    }
}

bool MatchingEngine::modifyOrder(const std::string& symbol, uint64_t orderId, uint64_t newQuantity) {
    try {
        auto& book = bookManager_->getBook(symbol);
        return book.modifyOrder(orderId, newQuantity);
    } catch (const std::exception& e) {
        return false;
    }
}

bool MatchingEngine::modifyOrder(const std::string& symbol, uint64_t orderId, 
                                double newPrice, uint64_t newQuantity) {
    try {
        auto& book = bookManager_->getBook(symbol);
        return book.modifyOrder(orderId, newPrice, newQuantity);
    } catch (const std::exception& e) {
        return false;
    }
}

OrderBook& MatchingEngine::getOrderBook(const std::string& symbol) {
    return bookManager_->getBook(symbol);
}

ProcessingStats MatchingEngine::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

void MatchingEngine::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = ProcessingStats();
}

size_t MatchingEngine::getQueueSize() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex_));
    return orderQueue_.size();
}

bool MatchingEngine::updateConfig(const Config& config) {
    // Only allow certain config changes while running
    if (isRunning_.load()) {
        // Can only change these settings while running
        config_.enableRiskChecks = config.enableRiskChecks;
        config_.enableValidation = config.enableValidation;
        config_.enablePerformanceStats = config.enablePerformanceStats;
        config_.maxOrdersPerSecond = config.maxOrdersPerSecond;
        config_.orderTimeout = config.orderTimeout;
        
        return true;
    } else {
        // Can change all settings when stopped
        config_ = config;
        return true;
    }
}

void MatchingEngine::setOrderValidator(std::unique_ptr<IOrderValidator> validator) {
    validator_ = std::move(validator);
}

void MatchingEngine::setRiskManager(std::unique_ptr<IRiskManager> riskManager) {
    riskManager_ = std::move(riskManager);
}

void MatchingEngine::setOrderProcessedCallback(OrderProcessedCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbackMutex_);
    orderProcessedCallback_ = std::move(callback);
}

void MatchingEngine::setTradeExecutedCallback(TradeExecutedCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbackMutex_);
    tradeExecutedCallback_ = std::move(callback);
}

void MatchingEngine::setOrderRejectedCallback(OrderRejectedCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbackMutex_);
    orderRejectedCallback_ = std::move(callback);
}

void MatchingEngine::setRiskBreachCallback(RiskBreachCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbackMutex_);
    riskBreachCallback_ = std::move(callback);
}

void MatchingEngine::setEngineStatusCallback(EngineStatusCallback callback) {
    std::unique_lock<std::shared_mutex> lock(callbackMutex_);
    engineStatusCallback_ = std::move(callback);
}

std::vector<std::string> MatchingEngine::getActiveSymbols() const {
    return bookManager_->getSymbols();
}

size_t MatchingEngine::getTotalOrderCount() const {
    return bookManager_->getTotalOrderCount();
}

bool MatchingEngine::isHealthy() const {
    // Check if engine is running and queue is not backing up
    if (!isRunning_.load()) {
        return false;
    }
    
    size_t queueSize = getQueueSize();
    if (queueSize > config_.maxQueueSize * 0.8) {  // 80% threshold
        return false;
    }
    
    return true;
}

std::chrono::duration<double> MatchingEngine::getUptime() const {
    return std::chrono::steady_clock::now() - startTime_;
}

void MatchingEngine::cleanupEmptyBooks() {
    // Get all symbols and check if their books are empty
    auto symbols = bookManager_->getSymbols();
    for (const auto& symbol : symbols) {
        try {
            auto& book = bookManager_->getBook(symbol);
            if (book.isEmpty()) {
                bookManager_->removeBook(symbol);
            }
        } catch (const std::exception&) {
            // Ignore errors during cleanup
        }
    }
}

std::string MatchingEngine::getDetailedStatus() const {
    std::ostringstream oss;
    auto stats = getStats();
    auto uptime = getUptime();
    
    oss << "=== Matching Engine Status ===\n";
    oss << "Running: " << (isRunning_.load() ? "Yes" : "No") << "\n";
    oss << "Paused: " << (isPaused_.load() ? "Yes" : "No") << "\n";
    oss << "Healthy: " << (isHealthy() ? "Yes" : "No") << "\n";
    oss << "Uptime: " << std::fixed << std::setprecision(2) << uptime.count() << " seconds\n";
    oss << "Worker Threads: " << config_.workerThreads << "\n";
    oss << "Queue Size: " << getQueueSize() << "/" << config_.maxQueueSize << "\n";
    oss << "Active Symbols: " << getActiveSymbols().size() << "\n";
    oss << "Total Orders: " << getTotalOrderCount() << "\n\n";
    
    oss << "=== Processing Statistics ===\n";
    oss << "Orders Processed: " << stats.ordersProcessed << "\n";
    oss << "Orders Accepted: " << stats.ordersAccepted << "\n";
    oss << "Orders Rejected: " << stats.ordersRejected << "\n";
    oss << "Trades Generated: " << stats.tradesGenerated << "\n";
    oss << "Total Volume: " << stats.totalVolume << "\n";
    oss << "Total Value: $" << std::fixed << std::setprecision(2) << stats.totalValue << "\n";
    
    if (stats.ordersProcessed > 0) {
        oss << "Average Processing Time: " << stats.avgProcessingTime.count() << " ns\n";
        oss << "Min Processing Time: " << stats.minProcessingTime.count() << " ns\n";
        oss << "Max Processing Time: " << stats.maxProcessingTime.count() << " ns\n";
    }
    
    return oss.str();
}

std::string MatchingEngine::dumpState(bool includeOrderBooks) const {
    std::ostringstream oss;
    
    oss << getDetailedStatus();
    
    if (includeOrderBooks) {
        oss << "\n=== Order Books ===\n";
        auto symbols = getActiveSymbols();
        for (const auto& symbol : symbols) {
            try {
                auto& book = bookManager_->getBook(symbol);
                oss << "\n" << symbol << ":\n";
                oss << "  Orders: " << book.getOrderCount() << "\n";
                oss << "  Best Bid: " << book.getBestBid() << " (" << book.getBestBidQuantity() << ")\n";
                oss << "  Best Ask: " << book.getBestAsk() << " (" << book.getBestAskQuantity() << ")\n";
                oss << "  Spread: " << book.getSpread() << "\n";
                oss << "  Mid Price: " << book.getMidPrice() << "\n";
            } catch (const std::exception& e) {
                oss << "  Error: " << e.what() << "\n";
            }
        }
    }
    
    return oss.str();
}

// ============================================================================
// Private Methods
// ============================================================================

void MatchingEngine::workerThreadFunction() {
    while (isRunning_.load()) {
        std::shared_ptr<OrderContext> context;
        
        // Wait for work
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCondition_.wait(lock, [this] {
                return !orderQueue_.empty() || !isRunning_.load();
            });
            
            if (!isRunning_.load()) {
                break;
            }
            
            if (orderQueue_.empty()) {
                continue;
            }
            
            context = orderQueue_.front();
            orderQueue_.pop();
        }
        
        // Check if paused
        while (isPaused_.load() && isRunning_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        if (!isRunning_.load()) {
            break;
        }
        
        // Process the order
        processOrderInternal(context);
    }
}

void MatchingEngine::processOrderInternal(std::shared_ptr<OrderContext> context) {
    if (!context || !context->order) {
        return;
    }
    
    context->processStartTime = std::chrono::high_resolution_clock::now().time_since_epoch();
    
    auto& order = context->order;
    
    try {
        // Step 1: Validate order
        if (config_.enableValidation && validator_) {
            context->validation = validator_->validate(*order);
            if (!context->validation.isValid) {
                context->result = OrderResult::REJECTED;
                notifyOrderRejected(*order, context->validation.errorMessage);
                updateStatistics(*context);
                notifyOrderProcessed(*context);
                return;
            }
        }
        
        // Step 2: Risk checks
        if (config_.enableRiskChecks && riskManager_) {
            context->riskCheck = riskManager_->checkRisk(*order, order->symbol);
            if (!context->riskCheck.passed) {
                context->result = OrderResult::REJECTED;
                notifyRiskBreach(*order, context->riskCheck);
                notifyOrderRejected(*order, context->riskCheck.riskReason);
                updateStatistics(*context);
                notifyOrderProcessed(*context);
                return;
            }
        }
        
        // Step 3: Process order through matching engine
        auto& book = bookManager_->getBook(order->symbol);
        
        // Add order to book and get resulting trades
        context->generatedTrades = book.addOrder(order);
        
        // Step 4: Update risk manager with executed trades
        if (config_.enableRiskChecks && riskManager_) {
            for (const auto& trade : context->generatedTrades) {
                riskManager_->updatePosition(trade, order->symbol);
            }
        }
        
        // Step 5: Determine final order result
        if (order->isFullyFilled()) {
            context->result = OrderResult::FULLY_FILLED;
        } else if (order->filledQuantity > 0) {
            context->result = OrderResult::PARTIALLY_FILLED;
        } else {
            context->result = OrderResult::ACCEPTED;  // Resting in book
        }
        
        // Step 6: Notify about trades
        for (const auto& trade : context->generatedTrades) {
            notifyTradeExecuted(trade, order->symbol);
        }
        
    } catch (const std::exception& e) {
        context->result = OrderResult::REJECTED;
        notifyOrderRejected(*order, "Internal error: " + std::string(e.what()));
    }
    
    context->processEndTime = std::chrono::high_resolution_clock::now().time_since_epoch();
    
    // Update statistics and notify
    updateStatistics(*context);
    notifyOrderProcessed(*context);
}

bool MatchingEngine::checkRateLimit() {
    std::lock_guard<std::mutex> lock(rateLimitMutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastSecond_);
    
    if (duration.count() >= 1) {
        // Reset counter for new second
        ordersThisSecond_.store(0);
        lastSecond_ = now;
    }
    
    uint64_t currentCount = ordersThisSecond_.fetch_add(1);
    return currentCount < config_.maxOrdersPerSecond;
}

void MatchingEngine::updateStatistics(const OrderContext& context) {
    if (!config_.enablePerformanceStats) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(statsMutex_);
    
    stats_.ordersProcessed++;
    
    if (context.result == OrderResult::REJECTED) {
        stats_.ordersRejected++;
    } else {
        stats_.ordersAccepted++;
    }
    
    stats_.tradesGenerated += context.generatedTrades.size();
    
    for (const auto& trade : context.generatedTrades) {
        stats_.totalVolume += trade.quantity;
        stats_.totalValue += trade.price * trade.quantity;
    }
    
    // Update VWAP
    if (stats_.totalVolume > 0) {
        stats_.vwap = stats_.totalValue / stats_.totalVolume;
    }
    
    // Update processing time statistics
    auto processingTime = context.processEndTime - context.processStartTime;
    
    if (stats_.ordersProcessed == 1) {
        stats_.minProcessingTime = processingTime;
        stats_.maxProcessingTime = processingTime;
        stats_.avgProcessingTime = processingTime;
    } else {
        stats_.minProcessingTime = std::min(stats_.minProcessingTime, processingTime);
        stats_.maxProcessingTime = std::max(stats_.maxProcessingTime, processingTime);
        
        // Update running average
        auto totalTime = stats_.avgProcessingTime * (stats_.ordersProcessed - 1) + processingTime;
        stats_.avgProcessingTime = totalTime / stats_.ordersProcessed;
    }
}

void MatchingEngine::notifyOrderProcessed(const OrderContext& context) {
    std::shared_lock<std::shared_mutex> lock(callbackMutex_);
    if (orderProcessedCallback_) {
        orderProcessedCallback_(context);
    }
}

void MatchingEngine::notifyTradeExecuted(const Trade& trade, const std::string& symbol) {
    std::shared_lock<std::shared_mutex> lock(callbackMutex_);
    if (tradeExecutedCallback_) {
        tradeExecutedCallback_(trade, symbol);
    }
}

void MatchingEngine::notifyOrderRejected(const Order& order, const std::string& reason) {
    std::shared_lock<std::shared_mutex> lock(callbackMutex_);
    if (orderRejectedCallback_) {
        orderRejectedCallback_(order, reason);
    }
}

void MatchingEngine::notifyRiskBreach(const Order& order, const RiskCheckResult& result) {
    std::shared_lock<std::shared_mutex> lock(callbackMutex_);
    if (riskBreachCallback_) {
        riskBreachCallback_(order, result);
    }
}

void MatchingEngine::notifyEngineStatus(const std::string& status, bool isHealthy) {
    std::shared_lock<std::shared_mutex> lock(callbackMutex_);
    if (engineStatusCallback_) {
        engineStatusCallback_(status, isHealthy);
    }
}

// ============================================================================
// MatchingEngineFactory Implementation
// ============================================================================

std::unique_ptr<MatchingEngine> MatchingEngineFactory::createHFTEngine() {
    MatchingEngine::Config config;
    config.workerThreads = 8;
    config.maxQueueSize = 50000;
    config.enableRiskChecks = true;
    config.enableValidation = true;
    config.enablePerformanceStats = true;
    config.orderTimeout = std::chrono::milliseconds(100);
    config.rejectOnQueueFull = true;
    config.maxOrdersPerSecond = 100000;
    
    return std::make_unique<MatchingEngine>(config);
}

std::unique_ptr<MatchingEngine> MatchingEngineFactory::createStatArbEngine() {
    MatchingEngine::Config config;
    config.workerThreads = 4;
    config.maxQueueSize = 20000;
    config.enableRiskChecks = true;
    config.enableValidation = true;
    config.enablePerformanceStats = true;
    config.orderTimeout = std::chrono::milliseconds(1000);
    config.rejectOnQueueFull = false;
    config.maxOrdersPerSecond = 50000;
    
    return std::make_unique<MatchingEngine>(config);
}

std::unique_ptr<MatchingEngine> MatchingEngineFactory::createMarketMakingEngine() {
    MatchingEngine::Config config;
    config.workerThreads = 6;
    config.maxQueueSize = 30000;
    config.enableRiskChecks = true;
    config.enableValidation = true;
    config.enablePerformanceStats = true;
    config.orderTimeout = std::chrono::milliseconds(500);
    config.rejectOnQueueFull = true;
    config.maxOrdersPerSecond = 75000;
    
    return std::make_unique<MatchingEngine>(config);
}

std::unique_ptr<MatchingEngine> MatchingEngineFactory::createTestEngine() {
    MatchingEngine::Config config;
    config.workerThreads = 2;
    config.maxQueueSize = 1000;
    config.enableRiskChecks = false;
    config.enableValidation = false;
    config.enablePerformanceStats = true;
    config.orderTimeout = std::chrono::milliseconds(10000);
    config.rejectOnQueueFull = false;
    config.maxOrdersPerSecond = 1000;
    
    return std::make_unique<MatchingEngine>(config);
}

} // namespace StatArb
