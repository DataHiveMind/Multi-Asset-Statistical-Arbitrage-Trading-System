#include "OrderBook.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace StatArb {

// ============================================================================
// BookLevel Implementation
// ============================================================================

void BookLevel::addOrder(std::shared_ptr<Order> order) {
    if (!order || order->getRemainingQuantity() == 0) {
        return;
    }
    
    // Add to the end of the queue (FIFO)
    orders_.push_back(order);
    auto it = std::prev(orders_.end());
    
    // Update mapping for fast lookup
    orderMap_[order->orderId] = it;
    
    // Update total quantity
    totalQuantity_ += order->getRemainingQuantity();
}

bool BookLevel::removeOrder(uint64_t orderId) {
    auto mapIt = orderMap_.find(orderId);
    if (mapIt == orderMap_.end()) {
        return false;
    }
    
    auto orderIt = mapIt->second;
    auto order = *orderIt;
    
    // Update total quantity
    totalQuantity_ -= order->getRemainingQuantity();
    
    // Remove from list and map
    orders_.erase(orderIt);
    orderMap_.erase(mapIt);
    
    return true;
}

bool BookLevel::modifyOrder(uint64_t orderId, uint64_t newQuantity) {
    auto mapIt = orderMap_.find(orderId);
    if (mapIt == orderMap_.end()) {
        return false;
    }
    
    auto order = *(mapIt->second);
    uint64_t oldRemainingQty = order->getRemainingQuantity();
    
    if (newQuantity <= order->filledQuantity) {
        // New quantity is less than or equal to filled quantity - remove order
        return removeOrder(orderId);
    }
    
    // Update order quantity
    order->quantity = newQuantity;
    
    // Update total quantity for this level
    uint64_t newRemainingQty = order->getRemainingQuantity();
    totalQuantity_ = totalQuantity_ - oldRemainingQty + newRemainingQty;
    
    return true;
}

std::vector<Trade> BookLevel::executeAgainst(std::shared_ptr<Order> incomingOrder, 
                                           uint64_t& tradeIdCounter) {
    std::vector<Trade> trades;
    
    if (!incomingOrder || incomingOrder->getRemainingQuantity() == 0) {
        return trades;
    }
    
    auto it = orders_.begin();
    while (it != orders_.end() && incomingOrder->getRemainingQuantity() > 0) {
        auto restingOrder = *it;
        
        // Calculate trade quantity
        uint64_t tradeQty = std::min(incomingOrder->getRemainingQuantity(), 
                                   restingOrder->getRemainingQuantity());
        
        if (tradeQty == 0) {
            ++it;
            continue;
        }
        
        // Create trade
        uint64_t buyOrderId, sellOrderId;
        if (incomingOrder->side == OrderSide::BUY) {
            buyOrderId = incomingOrder->orderId;
            sellOrderId = restingOrder->orderId;
        } else {
            buyOrderId = restingOrder->orderId;
            sellOrderId = incomingOrder->orderId;
        }
        
        trades.emplace_back(++tradeIdCounter, buyOrderId, sellOrderId, 
                           incomingOrder->symbol, price_, tradeQty);
        
        // Fill both orders
        incomingOrder->fill(tradeQty);
        restingOrder->fill(tradeQty);
        
        // Update total quantity
        totalQuantity_ -= tradeQty;
        
        // Remove fully filled orders
        if (restingOrder->isFullyFilled()) {
            orderMap_.erase(restingOrder->orderId);
            it = orders_.erase(it);
        } else {
            ++it;
        }
    }
    
    return trades;
}

// ============================================================================
// OrderBook Implementation
// ============================================================================

OrderBook::OrderBook(const std::string& symbol) 
    : symbol_(symbol), orderIdCounter_(0), tradeIdCounter_(0),
      cachedBestBid_(0.0), cachedBestAsk_(0.0), cacheValid_(false) {
    stats_.lastUpdateTime = std::chrono::high_resolution_clock::now().time_since_epoch();
}

void OrderBook::updateCache() {
    // Update best bid
    if (!bidLevels_.empty()) {
        cachedBestBid_.store(bidLevels_.begin()->first);
    } else {
        cachedBestBid_.store(0.0);
    }
    
    // Update best ask
    if (!askLevels_.empty()) {
        cachedBestAsk_.store(askLevels_.begin()->first);
    } else {
        cachedBestAsk_.store(0.0);
    }
    
    cacheValid_.store(true);
}

void OrderBook::invalidateCache() {
    cacheValid_.store(false);
}

std::vector<Trade> OrderBook::matchOrder(std::shared_ptr<Order> order) {
    std::vector<Trade> allTrades;
    
    if (!order || order->getRemainingQuantity() == 0) {
        return allTrades;
    }
    
    if (order->type == OrderType::MARKET || order->type == OrderType::LIMIT) {
        if (order->side == OrderSide::BUY) {
            // Match against asks (lowest price first)
            auto it = askLevels_.begin();
            while (it != askLevels_.end() && order->getRemainingQuantity() > 0) {
                double askPrice = it->first;
                
                // For limit orders, check if price is acceptable
                if (order->type == OrderType::LIMIT && askPrice > order->price) {
                    break;
                }
                
                auto& level = it->second;
                uint64_t tradeCounter = tradeIdCounter_.load();
                auto trades = level->executeAgainst(order, tradeCounter);
                tradeIdCounter_.store(tradeCounter);
                
                // Add trades to result
                allTrades.insert(allTrades.end(), trades.begin(), trades.end());
                
                // Remove empty levels
                if (level->isEmpty()) {
                    it = askLevels_.erase(it);
                } else {
                    ++it;
                }
            }
        } else { // SELL
            // Match against bids (highest price first)
            auto it = bidLevels_.begin();
            while (it != bidLevels_.end() && order->getRemainingQuantity() > 0) {
                double bidPrice = it->first;
                
                // For limit orders, check if price is acceptable
                if (order->type == OrderType::LIMIT && bidPrice < order->price) {
                    break;
                }
                
                auto& level = it->second;
                uint64_t tradeCounter = tradeIdCounter_.load();
                auto trades = level->executeAgainst(order, tradeCounter);
                tradeIdCounter_.store(tradeCounter);
                
                // Add trades to result
                allTrades.insert(allTrades.end(), trades.begin(), trades.end());
                
                // Remove empty levels
                if (level->isEmpty()) {
                    it = bidLevels_.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    
    return allTrades;
}

void OrderBook::updateStats(const Trade& trade) {
    stats_.totalTrades++;
    stats_.totalVolume += trade.quantity;
    stats_.totalValue += trade.price * trade.quantity;
    
    // Update VWAP
    if (stats_.totalVolume > 0) {
        stats_.vwap = stats_.totalValue / stats_.totalVolume;
    }
    
    stats_.lastUpdateTime = std::chrono::high_resolution_clock::now().time_since_epoch();
}

void OrderBook::notifyOrderUpdate(const Order& order) {
    if (orderCallback_) {
        orderCallback_(order);
    }
}

void OrderBook::notifyTradeUpdate(const Trade& trade) {
    if (tradeCallback_) {
        tradeCallback_(trade);
    }
    updateStats(trade);
}

void OrderBook::notifyDepthUpdate() {
    if (depthCallback_) {
        auto depth = getDepth();
        depthCallback_(depth);
    }
}

uint64_t OrderBook::addOrder(OrderSide side, OrderType type, double price, 
                            uint64_t quantity, const std::string& clientId) {
    if (quantity == 0) {
        return 0; // Invalid order
    }
    
    uint64_t orderId = ++orderIdCounter_;
    auto order = std::make_shared<Order>(orderId, symbol_, side, type, price, quantity, clientId);
    
    auto trades = addOrder(order);
    
    return orderId;
}

std::vector<Trade> OrderBook::addOrder(std::shared_ptr<Order> order) {
    if (!order || order->quantity == 0) {
        return {};
    }
    
    std::unique_lock<std::shared_mutex> lock(bookMutex_);
    
    std::vector<Trade> trades;
    
    // Add to orders map
    orders_[order->orderId] = order;
    stats_.totalOrders++;
    
    // Try to match the order
    trades = matchOrder(order);
    
    // If order has remaining quantity and is not a market order, add to book
    if (order->getRemainingQuantity() > 0 && order->type != OrderType::MARKET) {
        if (order->side == OrderSide::BUY) {
            // Add to bid side
            auto& level = bidLevels_[order->price];
            if (!level) {
                level = std::make_unique<BookLevel>(order->price);
            }
            level->addOrder(order);
            stats_.bidLevels = bidLevels_.size();
        } else {
            // Add to ask side
            auto& level = askLevels_[order->price];
            if (!level) {
                level = std::make_unique<BookLevel>(order->price);
            }
            level->addOrder(order);
            stats_.askLevels = askLevels_.size();
        }
    } else if (order->getRemainingQuantity() == 0) {
        // Order fully filled, mark as such
        order->status = OrderStatus::FILLED;
    } else if (order->type == OrderType::MARKET && order->getRemainingQuantity() > 0) {
        // Market order with remaining quantity gets cancelled
        order->status = OrderStatus::CANCELLED;
    }
    
    // Update cache and notify
    invalidateCache();
    notifyOrderUpdate(*order);
    
    // Notify about trades
    for (const auto& trade : trades) {
        notifyTradeUpdate(trade);
    }
    
    if (!trades.empty()) {
        notifyDepthUpdate();
    }
    
    return trades;
}

bool OrderBook::cancelOrder(uint64_t orderId) {
    std::unique_lock<std::shared_mutex> lock(bookMutex_);
    
    auto orderIt = orders_.find(orderId);
    if (orderIt == orders_.end()) {
        return false;
    }
    
    auto order = orderIt->second;
    
    // Can't cancel already filled or cancelled orders
    if (order->status == OrderStatus::FILLED || order->status == OrderStatus::CANCELLED) {
        return false;
    }
    
    bool removed = false;
    
    // Remove from appropriate side
    if (order->side == OrderSide::BUY) {
        auto levelIt = bidLevels_.find(order->price);
        if (levelIt != bidLevels_.end()) {
            removed = levelIt->second->removeOrder(orderId);
            if (levelIt->second->isEmpty()) {
                bidLevels_.erase(levelIt);
                stats_.bidLevels = bidLevels_.size();
            }
        }
    } else {
        auto levelIt = askLevels_.find(order->price);
        if (levelIt != askLevels_.end()) {
            removed = levelIt->second->removeOrder(orderId);
            if (levelIt->second->isEmpty()) {
                askLevels_.erase(levelIt);
                stats_.askLevels = askLevels_.size();
            }
        }
    }
    
    if (removed) {
        order->status = OrderStatus::CANCELLED;
        invalidateCache();
        notifyOrderUpdate(*order);
        notifyDepthUpdate();
    }
    
    return removed;
}

bool OrderBook::modifyOrder(uint64_t orderId, uint64_t newQuantity) {
    if (newQuantity == 0) {
        return cancelOrder(orderId);
    }
    
    std::unique_lock<std::shared_mutex> lock(bookMutex_);
    
    auto orderIt = orders_.find(orderId);
    if (orderIt == orders_.end()) {
        return false;
    }
    
    auto order = orderIt->second;
    
    // Can't modify filled or cancelled orders
    if (order->status == OrderStatus::FILLED || order->status == OrderStatus::CANCELLED) {
        return false;
    }
    
    // If new quantity is less than filled quantity, cancel the order
    if (newQuantity <= order->filledQuantity) {
        return cancelOrder(orderId);
    }
    
    bool modified = false;
    
    // Modify in appropriate level
    if (order->side == OrderSide::BUY) {
        auto levelIt = bidLevels_.find(order->price);
        if (levelIt != bidLevels_.end()) {
            modified = levelIt->second->modifyOrder(orderId, newQuantity);
        }
    } else {
        auto levelIt = askLevels_.find(order->price);
        if (levelIt != askLevels_.end()) {
            modified = levelIt->second->modifyOrder(orderId, newQuantity);
        }
    }
    
    if (modified) {
        notifyOrderUpdate(*order);
        notifyDepthUpdate();
    }
    
    return modified;
}

bool OrderBook::modifyOrder(uint64_t orderId, double newPrice, uint64_t newQuantity) {
    // For price changes, we need to cancel and re-add
    std::unique_lock<std::shared_mutex> lock(bookMutex_);
    
    auto orderIt = orders_.find(orderId);
    if (orderIt == orders_.end()) {
        return false;
    }
    
    auto order = orderIt->second;
    
    // Can't modify filled or cancelled orders
    if (order->status == OrderStatus::FILLED || order->status == OrderStatus::CANCELLED) {
        return false;
    }
    
    // If price hasn't changed, just modify quantity
    if (std::abs(newPrice - order->price) < 1e-8) {
        lock.unlock();
        return modifyOrder(orderId, newQuantity);
    }
    
    // Store order details
    OrderSide side = order->side;
    OrderType type = order->type;
    std::string clientId = order->clientId;
    uint64_t filledQty = order->filledQuantity;
    
    // Cancel existing order
    lock.unlock();
    if (!cancelOrder(orderId)) {
        return false;
    }
    
    // Create new order with remaining quantity
    uint64_t remainingQty = (newQuantity > filledQty) ? newQuantity - filledQty : 0;
    if (remainingQty > 0) {
        auto newOrder = std::make_shared<Order>(orderId, symbol_, side, type, newPrice, 
                                               newQuantity, clientId);
        newOrder->filledQuantity = filledQty;
        
        lock.lock();
        addOrder(newOrder);
    }
    
    return true;
}

double OrderBook::getBestBid() const {
    if (cacheValid_.load()) {
        return cachedBestBid_.load();
    }
    
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    if (bidLevels_.empty()) {
        return 0.0;
    }
    
    return bidLevels_.begin()->first;
}

double OrderBook::getBestAsk() const {
    if (cacheValid_.load()) {
        return cachedBestAsk_.load();
    }
    
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    if (askLevels_.empty()) {
        return 0.0;
    }
    
    return askLevels_.begin()->first;
}

uint64_t OrderBook::getBestBidQuantity() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    if (bidLevels_.empty()) {
        return 0;
    }
    
    return bidLevels_.begin()->second->getTotalQuantity();
}

uint64_t OrderBook::getBestAskQuantity() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    if (askLevels_.empty()) {
        return 0;
    }
    
    return askLevels_.begin()->second->getTotalQuantity();
}

double OrderBook::getSpread() const {
    double bid = getBestBid();
    double ask = getBestAsk();
    
    if (bid > 0.0 && ask > 0.0) {
        return ask - bid;
    }
    
    return 0.0;
}

double OrderBook::getMidPrice() const {
    double bid = getBestBid();
    double ask = getBestAsk();
    
    if (bid > 0.0 && ask > 0.0) {
        return (bid + ask) / 2.0;
    }
    
    return 0.0;
}

MarketDepth OrderBook::getDepth(size_t levels) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    MarketDepth depth;
    
    // Get best bid/ask
    depth.bidPrice = getBestBid();
    depth.askPrice = getBestAsk();
    depth.bidQuantity = getBestBidQuantity();
    depth.askQuantity = getBestAskQuantity();
    depth.spread = getSpread();
    depth.midPrice = getMidPrice();
    
    // Collect bid levels
    size_t count = 0;
    for (const auto& [price, level] : bidLevels_) {
        if (levels > 0 && count >= levels) break;
        depth.bids.emplace_back(price, level->getTotalQuantity());
        ++count;
    }
    
    // Collect ask levels
    count = 0;
    for (const auto& [price, level] : askLevels_) {
        if (levels > 0 && count >= levels) break;
        depth.asks.emplace_back(price, level->getTotalQuantity());
        ++count;
    }
    
    return depth;
}

OrderBookStats OrderBook::getStats() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    OrderBookStats currentStats = stats_;
    currentStats.bidLevels = bidLevels_.size();
    currentStats.askLevels = askLevels_.size();
    
    return currentStats;
}

std::shared_ptr<Order> OrderBook::getOrder(uint64_t orderId) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    auto it = orders_.find(orderId);
    if (it != orders_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::vector<std::shared_ptr<Order>> OrderBook::getOrdersByClient(const std::string& clientId) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    std::vector<std::shared_ptr<Order>> clientOrders;
    
    for (const auto& [orderId, order] : orders_) {
        if (order->clientId == clientId) {
            clientOrders.push_back(order);
        }
    }
    
    return clientOrders;
}

bool OrderBook::isEmpty() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    return bidLevels_.empty() && askLevels_.empty();
}

size_t OrderBook::getOrderCount() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    return orders_.size();
}

size_t OrderBook::getBidLevelCount() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    return bidLevels_.size();
}

size_t OrderBook::getAskLevelCount() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    return askLevels_.size();
}

void OrderBook::clear() {
    std::unique_lock<std::shared_mutex> lock(bookMutex_);
    
    bidLevels_.clear();
    askLevels_.clear();
    orders_.clear();
    
    stats_ = OrderBookStats();
    stats_.lastUpdateTime = std::chrono::high_resolution_clock::now().time_since_epoch();
    
    invalidateCache();
}

uint64_t OrderBook::getVolumeAtPrice(double price, OrderSide side) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    if (side == OrderSide::BUY) {
        auto it = bidLevels_.find(price);
        return (it != bidLevels_.end()) ? it->second->getTotalQuantity() : 0;
    } else {
        auto it = askLevels_.find(price);
        return (it != askLevels_.end()) ? it->second->getTotalQuantity() : 0;
    }
}

uint64_t OrderBook::getVolumeAroundPrice(double price, OrderSide side, bool above) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    uint64_t totalVolume = 0;
    
    if (side == OrderSide::BUY) {
        for (const auto& [levelPrice, level] : bidLevels_) {
            if ((above && levelPrice > price) || (!above && levelPrice < price)) {
                totalVolume += level->getTotalQuantity();
            }
        }
    } else {
        for (const auto& [levelPrice, level] : askLevels_) {
            if ((above && levelPrice > price) || (!above && levelPrice < price)) {
                totalVolume += level->getTotalQuantity();
            }
        }
    }
    
    return totalVolume;
}

double OrderBook::calculateVWAP(OrderSide side, size_t levels) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    double totalValue = 0.0;
    uint64_t totalVolume = 0;
    size_t count = 0;
    
    if (side == OrderSide::BUY) {
        for (const auto& [price, level] : bidLevels_) {
            if (levels > 0 && count >= levels) break;
            
            uint64_t volume = level->getTotalQuantity();
            totalValue += price * volume;
            totalVolume += volume;
            ++count;
        }
    } else {
        for (const auto& [price, level] : askLevels_) {
            if (levels > 0 && count >= levels) break;
            
            uint64_t volume = level->getTotalQuantity();
            totalValue += price * volume;
            totalVolume += volume;
            ++count;
        }
    }
    
    return (totalVolume > 0) ? totalValue / totalVolume : 0.0;
}

double OrderBook::simulateMarketImpact(OrderSide side, uint64_t quantity) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    if (quantity == 0) {
        return 0.0;
    }
    
    double totalValue = 0.0;
    uint64_t remainingQty = quantity;
    uint64_t executedQty = 0;
    
    if (side == OrderSide::BUY) {
        // Simulate buying against asks
        for (const auto& [price, level] : askLevels_) {
            if (remainingQty == 0) break;
            
            uint64_t levelQty = level->getTotalQuantity();
            uint64_t tradeQty = std::min(remainingQty, levelQty);
            
            totalValue += price * tradeQty;
            executedQty += tradeQty;
            remainingQty -= tradeQty;
        }
    } else {
        // Simulate selling against bids
        for (const auto& [price, level] : bidLevels_) {
            if (remainingQty == 0) break;
            
            uint64_t levelQty = level->getTotalQuantity();
            uint64_t tradeQty = std::min(remainingQty, levelQty);
            
            totalValue += price * tradeQty;
            executedQty += tradeQty;
            remainingQty -= tradeQty;
        }
    }
    
    return (executedQty > 0) ? totalValue / executedQty : 0.0;
}

void OrderBook::printBook(size_t levels) const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    auto depth = getDepth(levels);
    
    std::cout << "\n=== Order Book for " << symbol_ << " ===\n";
    std::cout << "Mid Price: " << std::fixed << std::setprecision(2) << depth.midPrice;
    std::cout << ", Spread: " << depth.spread << "\n\n";
    
    std::cout << "ASKS:\n";
    std::cout << std::setw(10) << "Price" << std::setw(15) << "Quantity" << "\n";
    std::cout << std::string(25, '-') << "\n";
    
    // Print asks in reverse order (highest first)
    for (auto it = depth.asks.rbegin(); it != depth.asks.rend(); ++it) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << it->first;
        std::cout << std::setw(15) << it->second << "\n";
    }
    
    std::cout << std::string(25, '=') << "\n";
    
    // Print bids (highest first)
    for (const auto& [price, qty] : depth.bids) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << price;
        std::cout << std::setw(15) << qty << "\n";
    }
    
    std::cout << std::string(25, '-') << "\n";
    std::cout << "BIDS:\n";
    std::cout << std::setw(10) << "Price" << std::setw(15) << "Quantity" << "\n\n";
    
    auto stats = getStats();
    std::cout << "Total Orders: " << stats.totalOrders;
    std::cout << ", Total Trades: " << stats.totalTrades;
    std::cout << ", VWAP: " << std::fixed << std::setprecision(2) << stats.vwap << "\n";
}

bool OrderBook::validateBook() const {
    std::shared_lock<std::shared_mutex> lock(bookMutex_);
    
    // Check that all orders in levels exist in orders_ map
    for (const auto& [price, level] : bidLevels_) {
        for (const auto& order : level->getOrders()) {
            auto it = orders_.find(order->orderId);
            if (it == orders_.end() || it->second != order) {
                return false;
            }
            
            // Check price consistency
            if (std::abs(order->price - price) > 1e-8) {
                return false;
            }
            
            // Check side consistency
            if (order->side != OrderSide::BUY) {
                return false;
            }
        }
    }
    
    for (const auto& [price, level] : askLevels_) {
        for (const auto& order : level->getOrders()) {
            auto it = orders_.find(order->orderId);
            if (it == orders_.end() || it->second != order) {
                return false;
            }
            
            // Check price consistency
            if (std::abs(order->price - price) > 1e-8) {
                return false;
            }
            
            // Check side consistency
            if (order->side != OrderSide::SELL) {
                return false;
            }
        }
    }
    
    // Check that bid prices are in descending order
    double lastBidPrice = std::numeric_limits<double>::max();
    for (const auto& [price, level] : bidLevels_) {
        if (price >= lastBidPrice) {
            return false;
        }
        lastBidPrice = price;
    }
    
    // Check that ask prices are in ascending order
    double lastAskPrice = 0.0;
    for (const auto& [price, level] : askLevels_) {
        if (price <= lastAskPrice) {
            return false;
        }
        lastAskPrice = price;
    }
    
    return true;
}

// ============================================================================
// OrderBookManager Implementation
// ============================================================================

OrderBook& OrderBookManager::getBook(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(managerMutex_);
    
    auto it = books_.find(symbol);
    if (it == books_.end()) {
        auto [newIt, inserted] = books_.emplace(symbol, std::make_unique<OrderBook>(symbol));
        return *newIt->second;
    }
    
    return *it->second;
}

bool OrderBookManager::hasBook(const std::string& symbol) const {
    std::shared_lock<std::shared_mutex> lock(managerMutex_);
    return books_.find(symbol) != books_.end();
}

bool OrderBookManager::removeBook(const std::string& symbol) {
    std::unique_lock<std::shared_mutex> lock(managerMutex_);
    
    auto it = books_.find(symbol);
    if (it != books_.end()) {
        books_.erase(it);
        return true;
    }
    
    return false;
}

std::vector<std::string> OrderBookManager::getSymbols() const {
    std::shared_lock<std::shared_mutex> lock(managerMutex_);
    
    std::vector<std::string> symbols;
    symbols.reserve(books_.size());
    
    for (const auto& [symbol, book] : books_) {
        symbols.push_back(symbol);
    }
    
    return symbols;
}

size_t OrderBookManager::getTotalOrderCount() const {
    std::shared_lock<std::shared_mutex> lock(managerMutex_);
    
    size_t totalOrders = 0;
    for (const auto& [symbol, book] : books_) {
        totalOrders += book->getOrderCount();
    }
    
    return totalOrders;
}

void OrderBookManager::clear() {
    std::unique_lock<std::shared_mutex> lock(managerMutex_);
    books_.clear();
}

} // namespace StatArb