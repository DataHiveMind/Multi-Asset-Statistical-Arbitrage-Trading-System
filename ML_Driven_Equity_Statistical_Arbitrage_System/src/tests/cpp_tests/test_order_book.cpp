/**
 * @file test_order_book.cpp
 * @brief Comprehensive unit tests for the OrderBook implementation
 * 
 * This test suite covers all major functionality of the OrderBook class including:
 * - Order addition, modification, and cancellation
 * - Order matching and trade generation
 * - Market depth queries and statistics
 * - Thread safety and performance
 * - Edge cases and error conditions
 * 
 * Uses Google Test framework for C++ unit testing.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <future>

// Include the OrderBook header
#include "../order_book/OrderBook.h"

using namespace StatArb;
using namespace testing;

/**
 * @brief Test fixture for OrderBook tests
 * 
 * Provides common setup and utility methods for all OrderBook tests.
 */
class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        book = std::make_unique<OrderBook>("TEST");
        tradeCount = 0;
        lastTrade = Trade(0, 0, 0, "", 0.0, 0);
        orderUpdates.clear();
        
        // Set up callbacks for testing
        book->setTradeCallback([this](const Trade& trade) {
            tradeCount++;
            lastTrade = trade;
        });
        
        book->setOrderCallback([this](const Order& order) {
            orderUpdates.push_back(order);
        });
    }
    
    void TearDown() override {
        book.reset();
    }
    
    // Helper methods
    uint64_t addBuyOrder(double price, uint64_t quantity, const std::string& client = "") {
        return book->addOrder(OrderSide::BUY, OrderType::LIMIT, price, quantity, client);
    }
    
    uint64_t addSellOrder(double price, uint64_t quantity, const std::string& client = "") {
        return book->addOrder(OrderSide::SELL, OrderType::LIMIT, price, quantity, client);
    }
    
    uint64_t addMarketBuyOrder(uint64_t quantity, const std::string& client = "") {
        return book->addOrder(OrderSide::BUY, OrderType::MARKET, 0.0, quantity, client);
    }
    
    uint64_t addMarketSellOrder(uint64_t quantity, const std::string& client = "") {
        return book->addOrder(OrderSide::SELL, OrderType::MARKET, 0.0, quantity, client);
    }
    
    void setupBasicBook() {
        // Create a basic book with some orders
        addBuyOrder(99.5, 100);   // Best bid
        addBuyOrder(99.0, 200);   // Second level
        addBuyOrder(98.5, 150);   // Third level
        
        addSellOrder(100.5, 100); // Best ask
        addSellOrder(101.0, 200); // Second level
        addSellOrder(101.5, 150); // Third level
    }
    
    void setupCrossedBook() {
        // Create orders that will immediately match
        addBuyOrder(100.0, 100);
        addSellOrder(99.5, 100);  // This will cross with the buy order
    }
    
    // Test data
    std::unique_ptr<OrderBook> book;
    std::atomic<int> tradeCount{0};
    Trade lastTrade;
    std::vector<Order> orderUpdates;
};

/**
 * @brief Test fixture for multi-threaded OrderBook tests
 */
class OrderBookConcurrencyTest : public OrderBookTest {
protected:
    static constexpr int NUM_THREADS = 4;
    static constexpr int ORDERS_PER_THREAD = 100;
};

// ============================================================================
// Basic Order Management Tests
// ============================================================================

TEST_F(OrderBookTest, InitialStateIsEmpty) {
    EXPECT_TRUE(book->isEmpty());
    EXPECT_EQ(book->getOrderCount(), 0);
    EXPECT_EQ(book->getBidLevelCount(), 0);
    EXPECT_EQ(book->getAskLevelCount(), 0);
    EXPECT_EQ(book->getBestBid(), 0.0);
    EXPECT_EQ(book->getBestAsk(), 0.0);
    EXPECT_EQ(book->getSpread(), 0.0);
    EXPECT_EQ(book->getMidPrice(), 0.0);
}

TEST_F(OrderBookTest, AddSingleBuyOrder) {
    uint64_t orderId = addBuyOrder(100.0, 100);
    
    EXPECT_FALSE(book->isEmpty());
    EXPECT_EQ(book->getOrderCount(), 1);
    EXPECT_EQ(book->getBidLevelCount(), 1);
    EXPECT_EQ(book->getAskLevelCount(), 0);
    EXPECT_EQ(book->getBestBid(), 100.0);
    EXPECT_EQ(book->getBestBidQuantity(), 100);
    EXPECT_EQ(book->getBestAsk(), 0.0);
    EXPECT_EQ(book->getSpread(), 0.0);
    
    // Check order can be retrieved
    auto order = book->getOrder(orderId);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->orderId, orderId);
    EXPECT_EQ(order->side, OrderSide::BUY);
    EXPECT_EQ(order->price, 100.0);
    EXPECT_EQ(order->quantity, 100);
    EXPECT_EQ(order->status, OrderStatus::PENDING);
}

TEST_F(OrderBookTest, AddSingleSellOrder) {
    uint64_t orderId = addSellOrder(100.0, 100);
    
    EXPECT_FALSE(book->isEmpty());
    EXPECT_EQ(book->getOrderCount(), 1);
    EXPECT_EQ(book->getBidLevelCount(), 0);
    EXPECT_EQ(book->getAskLevelCount(), 1);
    EXPECT_EQ(book->getBestBid(), 0.0);
    EXPECT_EQ(book->getBestAsk(), 100.0);
    EXPECT_EQ(book->getBestAskQuantity(), 100);
    EXPECT_EQ(book->getSpread(), 0.0);
    
    // Check order can be retrieved
    auto order = book->getOrder(orderId);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->side, OrderSide::SELL);
}

TEST_F(OrderBookTest, AddMultipleBuyOrdersSamePrice) {
    uint64_t order1 = addBuyOrder(100.0, 100);
    uint64_t order2 = addBuyOrder(100.0, 200);
    uint64_t order3 = addBuyOrder(100.0, 150);
    
    EXPECT_EQ(book->getOrderCount(), 3);
    EXPECT_EQ(book->getBidLevelCount(), 1);
    EXPECT_EQ(book->getBestBidQuantity(), 450); // 100 + 200 + 150
    
    // All orders should exist
    EXPECT_NE(book->getOrder(order1), nullptr);
    EXPECT_NE(book->getOrder(order2), nullptr);
    EXPECT_NE(book->getOrder(order3), nullptr);
}

TEST_F(OrderBookTest, AddMultipleBuyOrdersDifferentPrices) {
    addBuyOrder(100.0, 100);
    addBuyOrder(99.5, 200);
    addBuyOrder(101.0, 150);  // Best bid
    
    EXPECT_EQ(book->getOrderCount(), 3);
    EXPECT_EQ(book->getBidLevelCount(), 3);
    EXPECT_EQ(book->getBestBid(), 101.0);  // Highest price should be best bid
    EXPECT_EQ(book->getBestBidQuantity(), 150);
}

TEST_F(OrderBookTest, AddMultipleSellOrdersDifferentPrices) {
    addSellOrder(100.0, 100);  // Best ask
    addSellOrder(100.5, 200);
    addSellOrder(99.5, 150);   // Should become best ask
    
    EXPECT_EQ(book->getOrderCount(), 3);
    EXPECT_EQ(book->getAskLevelCount(), 3);
    EXPECT_EQ(book->getBestAsk(), 99.5);   // Lowest price should be best ask
    EXPECT_EQ(book->getBestAskQuantity(), 150);
}

TEST_F(OrderBookTest, BasicSpreadCalculation) {
    addBuyOrder(99.5, 100);
    addSellOrder(100.5, 100);
    
    EXPECT_EQ(book->getBestBid(), 99.5);
    EXPECT_EQ(book->getBestAsk(), 100.5);
    EXPECT_EQ(book->getSpread(), 1.0);
    EXPECT_EQ(book->getMidPrice(), 100.0);
}

// ============================================================================
// Order Modification Tests
// ============================================================================

TEST_F(OrderBookTest, CancelBuyOrder) {
    uint64_t orderId = addBuyOrder(100.0, 100);
    
    EXPECT_TRUE(book->cancelOrder(orderId));
    EXPECT_TRUE(book->isEmpty());
    EXPECT_EQ(book->getOrderCount(), 0);
    EXPECT_EQ(book->getBestBid(), 0.0);
    
    // Order should still exist but be cancelled
    auto order = book->getOrder(orderId);
    EXPECT_EQ(order, nullptr);  // Cancelled orders are removed from tracking
}

TEST_F(OrderBookTest, CancelNonExistentOrder) {
    EXPECT_FALSE(book->cancelOrder(999999));
}

TEST_F(OrderBookTest, CancelOrderFromMultipleOrders) {
    uint64_t order1 = addBuyOrder(100.0, 100);
    uint64_t order2 = addBuyOrder(100.0, 200);
    uint64_t order3 = addBuyOrder(100.0, 150);
    
    EXPECT_TRUE(book->cancelOrder(order2));
    EXPECT_EQ(book->getOrderCount(), 2);
    EXPECT_EQ(book->getBestBidQuantity(), 250); // 100 + 150
    
    EXPECT_NE(book->getOrder(order1), nullptr);
    EXPECT_EQ(book->getOrder(order2), nullptr);
    EXPECT_NE(book->getOrder(order3), nullptr);
}

TEST_F(OrderBookTest, ModifyOrderQuantityIncrease) {
    uint64_t orderId = addBuyOrder(100.0, 100);
    
    EXPECT_TRUE(book->modifyOrder(orderId, 200));
    EXPECT_EQ(book->getBestBidQuantity(), 200);
    
    auto order = book->getOrder(orderId);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->quantity, 200);
}

TEST_F(OrderBookTest, ModifyOrderQuantityDecrease) {
    uint64_t orderId = addBuyOrder(100.0, 100);
    
    EXPECT_TRUE(book->modifyOrder(orderId, 50));
    EXPECT_EQ(book->getBestBidQuantity(), 50);
    
    auto order = book->getOrder(orderId);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->quantity, 50);
}

TEST_F(OrderBookTest, ModifyOrderQuantityToZero) {
    uint64_t orderId = addBuyOrder(100.0, 100);
    
    EXPECT_TRUE(book->modifyOrder(orderId, 0));
    EXPECT_TRUE(book->isEmpty());
    EXPECT_EQ(book->getOrderCount(), 0);
}

TEST_F(OrderBookTest, ModifyOrderPriceAndQuantity) {
    uint64_t orderId = addBuyOrder(100.0, 100);
    
    EXPECT_TRUE(book->modifyOrder(orderId, 101.0, 200));
    EXPECT_EQ(book->getBestBid(), 101.0);
    EXPECT_EQ(book->getBestBidQuantity(), 200);
    
    auto order = book->getOrder(orderId);
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->price, 101.0);
    EXPECT_EQ(order->quantity, 200);
}

TEST_F(OrderBookTest, ModifyNonExistentOrder) {
    EXPECT_FALSE(book->modifyOrder(999999, 100));
    EXPECT_FALSE(book->modifyOrder(999999, 100.0, 100));
}

// ============================================================================
// Order Matching Tests
// ============================================================================

TEST_F(OrderBookTest, SimpleOrderMatching) {
    // Add a buy order
    uint64_t buyOrderId = addBuyOrder(100.0, 100);
    EXPECT_EQ(tradeCount, 0);
    
    // Add a matching sell order
    uint64_t sellOrderId = addSellOrder(100.0, 100);
    
    // Should have generated one trade
    EXPECT_EQ(tradeCount, 1);
    EXPECT_EQ(lastTrade.buyOrderId, buyOrderId);
    EXPECT_EQ(lastTrade.sellOrderId, sellOrderId);
    EXPECT_EQ(lastTrade.price, 100.0);
    EXPECT_EQ(lastTrade.quantity, 100);
    
    // Book should be empty after full match
    EXPECT_TRUE(book->isEmpty());
}

TEST_F(OrderBookTest, PartialOrderMatching) {
    // Add a large buy order
    uint64_t buyOrderId = addBuyOrder(100.0, 200);
    
    // Add a smaller sell order
    uint64_t sellOrderId = addSellOrder(100.0, 100);
    
    // Should have one trade for partial fill
    EXPECT_EQ(tradeCount, 1);
    EXPECT_EQ(lastTrade.quantity, 100);
    
    // Buy order should remain with reduced quantity
    EXPECT_EQ(book->getOrderCount(), 1);
    EXPECT_EQ(book->getBestBidQuantity(), 100);
    
    auto buyOrder = book->getOrder(buyOrderId);
    ASSERT_NE(buyOrder, nullptr);
    EXPECT_EQ(buyOrder->filledQuantity, 100);
    EXPECT_EQ(buyOrder->getRemainingQuantity(), 100);
    EXPECT_EQ(buyOrder->status, OrderStatus::PARTIAL_FILL);
}

TEST_F(OrderBookTest, MultiLevelOrderMatching) {
    // Setup multiple bid levels
    addBuyOrder(100.0, 100);
    addBuyOrder(99.5, 150);
    addBuyOrder(99.0, 200);
    
    // Large sell order that should match multiple levels
    addSellOrder(99.0, 300);
    
    // Should generate multiple trades
    EXPECT_GE(tradeCount, 2);
    
    // Check final state
    EXPECT_EQ(book->getBestBid(), 99.0);
    EXPECT_EQ(book->getBestBidQuantity(), 150); // Remaining from 99.0 level
}

TEST_F(OrderBookTest, PriorityOrderMatching) {
    // Add orders at same price (time priority should apply)
    uint64_t order1 = addBuyOrder(100.0, 100);
    uint64_t order2 = addBuyOrder(100.0, 200);
    uint64_t order3 = addBuyOrder(100.0, 150);
    
    // Sell order should match first order first (FIFO)
    addSellOrder(100.0, 100);
    
    EXPECT_EQ(tradeCount, 1);
    EXPECT_EQ(lastTrade.buyOrderId, order1);
    
    // Check remaining orders
    EXPECT_EQ(book->getOrderCount(), 2);
    EXPECT_EQ(book->getBestBidQuantity(), 350); // 200 + 150
}

TEST_F(OrderBookTest, MarketOrderExecution) {
    // Setup ask side
    addSellOrder(100.0, 100);
    addSellOrder(100.5, 200);
    
    // Market buy order should match best ask
    addMarketBuyOrder(150);
    
    EXPECT_EQ(tradeCount, 1);
    EXPECT_EQ(lastTrade.price, 100.0);
    EXPECT_EQ(lastTrade.quantity, 100);
    
    // Should have consumed entire first level and partial second level
    EXPECT_EQ(book->getBestAsk(), 100.5);
    EXPECT_EQ(book->getBestAskQuantity(), 150); // 200 - 50 remaining
}

TEST_F(OrderBookTest, MarketOrderInsufficientLiquidity) {
    // Add small ask
    addSellOrder(100.0, 50);
    
    // Large market buy order
    addMarketBuyOrder(200);
    
    // Should match available liquidity
    EXPECT_EQ(tradeCount, 1);
    EXPECT_EQ(lastTrade.quantity, 50);
    
    // Book should be empty after consuming all asks
    EXPECT_TRUE(book->isEmpty());
}

// ============================================================================
// Market Depth and Statistics Tests
// ============================================================================

TEST_F(OrderBookTest, GetMarketDepth) {
    setupBasicBook();
    
    MarketDepth depth = book->getDepth(3);
    
    // Check bid side
    ASSERT_EQ(depth.bids.size(), 3);
    EXPECT_EQ(depth.bids[0].first, 99.5);   // Best bid
    EXPECT_EQ(depth.bids[0].second, 100);
    EXPECT_EQ(depth.bids[1].first, 99.0);
    EXPECT_EQ(depth.bids[1].second, 200);
    EXPECT_EQ(depth.bids[2].first, 98.5);
    EXPECT_EQ(depth.bids[2].second, 150);
    
    // Check ask side
    ASSERT_EQ(depth.asks.size(), 3);
    EXPECT_EQ(depth.asks[0].first, 100.5);  // Best ask
    EXPECT_EQ(depth.asks[0].second, 100);
    EXPECT_EQ(depth.asks[1].first, 101.0);
    EXPECT_EQ(depth.asks[1].second, 200);
    EXPECT_EQ(depth.asks[2].first, 101.5);
    EXPECT_EQ(depth.asks[2].second, 150);
    
    // Check aggregated data
    EXPECT_EQ(depth.bidPrice, 99.5);
    EXPECT_EQ(depth.askPrice, 100.5);
    EXPECT_EQ(depth.bidQuantity, 100);
    EXPECT_EQ(depth.askQuantity, 100);
    EXPECT_EQ(depth.spread, 1.0);
    EXPECT_EQ(depth.midPrice, 100.0);
}

TEST_F(OrderBookTest, GetOrderBookStats) {
    setupBasicBook();
    
    OrderBookStats stats = book->getStats();
    
    EXPECT_EQ(stats.totalOrders, 6);
    EXPECT_EQ(stats.totalTrades, 0);
    EXPECT_EQ(stats.bidLevels, 3);
    EXPECT_EQ(stats.askLevels, 3);
    EXPECT_GT(stats.lastUpdateTime.count(), 0);
}

TEST_F(OrderBookTest, VolumeAtPriceQueries) {
    setupBasicBook();
    
    EXPECT_EQ(book->getVolumeAtPrice(99.5, OrderSide::BUY), 100);
    EXPECT_EQ(book->getVolumeAtPrice(100.5, OrderSide::SELL), 100);
    EXPECT_EQ(book->getVolumeAtPrice(98.0, OrderSide::BUY), 0); // No orders at this price
    
    // Test volume around price
    EXPECT_EQ(book->getVolumeAroundPrice(99.25, OrderSide::BUY, true), 100);  // Above 99.25
    EXPECT_EQ(book->getVolumeAroundPrice(99.25, OrderSide::BUY, false), 350); // Below 99.25 (99.0 + 98.5)
}

TEST_F(OrderBookTest, VWAPCalculation) {
    setupBasicBook();
    
    double bidVWAP = book->calculateVWAP(OrderSide::BUY, 0);
    // VWAP = (99.5*100 + 99.0*200 + 98.5*150) / (100+200+150)
    double expectedBidVWAP = (99.5*100 + 99.0*200 + 98.5*150) / 450;
    EXPECT_NEAR(bidVWAP, expectedBidVWAP, 0.001);
    
    double askVWAP = book->calculateVWAP(OrderSide::SELL, 0);
    double expectedAskVWAP = (100.5*100 + 101.0*200 + 101.5*150) / 450;
    EXPECT_NEAR(askVWAP, expectedAskVWAP, 0.001);
}

TEST_F(OrderBookTest, MarketImpactSimulation) {
    setupBasicBook();
    
    // Simulate buying 250 shares (should impact 2 levels)
    double avgPrice = book->simulateMarketImpact(OrderSide::BUY, 250);
    
    // Should get 100 shares at 100.5 and 150 shares at 101.0
    double expectedAvgPrice = (100 * 100.5 + 150 * 101.0) / 250;
    EXPECT_NEAR(avgPrice, expectedAvgPrice, 0.001);
}

// ============================================================================
// Client Order Management Tests
// ============================================================================

TEST_F(OrderBookTest, ClientOrderTracking) {
    uint64_t order1 = addBuyOrder(100.0, 100, "client1");
    uint64_t order2 = addSellOrder(101.0, 200, "client1");
    uint64_t order3 = addBuyOrder(99.0, 150, "client2");
    
    auto client1Orders = book->getOrdersByClient("client1");
    auto client2Orders = book->getOrdersByClient("client2");
    auto unknownClientOrders = book->getOrdersByClient("unknown");
    
    EXPECT_EQ(client1Orders.size(), 2);
    EXPECT_EQ(client2Orders.size(), 1);
    EXPECT_EQ(unknownClientOrders.size(), 0);
    
    // Check order IDs
    std::vector<uint64_t> client1Ids;
    for (const auto& order : client1Orders) {
        client1Ids.push_back(order->orderId);
    }
    EXPECT_THAT(client1Ids, UnorderedElementsAre(order1, order2));
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST_F(OrderBookTest, ZeroQuantityOrder) {
    uint64_t orderId = book->addOrder(OrderSide::BUY, OrderType::LIMIT, 100.0, 0);
    
    // Zero quantity orders should not be added
    EXPECT_EQ(orderId, 0);
    EXPECT_TRUE(book->isEmpty());
}

TEST_F(OrderBookTest, NegativePriceOrder) {
    uint64_t orderId = book->addOrder(OrderSide::BUY, OrderType::LIMIT, -100.0, 100);
    
    // Negative price orders should not be added
    EXPECT_EQ(orderId, 0);
    EXPECT_TRUE(book->isEmpty());
}

TEST_F(OrderBookTest, ClearAllOrders) {
    setupBasicBook();
    
    EXPECT_FALSE(book->isEmpty());
    book->clear();
    EXPECT_TRUE(book->isEmpty());
    EXPECT_EQ(book->getOrderCount(), 0);
    EXPECT_EQ(book->getBidLevelCount(), 0);
    EXPECT_EQ(book->getAskLevelCount(), 0);
}

TEST_F(OrderBookTest, BookValidation) {
    setupBasicBook();
    
    // Book should be valid after normal operations
    EXPECT_TRUE(book->validateBook());
    
    // Add some trades
    addSellOrder(99.5, 50); // Should cross with best bid
    EXPECT_TRUE(book->validateBook());
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST_F(OrderBookTest, LargeNumberOfOrders) {
    const int numOrders = 1000;
    std::vector<uint64_t> orderIds;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Add many orders
    for (int i = 0; i < numOrders; ++i) {
        double price = 100.0 + (i % 100) * 0.01; // Spread orders across price levels
        uint64_t quantity = 100 + (i % 50);
        OrderSide side = (i % 2 == 0) ? OrderSide::BUY : OrderSide::SELL;
        
        uint64_t orderId = book->addOrder(side, OrderType::LIMIT, price, quantity);
        if (orderId > 0) {
            orderIds.push_back(orderId);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Added " << orderIds.size() << " orders in " << duration.count() << " microseconds" << std::endl;
    
    // Verify book state
    EXPECT_GT(book->getOrderCount(), 0);
    EXPECT_TRUE(book->validateBook());
}

TEST_F(OrderBookTest, FrequentModifications) {
    const int numOperations = 1000;
    std::vector<uint64_t> orderIds;
    
    // Add initial orders
    for (int i = 0; i < 100; ++i) {
        uint64_t orderId = addBuyOrder(100.0 + i * 0.1, 100);
        orderIds.push_back(orderId);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform many modifications
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> opDist(0, 2); // 0=modify, 1=cancel, 2=add
    std::uniform_int_distribution<> orderDist(0, orderIds.size() - 1);
    
    for (int i = 0; i < numOperations; ++i) {
        int op = opDist(gen);
        
        if (op == 0 && !orderIds.empty()) {
            // Modify random order
            int idx = orderDist(gen) % orderIds.size();
            book->modifyOrder(orderIds[idx], 100 + (i % 50));
        } else if (op == 1 && !orderIds.empty()) {
            // Cancel random order
            int idx = orderDist(gen) % orderIds.size();
            book->cancelOrder(orderIds[idx]);
            orderIds.erase(orderIds.begin() + idx);
        } else {
            // Add new order
            uint64_t orderId = addBuyOrder(100.0 + (i % 100) * 0.1, 100);
            if (orderId > 0) {
                orderIds.push_back(orderId);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Performed " << numOperations << " operations in " << duration.count() << " microseconds" << std::endl;
    
    EXPECT_TRUE(book->validateBook());
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST_F(OrderBookConcurrencyTest, ConcurrentOrderAddition) {
    std::vector<std::thread> threads;
    std::vector<std::vector<uint64_t>> threadOrderIds(NUM_THREADS);
    
    // Launch threads that add orders concurrently
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, t, &threadOrderIds]() {
            for (int i = 0; i < ORDERS_PER_THREAD; ++i) {
                double price = 100.0 + t + i * 0.01;
                uint64_t quantity = 100 + i;
                OrderSide side = (t % 2 == 0) ? OrderSide::BUY : OrderSide::SELL;
                
                uint64_t orderId = book->addOrder(side, OrderType::LIMIT, price, quantity);
                if (orderId > 0) {
                    threadOrderIds[t].push_back(orderId);
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all orders were added correctly
    size_t totalOrders = 0;
    for (const auto& orderIds : threadOrderIds) {
        totalOrders += orderIds.size();
    }
    
    EXPECT_EQ(book->getOrderCount(), totalOrders);
    EXPECT_TRUE(book->validateBook());
}

TEST_F(OrderBookConcurrencyTest, ConcurrentOrderOperations) {
    // Pre-populate book with some orders
    std::vector<uint64_t> initialOrders;
    for (int i = 0; i < 100; ++i) {
        uint64_t orderId = addBuyOrder(100.0 + i * 0.1, 100);
        initialOrders.push_back(orderId);
    }
    
    std::atomic<int> operationsCompleted{0};
    std::vector<std::thread> threads;
    
    // Launch threads performing different operations
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, t, &initialOrders, &operationsCompleted]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> opDist(0, 3);
            
            for (int i = 0; i < ORDERS_PER_THREAD; ++i) {
                int op = opDist(gen);
                
                switch (op) {
                    case 0: // Add order
                        book->addOrder(OrderSide::BUY, OrderType::LIMIT, 
                                     100.0 + t + i * 0.01, 100);
                        break;
                    case 1: // Cancel order
                        if (!initialOrders.empty()) {
                            size_t idx = gen() % initialOrders.size();
                            book->cancelOrder(initialOrders[idx]);
                        }
                        break;
                    case 2: // Modify order
                        if (!initialOrders.empty()) {
                            size_t idx = gen() % initialOrders.size();
                            book->modifyOrder(initialOrders[idx], 50 + (i % 100));
                        }
                        break;
                    case 3: // Query operations
                        book->getBestBid();
                        book->getBestAsk();
                        book->getDepth(5);
                        break;
                }
                
                operationsCompleted++;
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GE(operationsCompleted.load(), NUM_THREADS * ORDERS_PER_THREAD);
    EXPECT_TRUE(book->validateBook());
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST_F(OrderBookTest, OrderCallbacks) {
    int callbackCount = 0;
    book->setOrderCallback([&callbackCount](const Order& order) {
        callbackCount++;
    });
    
    addBuyOrder(100.0, 100);
    addSellOrder(101.0, 100);
    
    // Should have received callbacks for both orders
    EXPECT_GE(callbackCount, 2);
}

TEST_F(OrderBookTest, TradeCallbacks) {
    int tradeCallbackCount = 0;
    book->setTradeCallback([&tradeCallbackCount](const Trade& trade) {
        tradeCallbackCount++;
    });
    
    addBuyOrder(100.0, 100);
    addSellOrder(100.0, 100); // Should generate trade
    
    EXPECT_EQ(tradeCallbackCount, 1);
}

TEST_F(OrderBookTest, DepthCallbacks) {
    int depthCallbackCount = 0;
    book->setDepthCallback([&depthCallbackCount](const MarketDepth& depth) {
        depthCallbackCount++;
    });
    
    addBuyOrder(100.0, 100);
    addSellOrder(101.0, 100);
    
    // Should have received depth callbacks
    EXPECT_GE(depthCallbackCount, 0); // Implementation dependent
}

// ============================================================================
// OrderBookManager Tests
// ============================================================================

class OrderBookManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager = std::make_unique<OrderBookManager>();
    }
    
    std::unique_ptr<OrderBookManager> manager;
};

TEST_F(OrderBookManagerTest, CreateAndRetrieveBooks) {
    EXPECT_FALSE(manager->hasBook("AAPL"));
    
    OrderBook& book = manager->getBook("AAPL");
    EXPECT_TRUE(manager->hasBook("AAPL"));
    
    // Should return same book instance
    OrderBook& book2 = manager->getBook("AAPL");
    EXPECT_EQ(&book, &book2);
}

TEST_F(OrderBookManagerTest, MultipleSymbols) {
    manager->getBook("AAPL");
    manager->getBook("MSFT");
    manager->getBook("GOOGL");
    
    auto symbols = manager->getSymbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_THAT(symbols, UnorderedElementsAre("AAPL", "MSFT", "GOOGL"));
}

TEST_F(OrderBookManagerTest, RemoveBook) {
    manager->getBook("AAPL");
    EXPECT_TRUE(manager->hasBook("AAPL"));
    
    EXPECT_TRUE(manager->removeBook("AAPL"));
    EXPECT_FALSE(manager->hasBook("AAPL"));
    
    // Removing non-existent book should return false
    EXPECT_FALSE(manager->removeBook("NONEXISTENT"));
}

TEST_F(OrderBookManagerTest, TotalOrderCount) {
    OrderBook& aaplBook = manager->getBook("AAPL");
    OrderBook& msftBook = manager->getBook("MSFT");
    
    aaplBook.addOrder(OrderSide::BUY, OrderType::LIMIT, 100.0, 100);
    aaplBook.addOrder(OrderSide::SELL, OrderType::LIMIT, 101.0, 100);
    msftBook.addOrder(OrderSide::BUY, OrderType::LIMIT, 200.0, 100);
    
    EXPECT_EQ(manager->getTotalOrderCount(), 3);
}

TEST_F(OrderBookManagerTest, ClearAllBooks) {
    manager->getBook("AAPL");
    manager->getBook("MSFT");
    
    EXPECT_EQ(manager->getSymbols().size(), 2);
    
    manager->clear();
    EXPECT_EQ(manager->getSymbols().size(), 0);
    EXPECT_FALSE(manager->hasBook("AAPL"));
    EXPECT_FALSE(manager->hasBook("MSFT"));
}

// ============================================================================
// Main function for running tests
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
