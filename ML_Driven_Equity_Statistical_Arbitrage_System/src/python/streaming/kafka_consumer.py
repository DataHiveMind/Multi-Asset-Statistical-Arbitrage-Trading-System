# Real-time streaming data consumer
import asyncio
from kafka import KafkaConsumer
import json
import pandas as pd
from typing import Dict, Any

class RealTimeDataConsumer:
    """
    High-performance streaming data consumer for real-time market data
    """
    
    def __init__(self, topics: list, bootstrap_servers: str = 'localhost:9092'):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=[bootstrap_servers],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.signal_cache = {}
        
    async def process_market_data(self, data: Dict[str, Any]):
        """Process real-time market data and generate signals"""
        symbol = data['symbol']
        price = data['price']
        volume = data['volume']
        timestamp = data['timestamp']
        
        # Update rolling calculations
        self.update_technical_indicators(symbol, price, volume, timestamp)
        
        # Generate real-time signals
        signals = self.generate_real_time_signals(symbol)
        
        return signals
    
    def update_technical_indicators(self, symbol: str, price: float, volume: int, timestamp: str):
        """Update technical indicators in real-time"""
        if symbol not in self.signal_cache:
            self.signal_cache[symbol] = {
                'prices': [],
                'volumes': [],
                'timestamps': [],
                'ma_20': None,
                'rsi': None,
                'bollinger_bands': None
            }
        
        cache = self.signal_cache[symbol]
        cache['prices'].append(price)
        cache['volumes'].append(volume)
        cache['timestamps'].append(timestamp)
        
        # Keep only last 100 data points for efficiency
        if len(cache['prices']) > 100:
            cache['prices'] = cache['prices'][-100:]
            cache['volumes'] = cache['volumes'][-100:]
            cache['timestamps'] = cache['timestamps'][-100:]
        
        # Update indicators
        if len(cache['prices']) >= 20:
            cache['ma_20'] = sum(cache['prices'][-20:]) / 20
            cache['rsi'] = self.calculate_rsi(cache['prices'])
