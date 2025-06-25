"""
Alternative Data Ingestion Module

This module provides scripts for connecting to, ingesting, and cleaning various
alternative data sources for statistical arbitrage strategies. It handles API calls,
data parsing, validation, and storage.

Classes:
    - BaseDataConnector: Abstract base class for all data connectors
    - SentimentDataConnector: Social media and news sentiment analysis
    - SatelliteDataConnector: Satellite imagery and geospatial data
    - CreditCardDataConnector: Consumer spending and transaction data
    - ESGDataConnector: Environmental, Social, and Governance data
    - EconomicDataConnector: Alternative economic indicators
    - WebScrapingConnector: Web scraping for custom data sources
    - AlternativeDataManager: Central manager for all alternative data sources

Functions:
    - Data validation and cleaning utilities
    - Storage mechanisms for various data formats
    - Rate limiting and error handling
"""

import abc
import requests
import pandas as pd
import numpy as np
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import os
from pathlib import Path
import warnings

# Web scraping libraries
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    warnings.warn("BeautifulSoup not available. Web scraping functionality will be limited.")

# Sentiment analysis libraries
try:
    from textblob import TextBlob
    import nltk
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    warnings.warn("TextBlob/NLTK not available. Sentiment analysis will be limited.")

# Image processing for satellite data
try:
    from PIL import Image
    import rasterio
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    warnings.warn("PIL/Rasterio not available. Satellite data processing will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDataConnector(abc.ABC):
    """
    Abstract base class for all alternative data connectors.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 rate_limit: int = 100, storage_path: Optional[str] = None):
        """
        Initialize the data connector.
        
        Args:
            api_key: API key for the data provider
            base_url: Base URL for API endpoints
            rate_limit: Maximum requests per minute
            storage_path: Path to store raw data files
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.storage_path = storage_path or "data/raw/alternative_data"
        
        # Create storage directory if it doesn't exist
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_interval = 60.0 / rate_limit if rate_limit > 0 else 0
        
        # Session for connection pooling
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def _rate_limit(self) -> None:
        """Implement rate limiting."""
        if self.request_interval > 0:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_interval:
                time.sleep(self.request_interval - time_since_last)
            self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None,
                     headers: Optional[Dict] = None, timeout: int = 30) -> requests.Response:
        """Make a rate-limited API request."""
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    @abc.abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from the alternative data source."""
        pass
    
    @abc.abstractmethod
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the fetched data."""
        pass
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'parquet') -> str:
        """
        Save data to storage.
        
        Args:
            data: DataFrame to save
            filename: Name of the file (without extension)
            format: File format ('parquet', 'csv', 'json')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{filename}_{timestamp}"
        
        if format == 'parquet':
            filepath = os.path.join(self.storage_path, f"{filename_with_timestamp}.parquet")
            data.to_parquet(filepath)
        elif format == 'csv':
            filepath = os.path.join(self.storage_path, f"{filename_with_timestamp}.csv")
            data.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = os.path.join(self.storage_path, f"{filename_with_timestamp}.json")
            data.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()


class SentimentDataConnector(BaseDataConnector):
    """
    Connector for social media and news sentiment analysis data.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sentiment_sources = {
            'news_api': 'https://newsapi.org/v2',
            'twitter_api': 'https://api.twitter.com/2',
            'reddit_api': 'https://www.reddit.com/api/v1',
            'alpha_sense': 'https://api.alphasense.com/v1'
        }
    
    def fetch_news_sentiment(self, symbols: List[str], days_back: int = 7) -> pd.DataFrame:
        """
        Fetch news sentiment data for given symbols.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
            
        Returns:
            DataFrame with news sentiment data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_articles = []
        
        for symbol in symbols:
            try:
                url = f"{self.sentiment_sources['news_api']}/everything"
                params = {
                    'q': f"{symbol} OR {symbol.replace('$', '')}",
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 100
                }
                
                response = self._make_request(url, params=params)
                data = response.json()
                
                for article in data.get('articles', []):
                    article_data = {
                        'symbol': symbol,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': article.get('publishedAt', ''),
                        'url': article.get('url', '')
                    }
                    
                    # Add sentiment analysis
                    if SENTIMENT_AVAILABLE:
                        full_text = f"{article_data['title']} {article_data['description']}"
                        blob = TextBlob(full_text)
                        article_data['sentiment_polarity'] = blob.sentiment.polarity
                        article_data['sentiment_subjectivity'] = blob.sentiment.subjectivity
                    
                    all_articles.append(article_data)
                
                time.sleep(0.1)  # Additional rate limiting for news API
                
            except Exception as e:
                logger.warning(f"Failed to fetch news for {symbol}: {e}")
        
        return pd.DataFrame(all_articles)
    
    def fetch_social_sentiment(self, symbols: List[str], platforms: List[str] = None) -> pd.DataFrame:
        """
        Fetch social media sentiment data.
        
        Args:
            symbols: List of stock symbols
            platforms: List of platforms ('twitter', 'reddit', 'stocktwits')
        """
        platforms = platforms or ['twitter', 'reddit']
        all_posts = []
        
        for symbol in symbols:
            for platform in platforms:
                try:
                    if platform == 'twitter':
                        posts = self._fetch_twitter_sentiment(symbol)
                    elif platform == 'reddit':
                        posts = self._fetch_reddit_sentiment(symbol)
                    elif platform == 'stocktwits':
                        posts = self._fetch_stocktwits_sentiment(symbol)
                    else:
                        continue
                    
                    all_posts.extend(posts)
                
                except Exception as e:
                    logger.warning(f"Failed to fetch {platform} sentiment for {symbol}: {e}")
        
        return pd.DataFrame(all_posts)
    
    def _fetch_twitter_sentiment(self, symbol: str) -> List[Dict]:
        """Fetch Twitter sentiment data."""
        # Implementation would require Twitter API v2
        # This is a placeholder for the structure
        return []
    
    def _fetch_reddit_sentiment(self, symbol: str) -> List[Dict]:
        """Fetch Reddit sentiment data."""
        posts = []
        
        try:
            # Search relevant subreddits
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting']
            
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    'q': symbol,
                    'sort': 'new',
                    'limit': 25,
                    't': 'week'
                }
                
                response = self._make_request(url, params=params)
                data = response.json()
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    post_info = {
                        'symbol': symbol,
                        'platform': 'reddit',
                        'subreddit': subreddit,
                        'title': post_data.get('title', ''),
                        'text': post_data.get('selftext', ''),
                        'score': post_data.get('score', 0),
                        'num_comments': post_data.get('num_comments', 0),
                        'created_utc': post_data.get('created_utc', 0),
                        'author': post_data.get('author', ''),
                        'url': post_data.get('url', '')
                    }
                    
                    # Add sentiment analysis
                    if SENTIMENT_AVAILABLE:
                        full_text = f"{post_info['title']} {post_info['text']}"
                        if full_text.strip():
                            blob = TextBlob(full_text)
                            post_info['sentiment_polarity'] = blob.sentiment.polarity
                            post_info['sentiment_subjectivity'] = blob.sentiment.subjectivity
                    
                    posts.append(post_info)
                
                time.sleep(0.5)  # Rate limiting for Reddit
        
        except Exception as e:
            logger.warning(f"Failed to fetch Reddit data for {symbol}: {e}")
        
        return posts
    
    def _fetch_stocktwits_sentiment(self, symbol: str) -> List[Dict]:
        """Fetch StockTwits sentiment data."""
        posts = []
        
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            response = self._make_request(url)
            data = response.json()
            
            for message in data.get('messages', []):
                post_info = {
                    'symbol': symbol,
                    'platform': 'stocktwits',
                    'text': message.get('body', ''),
                    'created_at': message.get('created_at', ''),
                    'user_followers': message.get('user', {}).get('followers', 0),
                    'likes': message.get('likes', {}).get('total', 0),
                    'official_sentiment': message.get('entities', {}).get('sentiment', {}).get('basic', '')
                }
                
                # Add our own sentiment analysis
                if SENTIMENT_AVAILABLE and post_info['text']:
                    blob = TextBlob(post_info['text'])
                    post_info['sentiment_polarity'] = blob.sentiment.polarity
                    post_info['sentiment_subjectivity'] = blob.sentiment.subjectivity
                
                posts.append(post_info)
        
        except Exception as e:
            logger.warning(f"Failed to fetch StockTwits data for {symbol}: {e}")
        
        return posts
    
    def fetch_data(self, symbols: List[str], data_types: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        Main method to fetch sentiment data.
        
        Args:
            symbols: List of stock symbols
            data_types: Types of sentiment data ('news', 'social')
        """
        data_types = data_types or ['news', 'social']
        all_data = []
        
        if 'news' in data_types:
            news_data = self.fetch_news_sentiment(symbols, **kwargs)
            all_data.append(news_data)
        
        if 'social' in data_types:
            social_data = self.fetch_social_sentiment(symbols, **kwargs)
            all_data.append(social_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate sentiment data."""
        if raw_data.empty:
            return raw_data
        
        cleaned_data = raw_data.copy()
        
        # Convert timestamps
        if 'published_at' in cleaned_data.columns:
            cleaned_data['published_at'] = pd.to_datetime(cleaned_data['published_at'], errors='coerce')
        
        if 'created_at' in cleaned_data.columns:
            cleaned_data['created_at'] = pd.to_datetime(cleaned_data['created_at'], errors='coerce')
        
        if 'created_utc' in cleaned_data.columns:
            cleaned_data['created_utc'] = pd.to_datetime(cleaned_data['created_utc'], unit='s', errors='coerce')
        
        # Remove duplicates
        if 'url' in cleaned_data.columns:
            cleaned_data = cleaned_data.drop_duplicates(subset=['url'])
        
        # Filter out entries with missing sentiment scores
        sentiment_cols = ['sentiment_polarity', 'sentiment_subjectivity']
        for col in sentiment_cols:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data.dropna(subset=[col])
        
        # Remove entries with empty text content
        text_cols = ['title', 'text', 'description', 'content']
        for col in text_cols:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data[cleaned_data[col].str.len() > 10]
        
        return cleaned_data


class SatelliteDataConnector(BaseDataConnector):
    """
    Connector for satellite imagery and geospatial data.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.satellite_sources = {
            'planet': 'https://api.planet.com/data/v1',
            'maxar': 'https://api.maxar.com/v1',
            'sentinel_hub': 'https://services.sentinel-hub.com/api/v1'
        }
    
    def fetch_parking_lot_data(self, locations: List[Dict], date_range: Tuple[str, str]) -> pd.DataFrame:
        """
        Fetch parking lot occupancy data from satellite imagery.
        
        Args:
            locations: List of dictionaries with 'name', 'lat', 'lon', 'symbol'
            date_range: Tuple of start and end dates
        """
        all_data = []
        
        for location in locations:
            try:
                # This would typically involve:
                # 1. Fetching satellite images for the location
                # 2. Running computer vision algorithms to count cars
                # 3. Correlating with retail/business performance
                
                data_point = {
                    'symbol': location.get('symbol'),
                    'location_name': location.get('name'),
                    'latitude': location.get('lat'),
                    'longitude': location.get('lon'),
                    'date': date_range[0],  # Simplified for example
                    'parking_occupancy': np.random.uniform(0.3, 0.9),  # Placeholder
                    'change_from_baseline': np.random.uniform(-0.2, 0.3)
                }
                
                all_data.append(data_point)
                
            except Exception as e:
                logger.warning(f"Failed to fetch satellite data for {location}: {e}")
        
        return pd.DataFrame(all_data)
    
    def fetch_agricultural_data(self, regions: List[Dict], metrics: List[str] = None) -> pd.DataFrame:
        """
        Fetch agricultural monitoring data.
        
        Args:
            regions: List of geographical regions to monitor
            metrics: List of metrics ('ndvi', 'precipitation', 'temperature')
        """
        metrics = metrics or ['ndvi', 'precipitation']
        all_data = []
        
        for region in regions:
            try:
                data_point = {
                    'region_name': region.get('name'),
                    'latitude': region.get('lat'),
                    'longitude': region.get('lon'),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'ndvi': np.random.uniform(0.2, 0.8),  # Normalized Difference Vegetation Index
                    'precipitation_mm': np.random.uniform(0, 50),
                    'temperature_c': np.random.uniform(15, 35),
                    'crop_health_score': np.random.uniform(0.5, 1.0)
                }
                
                all_data.append(data_point)
                
            except Exception as e:
                logger.warning(f"Failed to fetch agricultural data for {region}: {e}")
        
        return pd.DataFrame(all_data)
    
    def fetch_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """
        Main method to fetch satellite data.
        
        Args:
            data_type: Type of satellite data ('parking', 'agriculture', 'shipping')
        """
        if data_type == 'parking':
            return self.fetch_parking_lot_data(**kwargs)
        elif data_type == 'agriculture':
            return self.fetch_agricultural_data(**kwargs)
        else:
            raise ValueError(f"Unsupported satellite data type: {data_type}")
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate satellite data."""
        if raw_data.empty:
            return raw_data
        
        cleaned_data = raw_data.copy()
        
        # Validate coordinates
        if 'latitude' in cleaned_data.columns:
            cleaned_data = cleaned_data[(cleaned_data['latitude'] >= -90) & 
                                      (cleaned_data['latitude'] <= 90)]
        
        if 'longitude' in cleaned_data.columns:
            cleaned_data = cleaned_data[(cleaned_data['longitude'] >= -180) & 
                                      (cleaned_data['longitude'] <= 180)]
        
        # Convert dates
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
        
        # Remove invalid data points
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        cleaned_data = cleaned_data.dropna(subset=numeric_cols)
        
        return cleaned_data


class CreditCardDataConnector(BaseDataConnector):
    """
    Connector for consumer spending and credit card transaction data.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spending_sources = {
            'facteus': 'https://api.facteus.com/v1',
            'earnest': 'https://api.earnestresearch.com/v1',
            'yodlee': 'https://api.yodlee.com/ysl/v1'
        }
    
    def fetch_retail_spending(self, merchants: List[str], date_range: Tuple[str, str]) -> pd.DataFrame:
        """
        Fetch retail spending data for specific merchants.
        
        Args:
            merchants: List of merchant names or categories
            date_range: Tuple of start and end dates
        """
        all_data = []
        
        for merchant in merchants:
            try:
                # Simulate spending data
                dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
                
                for date in dates:
                    data_point = {
                        'merchant': merchant,
                        'date': date,
                        'transaction_count': np.random.poisson(1000),
                        'total_amount': np.random.normal(50000, 15000),
                        'avg_transaction_size': np.random.normal(45, 15),
                        'unique_customers': np.random.poisson(800),
                        'yoy_growth': np.random.normal(0.05, 0.2),
                        'category': self._categorize_merchant(merchant)
                    }
                    
                    all_data.append(data_point)
                
            except Exception as e:
                logger.warning(f"Failed to fetch spending data for {merchant}: {e}")
        
        return pd.DataFrame(all_data)
    
    def fetch_category_spending(self, categories: List[str], regions: List[str] = None) -> pd.DataFrame:
        """
        Fetch spending data by category and region.
        
        Args:
            categories: List of spending categories
            regions: List of geographical regions
        """
        regions = regions or ['US', 'CA', 'EU']
        all_data = []
        
        for category in categories:
            for region in regions:
                try:
                    data_point = {
                        'category': category,
                        'region': region,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'total_spending': np.random.exponential(1000000),
                        'transaction_count': np.random.poisson(10000),
                        'avg_transaction_size': np.random.normal(75, 25),
                        'growth_rate': np.random.normal(0.02, 0.15),
                        'seasonality_factor': np.random.uniform(0.8, 1.2)
                    }
                    
                    all_data.append(data_point)
                
                except Exception as e:
                    logger.warning(f"Failed to fetch category data for {category} in {region}: {e}")
        
        return pd.DataFrame(all_data)
    
    def _categorize_merchant(self, merchant: str) -> str:
        """Categorize merchant by type."""
        categories = {
            'grocery': ['walmart', 'target', 'costco', 'kroger'],
            'restaurant': ['mcdonalds', 'starbucks', 'chipotle', 'subway'],
            'retail': ['amazon', 'apple', 'nike', 'gap'],
            'gas': ['exxon', 'shell', 'bp', 'chevron'],
            'travel': ['uber', 'airbnb', 'booking', 'expedia']
        }
        
        merchant_lower = merchant.lower()
        for category, merchants in categories.items():
            if any(m in merchant_lower for m in merchants):
                return category
        
        return 'other'
    
    def fetch_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """
        Main method to fetch credit card data.
        
        Args:
            data_type: Type of spending data ('retail', 'category')
        """
        if data_type == 'retail':
            return self.fetch_retail_spending(**kwargs)
        elif data_type == 'category':
            return self.fetch_category_spending(**kwargs)
        else:
            raise ValueError(f"Unsupported spending data type: {data_type}")
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate credit card data."""
        if raw_data.empty:
            return raw_data
        
        cleaned_data = raw_data.copy()
        
        # Convert dates
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
        
        # Remove negative spending amounts
        amount_cols = ['total_amount', 'total_spending', 'avg_transaction_size']
        for col in amount_cols:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data[cleaned_data[col] >= 0]
        
        # Remove invalid transaction counts
        if 'transaction_count' in cleaned_data.columns:
            cleaned_data = cleaned_data[cleaned_data['transaction_count'] >= 0]
        
        # Cap extreme growth rates
        if 'growth_rate' in cleaned_data.columns:
            cleaned_data['growth_rate'] = cleaned_data['growth_rate'].clip(-0.5, 2.0)
        
        return cleaned_data


class ESGDataConnector(BaseDataConnector):
    """
    Connector for Environmental, Social, and Governance data.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.esg_sources = {
            'refinitiv': 'https://api.refinitiv.com/data/v1',
            'msci': 'https://api.msci.com/esg/v1',
            'sustainalytics': 'https://api.sustainalytics.com/v1'
        }
    
    def fetch_esg_scores(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch ESG scores for given symbols.
        
        Args:
            symbols: List of stock symbols
        """
        all_data = []
        
        for symbol in symbols:
            try:
                # Simulate ESG data
                data_point = {
                    'symbol': symbol,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'esg_score': np.random.uniform(30, 95),
                    'environmental_score': np.random.uniform(20, 100),
                    'social_score': np.random.uniform(25, 90),
                    'governance_score': np.random.uniform(40, 95),
                    'controversy_score': np.random.uniform(0, 10),
                    'carbon_emissions': np.random.exponential(100000),
                    'water_usage': np.random.exponential(50000),
                    'waste_generated': np.random.exponential(10000),
                    'board_diversity': np.random.uniform(0.1, 0.6),
                    'ceo_pay_ratio': np.random.exponential(200)
                }
                
                all_data.append(data_point)
                
            except Exception as e:
                logger.warning(f"Failed to fetch ESG data for {symbol}: {e}")
        
        return pd.DataFrame(all_data)
    
    def fetch_sustainability_metrics(self, symbols: List[str], metrics: List[str] = None) -> pd.DataFrame:
        """
        Fetch detailed sustainability metrics.
        
        Args:
            symbols: List of stock symbols
            metrics: Specific metrics to fetch
        """
        metrics = metrics or ['carbon_intensity', 'renewable_energy', 'waste_recycling']
        all_data = []
        
        for symbol in symbols:
            for metric in metrics:
                try:
                    data_point = {
                        'symbol': symbol,
                        'metric': metric,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'value': np.random.exponential(100),
                        'unit': self._get_metric_unit(metric),
                        'yoy_change': np.random.normal(0, 0.15),
                        'industry_percentile': np.random.uniform(10, 90),
                        'target_value': np.random.exponential(80),
                        'target_year': np.random.choice([2025, 2030, 2035, 2040])
                    }
                    
                    all_data.append(data_point)
                
                except Exception as e:
                    logger.warning(f"Failed to fetch {metric} for {symbol}: {e}")
        
        return pd.DataFrame(all_data)
    
    def _get_metric_unit(self, metric: str) -> str:
        """Get the unit for a given metric."""
        units = {
            'carbon_intensity': 'tCO2e/revenue',
            'renewable_energy': 'percentage',
            'waste_recycling': 'percentage',
            'water_usage': 'liters/revenue',
            'energy_consumption': 'kWh/revenue'
        }
        return units.get(metric, 'units')
    
    def fetch_data(self, data_type: str, symbols: List[str], **kwargs) -> pd.DataFrame:
        """
        Main method to fetch ESG data.
        
        Args:
            data_type: Type of ESG data ('scores', 'metrics')
            symbols: List of stock symbols
        """
        if data_type == 'scores':
            return self.fetch_esg_scores(symbols)
        elif data_type == 'metrics':
            return self.fetch_sustainability_metrics(symbols, **kwargs)
        else:
            raise ValueError(f"Unsupported ESG data type: {data_type}")
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate ESG data."""
        if raw_data.empty:
            return raw_data
        
        cleaned_data = raw_data.copy()
        
        # Convert dates
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'], errors='coerce')
        
        # Validate score ranges
        score_cols = ['esg_score', 'environmental_score', 'social_score', 'governance_score']
        for col in score_cols:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data[(cleaned_data[col] >= 0) & (cleaned_data[col] <= 100)]
        
        # Remove negative values for physical metrics
        physical_cols = ['carbon_emissions', 'water_usage', 'waste_generated']
        for col in physical_cols:
            if col in cleaned_data.columns:
                cleaned_data = cleaned_data[cleaned_data[col] >= 0]
        
        return cleaned_data


class WebScrapingConnector(BaseDataConnector):
    """
    Connector for web scraping custom data sources.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError("BeautifulSoup is required for web scraping")
    
    def scrape_earnings_transcripts(self, symbols: List[str], quarters: List[str] = None) -> pd.DataFrame:
        """
        Scrape earnings call transcripts.
        
        Args:
            symbols: List of stock symbols
            quarters: List of quarters to scrape
        """
        all_transcripts = []
        
        for symbol in symbols:
            try:
                # This would scrape from sites like SeekingAlpha, Yahoo Finance, etc.
                # Implementation would depend on specific site structure
                
                transcript_data = {
                    'symbol': symbol,
                    'quarter': '2024Q4',  # Placeholder
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'transcript_text': 'Sample earnings transcript...',  # Placeholder
                    'management_tone': np.random.uniform(-1, 1),  # Sentiment analysis
                    'word_count': np.random.randint(5000, 15000),
                    'key_phrases': ['growth', 'revenue', 'margin'],  # NLP extraction
                    'guidance_mentioned': np.random.choice([True, False])
                }
                
                all_transcripts.append(transcript_data)
                
            except Exception as e:
                logger.warning(f"Failed to scrape transcript for {symbol}: {e}")
        
        return pd.DataFrame(all_transcripts)
    
    def scrape_insider_trading(self, symbols: List[str]) -> pd.DataFrame:
        """
        Scrape insider trading data from SEC filings.
        
        Args:
            symbols: List of stock symbols
        """
        all_trades = []
        
        for symbol in symbols:
            try:
                # Scrape from SEC EDGAR or other sources
                trade_data = {
                    'symbol': symbol,
                    'filing_date': datetime.now().strftime('%Y-%m-%d'),
                    'insider_name': 'John Doe',  # Placeholder
                    'title': 'CEO',
                    'transaction_type': np.random.choice(['Purchase', 'Sale']),
                    'shares': np.random.randint(1000, 100000),
                    'price': np.random.uniform(50, 200),
                    'total_value': 0,  # Will be calculated
                    'shares_owned_after': np.random.randint(100000, 1000000)
                }
                
                trade_data['total_value'] = trade_data['shares'] * trade_data['price']
                all_trades.append(trade_data)
                
            except Exception as e:
                logger.warning(f"Failed to scrape insider trading for {symbol}: {e}")
        
        return pd.DataFrame(all_trades)
    
    def fetch_data(self, data_type: str, symbols: List[str], **kwargs) -> pd.DataFrame:
        """
        Main method to fetch web scraped data.
        
        Args:
            data_type: Type of data to scrape ('transcripts', 'insider_trading')
            symbols: List of stock symbols
        """
        if data_type == 'transcripts':
            return self.scrape_earnings_transcripts(symbols, **kwargs)
        elif data_type == 'insider_trading':
            return self.scrape_insider_trading(symbols)
        else:
            raise ValueError(f"Unsupported web scraping data type: {data_type}")
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate web scraped data."""
        if raw_data.empty:
            return raw_data
        
        cleaned_data = raw_data.copy()
        
        # Convert dates
        date_cols = ['date', 'filing_date']
        for col in date_cols:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
        
        # Remove entries with missing critical data
        if 'transcript_text' in cleaned_data.columns:
            cleaned_data = cleaned_data[cleaned_data['transcript_text'].str.len() > 100]
        
        # Validate financial data
        if 'price' in cleaned_data.columns:
            cleaned_data = cleaned_data[cleaned_data['price'] > 0]
        
        if 'shares' in cleaned_data.columns:
            cleaned_data = cleaned_data[cleaned_data['shares'] > 0]
        
        return cleaned_data


class AlternativeDataManager:
    """
    Central manager for all alternative data sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alternative data manager.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config
        self.connectors = {}
        
        # Initialize connectors based on config
        if 'sentiment' in config:
            self.connectors['sentiment'] = SentimentDataConnector(**config['sentiment'])
        
        if 'satellite' in config:
            self.connectors['satellite'] = SatelliteDataConnector(**config['satellite'])
        
        if 'credit_card' in config:
            self.connectors['credit_card'] = CreditCardDataConnector(**config['credit_card'])
        
        if 'esg' in config:
            self.connectors['esg'] = ESGDataConnector(**config['esg'])
        
        if 'web_scraping' in config:
            self.connectors['web_scraping'] = WebScrapingConnector(**config['web_scraping'])
        
        # Setup database for metadata
        self.db_path = config.get('database_path', 'data/alternative_data.db')
        self._setup_database()
    
    def _setup_database(self) -> None:
        """Setup SQLite database for tracking data ingestion."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    symbols TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    records_count INTEGER,
                    file_path TEXT,
                    ingestion_timestamp TEXT,
                    status TEXT
                )
            ''')
    
    def fetch_all_data(self, symbols: List[str], data_sources: List[str] = None,
                      start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all configured sources.
        
        Args:
            symbols: List of stock symbols
            data_sources: List of data sources to fetch from
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        data_sources = data_sources or list(self.connectors.keys())
        all_data = {}
        
        for source in data_sources:
            if source not in self.connectors:
                logger.warning(f"Connector for {source} not configured")
                continue
            
            try:
                logger.info(f"Fetching data from {source}")
                connector = self.connectors[source]
                
                if source == 'sentiment':
                    data = connector.fetch_data(symbols, data_types=['news', 'social'])
                elif source == 'satellite':
                    # Would need location data for symbols
                    locations = [{'name': f'{s} HQ', 'lat': 40.7128, 'lon': -74.0060, 'symbol': s} 
                               for s in symbols]
                    data = connector.fetch_data('parking', locations=locations, 
                                              date_range=(start_date, end_date))
                elif source == 'credit_card':
                    data = connector.fetch_data('retail', merchants=symbols, 
                                              date_range=(start_date, end_date))
                elif source == 'esg':
                    data = connector.fetch_data('scores', symbols=symbols)
                elif source == 'web_scraping':
                    data = connector.fetch_data('transcripts', symbols=symbols)
                else:
                    continue
                
                # Clean the data
                cleaned_data = connector.clean_data(data)
                all_data[source] = cleaned_data
                
                # Log the ingestion
                self._log_ingestion(source, 'mixed', symbols, start_date, end_date,
                                  len(cleaned_data), '', 'success')
                
                # Save the data
                if not cleaned_data.empty:
                    filepath = connector.save_data(cleaned_data, f"{source}_data")
                    logger.info(f"Saved {len(cleaned_data)} records from {source}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data from {source}: {e}")
                self._log_ingestion(source, 'mixed', symbols, start_date, end_date,
                                  0, '', 'failed')
        
        return all_data
    
    def _log_ingestion(self, source_type: str, data_type: str, symbols: List[str],
                      start_date: str, end_date: str, records_count: int,
                      file_path: str, status: str) -> None:
        """Log data ingestion to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO data_ingestion_log 
                (source_type, data_type, symbols, start_date, end_date, 
                 records_count, file_path, ingestion_timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (source_type, data_type, ','.join(symbols), start_date, end_date,
                  records_count, file_path, datetime.now().isoformat(), status))
    
    def get_ingestion_history(self, source_type: str = None, days_back: int = 30) -> pd.DataFrame:
        """Get ingestion history from the database."""
        query = '''
            SELECT * FROM data_ingestion_log 
            WHERE ingestion_timestamp >= ?
        '''
        params = [datetime.now() - timedelta(days=days_back)]
        
        if source_type:
            query += ' AND source_type = ?'
            params.append(source_type)
        
        query += ' ORDER BY ingestion_timestamp DESC'
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)


# Example configuration
EXAMPLE_CONFIG = {
    'sentiment': {
        'api_key': 'your_news_api_key',
        'rate_limit': 60,
        'storage_path': 'data/raw/sentiment'
    },
    'satellite': {
        'api_key': 'your_satellite_api_key',
        'rate_limit': 30,
        'storage_path': 'data/raw/satellite'
    },
    'credit_card': {
        'api_key': 'your_spending_api_key',
        'rate_limit': 100,
        'storage_path': 'data/raw/credit_card'
    },
    'esg': {
        'api_key': 'your_esg_api_key',
        'rate_limit': 50,
        'storage_path': 'data/raw/esg'
    },
    'web_scraping': {
        'rate_limit': 10,
        'storage_path': 'data/raw/web_scraping'
    },
    'database_path': 'data/alternative_data.db'
}