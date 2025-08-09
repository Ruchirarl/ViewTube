"""
Configuration Management Module

This module handles configuration settings for the YouTube analysis project.
"""

import os
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration management class for YouTube analysis project.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file (str, optional): Path to configuration file
        """
        self.config_file = config_file or 'config.json'
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            'api_key': os.getenv('YOUTUBE_API_KEY', ''),
            'search_query': 'day in the life vlogs',
            'max_results': 200,
            'output_file': 'youtube_data.csv',
            'data_dir': 'data',
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'models_dir': 'data/models',
            'charts_dir': 'charts',
            'log_level': 'INFO',
            'save_charts': True,
            'use_transformers': False,
            'sentiment_analysis': {
                'textblob': True,
                'vader': True,
                'transformers': False
            },
            'visualization': {
                'figsize': [12, 8],
                'dpi': 300,
                'style': 'seaborn-v0_8'
            },
            'analysis': {
                'engagement_threshold': 0.05,
                'viral_threshold_percentile': 95,
                'top_n_performers': 10
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    default_config.update(file_config)
                    logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}. Using defaults.")
        else:
            logger.info("No config file found. Using default configuration.")
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key (str): Configuration key
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key
            value (Any): Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_api_key(self) -> str:
        """
        Get YouTube API key.
        
        Returns:
            str: API key
        """
        api_key = self.get('api_key')
        if not api_key:
            logger.warning("No API key found in configuration")
        return api_key
    
    def get_search_query(self) -> str:
        """
        Get search query for video collection.
        
        Returns:
            str: Search query
        """
        return self.get('search_query', 'day in the life vlogs')
    
    def get_max_results(self) -> int:
        """
        Get maximum number of results to collect.
        
        Returns:
            int: Maximum results
        """
        return self.get('max_results', 200)
    
    def get_output_file(self) -> str:
        """
        Get output file name.
        
        Returns:
            str: Output file name
        """
        return self.get('output_file', 'youtube_data.csv')
    
    def get_data_dirs(self) -> Dict[str, str]:
        """
        Get data directory paths.
        
        Returns:
            Dict[str, str]: Directory paths
        """
        return {
            'data_dir': self.get('data_dir', 'data'),
            'raw_data_dir': self.get('raw_data_dir', 'data/raw'),
            'processed_data_dir': self.get('processed_data_dir', 'data/processed'),
            'models_dir': self.get('models_dir', 'data/models'),
            'charts_dir': self.get('charts_dir', 'charts')
        }
    
    def get_sentiment_config(self) -> Dict[str, bool]:
        """
        Get sentiment analysis configuration.
        
        Returns:
            Dict[str, bool]: Sentiment analysis settings
        """
        return self.get('sentiment_analysis', {
            'textblob': True,
            'vader': True,
            'transformers': False
        })
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration.
        
        Returns:
            Dict[str, Any]: Visualization settings
        """
        return self.get('visualization', {
            'figsize': [12, 8],
            'dpi': 300,
            'style': 'seaborn-v0_8'
        })
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis configuration.
        
        Returns:
            Dict[str, Any]: Analysis settings
        """
        return self.get('analysis', {
            'engagement_threshold': 0.05,
            'viral_threshold_percentile': 95,
            'top_n_performers': 10
        })
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            bool: True if valid, False otherwise
        """
        required_keys = ['api_key', 'search_query', 'max_results']
        
        for key in required_keys:
            if not self.get(key):
                logger.error(f"Missing required configuration: {key}")
                return False
        
        # Validate numeric values
        max_results = self.get('max_results')
        if not isinstance(max_results, int) or max_results <= 0:
            logger.error("max_results must be a positive integer")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        dirs = self.get_data_dirs()
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def get_log_level(self) -> str:
        """
        Get logging level.
        
        Returns:
            str: Log level
        """
        return self.get('log_level', 'INFO') 