"""
Data Processor for YouTube Analysis

This module handles data cleaning, processing, and feature engineering.
"""

import pandas as pd
import numpy as np
import isodate
from datetime import datetime
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A class to process and clean YouTube data.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    def clean_video_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process video data.
        
        Args:
            df (pd.DataFrame): Raw video data
            
        Returns:
            pd.DataFrame: Cleaned video data
        """
        logger.info("Cleaning video data...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Convert date columns and ensure timezone-naive
        df_clean['published_at'] = pd.to_datetime(df_clean['published_at']).dt.tz_localize(None)
        
        # Convert duration to seconds
        df_clean['duration_seconds'] = df_clean['duration'].apply(self._parse_duration)
        
        # Convert numeric columns
        numeric_columns = ['view_count', 'like_count', 'comment_count']
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Create engagement metrics
        df_clean = self._create_engagement_metrics(df_clean)
        
        # Clean text columns
        df_clean['title_clean'] = df_clean['title'].apply(self._clean_text)
        df_clean['description_clean'] = df_clean['description'].apply(self._clean_text)
        
        # Extract tags as string
        df_clean['tags_string'] = df_clean['tags'].apply(self._tags_to_string)
        
        # Create time-based features
        df_clean = self._create_time_features(df_clean)
        
        logger.info(f"Cleaned {len(df_clean)} videos")
        return df_clean
    
    def _parse_duration(self, duration: str) -> int:
        """
        Parse ISO 8601 duration to seconds.
        
        Args:
            duration (str): ISO 8601 duration string
            
        Returns:
            int: Duration in seconds
        """
        try:
            if pd.isna(duration) or duration == '':
                return 0
            return int(isodate.parse_duration(duration).total_seconds())
        except:
            return 0
    
    def _create_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement metrics from video data.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            pd.DataFrame: Data with engagement metrics
        """
        # Engagement rate (likes + comments) / views
        df['engagement_rate'] = 0.0  # Initialize with zeros
        mask = df['view_count'] > 0
        df.loc[mask, 'engagement_rate'] = (df.loc[mask, 'like_count'] + df.loc[mask, 'comment_count']) / df.loc[mask, 'view_count']
        
        # Like rate
        df['like_rate'] = 0.0  # Initialize with zeros
        df.loc[mask, 'like_rate'] = df.loc[mask, 'like_count'] / df.loc[mask, 'view_count']
        
        # Comment rate
        df['comment_rate'] = 0.0  # Initialize with zeros
        df.loc[mask, 'comment_rate'] = df.loc[mask, 'comment_count'] / df.loc[mask, 'view_count']
        
        # Views per day (since published)
        # Ensure both datetime objects are timezone-naive for calculation
        current_time = datetime.now()
        df['days_since_published'] = (current_time - df['published_at']).dt.days
        
        # Calculate views per day safely
        df['views_per_day'] = 0.0  # Initialize with zeros
        mask = df['days_since_published'] > 0
        df.loc[mask, 'views_per_day'] = df.loc[mask, 'view_count'] / df.loc[mask, 'days_since_published']
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tags_to_string(self, tags: List) -> str:
        """
        Convert tags list to string.
        
        Args:
            tags (List): List of tags
            
        Returns:
            str: Tags as comma-separated string
        """
        # Handle None, NaN, or empty list
        if tags is None or (isinstance(tags, list) and len(tags) == 0):
            return ""
        return ', '.join(tags)
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from published date.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            pd.DataFrame: Data with time features
        """
        df['published_year'] = df['published_at'].dt.year
        df['published_month'] = df['published_at'].dt.month
        df['published_day'] = df['published_at'].dt.day
        df['published_hour'] = df['published_at'].dt.hour
        df['published_day_of_week'] = df['published_at'].dt.dayofweek
        
        return df
    
    def clean_comments_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process comments data.
        
        Args:
            df (pd.DataFrame): Raw comments data
            
        Returns:
            pd.DataFrame: Cleaned comments data
        """
        logger.info("Cleaning comments data...")
        
        df_clean = df.copy()
        
        # Convert date columns and ensure timezone-naive
        df_clean['published_at'] = pd.to_datetime(df_clean['published_at']).dt.tz_localize(None)
        df_clean['updated_at'] = pd.to_datetime(df_clean['updated_at']).dt.tz_localize(None)
        
        # Clean comment text
        df_clean['text_clean'] = df_clean['text'].apply(self._clean_text)
        
        # Convert numeric columns
        df_clean['like_count'] = pd.to_numeric(df_clean['like_count'], errors='coerce').fillna(0)
        df_clean['total_reply_count'] = pd.to_numeric(df_clean['total_reply_count'], errors='coerce').fillna(0)
        
        # Create comment length feature
        df_clean['comment_length'] = df_clean['text_clean'].str.len()
        
        # Create time-based features
        df_clean['comment_year'] = df_clean['published_at'].dt.year
        df_clean['comment_month'] = df_clean['published_at'].dt.month
        df_clean['comment_day'] = df_clean['published_at'].dt.day
        
        logger.info(f"Cleaned {len(df_clean)} comments")
        return df_clean
    
    def filter_data(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """
        Filter data based on specified criteria.
        
        Args:
            df (pd.DataFrame): Data to filter
            filters (Dict): Filter criteria
            
        Returns:
            pd.DataFrame: Filtered data
        """
        df_filtered = df.copy()
        
        for column, condition in filters.items():
            if column in df_filtered.columns:
                if isinstance(condition, dict):
                    if 'min' in condition:
                        df_filtered = df_filtered[df_filtered[column] >= condition['min']]
                    if 'max' in condition:
                        df_filtered = df_filtered[df_filtered[column] <= condition['max']]
                    if 'values' in condition:
                        df_filtered = df_filtered[df_filtered[column].isin(condition['values'])]
                else:
                    df_filtered = df_filtered[df_filtered[column] == condition]
        
        logger.info(f"Filtered data: {len(df)} -> {len(df_filtered)} rows")
        return df_filtered
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, output_dir: str = 'data/processed'):
        """
        Save processed data to CSV file.
        
        Args:
            df (pd.DataFrame): Processed data
            filename (str): Output filename
            output_dir (str): Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"✅ Success! Processed data saved to '{filepath}'")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise 