"""
Helper Functions Module

This module contains utility functions for the YouTube analysis project.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with appropriate suffix (K, M, B).
    
    Args:
        num (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if num >= 1e9:
        return f"{num/1e9:.{decimals}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{decimals}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def clean_text(text: str) -> str:
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


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text (str): Text to extract keywords from
        min_length (int): Minimum keyword length
        
    Returns:
        List[str]: List of keywords
    """
    if pd.isna(text):
        return []
    
    # Clean text
    text = clean_text(text.lower())
    
    # Split into words
    words = text.split()
    
    # Filter by length and common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    return keywords


def calculate_percentile_rank(values: pd.Series, value: float) -> float:
    """
    Calculate percentile rank of a value in a series.
    
    Args:
        values (pd.Series): Series of values
        value (float): Value to find rank for
        
    Returns:
        float: Percentile rank (0-100)
    """
    return (values < value).mean() * 100


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in data using various methods.
    
    Args:
        data (pd.Series): Data series
        method (str): Method to use ('iqr', 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")


def create_summary_stats(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics for a dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to analyze
        numeric_columns (List[str], optional): Specific numeric columns to analyze
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': {}
    }
    
    for col in numeric_columns:
        if col in df.columns:
            summary['numeric_columns'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'missing': df[col].isnull().sum(),
                'outliers_iqr': detect_outliers(df[col], 'iqr').sum(),
                'outliers_zscore': detect_outliers(df[col], 'zscore').sum()
            }
    
    return summary


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that dataframe has required columns.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (List[str]): List of required columns
        
    Returns:
        bool: True if valid, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info("Dataframe validation passed")
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division by zero
        
    Returns:
        float: Result of division or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def convert_duration_to_minutes(duration_str: str) -> float:
    """
    Convert ISO 8601 duration string to minutes.
    
    Args:
        duration_str (str): ISO 8601 duration string
        
    Returns:
        float: Duration in minutes
    """
    try:
        import isodate
        duration = isodate.parse_duration(duration_str)
        return duration.total_seconds() / 60
    except:
        return 0.0


def get_top_n_by_metric(df: pd.DataFrame, metric: str, n: int = 10, 
                        group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Get top N rows by metric, optionally grouped.
    
    Args:
        df (pd.DataFrame): Dataframe
        metric (str): Metric column name
        n (int): Number of top rows to return
        group_by (str, optional): Column to group by
        
    Returns:
        pd.DataFrame: Top N rows
    """
    if group_by:
        return df.groupby(group_by)[metric].sum().nlargest(n).reset_index()
    else:
        return df.nlargest(n, metric)


def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create time-based features from date column.
    
    Args:
        df (pd.DataFrame): Dataframe
        date_column (str): Date column name
        
    Returns:
        pd.DataFrame: Dataframe with time features
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
    df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
    df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
    df_copy[f'{date_column}_hour'] = df_copy[date_column].dt.hour
    df_copy[f'{date_column}_day_of_week'] = df_copy[date_column].dt.dayofweek
    df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
    
    return df_copy


def calculate_rolling_stats(df: pd.DataFrame, value_column: str, time_column: str, 
                          window: int = 7) -> pd.DataFrame:
    """
    Calculate rolling statistics for time series data.
    
    Args:
        df (pd.DataFrame): Dataframe
        value_column (str): Value column to calculate stats for
        time_column (str): Time column
        window (int): Rolling window size
        
    Returns:
        pd.DataFrame: Dataframe with rolling statistics
    """
    df_copy = df.copy()
    df_copy[time_column] = pd.to_datetime(df_copy[time_column])
    df_copy = df_copy.sort_values(time_column)
    
    df_copy[f'{value_column}_rolling_mean'] = df_copy[value_column].rolling(window=window).mean()
    df_copy[f'{value_column}_rolling_std'] = df_copy[value_column].rolling(window=window).std()
    df_copy[f'{value_column}_rolling_min'] = df_copy[value_column].rolling(window=window).min()
    df_copy[f'{value_column}_rolling_max'] = df_copy[value_column].rolling(window=window).max()
    
    return df_copy


def print_summary_table(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """
    Print a formatted summary table.
    
    Args:
        df (pd.DataFrame): Dataframe to summarize
        title (str): Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing Values:")
    for col, missing in df.isnull().sum().items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    print(f"{'='*50}\n") 