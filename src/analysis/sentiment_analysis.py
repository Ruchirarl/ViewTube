"""
Sentiment Analysis Module

This module handles sentiment analysis of YouTube comments using multiple approaches.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on YouTube comments.
    """
    
    def __init__(self, use_transformers: bool = False):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            use_transformers (bool): Whether to use transformers for sentiment analysis
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.use_transformers = use_transformers
        
        if use_transformers:
            try:
                # Lazy import to avoid hard dependency when not used
                from transformers import pipeline  # type: ignore
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("Transformers pipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize transformers pipeline: {e}")
                self.use_transformers = False
    
    def analyze_comments(self, df: pd.DataFrame, text_column: str = 'text_clean') -> pd.DataFrame:
        """
        Perform sentiment analysis on comments.
        
        Args:
            df (pd.DataFrame): Comments data
            text_column (str): Column containing comment text
            
        Returns:
            pd.DataFrame: Comments data with sentiment scores
        """
        logger.info("Starting sentiment analysis...")
        
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in dataframe")
            return df
        
        df_with_sentiment = df.copy()
        
        # TextBlob sentiment
        df_with_sentiment['textblob_polarity'] = df_with_sentiment[text_column].apply(self._get_textblob_sentiment)
        df_with_sentiment['textblob_subjectivity'] = df_with_sentiment[text_column].apply(self._get_textblob_subjectivity)
        
        # VADER sentiment
        df_with_sentiment['vader_compound'] = df_with_sentiment[text_column].apply(self._get_vader_sentiment)
        df_with_sentiment['vader_positive'] = df_with_sentiment[text_column].apply(self._get_vader_positive)
        df_with_sentiment['vader_negative'] = df_with_sentiment[text_column].apply(self._get_vader_negative)
        df_with_sentiment['vader_neutral'] = df_with_sentiment[text_column].apply(self._get_vader_neutral)
        
        # Transformers sentiment (if available)
        if self.use_transformers:
            df_with_sentiment['transformer_sentiment'] = df_with_sentiment[text_column].apply(self._get_transformer_sentiment)
        
        # Create sentiment categories
        df_with_sentiment = self._create_sentiment_categories(df_with_sentiment)
        
        logger.info(f"Sentiment analysis completed for {len(df_with_sentiment)} comments")
        return df_with_sentiment
    
    def _get_textblob_sentiment(self, text: str) -> float:
        """
        Get TextBlob polarity score.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Polarity score (-1 to 1)
        """
        try:
            if pd.isna(text) or text.strip() == '':
                return 0.0
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0
    
    def _get_textblob_subjectivity(self, text: str) -> float:
        """
        Get TextBlob subjectivity score.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Subjectivity score (0 to 1)
        """
        try:
            if pd.isna(text) or text.strip() == '':
                return 0.0
            return TextBlob(text).sentiment.subjectivity
        except:
            return 0.0
    
    def _get_vader_sentiment(self, text: str) -> float:
        """
        Get VADER compound sentiment score.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Compound score (-1 to 1)
        """
        try:
            if pd.isna(text) or text.strip() == '':
                return 0.0
            return self.vader_analyzer.polarity_scores(text)['compound']
        except:
            return 0.0
    
    def _get_vader_positive(self, text: str) -> float:
        """Get VADER positive score."""
        try:
            if pd.isna(text) or text.strip() == '':
                return 0.0
            return self.vader_analyzer.polarity_scores(text)['pos']
        except:
            return 0.0
    
    def _get_vader_negative(self, text: str) -> float:
        """Get VADER negative score."""
        try:
            if pd.isna(text) or text.strip() == '':
                return 0.0
            return self.vader_analyzer.polarity_scores(text)['neg']
        except:
            return 0.0
    
    def _get_vader_neutral(self, text: str) -> float:
        """Get VADER neutral score."""
        try:
            if pd.isna(text) or text.strip() == '':
                return 0.0
            return self.vader_analyzer.polarity_scores(text)['neu']
        except:
            return 0.0
    
    def _get_transformer_sentiment(self, text: str) -> str:
        """
        Get transformer-based sentiment label.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Sentiment label
        """
        try:
            if pd.isna(text) or text.strip() == '':
                return 'neutral'
            
            # Truncate text if too long for transformer
            if len(text) > 500:
                text = text[:500]
            
            result = self.transformer_pipeline(text)[0]
            return result['label'].lower()
        except:
            return 'neutral'
    
    def _create_sentiment_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment categories based on scores.
        
        Args:
            df (pd.DataFrame): Data with sentiment scores
            
        Returns:
            pd.DataFrame: Data with sentiment categories
        """
        # TextBlob categories
        df['textblob_sentiment'] = pd.cut(
            df['textblob_polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        # VADER categories
        df['vader_sentiment'] = pd.cut(
            df['vader_compound'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Combined sentiment (using VADER as primary)
        df['overall_sentiment'] = df['vader_sentiment']
        
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sentiment analysis.
        
        Args:
            df (pd.DataFrame): Comments data with sentiment scores
            
        Returns:
            Dict: Sentiment summary
        """
        summary = {}
        
        # Sentiment distribution
        if 'overall_sentiment' in df.columns:
            sentiment_counts = df['overall_sentiment'].value_counts()
            summary['sentiment_distribution'] = sentiment_counts.to_dict()
            
            # Sentiment percentages
            total_comments = len(df)
            summary['sentiment_percentages'] = {
                'positive': (sentiment_counts.get('positive', 0) / total_comments) * 100,
                'neutral': (sentiment_counts.get('neutral', 0) / total_comments) * 100,
                'negative': (sentiment_counts.get('negative', 0) / total_comments) * 100
            }
        
        # Average scores
        score_columns = ['textblob_polarity', 'textblob_subjectivity', 'vader_compound']
        for col in score_columns:
            if col in df.columns:
                summary[f'avg_{col}'] = df[col].mean()
                summary[f'std_{col}'] = df[col].std()
        
        # Sentiment by engagement
        if 'like_count' in df.columns and 'overall_sentiment' in df.columns:
            sentiment_engagement = df.groupby('overall_sentiment')['like_count'].agg(['mean', 'count']).round(2)
            summary['sentiment_by_engagement'] = sentiment_engagement.to_dict('index')
        
        return summary
    
    def analyze_sentiment_trends(self, df: pd.DataFrame, time_column: str = 'published_at') -> Dict:
        """
        Analyze sentiment trends over time.
        
        Args:
            df (pd.DataFrame): Comments data with sentiment scores
            time_column (str): Column containing timestamp
            
        Returns:
            Dict: Sentiment trends analysis
        """
        if time_column not in df.columns:
            logger.warning(f"Time column '{time_column}' not found")
            return {}
        
        df_copy = df.copy()
        df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        
        # Daily sentiment trends
        daily_sentiment = df_copy.groupby(df_copy[time_column].dt.date).agg({
            'vader_compound': 'mean',
            'textblob_polarity': 'mean',
            'overall_sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral'
        }).rename(columns={'overall_sentiment': 'dominant_sentiment'})
        
        # Monthly sentiment trends
        monthly_sentiment = df_copy.groupby(df_copy[time_column].dt.to_period('M')).agg({
            'vader_compound': 'mean',
            'textblob_polarity': 'mean'
        })
        
        return {
            'daily_trends': daily_sentiment.to_dict('index'),
            'monthly_trends': monthly_sentiment.to_dict('index'),
            'overall_trend': {
                'trend_direction': 'increasing' if monthly_sentiment['vader_compound'].iloc[-1] > monthly_sentiment['vader_compound'].iloc[0] else 'decreasing',
                'trend_strength': abs(monthly_sentiment['vader_compound'].iloc[-1] - monthly_sentiment['vader_compound'].iloc[0])
            }
        }
    
    def get_sentiment_insights(self, summary: Dict) -> List[str]:
        """
        Generate insights from sentiment analysis.
        
        Args:
            summary (Dict): Sentiment summary
            
        Returns:
            List[str]: List of insights
        """
        insights = []
        
        if 'sentiment_percentages' in summary:
            percentages = summary['sentiment_percentages']
            
            if percentages['positive'] > 60:
                insights.append("😊 Excellent! Over 60% of comments are positive")
            elif percentages['positive'] < 30:
                insights.append("😔 Low positive sentiment - consider addressing audience concerns")
            
            if percentages['negative'] > 30:
                insights.append("⚠️ High negative sentiment - review content and address issues")
        
        if 'avg_vader_compound' in summary:
            avg_sentiment = summary['avg_vader_compound']
            if avg_sentiment > 0.2:
                insights.append("✅ Strong positive community sentiment")
            elif avg_sentiment < -0.2:
                insights.append("❌ Negative community sentiment detected")
        
        return insights 