"""
Video Analysis Module

This module handles video metrics analysis and performance evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    A class to analyze YouTube video data and extract insights.
    """
    
    def __init__(self):
        """Initialize the VideoAnalyzer."""
        pass
    
    def analyze_videos(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive analysis of video data.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Starting video analysis...")
        
        results = {
            'summary_stats': self._get_summary_statistics(df),
            'engagement_analysis': self._analyze_engagement(df),
            'performance_metrics': self._analyze_performance(df),
            'time_analysis': self._analyze_time_patterns(df),
            'category_analysis': self._analyze_categories(df),
            'top_performers': self._get_top_performers(df)
        }
        
        logger.info("Video analysis completed")
        return results
    
    def _get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for video data.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Summary statistics
        """
        numeric_columns = ['view_count', 'like_count', 'comment_count', 'duration_seconds']
        
        summary = {}
        for col in numeric_columns:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
        
        # Additional metrics
        summary['total_videos'] = len(df)
        summary['unique_channels'] = df['channel_id'].nunique()
        summary['date_range'] = {
            'earliest': df['published_at'].min(),
            'latest': df['published_at'].max()
        }
        
        return summary
    
    def _analyze_engagement(self, df: pd.DataFrame) -> Dict:
        """
        Analyze engagement metrics.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Engagement analysis
        """
        engagement_cols = ['engagement_rate', 'like_rate', 'comment_rate']
        
        analysis = {}
        for col in engagement_cols:
            if col in df.columns:
                analysis[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'top_10_percentile': df[col].quantile(0.9),
                    'distribution': {
                        'low': len(df[df[col] <= df[col].quantile(0.33)]),
                        'medium': len(df[(df[col] > df[col].quantile(0.33)) & (df[col] <= df[col].quantile(0.66))]),
                        'high': len(df[df[col] > df[col].quantile(0.66)])
                    }
                }
        
        # Engagement correlation analysis
        if 'view_count' in df.columns and 'engagement_rate' in df.columns:
            analysis['correlation_with_views'] = df['view_count'].corr(df['engagement_rate'])
        
        return analysis
    
    def _analyze_performance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze video performance metrics.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Performance analysis
        """
        analysis = {}
        
        # Views analysis
        if 'view_count' in df.columns:
            analysis['views'] = {
                'total_views': df['view_count'].sum(),
                'avg_views': df['view_count'].mean(),
                'views_per_day': df['views_per_day'].mean() if 'views_per_day' in df.columns else None,
                'viral_threshold': df['view_count'].quantile(0.95)  # Top 5% as viral
            }
        
        # Duration analysis
        if 'duration_seconds' in df.columns:
            analysis['duration'] = {
                'avg_duration_minutes': df['duration_seconds'].mean() / 60,
                'optimal_duration': df.groupby(pd.cut(df['duration_seconds'], bins=10))['view_count'].mean().idxmax(),
                'duration_engagement_correlation': df['duration_seconds'].corr(df['engagement_rate']) if 'engagement_rate' in df.columns else None
            }
        
        return analysis
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze time-based patterns in video publishing.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Time analysis
        """
        analysis = {}
        
        # Publishing time analysis
        if 'published_hour' in df.columns:
            hour_analysis = df.groupby('published_hour').agg({
                'view_count': 'mean',
                'engagement_rate': 'mean',
                'video_id': 'count'
            }).rename(columns={'video_id': 'video_count'})
            
            analysis['hourly_patterns'] = {
                'peak_hour_views': hour_analysis['view_count'].idxmax(),
                'peak_hour_engagement': hour_analysis['engagement_rate'].idxmax(),
                'most_active_hour': hour_analysis['video_count'].idxmax()
            }
        
        # Day of week analysis
        if 'published_day_of_week' in df.columns:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_analysis = df.groupby('published_day_of_week').agg({
                'view_count': 'mean',
                'engagement_rate': 'mean',
                'video_id': 'count'
            }).rename(columns={'video_id': 'video_count'})
            
            analysis['daily_patterns'] = {
                'best_day_views': day_names[day_analysis['view_count'].idxmax()],
                'best_day_engagement': day_names[day_analysis['engagement_rate'].idxmax()],
                'most_active_day': day_names[day_analysis['video_count'].idxmax()]
            }
        
        return analysis
    
    def _analyze_categories(self, df: pd.DataFrame) -> Dict:
        """
        Analyze video categories.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Category analysis
        """
        analysis = {}
        
        if 'category_id' in df.columns:
            category_stats = df.groupby('category_id').agg({
                'view_count': ['mean', 'sum', 'count'],
                'engagement_rate': 'mean',
                'like_rate': 'mean',
                'comment_rate': 'mean'
            }).round(4)
            
            # Flatten column names
            category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
            
            analysis['category_performance'] = category_stats.to_dict('index')
            analysis['top_categories_by_views'] = category_stats['view_count_sum'].nlargest(5).to_dict()
            analysis['top_categories_by_engagement'] = category_stats['engagement_rate_mean'].nlargest(5).to_dict()
        
        return analysis
    
    def _get_top_performers(self, df: pd.DataFrame) -> Dict:
        """
        Get top performing videos and channels.
        
        Args:
            df (pd.DataFrame): Video data
            
        Returns:
            Dict: Top performers
        """
        top_performers = {}
        
        # Top videos by views
        if 'view_count' in df.columns:
            top_performers['top_videos_by_views'] = df.nlargest(10, 'view_count')[['title', 'channel_title', 'view_count', 'engagement_rate']].to_dict('records')
        
        # Top videos by engagement
        if 'engagement_rate' in df.columns:
            top_performers['top_videos_by_engagement'] = df.nlargest(10, 'engagement_rate')[['title', 'channel_title', 'engagement_rate', 'view_count']].to_dict('records')
        
        # Top channels
        if 'channel_title' in df.columns and 'view_count' in df.columns:
            channel_stats = df.groupby('channel_title').agg({
                'view_count': 'sum',
                'video_id': 'count',
                'engagement_rate': 'mean'
            }).rename(columns={'video_id': 'video_count'})
            
            top_performers['top_channels_by_views'] = channel_stats.nlargest(10, 'view_count').to_dict('index')
            top_performers['top_channels_by_engagement'] = channel_stats.nlargest(10, 'engagement_rate').to_dict('index')
        
        return top_performers
    
    def generate_insights(self, analysis_results: Dict) -> List[str]:
        """
        Generate actionable insights from analysis results.
        
        Args:
            analysis_results (Dict): Analysis results
            
        Returns:
            List[str]: List of insights
        """
        insights = []
        
        # Engagement insights
        if 'engagement_analysis' in analysis_results:
            engagement = analysis_results['engagement_analysis']
            if 'engagement_rate' in engagement:
                avg_engagement = engagement['engagement_rate']['mean']
                insights.append(f"Average engagement rate: {avg_engagement:.4f}")
                
                if avg_engagement < 0.05:
                    insights.append("⚠️ Engagement rate is below 5% - consider improving content quality")
                elif avg_engagement > 0.1:
                    insights.append("✅ High engagement rate - content is resonating well with audience")
        
        # Performance insights
        if 'performance_metrics' in analysis_results:
            performance = analysis_results['performance_metrics']
            if 'views' in performance:
                avg_views = performance['views']['avg_views']
                insights.append(f"Average views per video: {avg_views:,.0f}")
                
                if avg_views < 1000:
                    insights.append("📈 Consider improving video titles and thumbnails to increase views")
                elif avg_views > 10000:
                    insights.append("🎉 Excellent view performance - content is highly discoverable")
        
        # Time pattern insights
        if 'time_analysis' in analysis_results:
            time_analysis = analysis_results['time_analysis']
            if 'hourly_patterns' in time_analysis:
                peak_hour = time_analysis['hourly_patterns']['peak_hour_views']
                insights.append(f"📅 Best time to publish: {peak_hour}:00")
        
        return insights 