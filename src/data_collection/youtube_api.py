"""
YouTube API Data Collector

This module handles all YouTube API interactions for data collection.
"""

import os
import pandas as pd
import time
from googleapiclient.discovery import build
from getpass import getpass
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDataCollector:
    """
    A class to collect YouTube video data using the YouTube Data API v3.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTube Data Collector.
        
        Args:
            api_key (str, optional): YouTube API key. If not provided, will prompt user.
        """
        self.api_key = api_key or self._get_api_key()
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
    def _get_api_key(self) -> str:
        """Get API key from environment or user input."""
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            api_key = getpass('Enter your YouTube API Key: ')
        return api_key
    
    def get_niche_videos(self, query: str, max_results: int = 200) -> Tuple[pd.DataFrame, Dict]:
        """
        Get videos for a specific niche/query.
        
        Args:
            query (str): Search query for the niche
            max_results (int): Maximum number of videos to collect
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Video data and category mapping
        """
        logger.info(f"Collecting videos for query: '{query}'")
        
        video_data = []
        category_map = {}
        
        try:
            # Search for videos with pagination
            video_ids = []
            next_page_token = None
            while len(video_ids) < max_results:
                search_response = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=min(50, max_results - len(video_ids)),  # API limit per request
                    type='video',
                    order='relevance',
                    pageToken=next_page_token
                ).execute()

                video_ids.extend([item['id']['videoId'] for item in search_response.get('items', [])])
                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            # Get detailed video information
            for i in range(0, len(video_ids), 50):  # Process in batches of 50
                batch_ids = video_ids[i:i+50]
                
                videos_response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch_ids)
                ).execute()
                
                for video in videos_response['items']:
                    video_info = self._extract_video_info(video)
                    video_data.append(video_info)
                    
                    # Store category mapping
                    category_id = video['snippet']['categoryId']
                    category_name = video['snippet'].get('categoryTitle', f'Category {category_id}')
                    category_map[category_id] = category_name
                
                # Light rate limiting
                time.sleep(0.1)
            
            df = pd.DataFrame(video_data)
            logger.info(f"Successfully collected {len(df)} videos")
            
            return df, category_map
            
        except Exception as e:
            logger.error(f"Error collecting videos: {str(e)}")
            raise
    
    def _extract_video_info(self, video: Dict) -> Dict:
        """
        Extract relevant information from a video object.
        
        Args:
            video (Dict): Video object from YouTube API
            
        Returns:
            Dict: Extracted video information
        """
        snippet = video['snippet']
        statistics = video.get('statistics', {})
        content_details = video.get('contentDetails', {})
        
        return {
            'video_id': video['id'],
            'title': snippet['title'],
            'description': snippet['description'],
            'channel_id': snippet['channelId'],
            'channel_title': snippet['channelTitle'],
            'published_at': snippet['publishedAt'],
            'category_id': snippet['categoryId'],
            'tags': snippet.get('tags', []),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'duration': content_details.get('duration', ''),
            'definition': content_details.get('definition', ''),
            'caption': content_details.get('caption', ''),
            'default_language': snippet.get('defaultLanguage', ''),
            'default_audio_language': snippet.get('defaultAudioLanguage', ''),
            'thumbnails': snippet.get('thumbnails', {}),
            'live_broadcast_content': snippet.get('liveBroadcastContent', 'none')
        }
    
    def get_video_comments(self, video_id: str, max_results: int = 100) -> pd.DataFrame:
        """
        Get comments for a specific video.
        
        Args:
            video_id (str): YouTube video ID
            max_results (int): Maximum number of comments to collect
            
        Returns:
            pd.DataFrame: Comments data
        """
        logger.info(f"Collecting comments for video: {video_id}")
        
        comments_data = []
        
        try:
            comments_response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(max_results, 100),  # API limit
                order='relevance'
            ).execute()
            
            for comment in comments_response['items']:
                snippet = comment['snippet']['topLevelComment']['snippet']
                
                comment_info = {
                    'comment_id': comment['snippet']['topLevelComment']['id'],
                    'video_id': video_id,
                    'author_name': snippet['authorDisplayName'],
                    'author_channel_id': snippet.get('authorChannelId', {}).get('value', ''),
                    'text': snippet['textDisplay'],
                    'like_count': snippet['likeCount'],
                    'published_at': snippet['publishedAt'],
                    'updated_at': snippet['updatedAt'],
                    'total_reply_count': comment['snippet']['totalReplyCount']
                }
                
                comments_data.append(comment_info)
            
            df = pd.DataFrame(comments_data)
            logger.info(f"Successfully collected {len(df)} comments")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting comments: {str(e)}")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str, output_dir: str = 'data/raw'):
        """
        Save data to CSV file.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
            output_dir (str): Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"✅ Success! Data saved to '{filepath}'")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise 