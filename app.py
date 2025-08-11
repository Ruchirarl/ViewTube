#!/usr/bin/env python3
"""
ViewTube

A Streamlit web application for analyzing YouTube video data.
Users can input a search query and get comprehensive analysis for specified number of videos.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from pathlib import Path
# (Optional) logging can be enabled for debugging
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_collection.youtube_api import YouTubeDataCollector
from src.data_collection.data_processor import DataProcessor
from src.analysis.video_analysis import VideoAnalyzer
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.analysis.keyword_analysis import KeywordAnalyzer
from src.utils.config import Config
from src.utils.helpers import format_number
from dotenv import load_dotenv

# Configure page with YouTube-inspired dark theme
st.set_page_config(
    page_title="ViewTube",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# YouTube-inspired dark theme CSS
st.markdown("""
<style>
    /* Main theme colors - YouTube Classic Palette */
    :root {
        --primary-accent: #FF0000;
        --accent-light: #FF4444;
        --accent-dark: #CC0000;
        --text-primary: #FFFFFF;
        --text-secondary: #CCCCCC;
        --background: #000000;
        --card-background: #1A1A1A;
        --borders-lines: #333333;
        --highlight-glow: #FF6666;
        --gradient-start: #FF0000;
        --gradient-end: #CC0000;
        --success-color: #00CC00;
        --warning-color: #FFCC00;
        --error-color: #FF0000;
        --youtube-dark: #000000;
        --youtube-gray: #1A1A1A;
        --youtube-light-gray: #CCCCCC;
        --youtube-white: #FFFFFF;
    }
    
    /* Global styles */
    .main {
        background-color: var(--background);
        color: var(--text-primary);
    }
    
    .stApp {
        background-color: var(--background);
    }
    
    /* Header styling */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.025em;
    }
    
    .youtube-text {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .analysis-text {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Input form styling */
    .input-container {
        background: linear-gradient(145deg, var(--card-background), rgba(30, 41, 59, 0.8));
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid var(--borders-lines);
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.05);
        position: relative;
        z-index: 1;
        backdrop-filter: blur(10px);
    }
    
    .form-title {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, var(--card-background), rgba(30, 41, 59, 0.6));
        padding: 2.5rem 2rem;
        border-radius: 20px;
        border: 1px solid var(--borders-lines);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
        opacity: 0.8;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
        border-color: var(--primary-accent);
    }
    
    .metric-number {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.75rem;
        letter-spacing: -0.025em;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    
    .metric-description {
        font-size: 0.95rem;
        color: var(--text-secondary);
        opacity: 0.8;
        font-weight: 400;
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(135deg, var(--success-color), rgba(16, 185, 129, 0.8));
        color: var(--text-primary);
        padding: 1.25rem;
        border-radius: 15px;
        border: 1px solid var(--success-color);
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    
    .error-message {
        background: linear-gradient(135deg, var(--error-color), rgba(239, 68, 68, 0.8));
        color: var(--text-primary);
        padding: 1.25rem;
        border-radius: 15px;
        border: 1px solid var(--error-color);
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-accent);
        color: var(--text-primary);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: var(--card-background);
        color: var(--text-primary);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        color: var(--text-primary);
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, var(--gradient-end), var(--gradient-start));
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: linear-gradient(145deg, var(--card-background), rgba(30, 41, 59, 0.8));
        color: var(--text-primary);
        border: 1px solid var(--borders-lines);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-accent);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        background: linear-gradient(145deg, var(--card-background), rgba(30, 41, 59, 0.9));
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .input-container {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--card-background);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--gradient-end), var(--gradient-start));
    }
    
    /* Custom tooltip styling */
    .custom-tooltip {
        background: linear-gradient(145deg, var(--card-background), rgba(30, 41, 59, 0.9));
        color: var(--text-primary);
        border: 1px solid var(--borders-lines);
        border-radius: 12px;
        padding: 12px;
        font-size: 13px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_collected' not in st.session_state:
    st.session_state.data_collected = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header"><span class="youtube-text">View</span><span class="analysis-text">Tube</span></h1>', unsafe_allow_html=True)
    
    # Instruction line
    st.markdown('<p class="header-subtitle">Enter your search query below to analyze YouTube video performance</p>', unsafe_allow_html=True)
    
    # API Key input (env or manual)
    api_key_default = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))
    api_key = st.text_input(
        "YouTube API Key",
        value=api_key_default,
        type="password",
        help="Set YOUTUBE_API_KEY in a .env file to avoid entering it each time",
        placeholder="Enter your API key"
    )

    # Create columns for horizontal layout
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input(
            "Search Query",
            value="day in the life vlogs",
            help="Enter the search term for YouTube videos",
            placeholder="Enter search query..."
        )
    
    with col2:
        max_results = st.slider(
            "Videos",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of videos to analyze"
        )
    
    
    # Collect data button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Collect & Analyze Data", type="primary", use_container_width=True):
            if not api_key:
                st.error("🔑 Please provide a YouTube API key (set YOUTUBE_API_KEY in .env or enter above).")
                st.stop()

            with st.spinner("Collecting and analyzing data..."):
                try:
                    # Store inputs for reporting
                    st.session_state.search_query = search_query
                    st.session_state.max_results = max_results
                    collect_and_analyze_data(
                        api_key, search_query, max_results, 
                        True  # Always include sentiment analysis
                    )
                    st.session_state.data_collected = True
                    st.success("Data collection and analysis completed!")
                except Exception as e:
                    error_msg = str(e)
                    if "quota" in error_msg.lower() or "quota exceeded" in error_msg.lower():
                        st.session_state["api_quota_exceeded"] = True
                        st.error("🚫 API Quota Exceeded! The daily YouTube API limit has been reached. Please try again tomorrow.")
                    elif "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                        st.error("🔑 Invalid API Key! Please check the API key configuration.")
                    else:
                        st.error(f"Error: {error_msg}")
    
    # Main content area
    if st.session_state.data_collected and st.session_state.processed_data is not None:
        display_results()
    else:
        display_welcome()
    
    # Footer
    st.markdown("---")
    with st.expander("About this dashboard"):
        st.markdown(
            """
            - **Engagement rate** = (Likes + Comments) / Views
            - **Like rate** = Likes / Views
            - **Comment rate** = Comments / Views
            - **Views/day** = Views divided by days since publish
            - Charts are interactive; hover for details and exact values
            - The app uses YouTube Data API v3 and may hit daily quota limits
            """
        )
    if st.session_state.get("api_quota_exceeded"):
        st.warning("API quota is exhausted for today. Try again after the daily reset.")

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_collect_videos(api_key: str, search_query: str, max_results: int):
    collector = YouTubeDataCollector(api_key=api_key)
    return collector.get_niche_videos(search_query, max_results)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_get_comments(api_key: str, video_id: str, max_comments: int):
    collector = YouTubeDataCollector(api_key=api_key)
    return collector.get_video_comments(video_id, max_results=max_comments)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_process_videos(videos_df):
    processor = DataProcessor()
    return processor.clean_video_data(videos_df)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_analyze_videos(processed_videos):
    analyzer = VideoAnalyzer()
    return analyzer.analyze_videos(processed_videos)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_process_comments(comments_df):
    processor = DataProcessor()
    return processor.clean_comments_data(comments_df)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_analyze_sentiment(processed_comments):
    sentiment_analyzer = SentimentAnalyzer(use_transformers=False)
    return sentiment_analyzer.analyze_comments(processed_comments)

def collect_and_analyze_data(api_key, search_query, max_results, include_sentiment):
    """Collect and analyze YouTube data."""
    
    # Initialize components
    config = Config()
    config.set('api_key', api_key)
    config.set('search_query', search_query)
    config.set('max_results', max_results)
    
    # Use cached helpers for API and compute
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Collect data
    status_text.text("Step 1/4: Collecting video data...")
    progress_bar.progress(25)
    
    try:
        videos_df, category_map = _cached_collect_videos(api_key, search_query, max_results)
        
        if videos_df.empty:
            raise Exception("No videos found for the given search query.")
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "quota exceeded" in error_msg.lower():
            raise Exception("API quota exceeded. Daily limit reached.")
        elif "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise Exception("Invalid API key configuration.")
        else:
            raise e
    
    # Step 2: Process data
    status_text.text("Step 2/4: Processing data...")
    progress_bar.progress(50)
    
    processed_videos = _cached_process_videos(videos_df)
    st.session_state.processed_data = processed_videos
    
    # Step 3: Analyze videos
    status_text.text("Step 3/4: Analyzing videos...")
    progress_bar.progress(75)
    
    analysis_results = _cached_analyze_videos(processed_videos)
    st.session_state.analysis_results = analysis_results
    
    # Step 4: Optional sentiment analysis
    if include_sentiment:
        status_text.text("Step 4/4: Analyzing sentiment...")
        progress_bar.progress(90)
        
        # Get comments for top videos
        top_videos = processed_videos.nlargest(min(5, len(processed_videos)), 'view_count')
        
        all_comments = []
        for _, video in top_videos.iterrows():
            try:
                comments_df = _cached_get_comments(api_key, video['video_id'], max_comments=50)
                if not comments_df.empty:
                    comments_df['video_title'] = video['title']
                    all_comments.append(comments_df)
            except Exception as e:
                st.warning(f"Could not fetch comments for video: {video['title']}")
        
        if all_comments:
            combined_comments = pd.concat(all_comments, ignore_index=True)
            processed_comments = _cached_process_comments(combined_comments)
            comments_with_sentiment = _cached_analyze_sentiment(processed_comments)
            
            st.session_state.comments_data = comments_with_sentiment
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")

def display_welcome():
    """Display welcome message and instructions."""
    
    st.markdown("""
    ## 🎯 Welcome to ViewTube
    
    This interactive dashboard allows you to analyze YouTube videos based on your search queries.
    You can get comprehensive insights about video performance, engagement metrics, and more!
    
    ### 📋 What you can do:
    - **Search Videos**: Enter any search query to find relevant YouTube videos
    - **Analyze Performance**: Get detailed analysis of view counts, likes, comments, and engagement
    - **▲ Analysis**: Analyze comment sentiment (optional)
    - **Visualizations**: View interactive charts and insights
    
    ### 🚀 Getting Started:
    1. **Search**: Enter any search query for YouTube videos
    2. **Configure**: Set the number of videos to analyze
    3. **Analyze**: Click "Collect & Analyze Data" to start the analysis
    4. **Explore**: View results in the interactive dashboard below
    
    ### 📊 Sample Analysis:
    - **Engagement Metrics**: Like rate, comment rate, engagement rate
    - **Performance Analysis**: View trends, viral videos, top performers
    - **Time Patterns**: Upload time analysis, duration patterns
    - **Category Insights**: Performance by video category
    - **Sentiment Trends**: Comment sentiment distribution
    
    ---
    """)

def display_results():
    """Display analysis results."""
    
    processed_data = st.session_state.processed_data
    analysis_results = st.session_state.analysis_results
    
    # Render all sections (no navigation buttons)
    def show(section_key: str) -> bool:
        return True

    # Helper to avoid duplicate element IDs for charts
    def next_plot_key() -> str:
        import time as _t
        return f"plt_{_t.time_ns()}"
    
    # Key Metrics Cards at the top
    if show('Key'):
        st.header("▲ Key Metrics")
    
    # First row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{len(processed_data)}</div>
            <div class="metric-label">Total Videos</div>
            <div class="metric-description">analyzed</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        avg_views = format_number(processed_data['view_count'].mean())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{avg_views}</div>
            <div class="metric-label">Avg Views</div>
            <div class="metric-description">per video</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_engagement = f"{processed_data['engagement_rate'].mean():.2%}"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{avg_engagement}</div>
            <div class="metric-label">Avg Engagement</div>
            <div class="metric-description">rate</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_duration_minutes = processed_data['duration_seconds'].mean() / 60
        avg_duration = f"{avg_duration_minutes:.1f} min"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{avg_duration}</div>
            <div class="metric-label">Avg Duration</div>
            <div class="metric-description">per video</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Add spacing between metric rows
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_views = format_number(processed_data['view_count'].max())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{max_views}</div>
            <div class="metric-label">Highest Views</div>
            <div class="metric-description">single video</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        max_engagement = f"{processed_data['engagement_rate'].max():.2%}"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{max_engagement}</div>
            <div class="metric-label">Best Engagement</div>
            <div class="metric-description">single video</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        total_likes = format_number(processed_data['like_count'].sum())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{total_likes}</div>
            <div class="metric-label">Total Likes</div>
            <div class="metric-description">all videos</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        total_comments = format_number(processed_data['comment_count'].sum())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{total_comments}</div>
            <div class="metric-label">Total Comments</div>
            <div class="metric-description">all videos</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Analysis sections ordered by importance
    st.markdown("---")

    # (Sentiment will appear later in its original place)
    
    # 1. Top Performing Videos (Most Important)
    if show('Top'):
        st.header("■ Top Performing Videos")
        st.markdown("Here are the videos with the highest view counts:")
    
    # Top videos by views
    top_videos = processed_data.nlargest(10, 'view_count')[['video_id','title', 'view_count', 'like_count', 'comment_count', 'engagement_rate']]
    
    # Format the data for display
    display_data = top_videos.copy()
    display_data['url'] = display_data['video_id'].apply(lambda vid: f"https://www.youtube.com/watch?v={vid}")
    display_data['thumbnail'] = display_data['video_id'].apply(lambda vid: f"https://img.youtube.com/vi/{vid}/hqdefault.jpg")
    display_data['view_count'] = display_data['view_count'].apply(format_number)
    display_data['like_count'] = display_data['like_count'].apply(format_number)
    display_data['comment_count'] = display_data['comment_count'].apply(format_number)
    display_data['engagement_rate'] = display_data['engagement_rate'].apply(lambda x: f"{x:.2%}")
    
    # Back to table with small thumbnails
    # Build markdown with HTML img tag for small thumbs
    table_df = display_data.copy()
    table_df['thumbnail'] = table_df['thumbnail'].apply(lambda src: f"<img src='{src}' width='80'>")
    table_df['title'] = table_df.apply(lambda r: f"<a href='{r['url']}' target='_blank'>{r['title'][:70]}</a>", axis=1)
    st.write(
        table_df[['thumbnail','title','view_count','like_count','comment_count','engagement_rate']].to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # 2. Engagement Analysis (Second Most Important)
    if show('Engagement'):
        st.header("🎯 Engagement Analysis")
        st.markdown("How well do these videos engage their audience?")
    
    # Engagement metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_engagement = processed_data['engagement_rate'].mean()
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{avg_engagement:.2%}</div>
            <div class="metric-label">Average Engagement</div>
            <div class="metric-description">across all videos</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        max_engagement = processed_data['engagement_rate'].max()
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{max_engagement:.2%}</div>
            <div class="metric-label">Highest Engagement</div>
            <div class="metric-description">best performing video</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        engagement_std = processed_data['engagement_rate'].std()
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-number">{engagement_std:.2%}</div>
            <div class="metric-label">Engagement Variation</div>
            <div class="metric-description">standard deviation</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Big engagement charts
    st.markdown("---")
    col_lr, col_cr = st.columns(2)

    with col_lr:
        st.subheader("↗ Like Rate Analysis")
        st.markdown("How many likes do videos get compared to their views?")
        fig = px.histogram(
            processed_data,
            x='like_rate',
            nbins=15,
            title="Distribution of Like Rates",
            labels={'like_rate': 'Like Rate (%)', 'count': 'Number of Videos'},
            color_discrete_sequence=['#FF0000'],
            opacity=0.8
        )
        fig.update_traces(hovertemplate='Like rate: %{x:.1%}<br>Count: %{y}<extra></extra>')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font_size=18,
            title_font_color='white',
            height=450,
            hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)', font_size=14, font_color='white')
        )
        fig.update_xaxes(tickformat='.0%', gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True, key=next_plot_key())

    with col_cr:
        st.subheader("◆ Comment Rate Analysis")
        st.markdown("How many comments do videos get compared to their views?")
        fig = px.histogram(
            processed_data,
            x='comment_rate',
            nbins=15,
            title="Distribution of Comment Rates",
            labels={'comment_rate': 'Comment Rate (%)', 'count': 'Number of Videos'},
            color_discrete_sequence=['#FF4444'],
            opacity=0.8
        )
        fig.update_traces(hovertemplate='Comment rate: %{x:.1%}<br>Count: %{y}<extra></extra>')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font_size=18,
            title_font_color='white',
            height=450,
            hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)', font_size=14, font_color='white')
        )
        fig.update_xaxes(tickformat='.0%', gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Engagement insights
    if 'engagement_insights' in analysis_results:
        st.markdown("---")
        st.subheader("💡 Key Insights")
        st.markdown("Here's what we found about engagement:")
        
        insights = analysis_results['engagement_insights']
        for insight in insights[:5]:  # Show top 5 insights
            st.info(f"• {insight}")
    
    st.markdown("---")
    
    # 3. View Analysis (Third Most Important)
    if show('Views'):
        st.header("📈 View Analysis")
        st.markdown("How do views perform across these videos?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("● View Distribution")
        st.markdown("How many views do most videos get?")
        
        # Ensure numeric
        df_views = processed_data.copy()
        df_views['view_count'] = pd.to_numeric(df_views['view_count'], errors='coerce')
        df_views = df_views[df_views['view_count'].notna()]

        if df_views.empty:
            st.info("No view data available to plot.")
        else:
            # Prefer ECDF; fallback to histogram if ECDF fails
            try:
                fig = px.ecdf(
                    df_views,
                    x='view_count',
                    title="Cumulative Distribution of Video Views",
                    labels={'view_count': 'Number of Views', 'y': 'Cumulative Fraction'},
                )
                fig.update_traces(line_color='#FF6666')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    title_font_size=18,
                    title_font_color='white',
                    height=400,
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)', tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Fallback: simple histogram (linear scale)
                fig = px.histogram(
                    df_views,
                    x='view_count',
                    nbins=30,
                    title="Distribution of Video Views",
                    labels={'view_count': 'Number of Views', 'count': 'Number of Videos'},
                    color_discrete_sequence=['#FF6666'],
                    opacity=0.85
                )
                fig.update_traces(hovertemplate='Views: %{x:,}<br>Count: %{y}<extra></extra>')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    title_font_size=18,
                    title_font_color='white',
                    height=400,
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("▼ Views vs Engagement")
        st.markdown("Do videos with more views get more engagement?")
        
        # Bubble chart for Views vs Engagement
        fig = px.scatter(
            processed_data,
            x='view_count',
            y='engagement_rate',
            hover_data=['title'],
            title="Views vs Engagement Rate",
            labels={'view_count': 'Number of Views', 'engagement_rate': 'Engagement Rate (%)'},
            color='engagement_rate',
            color_continuous_scale='Viridis',
            size='like_count',
            size_max=30
        )
        fig.update_traces(marker=dict(opacity=0.7), hovertemplate='<b>%{customdata[0]}</b><br>Views: %{x:,}<br>Engagement: %{y:.1%}<br>Likes: %{marker.size:,}<extra></extra>')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font_size=18,
            title_font_color='white',
            height=420,
            hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)', font_size=14, font_color='white')
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 4. Time Analysis (Fourth Most Important)
    if show('Time'):
        st.header("◯ Time Analysis")
        st.markdown("How do timing factors affect video performance?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("◊ Video Duration")
        st.markdown("How long are the videos?")
        
        # Convert duration_seconds to minutes for visualization
        data_with_duration = processed_data.copy()
        data_with_duration['duration_minutes'] = data_with_duration['duration_seconds'] / 60
        
        # Box plot for video durations
        fig = px.box(
            data_with_duration,
            x='duration_minutes',
            title="Video Duration Distribution (Box)",
            labels={'duration_minutes': 'Duration (minutes)'},
            color_discrete_sequence=['#FF0000']
        )
        fig.update_traces(hovertemplate='Duration: %{x:.1f} min<extra></extra>')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font_size=18,
            title_font_color='white',
            height=400,
            hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)', font_size=14, font_color='white')
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("♦ Duration vs Views")
        st.markdown("Do longer videos get more views?")
        
        # Scatter (keep) with improved tooltip
        fig = px.scatter(
            data_with_duration,
            x='duration_minutes',
            y='view_count',
            hover_data=['title'],
            title="Duration vs Views",
            labels={'duration_minutes': 'Duration (minutes)', 'view_count': 'Number of Views'},
            color='engagement_rate',
            color_continuous_scale='Viridis',
            size='like_count',
            size_max=22
        )
        fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Duration: %{x:.1f} min<br>Views: %{y:,}<br>Engagement: %{marker.color:.1f}%<br>Likes: %{marker.size:,}<extra></extra>')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font_size=18,
            title_font_color='white',
            height=420,
            hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)', font_size=14, font_color='white')
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Upload time analysis (if available)
    if 'published_hour' in processed_data.columns:
        st.markdown("---")
        st.subheader("◉ Upload Time Analysis")
        st.markdown("When are videos typically uploaded?")
        
        upload_hour_counts = processed_data['published_hour'].value_counts().sort_index()
        
        # Create a simple bar chart
        fig = px.bar(
            x=[f"{hour}:00" for hour in upload_hour_counts.index],
            y=upload_hour_counts.values,
            title="Videos Uploaded by Hour (24-hour clock)",
            labels={'x': 'Hour of Day', 'y': 'Number of Videos'},
            color=upload_hour_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font_size=18,
            title_font_color='white',
            height=450
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)

        # Content Calendar: Recommend top publish slots (hour, weekday)
        if 'published_day_of_week' in processed_data.columns:
            st.subheader("🗓️ Recommended Publish Windows")
            # Score by average views/day if available, else by views
            score_col = 'views_per_day' if 'views_per_day' in processed_data.columns else 'view_count'
            slot_scores = processed_data.groupby(['published_day_of_week','published_hour'])[score_col].mean().reset_index()
            # Top 5 slots
            top_slots = slot_scores.sort_values(score_col, ascending=False).head(5)
            day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            top_slots['slot'] = top_slots.apply(lambda r: f"{day_names[int(r['published_day_of_week'])]} @ {int(r['published_hour']):02d}:00", axis=1)
            st.dataframe(top_slots[['slot', score_col]].rename(columns={score_col: 'expected_avg'}), use_container_width=True)
            # Save for report
            st.session_state.report_top_slots = top_slots[['slot', score_col]].rename(columns={score_col: 'expected_avg'})
    

    
    # 5. Keyword Analysis (N-gram lift)
    if show('Keywords'):
        st.markdown("---")
        st.header("◆ Keyword Analysis")
        st.markdown("Which words or phrases in titles/descriptions/tags associate with higher views per day?")

    with st.expander("Configure keyword analysis"):
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            text_source = st.selectbox("Text Source", ["title", "description", "tags"], index=0)
        with col_cfg2:
            n_options = st.multiselect("n-gram sizes", options=[1, 2, 3], default=[1, 2])
        with col_cfg3:
            min_support = st.slider("Min support (videos)", min_value=2, max_value=20, value=3, step=1)

        col_sw1, col_sw2, col_sw3 = st.columns(3)
        with col_sw1:
            stopword_input = st.text_input(
                "Extra stopwords (comma-separated)",
                value="",
                help="These will be removed in addition to built-in common words",
            )
        with col_sw2:
            min_token_len = st.slider("Min token length", min_value=1, max_value=6, value=3, step=1)
        with col_sw3:
            remove_numeric = st.checkbox("Remove numeric tokens", value=True)

    try:
        kw_analyzer = KeywordAnalyzer()
        extra_sw = [s.strip().lower() for s in stopword_input.split(",") if s.strip()]

        # Run for selected n sizes and combine
        frames = []
        for n in (n_options or [1, 2]):
            df_part = kw_analyzer.compute_ngram_lift(
                processed_data,
                text_source=text_source,
                n=int(n),
                min_support=min_support,
                metric="views_per_day",
                max_results=50,
                stopwords=extra_sw,
                min_token_length=min_token_len,
                remove_numeric_tokens=remove_numeric,
            )
            if not df_part.empty:
                df_part["n"] = int(n)
                frames.append(df_part)

        kw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if not kw_df.empty:
            # Save for report
            st.session_state.report_kw = kw_df.copy()
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.bar(
                    kw_df.sort_values("lift", ascending=True),
                    x="lift",
                    y="ngram",
                    orientation="h",
                    title="Top n-grams by lift on views/day",
                    labels={"lift": "Lift vs baseline", "ngram": "n-gram"},
                    color="n" if ("n" in kw_df.columns) else "lift",
                    color_continuous_scale="Viridis" if ("n" not in kw_df.columns) else None,
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=12),
                    title_font_size=18,
                    title_font_color='white',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
                st.metric("Baseline views/day", f"{kw_df['baseline_avg'].iloc[0]:.1f}")
                st.metric("Best lift", f"{kw_df['lift'].max():.2f}×")

            # Opportunities: Topic recommendations (composite score)
            st.subheader("🚀 Opportunities: Topic Recommendations")
            kw_display = kw_df.copy()
            if 'count' not in kw_display.columns:
                # If count not present (should be), approximate by 1
                kw_display['count'] = 1
            kw_display['opportunity_score'] = (kw_display['lift'] * kw_display['count']).round(2)
            kw_display['estimated_uplift_%'] = ((kw_display['lift'] - 1.0) * 100.0).round(1)
            cols_to_show = [col for col in ['n','ngram','count','lift','estimated_uplift_%','opportunity_score'] if col in kw_display.columns]
            st.dataframe(
                kw_display.sort_values(['opportunity_score','lift','count'], ascending=[False, False, False])[cols_to_show].head(15),
                use_container_width=True
            )
            st.markdown("#### n-gram table")
            st.dataframe(
                kw_df.assign(
                    avg_metric=lambda d: d["avg_metric"].round(1),
                    baseline_avg=lambda d: d["baseline_avg"].round(1),
                    lift=lambda d: d["lift"].round(2),
                ),
                use_container_width=True,
            )
        else:
            st.info("No n-grams met the minimum support or baseline is zero.")
    except Exception as e:
        st.warning(f"Keyword analysis unavailable: {e}")

    # 6. Sentiment Analysis Section
    if show('Sentiment'):
        st.markdown("---")
        st.header("▲ Comment Sentiment Analysis")
        st.markdown("What do viewers think about these videos?")
    
    if 'comments_data' in st.session_state:
        comments_data = st.session_state.comments_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("◇ Sentiment Distribution")
            st.markdown("How positive or negative are the comments?")
            
            if 'overall_sentiment' in comments_data.columns:
                sentiment_counts = comments_data['overall_sentiment'].value_counts()
                
                # Create a simple pie chart
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Comment Sentiment Distribution",
                    color_discrete_sequence=['#FF0000', '#FF4444', '#CC0000']  # YouTube red, light red, dark red
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    title_font_size=18,
                    title_font_color='white',
                    height=480,
                    hoverlabel=dict(bgcolor='rgba(26,26,26,0.95)', font_size=14, font_color='white')
                )
                fig.update_traces(hovertemplate='%{label}: %{percent} (%{value})<extra></extra>')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sentiment categories not available.")
        
        with col2:
            st.subheader("◆ Sentiment Scores")
            st.markdown("Distribution of sentiment scores")
            
            if 'vader_compound' in comments_data.columns:
                fig = px.histogram(
                    comments_data,
                    x='vader_compound',
                    nbins=20,
                    title="Sentiment Score Distribution",
                    labels={'vader_compound': 'Sentiment Score', 'count': 'Number of Comments'},
                    color_discrete_sequence=['#FF6666'],  # YouTube red
                    opacity=0.8
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14),
                    title_font_size=18,
                    title_font_color='white',
                    height=480
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Sentiment scores not available.")
        
        # Sentiment by video
        if 'video_title' in comments_data.columns and 'vader_compound' in comments_data.columns:
            st.markdown("---")
            st.subheader("◐ Sentiment by Video")
            st.markdown("Which videos have the most positive comments?")
            
            video_sentiment = comments_data.groupby('video_title')['vader_compound'].mean().sort_values(ascending=False)
            
            # Create a horizontal bar chart for better readability
            fig = px.bar(
                x=video_sentiment.values,
                y=video_sentiment.index,
                orientation='h',
                title="Average Sentiment by Video",
                labels={'x': 'Average Sentiment Score', 'y': 'Video Title'},
                color=video_sentiment.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                title_font_size=18,
                title_font_color='white',
                height=400
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("💡 Sentiment analysis is being processed... Please wait for the analysis to complete.")
    
    st.markdown("---")
    
    # 7. Summary Section
    if show('Summary'):
        st.header("📋 Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Findings")
        
        total_videos = len(processed_data)
        avg_views = format_number(processed_data['view_count'].mean())
        avg_engagement = f"{processed_data['engagement_rate'].mean():.2%}"
        avg_duration = f"{processed_data['duration_seconds'].mean() / 60:.1f} minutes"
        
        st.markdown(f"""
        - **Total Videos Analyzed**: {total_videos}
        - **Average Views**: {avg_views}
        - **Average Engagement**: {avg_engagement}
        - **Average Duration**: {avg_duration}
        """)
    
    with col2:
        st.subheader("📊 Top Performers")
        top_3 = processed_data.nlargest(3, 'view_count')
        for i, (_, video) in enumerate(top_3.iterrows(), 1):
            st.markdown(f"**{i}.** {video['title'][:50]}... ({format_number(video['view_count'])} views)")
    
    st.markdown("---")
    st.markdown("**Analysis completed!** Use the configuration above to analyze different search queries.")

    # Download Report (HTML)
    st.subheader("⬇ Download Report (HTML)")
    def build_html_report():
        import html
        q = html.escape(str(st.session_state.get('search_query', '')))
        mr = st.session_state.get('max_results', 0)
        total_videos = len(processed_data)
        avg_views = processed_data['view_count'].mean()
        avg_eng = processed_data['engagement_rate'].mean()
        avg_dur_min = processed_data['duration_seconds'].mean() / 60 if 'duration_seconds' in processed_data.columns else None
        top_table = processed_data.nlargest(10, 'view_count')[['title','view_count','like_count','comment_count','engagement_rate']].copy()
        def fmt_num(x):
            try:
                return f"{int(x):,}"
            except:
                return str(x)
        top_table['view_count'] = top_table['view_count'].apply(fmt_num)
        top_table['like_count'] = top_table['like_count'].apply(fmt_num)
        top_table['comment_count'] = top_table['comment_count'].apply(fmt_num)
        top_table['engagement_rate'] = top_table['engagement_rate'].apply(lambda x: f"{x:.2%}")
        top_html_rows = "".join([
            f"<tr><td>{html.escape(str(r['title']))}</td><td>{r['view_count']}</td><td>{r['like_count']}</td><td>{r['comment_count']}</td><td>{r['engagement_rate']}</td></tr>"
            for _, r in top_table.iterrows()
        ])
        kw_df = st.session_state.get('report_kw', None)
        kw_html = ""
        if kw_df is not None and not kw_df.empty:
            kw_part = kw_df.sort_values(['lift','count'], ascending=[False,False]).head(15)[['ngram','count','lift']]
            kw_html_rows = "".join([
                f"<tr><td>{html.escape(str(r['ngram']))}</td><td>{int(r['count'])}</td><td>{r['lift']:.2f}×</td></tr>"
                for _, r in kw_part.iterrows()
            ])
            kw_html = f"""
            <h3>Top Keyword Opportunities</h3>
            <table border='1' cellpadding='6' cellspacing='0'>
              <tr><th>n‑gram</th><th>Support</th><th>Lift</th></tr>
              {kw_html_rows}
            </table>
            """
        slots_df = st.session_state.get('report_top_slots', None)
        slots_html = ""
        if slots_df is not None and not slots_df.empty:
            slot_rows = "".join([f"<tr><td>{html.escape(str(r['slot']))}</td><td>{r['expected_avg']:.1f}</td></tr>" for _, r in slots_df.iterrows()])
            slots_html = f"""
            <h3>Recommended Publish Windows</h3>
            <table border='1' cellpadding='6' cellspacing='0'>
              <tr><th>Slot</th><th>Expected Avg</th></tr>
              {slot_rows}
            </table>
            """
        html_report = f"""
        <html><head><meta charset='utf-8'><title>ViewTube Report</title></head>
        <body style='font-family:Arial, sans-serif;'>
          <h1>ViewTube Report</h1>
          <p><b>Query:</b> {q} &nbsp; <b>Videos requested:</b> {mr} &nbsp; <b>Analyzed:</b> {total_videos}</p>
          <h2>Key Metrics</h2>
          <ul>
            <li>Average views: {avg_views:,.0f}</li>
            <li>Average engagement: {avg_eng:.2%}</li>
            <li>Average duration: {avg_dur_min:.1f} minutes</li>
          </ul>
          <h2>Top Performing Videos</h2>
          <table border='1' cellpadding='6' cellspacing='0'>
            <tr><th>Title</th><th>Views</th><th>Likes</th><th>Comments</th><th>Engagement</th></tr>
            {top_html_rows}
          </table>
          {kw_html}
          {slots_html}
          <p style='margin-top:24px;'>Generated by ViewTube.</p>
        </body></html>
        """
        return html_report

    report_html = build_html_report()
    st.download_button(
        label="Download HTML Report",
        data=report_html,
        file_name="viewtube_report.html",
        mime="text/html"
    )

if __name__ == "__main__":
    main() 