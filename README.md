# ViewTube - YouTube analytics dashboard

<p align="center">
  <a href="https://streamlit.io/" target="_blank"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"></a>
  <a href="https://www.python.org/" target="_blank"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://plotly.com/python/" target="_blank"><img alt="Plotly" src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"></a>
  <a href="https://developers.google.com/youtube/v3" target="_blank"><img alt="YouTube Data API" src="https://img.shields.io/badge/YouTube%20Data%20API-v3-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
  <a href="https://pandas.pydata.org/" target="_blank"><img alt="Pandas" src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"></a>
  <a href="https://numpy.org/" target="_blank"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"></a>
</p>

An interactive Streamlit dashboard that collects, analyzes, and visualizes YouTube video data using the YouTube Data API v3. Built with a YouTube classic theme and focused on actionable insights for creators.

## Live Demo

<p align="center">
  <img width="1896" height="869" alt="Dashboard Preview" src="https://github.com/user-attachments/assets/f258e6a9-5865-4ec6-96b1-07394412dc77" />
  <br/>
  <a href="[https://viewtube.streamlit.app/]" target="_blank"><b>▶ Open the live dashboard</b></a>
</p>

## Sample Report

<p align="center">
  <img width="1903" height="991" alt="image" src="https://github.com/user-attachments/assets/067aa789-4bc0-46c0-8a75-fec117a1d878" />
</p>

## Overview

This project is a full, end‑to‑end YouTube analytics dashboard that helps you quickly understand what drives performance for a given topic. It collects videos via the YouTube Data API, cleans and enriches the data (engagement metrics, views/day, time features), and presents interactive visuals that highlight top performers, engagement patterns, keyword opportunities, sentiment trends, and recommended publish windows. The UI is optimized for clarity and fast decision‑making.

- Accelerates content strategy by identifying high‑lift keywords and top publish windows (weekday × hour) so creators can target the best topics at the best times.
- Surfaces engagement drivers (duration, title/tags, timing) to guide edits that increase watch‑through and interactions.
- Quantifies audience sentiment by category to spot at‑risk topics and prioritize fixes.
- Reduces manual analysis time with an executive‑style dashboard, enabling more experiments per week and faster iteration cycles.

## What’s Inside

- Streamlit UI with a YouTube classic theme for high readability
- Plotly interactive visuals
- YouTube Data API v3 ingestion with pagination (up to 200 videos/run)
- Processing: text cleaning, engagement metrics, views/day, time features (hour, weekday)
- Sentiment analysis: VADER/TextBlob, plus per‑video distributions
- Keyword lift and topic recommendations (opportunity score)
- One‑click HTML report export (executive summary: highlights, KPIs, top videos, categories, publish windows)
- Caching for faster re‑runs and fewer API calls; Streamlit Cloud–ready (Secrets for API key)

## Metrics & Formulas

- **Engagement rate**: (Likes + Comments) ÷ Views
- **Views/day**: Total views ÷ Days since publish
- **Keyword lift**: Avg(views/day for videos containing an n‑gram) ÷ Baseline avg(views/day)
- **Opportunity score**: lift × count, where count = number of videos that contain the n‑gram

## Project Structure

```
youtube-analysis/
├── src/
│   ├── data_collection/
│   │   ├── youtube_api.py
│   │   └── data_processor.py
│   ├── analysis/
│   │   ├── video_analysis.py
│   │   ├── sentiment_analysis.py
│   │   └── keyword_analysis.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── notebooks/
│   └── Youtube_Analysis.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
└── README.md
```
## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd youtube-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API key (recommended: .env):
```bash
cp .env.example .env
# Edit .env and set:
# YOUTUBE_API_KEY=your_api_key
```

## Usage

### Streamlit Dashboard
Run the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

In the app:
- **Interactive Search**: Enter any search query and specify number of videos
- **Real-time Analysis**: Get instant insights on video performance
- **Visualizations**: Interactive charts and graphs
- **Sentiment Analysis**: Optional comment sentiment analysis

> Programmatic usage examples were removed to keep this repo focused on the Streamlit app experience.

### Deploy on Streamlit Cloud

1. Push this repo to a public GitHub repository
2. Go to Streamlit Community Cloud → New app → select your repo/branch → file path: `app.py`
3. In Settings → Secrets, add:
   - `YOUTUBE_API_KEY = your_api_key`
4. Deploy. You’ll get a permanent URL like `https://your-app.streamlit.app`

## Configuration

- On Streamlit Cloud: set `YOUTUBE_API_KEY` in Secrets (Settings → Secrets).
- Local development: optionally create a `.env` file with:

## Analytics Overview

- Key Metrics: Total videos, avg views, engagement, avg duration
- Top Performers: Highest views, likes, comments
- Engagement: Like/comment rate distributions, insights; beeswarm by category
- Views: View distribution (log), views vs engagement
- Time: Duration distribution, duration vs views, upload hours; recommended publish windows
- Keywords: n‑gram lift (treemap) with opportunity score and estimated uplift
- Sentiment: Distribution, scores, per‑video sentiment; category
- Report: One‑click HTML report export with highlights, top videos, category performance, and best publish windows

## Notebook

- `notebooks/Youtube_Analysis.ipynb` contains a clean, sectioned walkthrough of the workflow (no outputs committed).

 
