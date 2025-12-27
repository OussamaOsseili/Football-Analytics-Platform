"""
Overview Page - Dashboard Homepage
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

st.title("🏠 Dashboard Overview")
st.markdown("### Key Performance Indicators")

@st.cache_data
def load_data():
    try:
        enhanced_path = settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv"
        if enhanced_path.exists():
            return pd.read_csv(enhanced_path)
        return pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats.csv")
    except FileNotFoundError:
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.warning("⚠️ No data available. Please run the ETL pipeline first.")
    st.code("python src\\etl\\etl_pipeline.py")
    st.stop()

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Total Players", len(data))

with col2:
    try:
        match_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        total_matches = match_df['match_id'].nunique()
    except:
        total_matches = data['matches_played'].sum() if 'matches_played' in data.columns else 0
    st.metric("⚽ Total Matches", int(total_matches))

with col3:
    total_goals = data['goals'].sum() if 'goals' in data.columns else 0
    st.metric("🥅 Total Goals", int(total_goals))

with col4:
    avg_goals_90 = data['goals_per90'].mean() if 'goals_per90' in data.columns else 0
    st.metric("📊 Avg Goals/90", f"{avg_goals_90:.3f}")

st.markdown("---")

# Top Scorers
st.subheader("⭐ Top Performers")

col1, col2 = st.columns(2)

with col1:
    if 'goals' in data.columns and 'matches_played' in data.columns:
        st.markdown("**🥇 Top Scorers (best goals/match ratio)**")
        
        # Filter players with at least 3 matches for meaningful ratio
        scorers = data[data['matches_played'] >= 3].copy()
        scorers['goals_per_match'] = scorers['goals'] / scorers['matches_played']
        
        top_scorers = scorers.nlargest(10, 'goals_per_match')[
            ['player_name', 'goals', 'matches_played', 'goals_per_match']
        ].copy()
        
        # Rename columns for display
        top_scorers.columns = ['Player', 'Goals', 'Matches', 'Goals/Match']
        
        # Format the ratio
        top_scorers['Goals/Match'] = top_scorers['Goals/Match'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(top_scorers, hide_index=True, width='stretch')

with col2:
    if 'assists' in data.columns and 'matches_played' in data.columns:
        st.markdown("**🎨 Top Assisters (best assists/match ratio)**")
        
        # Filter players with at least 3 matches
        assisters = data[data['matches_played'] >= 3].copy()
        assisters['assists_per_match'] = assisters['assists'] / assisters['matches_played']
        
        top_assisters = assisters.nlargest(10, 'assists_per_match')[
            ['player_name', 'assists', 'matches_played', 'assists_per_match']
        ].copy()
        
        # Rename columns
        top_assisters.columns = ['Player', 'Assists', 'Matches', 'Assists/Match']
        
        # Format the ratio
        top_assisters['Assists/Match'] = top_assisters['Assists/Match'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(top_assisters, hide_index=True, width='stretch')

st.markdown("---")

# Position Distribution
st.subheader("📊 Position Distribution")

if 'position_category' in data.columns:
    fig = px.pie(data, names='position_category', title='Players by Position', hole=0.3)
    st.plotly_chart(fig, width='stretch')
else:
    st.info("Position categorization will be available after running ML pipeline")

st.success("✅ Dashboard loaded successfully! Explore pages from sidebar →")
