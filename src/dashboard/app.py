"""
Football Analytics Dashboard - Main Application
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Football Analytics Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    /* Force metric labels to be dark */
    .stMetric label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    /* Force metric values to be dark blue */
    .stMetric [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    /* Alternative selector for values */
    .stMetric div[data-testid="stMetricValue"] > div {
        color: #1f77b4 !important;
    }
    h1 {
        color: #1f77b4;
    }
    h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("⚽ Navigation")
st.sidebar.markdown("---")

pages = {
    "🏠 Overview": "Overview",
    "👤 Player Profile": "Player_Profile",
    "⚔️ Match Analysis": "Match_Analysis",
    "🔄 Player Comparison": "Comparison",
    "🔮 Predictions": "Predictions",
    "🎯 Clusters & Styles": "Clusters",
    "🔍 Scouting": "Scouting",
    "🤝 Team Analysis": "Team_Analysis",
    "📊 Tactical Board": "Tactical_Board",
    "📈 Temporal Trends": "Temporal_Trends"
}

# Main page
st.title("⚽ Football Analytics Platform")
st.markdown("### Advanced Player Performance Analysis & Prediction")
st.markdown("---")

st.info("""
**Welcome to the Football Analytics Platform!**

This platform provides comprehensive analysis of football player performance using StatsBomb data (2022-2024).

**Features:**
- 📊 Multi-dimensional performance analysis
- 🎯 15+ playing style classifications
- 🤖 AI-generated insights
- 📈 Temporal trend analysis
- 🔮 Performance predictions
- 📄 Professional PDF reports
        
👈 **Select a page** from the sidebar to get started!
""") 

# Quick stats
@st.cache_data
def load_quick_stats():
    try:
        from config import settings
        
        # Load season stats for player count
        enhanced_path = settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv"
        if enhanced_path.exists():
            df = pd.read_csv(enhanced_path)
        else:
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats.csv")
        
        # Load match stats for UNIQUE match count
        match_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        unique_matches = match_df['match_id'].nunique()
        
        return {
            'total_players': len(df),
            'unique_matches': unique_matches,
            'total_goals': int(df['goals'].sum()) if 'goals' in df.columns else 0,
            'styles_count': df['primary_style'].nunique() if 'primary_style' in df.columns else 0
        }
    except:
        return None

stats = load_quick_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    from config import settings
    # Count total seasons or competitions
    comp_count = len(settings.SELECTED_COMPETITIONS)
    st.metric("📁 Competitions", str(comp_count), help="Active competitions in dataset")
    
with col2:
    matches_val = stats['unique_matches'] if stats else 308
    st.metric("⚽ Matches", f"{matches_val:,}" if isinstance(matches_val, int) else matches_val, 
              help="Total unique matches analyzed")
    
with col3:
    players_val = stats['total_players'] if stats else "Loading..."
    st.metric("👥 Players", f"{players_val:,}" if isinstance(players_val, int) else players_val, 
              help="Unique players in dataset")
    
with col4:
    styles_val = stats['styles_count'] if stats and stats['styles_count'] > 0 else "13"
    st.metric("🎯 Playing Styles", styles_val, help="Automated style classifications")

st.markdown("---")

# Getting started
st.subheader("🚀 Getting Started")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **For Scouts:**
    1. Go to 🔍 **Scouting** page
    2. Set your filters (position, age, style)
    3. Export shortlist or generate PDF reports
    
    **For Analysts:**
    1. Visit 👤 **Player Profile** for detailed analysis
    2. Check ⚔️ **Match Analysis** for game-by-game breakdown
    3. Use 🔮 **Predictions** for future performance forecasting
    """)
    
with col2:
    st.markdown("""
    **For Coaches:**
    1. Explore 🤝 **Team Analysis** for chemistry insights
    2. Review 📊 **Tactical Board** for formations
    3. Track 📈 **Temporal Trends** to monitor player form
    
    **For Data Scientists:**
    1. Check 🎯 **Clusters & Styles** for ML insights
    2. Compare players in 🔄 **Player Comparison**
    3. Analyze multi-dimensional performance scores
    """)

st.success("💡 **Tip**: All visualizations are interactive - hover, zoom, and click to explore!")
