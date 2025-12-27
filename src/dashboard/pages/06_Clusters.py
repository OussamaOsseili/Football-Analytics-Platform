"""
Clusters & Playing Styles Page - Visualize ML clustering results
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

st.title("🎯 Clusters & Playing Styles")
st.markdown("### ML-Powered Player Archetype Analysis")

@st.cache_data
def load_data():
    try:
        # Try enhanced data first
        enhanced_path = settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv"
        if enhanced_path.exists():
            return pd.read_csv(enhanced_path)
        else:
            return pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats.csv")
    except FileNotFoundError:
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.warning("⚠️ No data available.")
    st.stop()

# Position filter
st.sidebar.header("🎯 Filters")
positions = ['All'] + sorted(data['position_category'].unique().tolist()) if 'position_category' in data.columns else ['All']
selected_position = st.sidebar.selectbox("Filter by Position:", positions)

# Filter data
if selected_position != 'All' and 'position_category' in data.columns:
    filtered_data = data[data['position_category'] == selected_position]
else:
    filtered_data = data.copy()

# Playing styles distribution
st.subheader("📊 Playing Styles Distribution")

if 'primary_style' in filtered_data.columns:
    style_counts = filtered_data['primary_style'].value_counts().reset_index()
    style_counts.columns = ['Style', 'Count']
    
    fig = px.bar(
        style_counts.head(10),
        x='Style',
        y='Count',
        title=f"Top 10 Playing Styles{' - ' + selected_position if selected_position != 'All' else ''}",
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, width='stretch')
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Styles", len(style_counts))
    with col2:
        if not style_counts.empty:
            most_common = style_counts.iloc[0]
            st.metric("Most Common", most_common['Style'], f"{most_common['Count']} players")
        else:
            st.metric("Most Common", "N/A", "0 players")
    with col3:
        st.metric("Players Analyzed", len(filtered_data))

else:
    st.info("Run ML pipeline to generate playing style classifications: `python src/ml/train_pipeline.py`")

# 2D Visualization using PCA
st.markdown("---")
st.subheader("🗺️ Player Similarity Map (2D)")

# Select features for PCA
feature_cols = ['goals_per90', 'assists_per90', 'xg_per90', 'xa_per90',
                'progressive_passes_per90', 'tackles_per90', 'interceptions_per90']

available_features = [col for col in feature_cols if col in filtered_data.columns]

if len(available_features) >= 2 and len(filtered_data) > 5:
    # Prepare data
    X = filtered_data[available_features].fillna(0)
    
    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    # Create visualization dataframe
    viz_df = filtered_data.copy()
    viz_df['PC1'] = components[:, 0]
    viz_df['PC2'] = components[:, 1]
    
    # Color by style or position
    color_by = 'primary_style' if 'primary_style' in viz_df.columns else 'position_category'
    
    fig = px.scatter(
        viz_df,
        x='PC1',
        y='PC2',
        color=color_by,
        hover_data=['player_name'],
        title=f"Player Similarity Map (PCA)",
        height=600
    )
    
    fig.update_traces(marker=dict(size=8))
    st.plotly_chart(fig, width='stretch')
    
    # Explained variance
    st.info(f"📊 Variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# Style profiles
st.markdown("---")
st.subheader("📋 Style Profiles")

if 'primary_style' in filtered_data.columns:
    selected_style = st.selectbox("Select a style to analyze:", sorted(filtered_data['primary_style'].dropna().unique()))
    
    style_players = filtered_data[filtered_data['primary_style'] == selected_style]
    
    if len(style_players) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**{selected_style}** ({len(style_players)} players)")
            
            # Average stats
            st.markdown("**Average Statistics:**")
            avg_stats = {
                'Goals/90': style_players['goals_per90'].mean() if 'goals_per90' in style_players.columns else 0,
                'Assists/90': style_players['assists_per90'].mean() if 'assists_per90' in style_players.columns else 0,
                'xG/90': style_players['xg_per90'].mean() if 'xg_per90' in style_players.columns else 0,
                'Tackles/90': style_players['tackles_per90'].mean() if 'tackles_per90' in style_players.columns else 0,
            }
            
            for stat, value in avg_stats.items():
                st.write(f"• {stat}: {value:.2f}")
        
        with col2:
            # Dimension scores
            if 'offensive_score' in style_players.columns:
                st.markdown("**Avg Dimension Scores:**")
                dimensions = {
                    'Offensive': 'offensive_score',
                    'Creative': 'creative_score',
                    'Defensive': 'defensive_score'
                }
                
                for dim, col in dimensions.items():
                    if col in style_players.columns:
                        score = style_players[col].mean()
                        st.metric(dim, f"{score:.0f}/100")
        
        # Top players in this style
        st.markdown("**Top Players:**")
        
        sort_col = 'offensive_score' if 'offensive_score' in style_players.columns else 'goals_per90'
        if sort_col in style_players.columns:
            top_in_style = style_players.nlargest(5, sort_col)[['player_name', sort_col]]
            
            for idx, row in top_in_style.iterrows():
                st.write(f"⭐ {row['player_name']} ({row[sort_col]:.1f})")

# Similar players finder
st.markdown("---")
st.subheader("🔍 Find Similar Players")

player_search = st.selectbox("Select a player:", [''] + sorted(filtered_data['player_name'].unique()))

if player_search:
    player_data = filtered_data[filtered_data['player_name'] == player_search].iloc[0]
    
    st.markdown(f"**Finding players similar to {player_search}...**")
    
    if 'primary_style' in player_data:
        st.write(f"🎯 Style: {player_data['primary_style']}")
    
    # Simple similarity: same position + similar stats
    same_position = filtered_data[filtered_data['position_category'] == player_data.get('position_category', '')]
    
    if len(same_position) > 1 and len(available_features) > 0:
        # Calculate distances
        player_stats = player_data[available_features].fillna(0).values
        
        distances = []
        for idx, row in same_position.iterrows():
            if row['player_name'] != player_search:
                row_stats = row[available_features].fillna(0).values
                dist = ((player_stats - row_stats) ** 2).sum() ** 0.5
                distances.append((row['player_name'], dist))
        
        # Sort by distance
        similar = sorted(distances, key=lambda x: x[1])[:5]
        
        st.markdown("**Most Similar Players:**")
        for i, (name, dist) in enumerate(similar, 1):
            similarity = max(0, 100 - dist * 10)  # Convert distance to similarity %
            st.write(f"{i}. **{name}** (Similarity: {similarity:.0f}%)")
