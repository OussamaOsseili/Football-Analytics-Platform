"""
Player Comparison Page - Compare multiple players side-by-side
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

st.set_page_config(page_title="Player Comparison", layout="wide")

st.title("🔄 Player Comparison")
st.markdown("### Compare multiple players side-by-side")

# Initialize session state for selected players if not exists
if 'comparison_list' not in st.session_state:
    st.session_state['comparison_list'] = []

# ================= DATA LOADING (Copied from Player Profile) =================
@st.cache_data
def load_match_data_with_names():
    try:
        # Load player match stats
        df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        
        # Build team, competition mappings from StatsBomb data
        team_mapping = {}
        comp_mapping = {}
        match_comp_mapping = {}
        
        data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
        
        # Iterate through competition folders
        for comp_folder in data_path.iterdir():
            if comp_folder.is_dir():
                for season_file in comp_folder.glob("*.json"):
                    try:
                        with open(season_file, 'r', encoding='utf-8') as f:
                            matches = json.load(f)
                            for match in matches:
                                team_mapping[match['home_team']['home_team_id']] = match['home_team']['home_team_name']
                                team_mapping[match['away_team']['away_team_id']] = match['away_team']['away_team_name']
                                comp_name = match['competition']['competition_name']
                                season_name = match['season']['season_name']
                                comp_full = f"{comp_name} - {season_name}"
                                comp_mapping[match['competition']['competition_id']] = comp_full
                                match_comp_mapping[match['match_id']] = comp_full
                    except:
                        continue
        
        if 'team_id' in df.columns:
            df['team_name'] = df['team_id'].map(team_mapping)
        
        if 'match_id' in df.columns:
            df['competition_name'] = df['match_id'].map(match_comp_mapping)
        
        return df
    except FileNotFoundError:
        return pd.DataFrame()

match_data = load_match_data_with_names()

if match_data.empty:
    st.error("⚠️ Data not loaded. This usually means 'players_match_stats.csv' is missing or corrupted.")
    st.info(f"Debug: Project Root is {settings.PROJECT_ROOT}")
    st.info(f"Debug: Looking for {settings.PROCESSED_DATA_DIR / 'players_match_stats.csv'}")
    st.stop()

# Definition of columns to sum (Global scope)
COLS_TO_SUM = [
    'goals', 'assists', 'xg', 'xa', 'shots', 'key_passes',
    'passes_completed', 'passes', 'progressive_passes',
    'dribbles_completed', 'carries', 'progressive_carries',
    'tackles', 'interceptions', 'blocks', 'clearances', 'pressures',
    'minutes_played'
]

@st.cache_data
def calculate_aggregated_season_data(match_df):
    if match_df.empty:
        return pd.DataFrame()
        
    existing_cols = [c for c in COLS_TO_SUM if c in match_df.columns]
    season_agg = match_df.groupby(['player_name']).agg({c: 'sum' for c in existing_cols}).reset_index()
    
    # Matches count
    matches_count = match_df.groupby('player_name').size().reset_index(name='matches_played')
    season_agg = season_agg.merge(matches_count, on='player_name', how='left')
    
    # Position mapping
    if 'position' in match_df.columns:
        pos_map = match_df.groupby('player_name')['position'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        season_agg['position'] = season_agg['player_name'].map(pos_map)
        
        def categorize_pos(p):
            p = str(p).lower()
            if 'keeper' in p: return 'Goalkeeper'
            if 'back' in p or 'defender' in p: return 'Defender'
            if 'midfield' in p: return 'Midfielder'
            if 'wing' in p or 'forward' in p or 'striker' in p: return 'Forward'
            return 'Unknown'
            
        season_agg['position_category'] = season_agg['position'].apply(categorize_pos)

    # Per 90s
    for col in existing_cols:
        if col != 'minutes_played':
            season_agg[f"{col}_per90"] = (season_agg[col] / season_agg['minutes_played'] * 90).fillna(0)
            
    if 'passes' in season_agg.columns and 'passes_completed' in season_agg.columns:
        season_agg['pass_completion_rate'] = (season_agg['passes_completed'] / season_agg['passes']).fillna(0)

    # Aliases
    if 'xg' in season_agg.columns: season_agg['xg_total'] = season_agg['xg']
    if 'xa' in season_agg.columns: season_agg['xa_total'] = season_agg['xa']
    
    return season_agg

season_data = calculate_aggregated_season_data(match_data)

season_data = calculate_aggregated_season_data(match_data)

# ================= PLAYER SELECTION (MAIN PAGE) =================
# Collapsible filters for adding players
with st.expander("🔍 Add Players to Comparison", expanded=True):
    col_comp, col_team, col_player, col_add = st.columns([2, 2, 2, 1])

    # 1. Filters to find player
    if 'competition_name' in match_data.columns:
        competitions = sorted([c for c in match_data['competition_name'].unique() if pd.notna(c)])
        with col_comp:
            sel_comp = st.selectbox("Competition", ["All Competitions"] + competitions, key="comp_select")
        
        filtered_data = match_data.copy()
        if sel_comp != "All Competitions":
            filtered_data = filtered_data[filtered_data['competition_name'] == sel_comp]
    else:
        filtered_data = match_data

    if 'team_name' in filtered_data.columns:
        teams = sorted([t for t in filtered_data['team_name'].unique() if pd.notna(t)])
        with col_team:
            sel_team = st.selectbox("Team", ["All Teams"] + teams, key="team_select")
        
        if sel_team != "All Teams":
            filtered_data = filtered_data[filtered_data['team_name'] == sel_team]

    players = sorted(filtered_data['player_name'].unique())
    with col_player:
        player_to_add = st.selectbox("Select Player", ["None"] + players, key="player_add_select")

    with col_add:
        st.write("") # Spacer to align button
        st.write("") 
        if st.button("➕ Add", use_container_width=True):
            if player_to_add != "None" and player_to_add not in st.session_state['comparison_list']:
                if len(st.session_state['comparison_list']) < 5:
                    st.session_state['comparison_list'].append(player_to_add)
                    st.success(f"Added!")
                else:
                    st.warning("Max 5")
            elif player_to_add in st.session_state['comparison_list']:
                st.warning("Added")

st.markdown("### 📋 Selected Players")

# Manage selected list (Horizontal display)
if st.session_state['comparison_list']:
    cols = st.columns(len(st.session_state['comparison_list']) + 1)
    players_to_remove = []
    
    for i, p in enumerate(st.session_state['comparison_list']):
        with cols[i]:
            st.info(f"**{p}**")
            if st.button("🗑️ Remove", key=f"del_{p}"):
                players_to_remove.append(p)
                
    # Process removal
    if players_to_remove:
        for p in players_to_remove:
            st.session_state['comparison_list'].remove(p)
        st.rerun()
else:
    st.info("👈 Use the filters above to add players.")

# Use the session list as selected_players
selected_players = st.session_state['comparison_list']

if len(selected_players) < 2:
    st.info("👈 Use the filters above to add at least 2 players!")
    st.stop()

# ================= RE-CALCULATE DATA WITH CONTEXT =================
# We want to show stats for the players IN THE SELECTED COMPETITION context
# If user selected "World Cup", we show stats for that.
# If they selected "All", we show aggregated.

# Key step: Re-run aggregation on the ALREADY FILTERED data (e.g. filtered_data)
# But wait, filtered_data is also filtered by Team. 
# We want matches for the selected players in the SELECTED COMPETITION (regardless of team).

context_df = match_data.copy()
if 'competition_name' in match_data.columns and 'sel_comp' in locals() and sel_comp != "All Competitions":
    context_df = context_df[context_df['competition_name'] == sel_comp]

# Now aggregate specifically for our comparison table
comparison_source = calculate_aggregated_season_data(context_df)

# Get player data from our robust aggregated dataset
comparison_data = comparison_source[comparison_source['player_name'].isin(selected_players)]

# Comparison table
st.subheader("📊 Statistics Comparison")

display_cols = ['player_name', 'position_category', 'matches_played', 'minutes_played',
               'goals', 'assists', 'goals_per90', 'assists_per90', 'xg_per90',
               'offensive_score', 'creative_score', 'defensive_score']

display_cols = [col for col in display_cols if col in comparison_data.columns]

st.dataframe(
    comparison_data[display_cols].set_index('player_name'),
    width='stretch'
)

# Radar comparison
st.subheader("🎯 Performance Radar Comparison")

radar_metrics = {
    'Goals/90': 'goals_per90',
    'Assists/90': 'assists_per90',
    'xG/90': 'xg_per90',
    'Prog Passes/90': 'progressive_passes_per90',
    'Tackles/90': 'tackles_per90',
    'Interceptions/90': 'interceptions_per90'
}

# Filter available metrics
available_metrics = {k: v for k, v in radar_metrics.items() if v in comparison_data.columns}

if available_metrics:
    fig = go.Figure()
    
    for _, player_row in comparison_data.iterrows():
        player_name = player_row['player_name']
        
        # Calculate percentiles
        radar_values = []
        for label, col in available_metrics.items():
            value = player_row[col]
            percentile = (season_data[col] < value).mean() * 100 if col in season_data.columns else 50
            radar_values.append(percentile)
        
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=list(available_metrics.keys()),
            fill='toself',
            name=player_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')

# Dimension scores comparison
st.subheader("📊 Multi-Dimensional Scores")

dimension_cols = ['offensive_score', 'creative_score', 'defensive_score', 'workrate_score', 'discipline_score']
available_dimensions = [col for col in dimension_cols if col in comparison_data.columns]

if available_dimensions:
    dimension_labels = [col.replace('_score', '').title() for col in available_dimensions]
    
    fig = go.Figure()
    
    for _, player in comparison_data.iterrows():
        scores = [player[col] for col in available_dimensions]
        
        fig.add_trace(go.Bar(
            name=player['player_name'],
            x=dimension_labels,
            y=scores
        ))
    
    fig.update_layout(
        barmode='group',
        yaxis_range=[0, 100],
        height=400,
        yaxis_title="Score (0-100)"
    )
    
    st.plotly_chart(fig, width='stretch')

# Key differences
st.subheader("🔍 Key Differences")

if len(selected_players) == 2 and len(comparison_data) >= 2:
    player1 = comparison_data.iloc[0]
    player2 = comparison_data.iloc[1]
    
    st.markdown(f"**{player1['player_name']}** vs **{player2['player_name']}**")
    
    col1, col2 = st.columns(2)

    # Detailed metrics to compare
    comparison_metrics = {
        'Attacking': {
            'goals_per90': 'Goal Scoring',
            'xg_per90': 'Expected Goals (xG)',
            'shots_per90': 'Shot Volume'
        },
        'Playmaking': {
            'assists_per90': 'Assists',
            'xa_per90': 'Expected Assists (xA)',
            'key_passes_per90': 'Key Passes'
        },
        'Possession': {
            'dribbles_completed_per90': 'Dribbling',
            'progressive_carries_per90': 'Ball Progression (Carry)',
            'pass_completion_rate': 'Pass Accuracy'
        },
        'Defending': {
            'tackles_per90': 'Tackling',
            'interceptions_per90': 'Interceptions',
            'blocks_per90': 'Blocks',
            'pressures_per90': 'Pressing'
        }
    }
    
    # Helper to format value
    def fmt(val, key):
        if 'rate' in key or 'completion' in key:
            return f"{val*100:.1f}%"
        return f"{val:.2f}"

    with col1:
        st.markdown(f"**{player1['player_name']} Strengths:**")
        found_p1 = False
        
        # Check scores first
        if player1.get('offensive_score', 0) > player2.get('offensive_score', 0) * 1.05:
            st.write(f"✅ Higher Offensive Score ({player1.get('offensive_score', 0):.0f})")
            found_p1 = True
            
        # Check detailed metrics
        for category, metrics in comparison_metrics.items():
            for key, label in metrics.items():
                v1 = player1.get(key, 0)
                v2 = player2.get(key, 0)
                
                # Only show if value is non-negligible and noticeable difference
                if v1 > v2 and v1 > 0:
                    diff_pct = (v1 - v2) / ((v1 + v2)/2) if (v1+v2) > 0 else 0
                    if diff_pct > 0.10: # 10% difference
                         st.write(f"✅ Better {label} ({fmt(v1, key)} vs {fmt(v2, key)})")
                         found_p1 = True
                         
        if not found_p1:
            st.write("No significant statistical advantages found.")

    with col2:
        st.markdown(f"**{player2['player_name']} Strengths:**")
        found_p2 = False
        
        # Check scores
        if player2.get('offensive_score', 0) > player1.get('offensive_score', 0) * 1.05:
            st.write(f"✅ Higher Offensive Score ({player2.get('offensive_score', 0):.0f})")
            found_p2 = True

        for category, metrics in comparison_metrics.items():
            for key, label in metrics.items():
                v1 = player1.get(key, 0)
                v2 = player2.get(key, 0)
                
                if v2 > v1 and v2 > 0:
                    diff_pct = (v2 - v1) / ((v1 + v2)/2) if (v1+v2) > 0 else 0
                    if diff_pct > 0.10:
                         st.write(f"✅ Better {label} ({fmt(v2, key)} vs {fmt(v1, key)})")
                         found_p2 = True

        if not found_p2:
            st.write("No significant statistical advantages found.")
