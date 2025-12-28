"""
Tactical Board Page - Interactive Lineup Builder & Analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from dashboard.utils import load_css

# Load Custom CSS
load_css()

st.title("♟️ Tactical Board & Lineup Builder")
st.markdown("### Design your optimal XI and analyze squad balance")

# -----------------------------------------------------------------------------
# DATA LOADING (Copied from 07_Team_Analysis.py for consistency)
# -----------------------------------------------------------------------------

@st.cache_data
def load_match_data_v2():
    try:
        match_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        
        # Load team and competition names from JSON mappings if not in CSV
        if 'team_name' not in match_df.columns or 'competition_name' not in match_df.columns:
            import json
            team_mapping = {}
            comp_mapping = {}
            
            # Try loading pre-computed lookups first (Deployment Optimization)
            lookup_file = settings.PROCESSED_DATA_DIR / "match_lookups.json"
            loaded_from_json = False
            
            if lookup_file.exists():
                try:
                    with open(lookup_file, 'r', encoding='utf-8') as f:
                        lookups = json.load(f)
                        def int_keys(d): return {int(k): v for k, v in d.items()}
                        team_mapping = int_keys(lookups.get('team_mapping', {}))
                        comp_mapping = int_keys(lookups.get('match_comp_mapping', {}))
                        
                        if 'team_name' not in match_df.columns:
                            match_df['team_name'] = match_df['team_id'].map(team_mapping)
                        if 'competition_name' not in match_df.columns:
                            match_df['competition_name'] = match_df['match_id'].map(comp_mapping)
                        
                        loaded_from_json = True
                except Exception as e:
                    print(f"Error loading lookups: {e}")
            
            if not loaded_from_json:
                data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
                if data_path.exists():
                    for comp_folder in data_path.iterdir():
                        if comp_folder.is_dir():
                            for season_file in comp_folder.glob("*.json"):
                                try:
                                    with open(season_file, 'r', encoding='utf-8') as f:
                                        matches = json.load(f)
                                        for match in matches:
                                            if "Women's" in match['competition']['competition_name']: continue
                                            match_id = match['match_id']
                                            team_mapping[match['home_team']['home_team_id']] = match['home_team']['home_team_name']
                                            team_mapping[match['away_team']['away_team_id']] = match['away_team']['away_team_name']
                                            comp_mapping[match_id] = match['competition']['competition_name']
                                except: continue
                
                if 'team_name' not in match_df.columns:
                    match_df['team_name'] = match_df['team_id'].map(team_mapping)
                if 'competition_name' not in match_df.columns:
                    match_df['competition_name'] = match_df['match_id'].map(comp_mapping)
                
        return match_df
    except FileNotFoundError:
        return pd.DataFrame()

COLS_TO_SUM = ['goals', 'assists', 'tackles', 'interceptions', 'clearances', 'blocks',
               'shots', 'key_passes', 'passes_completed', 'minutes_played']

@st.cache_data(ttl=60)
def build_team_data(match_df):
    if match_df.empty: return pd.DataFrame()
    
    # Aggregate
    groupby_cols = ['player_name', 'team_name']
    existing = [c for c in COLS_TO_SUM if c in match_df.columns]
    agg_dict = {c: 'sum' for c in existing}
    agg_dict['minutes_played'] = 'sum'
    agg_dict['position'] = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    
    season = match_df.groupby(groupby_cols).agg(agg_dict).reset_index()
    
    # Position Category
    def cat_pos(p):
        p = str(p).lower()
        if 'keeper' in p: return 'GK'
        if 'midfield' in p: return 'Midfielder'
        if 'back' in p or 'defender' in p: return 'Defender'
        if 'wing' in p or 'forward' in p: return 'Forward'
        return 'Unknown'
    season['position_category'] = season['position'].apply(cat_pos)
    
    # Raw scores fallback
    season['offensive_score_raw'] = ((season.get('goals',0)/season.get('minutes_played',1)*90)*40 + 
                                  (season.get('assists',0)/season.get('minutes_played',1)*90)*30).clip(0, 100)
    season['defensive_score_raw'] = ((season.get('tackles',0)/season.get('minutes_played',1)*90)*30 + 
                                  (season.get('interceptions',0)/season.get('minutes_played',1)*90)*30).clip(0, 100)
    
    # Merge Enhanced Scores (The FIX)
    try:
        enhanced = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv")
        cols_to_merge = ['player_name', 'primary_style', 'offensive_score', 'defensive_score', 'creative_score']
        valid_cols = [c for c in cols_to_merge if c in enhanced.columns]
        
        if 'primary_style' in enhanced.columns:
            season = season.merge(enhanced[valid_cols], on='player_name', how='left')
            season['primary_style'] = season['primary_style'].fillna('Unknown')
            
            # Fill missing scores with raw
            for score in ['offensive_score', 'defensive_score']:
                if score in season.columns:
                    season[score] = season[score].fillna(season[f'{score}_raw'])
                else:
                    season[score] = season[f'{score}_raw']
        else:
            season['primary_style'] = 'Unknown'
            season['offensive_score'] = season['offensive_score_raw']
            season['defensive_score'] = season['defensive_score_raw']
    except:
        season['primary_style'] = 'Unknown'
        season['offensive_score'] = season['offensive_score_raw']
        season['defensive_score'] = season['defensive_score_raw']
        
    return season

# Load Data
with st.spinner("Loading tactical data..."):
    match_data = load_match_data_v2()
    data = build_team_data(match_data)

if data.empty:
    st.error("No data available.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------

st.sidebar.header("📋 Settings")

# Team Selection
teams = sorted(data['team_name'].dropna().unique())
selected_team = st.sidebar.selectbox("Select Team", teams, index=teams.index("Paris Saint-Germain") if "Paris Saint-Germain" in teams else 0)

# Filter Squad
squad = data[data['team_name'] == selected_team].copy()
squad = squad.sort_values('offensive_score', ascending=False)

# Formation Selection
formations = {
    "4-3-3": {
        "GK": [50, 5],
        "LB": [15, 25], "CB1": [38, 20], "CB2": [62, 20], "RB": [85, 25],
        "CDM": [50, 40], "CM1": [30, 55], "CM2": [70, 55],
        "LW": [15, 85], "ST": [50, 90], "RW": [85, 85]
    },
    "4-4-2": {
        "GK": [50, 5],
        "LB": [15, 25], "CB1": [38, 20], "CB2": [62, 20], "RB": [85, 25],
        "LM": [15, 60], "CM1": [38, 55], "CM2": [62, 55], "RM": [85, 60],
        "ST1": [38, 85], "ST2": [62, 85]
    },
    "3-5-2": {
        "GK": [50, 5],
        "CB1": [25, 20], "CB2": [50, 20], "CB3": [75, 20],
        "LWB": [10, 50], "CDM1": [40, 45], "CDM2": [60, 45], "RWB": [90, 50], "CAM": [50, 65],
        "ST1": [35, 85], "ST2": [65, 85]
    }
}

selected_formation_name = st.sidebar.selectbox("Select Formation", list(formations.keys()))
formation_coords = formations[selected_formation_name]

# -----------------------------------------------------------------------------
# LINEUP BUILDER
# -----------------------------------------------------------------------------

col_pitch, col_controls = st.columns([2, 1])

# Initialize lineup in session state if not exists or team changed
if 'tactical_team' not in st.session_state or st.session_state.tactical_team != selected_team:
    st.session_state.tactical_team = selected_team
    st.session_state.lineup = {}

# Helper to filter players by role
def get_players_by_role(role_name):
    role_name = role_name.lower()
    if 'gk' in role_name: return squad[squad['position_category'] == 'GK']
    if 'cb' in role_name or 'lb' in role_name or 'rb' in role_name: return squad[squad['position_category'] == 'Defender']
    if 'cm' in role_name or 'cdm' in role_name or 'lm' in role_name or 'rm' in role_name: return squad[squad['position_category'] == 'Midfielder']
    if 'st' in role_name or 'lw' in role_name or 'rw' in role_name: return squad[squad['position_category'] == 'Forward']
    return squad

with col_controls:
    st.subheader("🛠️ Select Players")
    
    selected_players_stats = []
    
    # Generate selectboxes for each position in formation
    for role, coords in formation_coords.items():
        candidates = get_players_by_role(role)
        candidate_names = ["Empty"] + candidates['player_name'].tolist()
        
        # Default selection (first available valid player not already picked ideally, but keeping simple for now)
        key = f"pos_{role}"
        
        # Try to auto-select best player for first load
        default_idx = 0
        if role not in st.session_state.lineup:
            if not candidates.empty:
                st.session_state.lineup[role] = candidates.iloc[0]['player_name']
        
        current_selection = st.session_state.lineup.get(role, "Empty")
        if current_selection not in candidate_names: current_selection = "Empty"
        
        choice = st.selectbox(f"{role}", candidate_names, 
                             index=candidate_names.index(current_selection),
                             key=key)
        
        st.session_state.lineup[role] = choice
        
        if choice != "Empty":
            player_stat = squad[squad['player_name'] == choice].iloc[0]
            selected_players_stats.append({
                'role': role,
                'name': choice,
                'x': coords[0],
                'y': coords[1],
                'off': player_stat.get('offensive_score', 0),
                'def': player_stat.get('defensive_score', 0),
                'style': player_stat.get('primary_style', 'Unknown')
            })

# -----------------------------------------------------------------------------
# PITCH VISUALIZATION
# -----------------------------------------------------------------------------

with col_pitch:
    fig = go.Figure()

    # Draw Pitch (Vertical)
    # Pitch dimensions: 0-100 x, 0-100 y (mapped from formation coords)
    
    # Grass
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color="white", width=2), fillcolor="#2E8B57", layer="below")
    
    # Penalty Areas
    fig.add_shape(type="rect", x0=20, y0=0, x1=80, y1=16, line=dict(color="white", width=2)) # Bottom
    fig.add_shape(type="rect", x0=20, y0=84, x1=80, y1=100, line=dict(color="white", width=2)) # Top
    
    # Center Circle
    fig.add_shape(type="circle", x0=35, y0=35, x1=65, y1=65, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50, line=dict(color="white", width=2))

    # Draw Players
    for p in selected_players_stats:
        # Determine color by role
        color = 'white'
        if 'GK' in p['role']: color = '#FFFF00' # Yellow
        elif 'CB' in p['role'] or 'LB' in p['role'] or 'RB' in p['role']: color = '#1E90FF' # Blue
        elif 'CM' in p['role'] or 'CDM' in p['role']: color = '#32CD32' # Green
        elif 'ST' in p['role'] or 'LW' in p['role'] or 'RW' in p['role']: color = '#FF4500' # Red
        
        # Player Marker
        fig.add_trace(go.Scatter(
            x=[p['x']], y=[p['y']],
            mode='markers+text',
            marker=dict(size=30, color=color, line=dict(color='black', width=1)),
            text=[f"<b>{p['role']}</b><br>{p['name'].split()[-1]}"], # Last name
            textposition="top center",
            textfont=dict(color='white', size=11, family="Arial Black"),
            hoverinfo='text',
            hovertext=f"<b>{p['name']}</b><br>Off: {p['off']:.0f}<br>Def: {p['def']:.0f}<br>Style: {p['style']}",
            name=p['role']
        ))

    fig.update_layout(
        title=f"Tactical Setup: {selected_formation_name}",
        xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, visible=False),
        height=600,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# ANALYSIS PANEL
# -----------------------------------------------------------------------------

if selected_players_stats:
    df_lineup = pd.DataFrame(selected_players_stats)
    
    st.markdown("---")
    st.subheader("📊 Squad Analysis")
    
    m1, m2, m3, m4 = st.columns(4)
    
    avg_off = df_lineup['off'].mean()
    avg_def = df_lineup['def'].mean()
    
    m1.metric("Avg Offensive Score", f"{avg_off:.1f}")
    m2.metric("Avg Defensive Score", f"{avg_def:.1f}")
    m3.metric("Selected Players", f"{len(df_lineup)}/11")
    
    # Balance Check
    ratio = avg_off / (avg_def + 1) # Avoid div by zero
    if ratio > 1.5: status = "🔥 High Offense"
    elif ratio < 0.6: status = "🛡️ High Defense"
    else: status = "⚖️ Balanced"
    
    m4.metric("Team Balance", status)
    
    # Detail Table
    with st.expander("View Detailed Stats"):
        st.dataframe(df_lineup[['role', 'name', 'off', 'def', 'style']].style.background_gradient(cmap='Blues'))
