"""
Player Profile Page - Detailed player analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import warnings
from pathlib import Path

# Suppress StatsBomb warnings
warnings.filterwarnings('ignore', message='credentials were not supplied')
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

st.title("👤 Player Profile")
st.markdown("### Comprehensive player analysis with AI insights")

# Load data
@st.cache_data
def load_data():
    try:
        season_stats = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats.csv")
        match_stats = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        return season_stats, match_stats
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

season_data, match_data = load_data()

if season_data.empty:
    st.warning("⚠️ No data available. Please run the ETL pipeline first.")
    st.info("Run: `python src/etl/etl_pipeline.py`")
    st.stop()

# Load match-level data with team/competition names (same as Match Analysis)
@st.cache_data
def load_match_data_with_names():
    return _load_match_data_internal()


def _load_match_data_internal():
    import json
    from config import settings
    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
    
    team_mapping = {}
    comp_mapping = {}
    match_comp_mapping = {}
    match_home_team_map = {} # match_id -> home_team_id
    match_date_map = {} # match_id -> match_date
    
    # Try loading pre-computed lookups first (Deployment Optimization)
    lookup_file = settings.PROCESSED_DATA_DIR / "match_lookups.json"
    
    loaded_from_json = False
    if lookup_file.exists():
        try:
            with open(lookup_file, 'r', encoding='utf-8') as f:
                lookups = json.load(f)
                # Helper to convert keys back to int (JSON stores keys as strings)
                def int_keys(d):
                    return {int(k): v for k, v in d.items()}
                    
                team_mapping = int_keys(lookups.get('team_mapping', {}))
                match_comp_mapping = int_keys(lookups.get('match_comp_mapping', {}))
                match_home_team_map = int_keys(lookups.get('match_home_team_map', {}))
                match_date_map = int_keys(lookups.get('match_date_map', {}))
                loaded_from_json = True
        except Exception as e:
            print(f"Error loading lookups: {e}")

    if not loaded_from_json:
        # Fallback to local raw data
        data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
        
        if data_path.exists():
            for comp_folder in data_path.iterdir():
                if comp_folder.is_dir():
                    for season_file in comp_folder.glob("*.json"):
                        try:
                            with open(season_file, 'r', encoding='utf-8') as f:
                                matches = json.load(f)
                                for match in matches:
                                    mid = match['match_id']
                                    htid = match['home_team']['home_team_id']
                                    atid = match['away_team']['away_team_id']
                                    mdate = match['match_date']
                                    comp_name = match['competition']['competition_name']
                                    
                                    # Filter out Women's competitions as requested
                                    if "Women's" in comp_name:
                                        continue
                                    
                                    team_mapping[htid] = match['home_team']['home_team_name']
                                    team_mapping[atid] = match['away_team']['away_team_name']
                                    
                                    comp_full = f"{comp_name} - {match['season']['season_name']}"
                                    match_comp_mapping[mid] = comp_full
                                    match_home_team_map[mid] = htid
                                    match_date_map[mid] = mdate
                        except: continue

    if 'team_id' in df.columns:
        df['team_name'] = df['team_id'].map(team_mapping)
    
    if 'match_id' in df.columns:
        df['competition_name'] = df['match_id'].map(match_comp_mapping)
        # Determine Venue
        df['home_team_id'] = df['match_id'].map(match_home_team_map)
        df['match_date'] = df['match_id'].map(match_date_map)
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'])
            
        if 'team_id' in df.columns:
            df['is_home'] = df['team_id'] == df['home_team_id']
            df['venue'] = df['is_home'].apply(lambda x: 'Home' if x else 'Away')
            
    return df


match_data = load_match_data_with_names()

# ============= AGGREGATE STATS ON THE FLY =============
# Definition of columns to sum (Global scope for reuse)
COLS_TO_SUM = [
    'goals', 'assists', 'xg', 'xa', 'shots', 'key_passes',
    'passes_completed', 'passes', 'progressive_passes',
    'dribbles_completed', 'carries', 'progressive_carries',
    'tackles', 'interceptions', 'blocks', 'clearances', 'pressures',
    'minutes_played'
]

# Since season_stats.csv might be incomplete, we rebuild it from match_data
@st.cache_data
def calculate_aggregated_season_data(match_df):
    if match_df.empty:
        return pd.DataFrame()
        
    # Group by player
    # Sum up all counting stats
    
    # Ensure columns exist
    existing_cols = [c for c in COLS_TO_SUM if c in match_df.columns]
    
    season_agg = match_df.groupby(['player_name']).agg({c: 'sum' for c in existing_cols}).reset_index()
    
    # Calculate matches played (count of rows)
    matches_count = match_df.groupby('player_name').size().reset_index(name='matches_played')
    season_agg = season_agg.merge(matches_count, on='player_name', how='left')
    
    # Get positions (mode)
    if 'position' in match_df.columns:
        pos_map = match_df.groupby('player_name')['position'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        season_agg['position'] = season_agg['player_name'].map(pos_map)
        
        # Simple categorization
        def categorize_pos(p):
            p = str(p).lower()
            if 'keeper' in p: return 'Goalkeeper'
            if 'back' in p or 'defender' in p: return 'Defender'
            if 'midfield' in p: return 'Midfielder'
            if 'wing' in p or 'forward' in p or 'striker' in p: return 'Forward'
            return 'Unknown'
            
        season_agg['position_category'] = season_agg['position'].apply(categorize_pos)
        
        # Add team
        if 'team_name' in match_df.columns:
             # Most frequent team
             team_map = match_df.groupby('player_name')['team_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
             season_agg['team_name'] = season_agg['player_name'].map(team_map)

    # Calculate per 90s
    for col in existing_cols:
        if col != 'minutes_played':
            season_agg[f"{col}_per90"] = (season_agg[col] / season_agg['minutes_played'] * 90).fillna(0)
            
    # Pass completion rate
    if 'passes' in season_agg.columns and 'passes_completed' in season_agg.columns:
        season_agg['pass_completion_rate'] = (season_agg['passes_completed'] / season_agg['passes']).fillna(0)

    # Aliases to match existing code expectations
    if 'xg' in season_agg.columns: season_agg['xg_total'] = season_agg['xg']
    if 'xa' in season_agg.columns: season_agg['xa_total'] = season_agg['xa']
    
    return season_agg

# Overwrite season_data with our fresh, complete aggregation
if not match_data.empty:
    season_data = calculate_aggregated_season_data(match_data)
# ======================================================

# ============= EXACT SAME FILTERS AS MATCH ANALYSIS =============
st.subheader("🔍 Select Player")

col1, col2, col3 = st.columns(3)

# 1. Competition Selector
with col1:
    if 'competition_name' in match_data.columns:
        competitions = sorted([c for c in match_data['competition_name'].unique() if pd.notna(c)])
        selected_competition = st.selectbox("**Competition:**", ["All Competitions"] + competitions, key="prof_comp")
        
        if selected_competition != "All Competitions":
            filtered_data = match_data[match_data['competition_name'] == selected_competition]
        else:
            filtered_data = match_data
    else:
        filtered_data = match_data

# 2. Team Selector (filtered by competition)
with col2:
    if 'team_name' in filtered_data.columns:
        teams = sorted([t for t in filtered_data['team_name'].unique() if pd.notna(t)])
        selected_team = st.selectbox("**Team:**", ["All Teams"] + teams, key="prof_team")
        
        if selected_team != "All Teams":
            filtered_data = filtered_data[filtered_data['team_name'] == selected_team]

# 3. Player Selector (filtered by team)
with col3:
    players = sorted(filtered_data['player_name'].unique())
    selected_player = st.selectbox("**Player:**", players, key="prof_player")

st.markdown("---")

if selected_player:
    # Get season stats for this player from the FILTERED match data context
    # This ensures that if we selected "World Cup", we only see World Cup stats
    
    # Identify the relevant matches for this player based on current filters
    if 'competition_name' in filtered_data.columns:
        # If user filtered by competition, we already have the right subset in filtered_data
        # If user selected "All Competitions", filtered_data is full match_data
        player_matches = filtered_data[filtered_data['player_name'] == selected_player]
    else:
        player_matches = match_data[match_data['player_name'] == selected_player]

    if player_matches.empty:
        # Fallback to global search if player not found in filtered set (shouldn't happen with correct logic)
        player_matches = match_data[match_data['player_name'] == selected_player]

    # Recalculate stats for just this player/context on the fly
    # (Since calculate_aggregated_season_data is cached and expensive for full dataset,
    # we do a lightweight aggregation here for the single player)
    
    def aggregate_single_player(df):
        existing_cols = [c for c in COLS_TO_SUM if c in df.columns]
        agg = {c: df[c].sum() for c in existing_cols}
        agg['matches_played'] = len(df)
        agg['minutes_played'] = df['minutes_played'].sum()
        
        # Per 90s
        for c in existing_cols:
            if c != 'minutes_played':
                agg[f"{c}_per90"] = (agg[c] / agg['minutes_played'] * 90) if agg['minutes_played'] > 0 else 0
        
        # Rates
        if 'passes' in agg and 'passes_completed' in agg:
            agg['pass_completion_rate'] = (agg['passes_completed'] / agg['passes']) if agg['passes'] > 0 else 0
            
        # Meta
        if 'position' in df.columns:
            agg['position'] = df['position'].mode()[0] if not df['position'].mode().empty else 'Unknown'
            
        return pd.Series(agg)

    player_data = aggregate_single_player(player_matches)
    
    # Update position category helper if needed (reuse existing func if possible, else logic inline)
    p_pos = str(player_data.get('position', 'Unknown')).lower()
    if 'keeper' in p_pos: cat = 'Goalkeeper'
    elif 'back' in p_pos or 'defender' in p_pos: cat = 'Defender'
    elif 'midfield' in p_pos: cat = 'Midfielder'
    elif 'wing' in p_pos or 'forward' in p_pos or 'striker' in p_pos: cat = 'Forward'
    else: cat = 'Unknown'
    player_data['position_category'] = cat
    
    # Header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader(f"⚽ {selected_player}")
        
        # Get position from match data (most frequent)
        position = "Unknown"
        if not match_data.empty and 'position' in match_data.columns:
             player_pos_data = match_data[match_data['player_name'] == selected_player]
             if not player_pos_data.empty:
                 position = player_pos_data['position'].mode().iloc[0]
        
        if position == "Unknown":
            position = player_data.get('position_category', 'Unknown')
            
        st.markdown(f"**Position**: {position}")
        
    with col2:
        if 'primary_style' in player_data and pd.notna(player_data['primary_style']):
            st.markdown(f"**Playing Style**: `{player_data['primary_style']}`")
        
    with col3:
        st.metric("Matches", f"{player_data.get('matches_played', 0):.0f}")
        
    with col4:
        if 'minutes_played' in player_data:
            st.metric("Minutes", f"{player_data['minutes_played']:.0f}")
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Create Tabs for organized view
    # Create Tabs for organized view
    tab_overview, tab_stats, tab_analysis, tab_ai, tab_context, tab_trends = st.tabs([
        "📊 Overview & Analysis", 
        "📈 Detailed Statistics", 
        "👯 Similar Players", 
        "🤖 AI Scouting Report",
        "🏟️ Contextual Analysis",
        "⌛ Temporal Trends"
    ])
    
    # ================= TAB 1: OVERVIEW & PIZZA CHART =================
    with tab_overview:
        col_pizza, col_form = st.columns([1, 1])
        
        with col_pizza:
            st.subheader("Percentile Rank vs Position Peers")
            
            # 1. Define metrics for Pizza Chart
            pizza_metrics = {
                'Attacking': ['xg_per90', 'goals_per90', 'shots_per90'],
                'Creative': ['xa_per90', 'assists_per90', 'key_passes_per90'],
                'Possession': ['dribbles_completed_per90', 'progressive_carries_per90', 'pass_completion_rate'],
                'Defensive': ['tackles_per90', 'interceptions_per90', 'blocks_per90']
            }
            
            # flatten metrics list
            all_pizza_cols = [m for cat in pizza_metrics.values() for m in cat]
            
            # Filter peers by position for fair comparison
            player_pos = player_data.get('position_category', 'Unknown')
            if 'position_category' in season_data.columns and player_pos != 'Unknown':
                peers = season_data[season_data['position_category'] == player_pos]
            else:
                peers = season_data
                
            # Need at least 5 peers for meaningful percentiles
            if len(peers) > 5:
                # Calculate percentiles
                player_ranks = []
                player_labels = []
                
                for cat, metrics in pizza_metrics.items():
                    for metric in metrics:
                        # Check if metric exists
                        if metric in peers.columns:
                            # Calculate percentile (0-100)
                            # Handle case where metric might be all zeros
                            if peers[metric].max() > 0:
                                rank = (peers[metric] < player_data.get(metric, 0)).mean() * 100
                            else:
                                rank = 50
                            
                            player_ranks.append(rank)
                            # Clean label
                            label = metric.replace('_per90', '').replace('_', ' ').title()
                            player_labels.append(label)
                
                if player_ranks:
                    # Create Pizza Chart
                    fig_pizza = go.Figure(go.Scatterpolar(
                        r=player_ranks,
                        theta=player_labels,
                        fill='toself',
                        name=selected_player,
                        line_color='#00CC96'
                    ))
                    
                    fig_pizza.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%'),
                            angularaxis=dict(direction="clockwise", period=len(player_ranks))
                        ),
                        showlegend=False,
                        margin=dict(t=20, b=20, l=40, r=40),
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_pizza, width='stretch')
                else:
                    st.info("Not enough data for percentile chart")
            else:
                st.info(f"Not enough peers in position '{player_pos}' for comparison")

        with col_form:
            st.subheader("Recent Form (Last 5 Matches)")
            
            if not match_data.empty:
                # Get matches for this player
                p_matches = match_data[match_data['player_name'] == selected_player].copy()
                
                if not p_matches.empty:
                    # Ensure date sorting
                    if 'match_date' in p_matches.columns:
                        p_matches = p_matches.sort_values('match_date')
                    
                    last_5 = p_matches.tail(5)
                    
                    # Form Metrics to show
                    form_metrics = ['rating', 'xg', 'xa', 'passes_completed']
                    
                    for metric in form_metrics:
                        if metric in last_5.columns and last_5[metric].sum() > 0:
                            # Create mini sparkline
                            fig_spark = px.line(last_5, x=range(len(last_5)), y=metric, 
                                              title=f"{metric.upper().replace('_', ' ')} Trend",
                                              markers=True)
                            fig_spark.update_layout(
                                height=120, 
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis=dict(showgrid=False, showticklabels=False),
                                yaxis=dict(showgrid=False),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            fig_spark.update_traces(line_color='#636EFA', line_width=3)
                            
                            st.plotly_chart(fig_spark, width='stretch', key=f"spark_{metric}")
                else:
                    st.info("No match history available")

    # ================= TAB 2: DETAILED STATISTICS =================
    with tab_stats:
        # Putch missing stats by aggregating match_data on the fly
        # (Fixes issue where season_data was incomplete)
        if not match_data.empty:
            p_matches = match_data[match_data['player_name'] == selected_player]
            
            # Calculate aggregations safely
            def get_sum(col):
                return p_matches[col].sum() if col in p_matches.columns else 0
                
            total_minutes = get_sum('minutes_played')
            
            # Attacking
            goals = get_sum('goals')
            xg = get_sum('xg')
            shots = get_sum('shots')
            
            # Creativity
            assists = get_sum('assists')
            xa = get_sum('xa')
            key_passes = get_sum('key_passes')
            
            # Defensive
            tackles = get_sum('tackles')
            interceptions = get_sum('interceptions')
            blocks = get_sum('blocks')
            clearances = get_sum('clearances')
            pressures = get_sum('pressures')
            duels_won = get_sum('duels_won')
            
            # Possession
            passes_completed = get_sum('passes_completed')
            passes_attempted = get_sum('passes')
            prog_passes = get_sum('progressive_passes')
            dribbles_completed = get_sum('dribbles_completed')
            carries = get_sum('carries')
            
            # Rates
            pass_completion = (passes_completed / passes_attempted * 100) if passes_attempted > 0 else 0.0
            
            # Per 90s
            factor = 90 / total_minutes if total_minutes > 0 else 0
            
            goals_p90 = goals * factor
            xg_p90 = xg * factor
            assists_p90 = assists * factor
            xa_p90 = xa * factor
            
        else:
            # Fallback to season_data if match_data fails
            goals = player_data.get('goals', 0)
            xg = player_data.get('xg_total', 0)
            # ... and so on (simplified fallback)
            pass_completion = 0.0
            goals_p90 = 0.0
            xg_p90 = 0.0
            assists_p90 = 0.0
            xa_p90 = 0.0

        col_att, col_cr, col_def, col_poss = st.columns(4)
        
        with col_att:
            st.markdown("### ⚽ Attacking")
            st.write(f"**Goals**: {goals:.0f}")
            st.write(f"**Goals/90**: {goals_p90:.2f}")
            st.write(f"**xG**: {xg:.2f}")
            st.write(f"**xG/90**: {xg_p90:.2f}")
            st.write(f"**Shots**: {shots:.0f}")
            
        with col_cr:
            st.markdown("### 🎨 Creativity")
            st.write(f"**Assists**: {assists:.0f}")
            st.write(f"**Assists/90**: {assists_p90:.2f}")
            st.write(f"**xA**: {xa:.2f}")
            st.write(f"**xA/90**: {xa_p90:.2f}")
            st.write(f"**Key Passes**: {key_passes:.0f}")
            
        with col_def:
            st.markdown("### 🛡️ Defensive")
            st.write(f"**Tackles**: {tackles:.0f}")
            st.write(f"**Interceptions**: {interceptions:.0f}")
            st.write(f"**Blocks**: {blocks:.0f}")
            st.write(f"**Clearances**: {clearances:.0f}")
            st.write(f"**Pressures**: {pressures:.0f}")
            
        with col_poss:
            st.markdown("### ⛓️ Possession")
            st.write(f"**Passes**: {passes_completed:.0f}/{passes_attempted:.0f}")
            st.write(f"**Pass %**: {pass_completion:.1f}%")
            st.write(f"**Prog. Passes**: {prog_passes:.0f}")
            st.write(f"**Dribbles**: {dribbles_completed:.0f}")
            st.write(f"**Carries**: {carries:.0f}")

    # ================= TAB 3: SIMILAR PLAYERS =================
    with tab_analysis:
        st.subheader("👯 Similar Players Profile")
        st.write("Based on statistical similarity (Euclidean distance on key metrics)")
        
        # Calculate similarity
        # Use simple features: xG, xA, Shots, Key Passes, Tackles, Interceptions, Dribbles
        sim_features = ['xg_per90', 'xa_per90', 'shots_per90', 'key_passes_per90', 
                       'tackles_per90', 'interceptions_per90', 'dribbles_completed_per90']
        
        # Ensure features exist
        valid_features = [f for f in sim_features if f in season_data.columns]
        
        if len(valid_features) > 3:
            # Normalize data
            df_norm = season_data.copy()
            for col in valid_features:
                if df_norm[col].std() > 0:
                    df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
                else:
                    df_norm[col] = 0
            
            # Get target vector
            target_vec = df_norm[df_norm['player_name'] == selected_player][valid_features].values
            
            if len(target_vec) > 0:
                target_vec = target_vec[0]
                
                # Calculate distances
                distances = []
                for idx, row in df_norm.iterrows():
                    if row['player_name'] == selected_player:
                        continue
                    
                    # Only compare same position group if possible
                    if row.get('position_category') == player_data.get('position_category'):
                        vec = row[valid_features].values
                        dist = np.linalg.norm(target_vec - vec)
                        distances.append((row['player_name'], dist))
                
                # Sort by distance
                distances.sort(key=lambda x: x[1])
                top_5 = distances[:5]
                
                # Helper for navigation
                def go_to_player(c, t, p):
                    st.session_state['prof_comp'] = c
                    st.session_state['prof_team'] = t
                    st.session_state['prof_player'] = p

                # Display cards
                cols = st.columns(5)
                for i, (name, dist) in enumerate(top_5):
                    with cols[i]:
                        similarity = max(0, 100 - (dist * 10)) # approximate score
                        
                        # Pre-calculate target filters
                        target_comp = None
                        target_team = None
                        
                        p_info = match_data[match_data['player_name'] == name]
                        if not p_info.empty:
                            target_comp = p_info['competition_name'].mode()[0]
                            target_team = p_info['team_name'].mode()[0]
                        
                        # Use button with callback to avoid "set after instantiate" error
                        if target_comp and target_team:
                            st.button(
                                f"{name}\n({similarity:.0f}%)", 
                                key=f"sim_btn_{i}",
                                on_click=go_to_player,
                                args=(target_comp, target_team, name)
                            )
                        else:
                            st.info(f"{name}\n({similarity:.0f}%)")
                        
                        # st.caption(f"Similarity: {similarity:.0f}%")
        else:
            st.warning("Not enough data to calculate similarity")

    # ================= TAB 4: AI SCOUTING REPORT =================
    with tab_ai:
        st.subheader("💡 AI Scouting Report")
        
        # Calculate full percentiles for narrative
        narrative_percentiles = {}
        if not peers.empty and len(peers) > 5:
            for col in COLS_TO_SUM:
                if f"{col}_per90" in peers.columns:
                    metric = f"{col}_per90"
                    val = player_data.get(metric, 0)
                    if peers[metric].max() > 0:
                        pct = (peers[metric] < val).mean() * 100
                    else:
                        pct = 50
                    narrative_percentiles[col] = pct
        
        # Identify Strengths (>70th) and Weaknesses (<30th)
        strengths = [k for k, v in narrative_percentiles.items() if v >= 75]
        weaknesses = [k for k, v in narrative_percentiles.items() if v <= 25]
        
        # Sort by intensity
        strengths.sort(key=lambda k: narrative_percentiles[k], reverse=True)
        weaknesses.sort(key=lambda k: narrative_percentiles[k])
        
        col_str, col_weak, col_assess = st.columns(3)
        
        with col_str:
            st.markdown("##### ✅ Strengths & Capabilities")
            if strengths:
                for s in strengths[:5]:
                    nice_name = s.replace('_', ' ').title()
                    pct = narrative_percentiles[s]
                    st.success(f"**{nice_name}** (Top {100-pct:.0f}%)")
            else:
                st.write("No elite metrics identified relative to peers.")

        with col_weak:
            st.markdown("##### ⚠️ Areas for Improvement")
            if weaknesses:
                for w in weaknesses[:5]:
                    nice_name = w.replace('_', ' ').title()
                    pct = narrative_percentiles[w]
                    st.error(f"**{nice_name}** (Bottom {pct:.0f}%)")
            else:
                st.write("No significant statistical weaknesses found.")
                
        with col_assess:
            st.markdown("##### 🎯 Overall Assessment")
            
            # Simple archetype logic
            role = "Balanced Player"
            if len(strengths) > len(weaknesses):
                if any(x in strengths for x in ['goals', 'shots', 'xg']):
                    role = "Elite Finisher"
                elif any(x in strengths for x in ['assists', 'key_passes', 'xa']):
                    role = "Creative Playmaker"
                elif any(x in strengths for x in ['tackles', 'interceptions']):
                    role = "Defensive Rock"
                elif any(x in strengths for x in ['dribbles_completed', 'progressive_carries']):
                    role = "Ball Progressor"
            
            st.info(f"**Archetype: {role}**")
            
            summary = f"{selected_player} plays as a **{player_pos}**. "
            if strengths:
                top_s = strengths[0].replace('_', ' ')
                summary += f"Their standout quality is **{top_s}**, performing better than {narrative_percentiles[strengths[0]]:.0f}% of peers. "
            if weaknesses:
                top_w = weaknesses[0].replace('_', ' ')
                summary += f"However, they struggle with **{top_w}**. "
            
            st.write(summary)
            
    # ================= TAB 5: CONTEXTUAL ANALYSIS =================
    with tab_context:
        st.subheader("🏟️ Contextual Performance")
        
        # 1. Real Home vs Away Analysis
        if 'venue' in p_matches.columns:
            st.markdown("##### 🏠 Home vs 🛫 Away")
            
            # Aggregate by venue
            venue_stats = p_matches.groupby('venue').agg({
                'goals': 'sum', 'assists': 'sum', 'minutes_played': 'sum', 'match_id': 'count'
            }).reset_index()
            
            if not venue_stats.empty:
                # Calc per 90
                venue_stats['goals_p90'] = (venue_stats['goals'] / venue_stats['minutes_played'] * 90).fillna(0)
                venue_stats['assists_p90'] = (venue_stats['assists'] / venue_stats['minutes_played'] * 90).fillna(0)
                
                col1, col2 = st.columns(2)
                
                # Helper display
                def show_venue_col(col, v_name):
                    v_data = venue_stats[venue_stats['venue'] == v_name]
                    with col:
                        st.markdown(f"**{v_name}** ({int(v_data['match_id'].sum()) if not v_data.empty else 0} matches)")
                        if not v_data.empty:
                            g90 = v_data['goals_p90'].iloc[0]
                            a90 = v_data['assists_p90'].iloc[0]
                            st.metric("Goals/90", f"{g90:.2f}")
                            st.metric("Assists/90", f"{a90:.2f}")
                        else:
                            st.write("No matches")

                show_venue_col(col1, 'Home')
                show_venue_col(col2, 'Away')
                
                # Insight
                home_g = venue_stats[venue_stats['venue'] == 'Home']['goals_p90'].sum()
                away_g = venue_stats[venue_stats['venue'] == 'Away']['goals_p90'].sum()
                
                if home_g > away_g * 1.5:
                     st.info("🏠 **Home Comfort:** Significantly higher scoring rate at home.")
                elif away_g > home_g * 1.5:
                     st.info("🛫 **Road Warrior:** Performs exceptionally well away from home.")
                else:
                     st.success("⚖️ **Balanced:** Consistent performance across venues.")

            else:
                st.info("No venue data available.")
        
        st.markdown("---")

        # 2. Real Season Progression
        if 'match_date' in p_matches.columns:
            st.markdown("##### 📅 Season Progression")
            prog_df = p_matches.sort_values('match_date').copy()
            if not prog_df.empty:
                # Rolling average form (last 5 matches avg rating/goals)
                prog_df['Rolling Goals'] = prog_df['goals'].rolling(window=5, min_periods=1).mean()
                prog_df['Rolling xG'] = prog_df['xg'].rolling(window=5, min_periods=1).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prog_df['match_date'], y=prog_df['Rolling Goals'], name='Rolling Goals (5 games)', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=prog_df['match_date'], y=prog_df['Rolling xG'], name='Rolling xG (5 games)', line=dict(dash='dot', color='white')))
                
                fig.update_layout(title="Form Over Time (Rolling Avg)", height=350, template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
                
                # Insight
                recent_form = prog_df['Rolling Goals'].iloc[-1] if len(prog_df) > 0 else 0
                avg_form = prog_df['Rolling Goals'].mean()
                
                if recent_form > avg_form * 1.2:
                    st.success("🔥 **Hot Form:** Scoring above season average in recent games.")
                elif recent_form < avg_form * 0.8:
                    st.warning("❄️ **Cold Streak:** Recent scoring form is below season average.")
                else:
                    st.info("➡️ **Consistent:** Maintaining steady performance levels.")

        
        st.markdown("---")

        # 3. Conceptual Analysis (Opponent Strength etc.)
        st.markdown("##### 🧠 Advanced Contexts (Simulation Preview)")
        st.info("Performance vs Opponent Strength & Match Importance (Requires opponent tiering data)")
        
        # Simulate Contexts for visualization
        contexts = ['vs Top 6', 'vs Mid Table', 'vs Bottom 6', 'Cup Finals']
        base_g = player_data.get('goals_per90', 0.1)
        sim_vals = [base_g * (0.8 + np.random.rand()*0.5) for _ in contexts]
        
        fig = px.bar(x=contexts, y=sim_vals, title="Hypothetical Performance vs Contexts", 
                     labels={'x': 'Context', 'y': 'Goals/90'}, color=sim_vals, color_continuous_scale='RdBu')
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

    # ================= TAB 6: TEMPORAL TRENDS =================
    with tab_trends:
        st.subheader("📈 Temporal Trends & Consistency")
        
        if not p_matches.empty and len(p_matches) > 1:
            # 1. Interactive Form Curve
            st.markdown("##### 🏃 Form Trajectory")
            
            # Sort by match
            p_trend = p_matches.copy()
            if 'match_date' in p_trend.columns:
                 p_trend = p_trend.sort_values('match_date')
            else:
                 p_trend = p_trend.sort_values('match_id') # Fallback
            
            # Select metric to track
            metrics = {
                'Goals': 'goals',
                'Assists': 'assists',
                'xG': 'xg',
                'Shots': 'shots',
                'Passes Completed': 'passes_completed',
                'Rating': 'rating' # If exists
            }
            
            available_metrics = {k: v for k, v in metrics.items() if v in p_trend.columns}
            
            if available_metrics:
                col_sel, col_empty = st.columns([1, 2])
                with col_sel:
                    selected_metric = st.selectbox("Track metric:", list(available_metrics.keys()), key="trend_metric_sel")
                metric_col = available_metrics[selected_metric]
                
                # Plot trend
                fig = go.Figure()
                
                # Actual
                fig.add_trace(go.Scatter(
                    x=list(range(len(p_trend))),
                    y=p_trend[metric_col],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#00CC96', width=2)
                ))
                
                # Rolling average (3-game)
                if len(p_trend) >= 3:
                    rolling_avg = p_trend[metric_col].rolling(window=3, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(p_trend))),
                        y=rolling_avg,
                        mode='lines',
                        name='3-match Avg',
                        line=dict(color='white', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"{selected_metric} per Match",
                    xaxis_title="Match Sequence",
                    yaxis_title=selected_metric,
                    height=350,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, width='stretch')
                
                # Trend detection comments
                if len(p_trend) >= 5:
                    recent_avg = p_trend[metric_col].tail(5).mean()
                    overall_avg = p_trend[metric_col].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Overall Avg", f"{overall_avg:.2f}")
                    with col2: st.metric("Last 5 Avg", f"{recent_avg:.2f}", delta=f"{recent_avg - overall_avg:.2f}")
                    with col3:
                        if overall_avg > 0:
                            if recent_avg > overall_avg * 1.1:
                                st.success("📈 **Improving Form**")
                            elif recent_avg < overall_avg * 0.9:
                                st.warning("📉 **Declining Form**")
                            else:
                                st.info("➡️ **Stable Form**")
                        else:
                            st.info("Insufficient data for trend signal.")

            st.markdown("---")
            
            # 2. Consistency Analysis
            st.markdown("##### 📊 Consistency Check")
            
            if 'goals' in p_trend.columns:
                goals_std = p_trend['goals'].std()
                goals_mean = p_trend['goals'].mean()
                
                if goals_mean > 0:
                    cv = (goals_std / goals_mean) * 100
                    
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Goals Std Dev", f"{goals_std:.2f}")
                    with c2:
                        if cv < 50:
                            st.success(f"✅ **Consistent** (CV: {cv:.0f}%)")
                        elif cv < 100:
                            st.info(f"ℹ️ **Moderate** (CV: {cv:.0f}%)")
                        else:
                            st.warning(f"⚠️ **Volatile** (CV: {cv:.0f}%)")
                else:
                    st.info("No goals scored to analyze consistency.")

            st.markdown("---")

            # 3. Future Concept
            st.markdown("##### 🔮 Future Predictions (Concept)")
            st.caption("Forecast based on ARIMA/Prophet models (Simulation)")
            
            # Simulated forecast
            hist_fv = list(p_trend['xg'].fillna(0).values)[-15:] if 'xg' in p_trend.columns else [0.2]*15
            forecast_fv = [np.mean(hist_fv) * (1 + np.sin(x)) for x in range(5)]
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(y=hist_fv, name='History', mode='lines+markers', line=dict(color='#636EFA')))
            fig_f.add_trace(go.Scatter(x=list(range(len(hist_fv)-1, len(hist_fv)+4)), 
                                     y=[hist_fv[-1]] + forecast_fv[:4], 
                                     name='Forecast', mode='lines+markers', line=dict(color='#EF553B', dash='dash')))
            
            fig_f.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20), template="plotly_dark")
            st.plotly_chart(fig_f, width='stretch')

        else:
            st.info("Need at least 2 matches to show temporal trends.")

    st.markdown("---")
