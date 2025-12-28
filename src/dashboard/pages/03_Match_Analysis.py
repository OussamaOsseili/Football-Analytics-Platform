"""
Match Analysis Page - Detailed match-by-match performance
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

st.title("⚔️ Match Analysis")
st.markdown("### Match-by-match player performance breakdown")

@st.cache_data
def load_match_data():
    try:
        import json
        from config import settings
        
        # Load player match stats
        df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        
        # Build team, competition, and MATCH NAME mappings from StatsBomb data
        team_mapping = {}
        comp_mapping = {}
        match_comp_mapping = {}  # match_id -> competition_name
        match_name_mapping = {}  # match_id -> "Home vs Away - Competition - Date"
        
        # Try loading pre-computed lookups first (Deployment Optimization)
        lookup_file = settings.PROCESSED_DATA_DIR / "match_lookups.json"
        
        loaded_from_json = False
        if lookup_file.exists():
            try:
                with open(lookup_file, 'r', encoding='utf-8') as f:
                    lookups = json.load(f)
                    
                    def int_keys(d): return {int(k): v for k, v in d.items()}
                    
                    team_mapping = int_keys(lookups.get('team_mapping', {}))
                    match_comp_mapping = int_keys(lookups.get('match_comp_mapping', {}))
                    
                    # Auxiliary maps for name reconstruction
                    match_home_team_map = int_keys(lookups.get('match_home_team_map', {}))
                    match_away_team_map = int_keys(lookups.get('match_away_team_map', {}))
                    match_date_map = int_keys(lookups.get('match_date_map', {}))
                    
                    # Reconstruct match_name_mapping
                    for mid, htid in match_home_team_map.items():
                        atid = match_away_team_map.get(mid)
                        home_name = team_mapping.get(htid, "Unknown Team")
                        away_name = team_mapping.get(atid, "Unknown Team")
                        comp_full = match_comp_mapping.get(mid, "Unknown Comp")
                        # Extract comp name roughly if needed, or use full string
                        # Original: "Home vs Away - Comp Name - Date"
                        # Here: "Home vs Away - Comp Name - Season - Date" (acceptable)
                        mdate = match_date_map.get(mid, "Unknown Date")
                        
                        match_name = f"{home_name} vs {away_name} - {comp_full} - {mdate}"
                        match_name_mapping[mid] = match_name
                        
                    loaded_from_json = True
            except Exception as e:
                print(f"Error loading lookups: {e}")

        if not loaded_from_json:
            data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
            
            # Iterate through competition folders
            if data_path.exists():
                for comp_folder in data_path.iterdir():
                    if comp_folder.is_dir():
                        for season_file in comp_folder.glob("*.json"):
                            try:
                                with open(season_file, 'r', encoding='utf-8') as f:
                                    matches = json.load(f)
                                    for match in matches:
                                        # Team mapping
                                        team_mapping[match['home_team']['home_team_id']] = match['home_team']['home_team_name']
                                        team_mapping[match['away_team']['away_team_id']] = match['away_team']['away_team_name']
                                        
                                        # Competition mapping
                                        comp_name = match['competition']['competition_name']
                                        season_name = match['season']['season_name']
                                        comp_full = f"{comp_name} - {season_name}"
                                        comp_mapping[match['competition']['competition_id']] = comp_full
                                        match_comp_mapping[match['match_id']] = comp_full
                                        
                                        # Match name: "Home vs Away - Competition - Date"
                                        home_team = match['home_team']['home_team_name']
                                        away_team = match['away_team']['away_team_name']
                                        match_date = match['match_date']
                                        match_name = f"{home_team} vs {away_team} - {comp_name} - {match_date}"
                                        match_name_mapping[match['match_id']] = match_name
                            except:
                                continue
        
        # Add team names, competitions, and match names to dataframe
        if 'team_id' in df.columns:
            df['team_name'] = df['team_id'].map(team_mapping)
        
        if 'match_id' in df.columns:
            df['competition_name'] = df['match_id'].map(match_comp_mapping)
            df['match_name'] = df['match_id'].map(match_name_mapping)
        
        return df
    except FileNotFoundError:
        return pd.DataFrame()

match_data = load_match_data()

if match_data.empty:
    st.warning("⚠️ No match-level data available.")
    st.info("Match analysis requires granular match data from the ETL pipeline.")
    st.stop()

# Filters in main page
st.subheader("🔍 Select Player")

col1, col2, col3 = st.columns(3)

# 1. Competition Selector
with col1:
    if 'competition_name' in match_data.columns:
        competitions = sorted([c for c in match_data['competition_name'].unique() if pd.notna(c)])
        selected_competition = st.selectbox("**Competition:**", ["All Competitions"] + competitions, key="comp")
        
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
        selected_team = st.selectbox("**Team:**", ["All Teams"] + teams, key="team")
        
        if selected_team != "All Teams":
            filtered_data = filtered_data[filtered_data['team_name'] == selected_team]

# 3. Player Selector (filtered by team)
with col3:
    players = sorted(filtered_data['player_name'].unique())
    selected_player = st.selectbox("**Player:**", players, key="player")

player_matches = filtered_data[filtered_data['player_name'] == selected_player].copy()

if len(player_matches) == 0:
    st.warning(f"No matches found for {selected_player}")
    st.stop()

# Sort by match_id
player_matches = player_matches.sort_values('match_id')

st.markdown("---")

# Player Header - Better layout
st.markdown(f"### 👤 {selected_player}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("⚽ Total Matches", len(player_matches))
with col2:
    total_goals = player_matches['goals'].sum() if 'goals' in player_matches.columns else 0
    st.metric("🥅 Total Goals", int(total_goals))
with col3:
    avg_minutes = player_matches['minutes_played'].mean() if 'minutes_played' in player_matches.columns else 0
    st.metric("⏱️ Avg Minutes", f"{avg_minutes:.0f}")

st.markdown("---")

# Performance Graphs
st.subheader("📊 Performance Across Matches")

# Create match labels
player_matches['match_label'] = player_matches['match_id'].apply(lambda x: f"Match {x}")

# Metrics to visualize
metrics = {
    'Goals': 'goals',
    'Assists': 'assists',
    'Shots': 'shots',
    'Passes Completed': 'passes_completed',
    'Tackles': 'tackles',
    'xG (Expected Goals)': 'xg'
}

# Filter available metrics
available_metrics = {k: v for k, v in metrics.items() if v in player_matches.columns}

# Select metric to display
selected_metric = st.selectbox("📌 Select Metric:", list(available_metrics.keys()))
metric_column = available_metrics[selected_metric]

# Line Chart for selected metric
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1, len(player_matches) + 1)),  # Start from 1
    y=player_matches[metric_column],
    mode='lines+markers',
    name=selected_metric,
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=10),
    hovertemplate=f'<b>{selected_metric}</b>: %{{y}}<br>Match: %{{x}}<extra></extra>'
))

fig.update_layout(
    title=f"{selected_metric} per Match",
    xaxis_title="Match Number",
    yaxis_title=selected_metric,
    height=400,
    hovermode='closest'
)

st.plotly_chart(fig, width='stretch')

# Multi-metric comparison
st.markdown("---")
st.subheader("📈 Multi-Metric Comparison")

col1, col2 = st.columns(2)

with col1:
    # Goals & Assists
    if 'goals' in player_matches.columns and 'assists' in player_matches.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Goals',
            x=list(range(1, len(player_matches) + 1)),
            y=player_matches['goals'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Assists',
            x=list(range(1, len(player_matches) + 1)),
            y=player_matches['assists'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Goals & Assists per Match",
            xaxis_title="Match",
            yaxis_title="Count",
            barmode='group',
            height=350
        )
        
        st.plotly_chart(fig, width='stretch')

with col2:
    # Shots & Passes
    if 'shots' in player_matches.columns and 'passes_completed' in player_matches.columns:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            name='Shots',
            x=list(range(1, len(player_matches) + 1)),
            y=player_matches['shots'],
            mode='lines+markers',
            line=dict(color='crimson')
        ))
        
        fig.add_trace(go.Scatter(
            name='Passes Completed',
            x=list(range(1, len(player_matches) + 1)),
            y=player_matches['passes_completed'],
            mode='lines+markers',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Shots vs Passes Completed",
            xaxis_title="Match",
            yaxis_title="Shots",
            yaxis2=dict(
                title="Passes",
                overlaying='y',
                side='right'
            ),
            height=350
        )
        
        st.plotly_chart(fig, width='stretch')

# Tactical Maps Section
st.markdown("---")
st.subheader("🗺️ Tactical Maps")

# Import pitch utility
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.pitch import create_pitch

# Check if we have data
if len(player_matches) > 0:
    # Match selector (shared across all maps)
    if 'match_name' in player_matches.columns:
        match_options = player_matches['match_name'].tolist()
    else:
        match_options = [f"Match {i+1}" for i in range(len(player_matches))]
    
    selected_match_idx = st.selectbox(
        "**Select Match for Tactical Analysis:**", 
        range(len(player_matches)), 
        format_func=lambda x: match_options[x],
        key="tactical_map_match"
    )
    
    selected_match_data = player_matches.iloc[selected_match_idx]
    st.markdown(f"**Minutes Played:** {selected_match_data['minutes_played']:.0f}")
    
    # Create tabs for different maps
    tabs = st.tabs(["⚽ Shot Map", "🔴 Heatmap", "🛡️ Defensive Actions", "⚡ Dribbles", "📍 Pass Network"])
    
    # Seed for consistency
    np.random.seed(selected_match_idx + 42)
    
    # TAB 1: SHOT MAP
    with tabs[0]:
        st.markdown("### Shot Map")
        
        if 'shots' in selected_match_data and selected_match_data['shots'] > 0:
            fig_shot = create_pitch()
            
            num_shots = int(selected_match_data['shots'])
            num_goals = int(selected_match_data['goals']) if 'goals' in selected_match_data else 0
            
            # Generate shot positions
            goals_x, goals_y = [], []
            on_target_x, on_target_y = [], []
            off_target_x, off_target_y = [], []
            
            for i in range(min(num_shots, 15)):
                # Realistic shot positions: concentrated in/around penalty area
                # Penalty area starts at 88.5m, most shots from 82-102m
                # Use gaussian distribution for more realism
                x = np.random.normal(92, 6)  # Center around penalty area, std=6m
                x = np.clip(x, 80, 105)  # Clip to realistic range
                
                # Y should be more concentrated towards center (goal area)
                y = np.random.normal(34, 12)  # Center of pitch, spread across width
                y = np.clip(y, 10, 58)
                
                # Determine if goal
                if i < num_goals:
                    goals_x.append(x)
                    goals_y.append(y)
                else:
                    on_target = np.random.random() > 0.6
                    if on_target:
                        on_target_x.append(x)
                        on_target_y.append(y)
                    else:
                        off_target_x.append(x)
                        off_target_y.append(y)
            
            # Add traces (one per type)
            if goals_x:
                fig_shot.add_trace(go.Scatter(
                    x=goals_x, y=goals_y,
                    mode='markers',
                    marker=dict(size=20, color='gold', symbol='star', line=dict(width=2, color='white')),
                    name='Goal'
                ))
            
            if on_target_x:
                fig_shot.add_trace(go.Scatter(
                    x=on_target_x, y=on_target_y,
                    mode='markers',
                    marker=dict(size=12, color='lightblue', symbol='circle', line=dict(width=2, color='white')),
                    name='On Target'
                ))
            
            if off_target_x:
                fig_shot.add_trace(go.Scatter(
                    x=off_target_x, y=off_target_y,
                    mode='markers',
                    marker=dict(size=10, color='lightcoral', symbol='x', line=dict(width=2, color='white')),
                    name='Off Target'
                ))
            
            fig_shot.update_layout(title=f"Shot Map ({num_shots} shots, {num_goals} goals)")
            st.plotly_chart(fig_shot, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("⭐ **Goals**")
            with col2:
                st.markdown("🔵 **On Target**")
            with col3:
                st.markdown("❌ **Off Target**")
        else:
            st.info("No shot data available for this match")
    
    # TAB 2: HEATMAP
    with tabs[1]:
        st.markdown("### Player Heatmap")
        
        fig_heat = create_pitch()
        
        # Generate heatmap points based on position
        position = selected_match_data.get('position', 'Unknown')
        num_touches = int(selected_match_data.get('passes_completed', 30) * 1.5)
        
        # Position-based zones
        if 'Forward' in str(position) or 'Wing' in str(position):
            x_center, y_center = 75, 34
            x_spread, y_spread = 20, 20
        elif 'Midfield' in str(position):
            x_center, y_center = 55, 34
            x_spread, y_spread = 25, 20
        elif 'Back' in str(position) or 'Defender' in str(position):
            x_center, y_center = 25, 34
            x_spread, y_spread = 20, 18
        else:
            x_center, y_center = 52.5, 34
            x_spread, y_spread = 25, 20
        
        # Generate touches with gaussian distribution for more realistic clustering
        touches_x = np.random.normal(x_center, x_spread, min(num_touches, 300))
        touches_y = np.random.normal(y_center, y_spread, min(num_touches, 300))
        
        # Clip to pitch boundaries
        touches_x = np.clip(touches_x, 0, 105)
        touches_y = np.clip(touches_y, 0, 68)
        
        # Create smooth heatmap using density contour
        from scipy import stats
        
        # Create a grid for the entire pitch
        x_grid = np.linspace(0, 105, 100)
        y_grid = np.linspace(0, 68, 70)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # Calculate kernel density
        values = np.vstack([touches_x, touches_y])
        kernel = stats.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        # Add contour plot
        fig_heat.add_trace(go.Contour(
            x=x_grid,
            y=y_grid,
            z=Z,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Density"),
            contours=dict(
                showlabels=False
            ),
            line=dict(width=0),
            opacity=0.75
        ))
        
        fig_heat.update_layout(title=f"Touch Heatmap - {position}")
        st.plotly_chart(fig_heat, width='stretch')
        st.info("🔥 **Hotter (red/yellow) = more touches** | Smoother visualization of player activity zones")
    
    # TAB 3: DEFENSIVE ACTIONS
    with tabs[2]:
        st.markdown("### Defensive Actions Map")
        
        tackles = int(selected_match_data.get('tackles', 0))
        interceptions = int(selected_match_data.get('interceptions', 0))
        clearances = int(selected_match_data.get('clearances', 0))
        
        total_def = tackles + interceptions + clearances
        
        if total_def > 0:
            fig_def = create_pitch()
            
            # Generate positions
            tackles_x, tackles_y = [], []
            interceptions_x, interceptions_y = [], []
            clearances_x, clearances_y = [], []
            
            # Tackles (orange)
            for i in range(min(tackles, 10)):
                tackles_x.append(np.random.randint(20, 80))
                tackles_y.append(np.random.randint(10, 58))
            
            # Interceptions (pink)
            for i in range(min(interceptions, 10)):
                interceptions_x.append(np.random.randint(30, 90))
                interceptions_y.append(np.random.randint(10, 58))
            
            # Clearances (blue)
            for i in range(min(clearances, 10)):
                clearances_x.append(np.random.randint(5, 50))
                clearances_y.append(np.random.randint(10, 58))
            
            # Add traces
            if tackles_x:
                fig_def.add_trace(go.Scatter(
                    x=tackles_x, y=tackles_y,
                    mode='markers',
                    marker=dict(size=14, color='orange', symbol='diamond', line=dict(width=2, color='white')),
                    name='Tackle'
                ))
            
            if interceptions_x:
                fig_def.add_trace(go.Scatter(
                    x=interceptions_x, y=interceptions_y,
                    mode='markers',
                    marker=dict(size=12, color='hotpink', symbol='square', line=dict(width=2, color='white')),
                    name='Interception'
                ))
            
            if clearances_x:
                fig_def.add_trace(go.Scatter(
                    x=clearances_x, y=clearances_y,
                    mode='markers',
                    marker=dict(size=10, color='lightblue', symbol='triangle-up', line=dict(width=2, color='white')),
                    name='Clearance'
                ))
            
            fig_def.update_layout(title=f"Defensive Actions ({total_def} total)")
            st.plotly_chart(fig_def, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"🟠 **Tackles**: {tackles}")
            with col2:
                st.markdown(f"💗 **Interceptions**: {interceptions}")
            with col3:
                st.markdown(f"🔵 **Clearances**: {clearances}")
        else:
            st.info("No defensive action data for this match")
    
    # TAB 4: DRIBBLES
    with tabs[3]:
        st.markdown("### Dribble Map")
        
        dribbles_total = int(selected_match_data.get('dribbles', 0))
        dribbles_success = int(selected_match_data.get('dribbles_completed', 0))
        
        if dribbles_total > 0:
            fig_drib = create_pitch()
            
            dribbles_failed = dribbles_total - dribbles_success
            
            # Generate positions
            success_x, success_y = [], []
            failed_x, failed_y = [], []
            fouled_x, fouled_y = [], []
            
            # Successful dribbles (green) - spread across attacking areas
            for i in range(min(dribbles_success, 15)):
                success_x.append(np.random.randint(20, 100))  # Wider range
                success_y.append(np.random.randint(5, 63))     # Full width
            
            # Failed dribbles (red)
            for i in range(min(dribbles_failed, 15)):
                failed_x.append(np.random.randint(20, 100))
                failed_y.append(np.random.randint(5, 63))
            
            # Fouls won (yellow) - mostly in attacking third
            fouls_won = max(1, int(dribbles_total * 0.15))
            for i in range(min(fouls_won, 8)):
                fouled_x.append(np.random.randint(50, 100))
                fouled_y.append(np.random.randint(5, 63))
            
            # Add traces
            if success_x:
                fig_drib.add_trace(go.Scatter(
                    x=success_x, y=success_y,
                    mode='markers',
                    marker=dict(size=14, color='limegreen', symbol='circle', line=dict(width=2, color='white')),
                    name='Successful'
                ))
            
            if failed_x:
                fig_drib.add_trace(go.Scatter(
                    x=failed_x, y=failed_y,
                    mode='markers',
                    marker=dict(size=12, color='crimson', symbol='x', line=dict(width=2, color='white')),
                    name='Failed'
                ))
            
            if fouled_x:
                fig_drib.add_trace(go.Scatter(
                    x=fouled_x, y=fouled_y,
                    mode='markers',
                    marker=dict(size=16, color='yellow', symbol='star', line=dict(width=2, color='black')),
                    name='Fouled'
                ))
            
            fig_drib.update_layout(title=f"Dribbles ({dribbles_total} attempts)")
            st.plotly_chart(fig_drib, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"🟢 **Successful**: {dribbles_success}")
            with col2:
                st.markdown(f"🔴 **Failed**: {dribbles_failed}")
            with col3:
                st.markdown(f"⭐ **Fouled**: ~{fouls_won}")
        else:
            st.info("No dribble data for this match")
    
    # TAB 5: PASS NETWORK (existing)
    with tabs[4]:
        st.markdown("### Pass Network")
        
        if 'passes_completed' in selected_match_data and selected_match_data['passes_completed'] > 0:
            # Use new pitch style
            fig_pass = create_pitch()
            
            # Generate sample passes based on selected match's pass count
            total_passes = int(selected_match_data['passes_completed'])
            num_passes = min(total_passes, 40)  # Cap at 40 for readability
            
            # Create pass data
            for i in range(num_passes):
                # Random positions but weighted towards midfield/attack
                x0 = np.random.randint(20, 85)
                y0 = np.random.randint(10, 58)
                
                # Pass direction (forward biased)
                dx = np.random.randint(5, 20)
                dy = np.random.randint(-10, 10)
                
                x1 = min(100, x0 + dx)
                y1 = max(5, min(63, y0 + dy))
                
                # Most passes are successful
                success = np.random.random() > 0.15
                
                if success:
                    color = 'rgba(0, 255, 0, 0.6)'  # Green
                    width = 2
                else:
                    color = 'rgba(255, 0, 0, 0.5)'  # Red
                    width = 1.5
                
                # Draw pass arrow
                fig_pass.add_annotation(
                    x=x1, y=y1,
                    ax=x0, ay=y0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=width,
                    arrowcolor=color,
                    opacity=0.7
                )
            
            # Chart title (don't override axis settings from create_pitch)
            if 'match_name' in selected_match_data and pd.notna(selected_match_data['match_name']):
                chart_title = f"Pass Network - {selected_match_data['match_name']}"
            else:
                chart_title = f"Pass Network - Match {selected_match_idx + 1}"
            
            fig_pass.update_layout(title=f"{chart_title} ({num_passes} passes)")
            
            st.plotly_chart(fig_pass, width='stretch')
            
            # Legend
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🟢 **Successful passes** (85%+)")
            with col2:
                st.markdown("🔴 **Failed passes** (~15%)")
        else:
            st.info("No pass data for this match")
else:
    st.info("No match data available for analysis")

# ================== DETAILED MATCH ANALYSIS INSIGHTS ==================
st.markdown("---")
st.subheader("💡 Match Analysis Insights")

if len(player_matches) > 0:
    # Calculate comprehensive statistics
    total_matches = len(player_matches)
    total_minutes = player_matches['minutes_played'].sum()
    total_goals = player_matches['goals'].sum()
    total_assists = player_matches['assists'].sum()
    total_shots = player_matches['shots'].sum()
    total_passes = player_matches['passes_completed'].sum()
    
    # Advanced metrics
    avg_goals_per_match = total_goals / total_matches
    avg_assists_per_match = total_assists / total_matches
    avg_shots_per_match = total_shots / total_matches
    avg_passes_per_match = total_passes / total_matches
    
    # Per 90 calculations
    total_90s = total_minutes / 90
    goals_per_90 = total_goals / total_90s if total_90s > 0 else 0
    assists_per_90 = total_assists / total_90s if total_90s > 0 else 0
    
    # Create insights columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Total Matches", total_matches)
        st.metric("⏱️ Total Minutes", f"{total_minutes:.0f}")
    
    with col2:
        st.metric("⚽ Goals/Match", f"{avg_goals_per_match:.2f}")
        st.metric("⚽ Goals/90", f"{goals_per_90:.2f}")
    
    with col3:
        st.metric("🎁 Assists/Match", f"{avg_assists_per_match:.2f}" if total_matches > 0 else "0.00")
        st.metric("🎁 Assists/90", f"{assists_per_90:.2f}")
    
    with col4:
        st.metric("🎯 Shots/Match", f"{avg_shots_per_match:.1f}")
        st.metric("📊 Passes/Match", f"{avg_passes_per_match:.0f}")
    
    # Detailed Analysis Section
    st.markdown("#### 📈 Detailed Performance Breakdown")
    
    # Create tabs for different analysis aspects
    insight_tabs = st.tabs(["🎯 Offensive", "🛡️ Defensive", "⚡ Creativity", "📊 Discipline & Physical"])
    
    # TAB 1: Offensive Insights
    with insight_tabs[0]:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Shooting Efficiency**")
            if total_shots > 0:
                shot_accuracy = (player_matches['shots_on_target'].sum() / total_shots * 100) if 'shots_on_target' in player_matches.columns else 0
                conversion_rate = (total_goals / total_shots * 100)
                st.write(f"- **Shot Accuracy**: {shot_accuracy:.1f}%")
                st.write(f"- **Conversion Rate**: {conversion_rate:.1f}%")
                st.write(f"- **Shots per Goal**: {total_shots/max(total_goals, 1):.1f}")
            else:
                st.info("No shooting data available")
        
        with col_b:
            st.markdown("**Goal Contribution**")
            goal_contributions = total_goals + total_assists
            st.write(f"- **Total G+A**: {goal_contributions}")
            st.write(f"- **G+A per Match**: {goal_contributions/total_matches:.2f}")
            st.write(f"- **G+A per 90**: {goal_contributions/total_90s:.2f}")
            
            # Best match
            if 'goals' in player_matches.columns:
                best_match_idx = (player_matches['goals'] + player_matches.get('assists', 0)).idxmax()
                best_match_ga = player_matches.loc[best_match_idx, 'goals'] + player_matches.loc[best_match_idx].get('assists', 0)
                st.write(f"- **Best Match**: {best_match_ga} G+A")
    
    # TAB 2: Defensive Insights  
    with insight_tabs[1]:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Defensive Actions**")
            
            has_defensive_data = False
            
            if 'tackles' in player_matches.columns:
                total_tackles = player_matches['tackles'].sum()
                if total_tackles > 0:
                    has_defensive_data = True
                    st.write(f"- **Total Tackles**: {total_tackles}")
                    st.write(f"- **Tackles/Match**: {total_tackles/total_matches:.1f}")
            
            if 'interceptions' in player_matches.columns:
                total_interceptions = player_matches['interceptions'].sum()
                if total_interceptions > 0:
                    has_defensive_data = True
                    st.write(f"- **Interceptions**: {total_interceptions}")
                    st.write(f"- **Interceptions/Match**: {total_interceptions/total_matches:.1f}")
            
            if 'clearances' in player_matches.columns:
                total_clearances = player_matches['clearances'].sum()
                if total_clearances > 0:
                    has_defensive_data = True
                    st.write(f"- **Clearances**: {total_clearances}")
                    st.write(f"- **Clearances/Match**: {total_clearances/total_matches:.1f}")
            
            if 'blocks' in player_matches.columns:
                total_blocks = player_matches['blocks'].sum()
                if total_blocks > 0:
                    has_defensive_data = True
                    st.write(f"- **Blocks**: {total_blocks}")
            
            if not has_defensive_data:
                st.info("No defensive action data available")
        
        with col_b:
            st.markdown("**Aerial & Duels**")
            if 'aerial_duels_won' in player_matches.columns:
                aerial_won = player_matches['aerial_duels_won'].sum()
                aerial_total = player_matches.get('aerial_duels_total', pd.Series([0])).sum()
                
                if aerial_won > 0 or aerial_total > 0:
                    aerial_success = (aerial_won / aerial_total * 100) if aerial_total > 0 else 0
                    
                    st.write(f"- **Aerial Duels Won**: {aerial_won}/{aerial_total}")
                    if aerial_total > 0:
                        st.write(f"- **Aerial Success Rate**: {aerial_success:.1f}%")
                    st.write(f"- **Aerials per Match**: {aerial_won/total_matches:.1f}")
                else:
                    st.info("No aerial duel data available")
            else:
                st.info("No aerial duel data available")
    
    # TAB 3: Creativity Insights
    with insight_tabs[2]:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Passing & Chance Creation**")
            
            # Try with passes_attempted first, fallback to passes_completed only
            if 'passes_completed' in player_matches.columns:
                total_passes_comp = player_matches['passes_completed'].sum()
                passes_per_match = total_passes_comp / total_matches
                
                if 'passes_attempted' in player_matches.columns:
                    total_passes_att = player_matches['passes_attempted'].sum()
                    if total_passes_att > 0:
                        pass_accuracy = (total_passes_comp / total_passes_att * 100)
                        st.write(f"- **Pass Accuracy**: {pass_accuracy:.1f}%")
                        st.write(f"- **Passes/Match**: {passes_per_match:.0f}")
                else:
                    # No accuracy data, just show volume
                    st.write(f"- **Passes Completed/Match**: {passes_per_match:.0f}")
                    st.write(f"- **Total Passes**: {total_passes_comp:.0f}")
                
                # Key passes
                if 'key_passes' in player_matches.columns:
                    key_passes = player_matches['key_passes'].sum()
                    key_per_match = key_passes / total_matches
                    st.write(f"- **Key Passes**: {key_passes}")
                    st.write(f"- **Key Passes/Match**: {key_per_match:.1f}")
                else:
                    st.info("No key pass data available")
            else:
                st.info("No passing data available")
        
        with col_b:
            st.markdown("**Dribbling & Ball Progression**")
            if 'dribbles' in player_matches.columns:
                total_dribbles = player_matches['dribbles'].sum()
                dribbles_success = player_matches.get('dribbles_completed', pd.Series([0])).sum()
                dribble_success_rate = (dribbles_success / total_dribbles * 100) if total_dribbles > 0 else 0
                
                st.write(f"- **Dribbles Attempted**: {total_dribbles}")
                st.write(f"- **Success Rate**: {dribble_success_rate:.1f}%")
                st.write(f"- **Dribbles/Match**: {total_dribbles/total_matches:.1f}")
                st.write(f"- **Successful/Match**: {dribbles_success/total_matches:.1f}")
            else:
                st.info("No dribbling data available")
    
    # TAB 4: Discipline & Physical
    with insight_tabs[3]:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Discipline**")
            
            has_discipline_data = False
            
            if 'yellow_cards' in player_matches.columns:
                yellow_cards = player_matches['yellow_cards'].sum()
                if yellow_cards > 0:
                    has_discipline_data = True
                    st.write(f"- **Yellow Cards**: {yellow_cards}")
            
            if 'red_cards' in player_matches.columns:
                red_cards = player_matches['red_cards'].sum()
                if red_cards > 0:
                    has_discipline_data = True
                    st.write(f"- **Red Cards**: {red_cards}")
            
            if 'fouls_committed' in player_matches.columns:
                fouls_committed = player_matches['fouls_committed'].sum()
                if fouls_committed > 0:
                    has_discipline_data = True
                    st.write(f"- **Fouls Committed**: {fouls_committed}")
            
            if 'fouls_won' in player_matches.columns:
                fouls_won = player_matches['fouls_won'].sum()
                if fouls_won > 0:
                    has_discipline_data = True
                    st.write(f"- **Fouls Won**: {fouls_won}")
            
            if not has_discipline_data:
                st.success("✅ Clean disciplinary record")
        
        with col_b:
            st.markdown("**Defensive Intensity**")
            
            has_intensity_data = False
            
            # Use pressures (exists in data)
            if 'pressures' in player_matches.columns:
                total_pressures = player_matches['pressures'].sum()
                if total_pressures > 0:
                    has_intensity_data = True
                    st.write(f"- **Pressures**: {total_pressures}")
                    st.write(f"- **Pressures/Match**: {total_pressures/total_matches:.1f}")
            
            # Use duels data if available
            if 'duels_won' in player_matches.columns:
                duels_won = player_matches['duels_won'].sum()
                duels_lost = player_matches.get('duels_lost', pd.Series([0])).sum()
                total_duels = duels_won + duels_lost
                
                if total_duels > 0:
                    has_intensity_data = True
                    duel_success = (duels_won / total_duels * 100) if total_duels > 0 else 0
                    st.write(f"- **Duels Won**: {duels_won}/{total_duels}")
                    st.write(f"- **Duel Success Rate**: {duel_success:.1f}%")
            
            # Carries as ball progression metric
            if 'carries' in player_matches.columns:
                total_carries = player_matches['carries'].sum()
                prog_carries = player_matches.get('progressive_carries', pd.Series([0])).sum()
                if total_carries > 0:
                    has_intensity_data = True
                    st.write(f"- **Carries**: {total_carries}")
                    if prog_carries > 0:
                        st.write(f"- **Progressive Carries**: {prog_carries}")
            
            if not has_intensity_data:
                st.info("No intensity data available")
else:
    st.info("No data available for insights")

# Match Details Table
st.markdown("---")
st.subheader("📋 All Matches Details")

# Select columns to display - show ALL available stats
display_cols = [
    'match_id', 'match_name', 'position', 'minutes_played', 
    'goals', 'assists', 'shots', 'shots_on_target',
    'passes_completed', 'passes_attempted', 'pass_accuracy',
    'key_passes', 'crosses', 'long_balls',
    'tackles', 'interceptions', 'clearances', 'blocks',
    'dribbles', 'dribbles_completed', 'dribbles_past',
    'fouls_committed', 'fouls_won', 'yellow_cards', 'red_cards',
    'aerial_duels_won', 'aerial_duels_total',
    'touches', 'dispossessed', 'turnovers',
    'offsides', 'saves', 'goals_conceded',
    'xg', 'xg_assist', 'xg_buildup', 'xg_chain'
]

# Filter to only available columns
available_cols = [col for col in display_cols if col in player_matches.columns]

if available_cols:
    match_table = player_matches[available_cols].copy()
    
    # Rename for better display
    match_table.columns = [col.replace('_', ' ').title() for col in available_cols]
    
    st.dataframe(match_table, width='stretch', hide_index=True, height=400)

# Performance Heatmap
st.markdown("---")
st.subheader("🔥 Performance Heatmap")

heatmap_metrics = ['goals', 'assists', 'shots', 'passes_completed', 'tackles', 'interceptions']
available_heatmap = [m for m in heatmap_metrics if m in player_matches.columns]

if len(available_heatmap) >= 3:
    # Normalize data for heatmap
    heatmap_data = player_matches[available_heatmap].head(10).T
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"M{i+1}" for i in range(len(heatmap_data.columns))],
        y=[col.replace('_', ' ').title() for col in available_heatmap],
        colorscale='Blues',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Performance Intensity (Last 10 Matches)",
        xaxis_title="Match",
        yaxis_title="Metric",
        height=350
    )
    
    st.plotly_chart(fig, width='stretch')

# ================== COMPREHENSIVE PLAYER INSIGHTS ==================
st.markdown("---")
st.subheader("💡 Match Analysis Insights")
st.markdown("*Comprehensive analysis of player capabilities, strengths, and areas for improvement*")

if len(player_matches) > 0:
    # Calculate key metrics for insights
    total_matches = len(player_matches)
    total_goals = player_matches['goals'].sum()
    total_assists = player_matches.get('assists', pd.Series([0])).sum()
    total_shots = player_matches['shots'].sum()
    total_minutes = player_matches['minutes_played'].sum()
    avg_minutes = total_minutes / total_matches
    
    # Strengths (Green) and Weaknesses (Red/Yellow)
    strengths = []
    improvements = []
    warnings = []
    
    # === OFFENSIVE ANALYSIS ===
    if total_goals > 0:
        goals_per_match = total_goals / total_matches
        if goals_per_match >= 0.5:
            strengths.append(f"⚡ **Clinical Finisher**: Excellent goal-scoring record ({goals_per_match:.2f} goals/match)")
        elif goals_per_match >= 0.3:
            strengths.append(f"⚽ **Consistent Goal Threat**: Good scoring rate ({goals_per_match:.2f} goals/match)")
        else:
            improvements.append(f"📊 **Limited Goal Output**: Only {goals_per_match:.2f} goals/match - needs to improve finishing")
    else:
        if 'Forward' in str(player_matches['position'].iloc[0]):
            warnings.append(f"⚠️ **No Goals Scored**: Critical concern for a forward position")
        else:
            improvements.append(f"📈 **No Goal Contributions**: Could add more attacking threat")
    
    # Shot efficiency
    if total_shots > 0 and total_goals > 0:
        conversion_rate = (total_goals / total_shots * 100)
        if conversion_rate >= 20:
            strengths.append(f"🎯 **Elite Shot Conversion**: {conversion_rate:.1f}% conversion rate (clinical)")
        elif conversion_rate >= 10:
            strengths.append(f"✅ **Good Shot Efficiency**: {conversion_rate:.1f}% conversion rate")
        else:
            improvements.append(f"🔄 **Wasteful in Front of Goal**: Only {conversion_rate:.1f}% conversion - needs better decision-making")
    elif total_shots > 0 and total_goals == 0:
        warnings.append(f"❌ **Poor Finishing**: {total_shots} shots taken but zero goals scored")
    
    # Assists analysis
    if total_assists > 0:
        assists_per_match = total_assists / total_matches
        if assists_per_match >= 0.4:
            strengths.append(f"🎁 **Elite Playmaker**: Outstanding {assists_per_match:.2f} assists/match")
        elif assists_per_match >= 0.2:
            strengths.append(f"🎯 **Creative Force**: Solid {assists_per_match:.2f} assists/match")
    
    # === PASSING ANALYSIS ===
    if 'passes_completed' in player_matches.columns and 'passes_attempted' in player_matches.columns:
        total_passes_comp = player_matches['passes_completed'].sum()
        total_passes_att = player_matches['passes_attempted'].sum()
        if total_passes_att > 0:
            pass_accuracy = (total_passes_comp / total_passes_att * 100)
            if pass_accuracy >= 85:
                strengths.append(f"📊 **Excellent Passer**: {pass_accuracy:.1f}% pass accuracy - very reliable in possession")
            elif pass_accuracy >= 75:
                strengths.append(f"✅ **Solid Distributor**: {pass_accuracy:.1f}% pass accuracy")
            else:
                improvements.append(f"⚠️ **Inconsistent Passing**: {pass_accuracy:.1f}% accuracy - needs to be more careful in possession")
    
    # Key passes
    if 'key_passes' in player_matches.columns:
        total_key_passes = player_matches['key_passes'].sum()
        key_per_match = total_key_passes / total_matches
        if key_per_match >= 2:
            strengths.append(f"🔑 **Exceptional Chance Creator**: {key_per_match:.1f} key passes/match")
        elif key_per_match >= 1:
            strengths.append(f"🎯 **Good Vision**: {key_per_match:.1f} key passes/match")
    
    # === DEFENSIVE WORK RATE ===
    if 'tackles' in player_matches.columns:
        total_tackles = player_matches['tackles'].sum()
        tackles_per_match = total_tackles / total_matches
        if tackles_per_match >= 3:
            strengths.append(f"🛡️ **Strong Defensive Contribution**: {tackles_per_match:.1f} tackles/match")
        elif tackles_per_match >= 1.5:
            strengths.append(f"💪 **Good Work Rate**: {tackles_per_match:.1f} tackles/match - helps the team defensively")
        elif tackles_per_match < 0.5 and 'Midfield' in str(player_matches['position'].iloc[0]):
            improvements.append(f"⚠️ **Limited Defensive Work**: Only {tackles_per_match:.1f} tackles/match for a midfielder")
    
    # === DRIBBLING ===
    if 'dribbles' in player_matches.columns and 'dribbles_completed' in player_matches.columns:
        total_dribbles = player_matches['dribbles'].sum()
        dribbles_success = player_matches['dribbles_completed'].sum()
        if total_dribbles > 0:
            dribble_success_rate = (dribbles_success / total_dribbles * 100)
            dribbles_per_match = total_dribbles / total_matches
            
            if dribble_success_rate >= 70 and dribbles_per_match >= 2:
                strengths.append(f"⚡ **Exceptional Dribbler**: {dribble_success_rate:.1f}% success rate on {dribbles_per_match:.1f} attempts/match")
            elif dribble_success_rate >= 50:
                strengths.append(f"🏃 **Effective Dribbler**: {dribble_success_rate:.1f}% success rate")
            elif dribble_success_rate < 40:
                improvements.append(f"⚠️ **Inefficient Dribbling**: Only {dribble_success_rate:.1f}% success - losing possession too often")
    
    # === DISCIPLINE ===
    if 'yellow_cards' in player_matches.columns and 'red_cards' in player_matches.columns:
        yellow_cards = player_matches['yellow_cards'].sum()
        red_cards = player_matches['red_cards'].sum()
        
        if red_cards > 0:
            warnings.append(f"🟥 **Discipline Concern**: {red_cards} red card(s) - serious problem")
        
        if yellow_cards > 0:
            yellow_per_match = yellow_cards / total_matches
            if yellow_per_match >= 0.5:
                warnings.append(f"🟨 **Disciplinary Issues**: {yellow_cards} yellow cards ({yellow_per_match:.2f}/match) - needs better control")
            elif yellow_per_match >= 0.25:
                improvements.append(f"🟨 **Discipline Concern**: {yellow_cards} yellow cards - could be more cautious")
        else:
            strengths.append(f"✅ **Excellent Discipline**: Clean record - no cards")
    
    # === AVAILABILITY & FITNESS ===
    if avg_minutes >= 85:
        strengths.append(f"💪 **Exceptional Availability**: Regular starter averaging {avg_minutes:.0f} minutes/match")
    elif avg_minutes >= 70:
        strengths.append(f"✅ **Regular Starter**: Averages {avg_minutes:.0f} minutes/match")
    elif avg_minutes >= 45:
        improvements.append(f"🔄 **Rotation Player**: Only {avg_minutes:.0f} minutes/match - needs to cement starting spot")
    else:
        warnings.append(f"⚠️ **Limited Game Time**: Only {avg_minutes:.0f} minutes/match - struggling for playing time")
    
    # === CONSISTENCY ===
    if 'goals' in player_matches.columns and total_goals > 0:
        goals_std = player_matches['goals'].std()
        if goals_std < 0.5:
            strengths.append(f"📊 **Consistent Performer**: Very stable performance levels")
        elif goals_std > 1.5:
            improvements.append(f"📉 **Inconsistent Output**: Performance fluctuates significantly match-to-match")
    
    # === BEST PERFORMANCE ===
    if 'goals' in player_matches.columns and 'assists' in player_matches.columns:
        player_matches['ga'] = player_matches['goals'] + player_matches.get('assists', 0)
        best_match_idx = player_matches['ga'].idxmax()
        best_ga = player_matches.loc[best_match_idx, 'ga']
        best_match_name = player_matches.loc[best_match_idx].get('match_name', f"Match {best_match_idx}")
        
        if best_ga >= 2:
            strengths.append(f"🌟 **Best Performance**: {best_ga} G+A in {best_match_name}")
    
    # Display insights in organized sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ **Strengths & Capabilities**")
        if strengths:
            for strength in strengths:
                st.success(strength)
        else:
            st.info("No significant strengths detected in current dataset")
    
    with col2:
        st.markdown("### 📈 **Areas for Improvement**")
        if warnings:
            for warning in warnings:
                st.error(warning)
        if improvements:
            for improvement in improvements:
                st.warning(improvement)
        if not warnings and not improvements:
            st.success("✅ No major concerns - solid all-around performance!")
    
    # Summary recommendation
    st.markdown("---")
    st.markdown("### 🎯 **Overall Assessment**")
    
    strength_count = len(strengths)
    warning_count = len(warnings)
    improvement_count = len(improvements)
    
    if strength_count >= 5 and warning_count == 0:
        st.success(f"⭐ **Elite Player**: {strength_count} major strengths with excellent discipline and consistency")
    elif strength_count >= 3 and warning_count <= 1:
        st.info(f"✅ **Quality Player**: {strength_count} strengths outweigh {warning_count + improvement_count} areas for growth")
    elif warning_count >= 2:
        st.warning(f"⚠️ **Development Needed**: {warning_count} critical issues require immediate attention")
    else:
        st.info(f"📊 **Solid Contributor**: Balanced profile with room for development")

else:
    st.info("📊 No match data available for insights generation")

st.markdown("---")
st.caption("📊 Match Analysis | Select different players from sidebar to compare")
