"""
Team Analysis Page - Team chemistry and partnerships
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

import json
from config import settings

st.set_page_config(page_title="Team Analysis", layout="wide")
st.title("🤝 Team Analysis")
st.markdown("### Team chemistry, partnerships, and balance")

# ================= DATA LOADING =================
@st.cache_data
def load_match_data_v2():
    try:
        df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        team_mapping = {}
        # Try loading pre-computed lookups first (Deployment Optimization)
        lookup_file = settings.PROCESSED_DATA_DIR / "match_lookups.json"
        
        loaded_from_json = False
        if lookup_file.exists():
            try:
                with open(lookup_file, 'r', encoding='utf-8') as f:
                    lookups = json.load(f)
                    def int_keys(d): return {int(k): v for k, v in d.items()}
                    
                    team_mapping = int_keys(lookups.get('team_mapping', {}))
                    # comp_mapping works on match_id in the original code, but here we mapped match_id -> comp_name in lookups
                    # origin: comp_mapping[match_id] = comp_name
                    # JSON: match_comp_mapping[match_id] = comp_name
                    match_comp_mapping = int_keys(lookups.get('match_comp_mapping', {}))
                    
                    if 'team_name' not in df.columns and 'team_id' in df.columns:
                        df['team_name'] = df['team_id'].map(team_mapping)
                    
                    if 'competition_name' not in df.columns and 'match_id' in df.columns:
                        df['competition_name'] = df['match_id'].map(match_comp_mapping)
                        
                    loaded_from_json = True
            except Exception as e:
                print(f"Error loading lookups: {e}")

        if not loaded_from_json:
            data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
            
            # Ensure team names and competition names
            if 'team_name' not in df.columns or 'competition_name' not in df.columns:
                 comp_mapping = {}
                 if data_path.exists():
                     for comp_folder in data_path.iterdir():
                        if comp_folder.is_dir():
                            for season_file in comp_folder.glob("*.json"):
                                try:
                                    with open(season_file, 'r', encoding='utf-8') as f:
                                        matches = json.load(f)
                                        for match in matches:
                                            # Filter Women's competitions (Safety check)
                                            if "Women's" in match['competition']['competition_name']:
                                                continue
                                                
                                            match_id = match['match_id']
                                            team_mapping[match['home_team']['home_team_id']] = match['home_team']['home_team_name']
                                            team_mapping[match['away_team']['away_team_id']] = match['away_team']['away_team_name']
                                            
                                            # Map match to competition
                                            comp_mapping[match_id] = match['competition']['competition_name']
                                except: continue
                 
                 if 'team_name' not in df.columns:
                     df['team_name'] = df['team_id'].map(team_mapping)
                 
                 if 'competition_name' not in df.columns:
                     df['competition_name'] = df['match_id'].map(comp_mapping)

        return df
    except FileNotFoundError:
        return pd.DataFrame()

COLS_TO_SUM = [
    'goals', 'assists', 'xg', 'xa', 'shots', 'key_passes',
    'passes_completed', 'passes', 'progressive_passes',
    'dribbles_completed', 'carries', 'progressive_carries',
    'tackles', 'interceptions', 'blocks', 'clearances', 'pressures',
    'minutes_played'
]

# Removed cache temporarily to force refresh after position categorization fix
def build_team_data_with_styles(match_df):
    """Build team data by aggregating match data directly - PERMANENT FIX"""
    if match_df.empty:
        return pd.DataFrame()
    
    # Aggregate by player AND team to get per-team stats
    groupby_cols = ['player_name', 'team_name']
    existing = [c for c in COLS_TO_SUM if c in match_df.columns]
    
    agg_dict = {c: 'sum' for c in existing}
    agg_dict['minutes_played'] = 'sum'
    agg_dict['match_id'] = 'count'
    agg_dict['position'] = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    agg_dict['competition_name'] = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    
    season = match_df.groupby(groupby_cols).agg(agg_dict).reset_index()
    season.rename(columns={'match_id': 'matches_played'}, inplace=True)
    
    # Position category
    def cat_pos(p):
        p = str(p).lower()
        if 'keeper' in p: return 'GK'
        if 'midfield' in p: return 'Midfielder'
        if 'back' in p or 'defender' in p: return 'Defender'
        if 'wing' in p or 'forward' in p: return 'Forward'
        return 'Unknown'
    season['position_category'] = season['position'].apply(cat_pos)
    
    # Calculate scores from match stats (FALLBACK only)
    season['offensive_score_raw'] = ((season.get('goals',0)/season.get('minutes_played',1)*90)*40 + 
                                  (season.get('assists',0)/season.get('minutes_played',1)*90)*30).clip(0, 100)
    season['defensive_score_raw'] = ((season.get('tackles',0)/season.get('minutes_played',1)*90)*30 + 
                                  (season.get('interceptions',0)/season.get('minutes_played',1)*90)*30).clip(0, 100)
    season['creative_score_raw'] = ((season.get('key_passes',0)/season.get('minutes_played',1)*90)*30 + 
                                 (season.get('xa',0)/season.get('minutes_played',1)*90)*40).clip(0, 100)
    
    # Merge playing styles AND percentile-based SCORES from enhanced CSV
    # These scores are much better (0-100 distribution) than the raw calc above
    try:
        enhanced = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv")
        # Ensure columns exist
        cols_to_merge = ['player_name', 'primary_style']
        score_cols = ['offensive_score', 'defensive_score', 'creative_score']
        valid_score_cols = [c for c in score_cols if c in enhanced.columns]
        cols_to_merge.extend(valid_score_cols)
        
        if 'primary_style' in enhanced.columns:
            season = season.merge(
                enhanced[cols_to_merge],
                on='player_name',
                how='left'
            )
            season['primary_style'] = season['primary_style'].fillna('Unknown')
            
            # Use raw scores if enhanced scores are missing (NaN)
            if 'offensive_score' in season.columns:
                 season['offensive_score'] = season['offensive_score'].fillna(season['offensive_score_raw'])
            else:
                 season['offensive_score'] = season['offensive_score_raw']
                 
            if 'defensive_score' in season.columns:
                 season['defensive_score'] = season['defensive_score'].fillna(season['defensive_score_raw'])
            else:
                 season['defensive_score'] = season['defensive_score_raw']

            if 'creative_score' in season.columns:
                 season['creative_score'] = season['creative_score'].fillna(season['creative_score_raw'])
            else:
                 season['creative_score'] = season['creative_score_raw']

        else:
            season['primary_style'] = 'Unknown'
            season['offensive_score'] = season['offensive_score_raw']
            season['defensive_score'] = season['defensive_score_raw']
            season['creative_score'] = season['creative_score_raw']
            
    except Exception as e:
        print(f"Error merging enhanced stats: {e}")
        season['primary_style'] = 'Unknown'
        season['offensive_score'] = season['offensive_score_raw']
        season['defensive_score'] = season['defensive_score_raw']
        season['creative_score'] = season['creative_score_raw']
    
    return season

match_data = load_match_data_v2()
full_data = build_team_data_with_styles(match_data)

# Ensure we have data
if full_data.empty:
    st.warning("⚠️ No data available.")
    st.stop()

# ================= TEAM SELECTION =================
# ================= TEAM SELECTION =================
with st.expander("🔍 Filter Teams", expanded=True):
    col1, col2 = st.columns(2)
    
    # 1. Competition (if available in aggregation, or mapped)
    # Note: 'players_season_stats' might not have 'competition_name'.
    # We try to infer or fallback.
    available_comps = []
    if 'competition_name' in full_data.columns:
        available_comps = sorted(full_data['competition_name'].dropna().unique())
    
    with col1:
        sel_comp = "All Competitions"
        if available_comps:
            sel_comp = st.selectbox("Competition", ["All Competitions"] + available_comps)
    
    # Filter data by comp
    active_data = full_data.copy()
    if sel_comp != "All Competitions":
        active_data = active_data[active_data['competition_name'] == sel_comp]
    
    # 2. Team Selection
    available_teams = sorted(active_data['team_name'].dropna().unique()) if 'team_name' in active_data.columns else []
    
    with col2:
        selected_team = st.selectbox("Select Team", ["All Teams"] + available_teams)
    
    # Filter by team
    if selected_team != "All Teams":
        data = active_data[active_data['team_name'] == selected_team]
    else:
        data = active_data
        if len(available_teams) > 1 and selected_team == "All Teams":
            st.info("💡 Select a specific team to see optimal XI and weaknesses.")

if data.empty:
    st.warning("No players found for this selection.")
    st.stop()

# Team balance analysis
st.subheader("⚖️ Team Balance Analysis")

# Aggregate by position
if 'position_category' in data.columns:
    position_balance = data.groupby('position_category').agg({
        'offensive_score': 'mean',
        'defensive_score': 'mean',
        'creative_score': 'mean'
    }).reset_index()
    
    if not position_balance.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Offensive',
            x=position_balance['position_category'],
            y=position_balance['offensive_score']
        ))
        
        fig.add_trace(go.Bar(
            name='Defensive',
            x=position_balance['position_category'],
            y=position_balance['defensive_score']
        ))
        
        fig.add_trace(go.Bar(
            name='Creative',
            x=position_balance['position_category'],
            y=position_balance['creative_score']
        ))
        
        fig.update_layout(
            barmode='group',
            title="Average Scores by Position",
            yaxis_title="Score (0-100)",
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')

# Team Form/Momentum Chart
st.markdown("---")
st.subheader("📈 Team Form & Momentum")

if selected_team != "All Teams":
    try:
        import json
        from pathlib import Path
        
        # Load match results from JSON files
        # Load match results from lookups (Cloud optimized)
        lookup_file = settings.PROCESSED_DATA_DIR / "match_lookups.json"
        
        team_matches = []
        loaded_matches = False
        
        if lookup_file.exists():
            try:
                with open(lookup_file, 'r', encoding='utf-8') as f:
                    lookups = json.load(f)
                    
                    # Maps (String keys)
                    home_map = lookups.get('match_home_team_map', {})
                    away_map = lookups.get('match_away_team_map', {})
                    home_score_map = lookups.get('match_home_score_map', {})
                    away_score_map = lookups.get('match_away_score_map', {})
                    date_map = lookups.get('match_date_map', {})
                    team_map = lookups.get('team_mapping', {}) # ID -> Name
                    
                    # Iterate all matches in lookup
                    for mid_str in home_map.keys():
                        m_date = date_map.get(mid_str, '')
                        if m_date < '2020-01-01': continue
                        
                        htid_str = str(home_map[mid_str])
                        atid_str = str(away_map[mid_str])
                        
                        h_name = team_map.get(htid_str, "Unknown")
                        a_name = team_map.get(atid_str, "Unknown")
                        
                        # Check if selected team participated (partial match for safety)
                        if selected_team in h_name:
                             team_matches.append({
                                'date': m_date,
                                'goals_for': home_score_map.get(mid_str, 0),
                                'goals_against': away_score_map.get(mid_str, 0)
                            })
                        elif selected_team in a_name:
                             team_matches.append({
                                'date': m_date,
                                'goals_for': away_score_map.get(mid_str, 0),
                                'goals_against': home_score_map.get(mid_str, 0)
                            })
                    loaded_matches = True
            except Exception as e:
                print(f"Error loading score lookups: {str(e)}")
        
        if not loaded_matches:
            data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
            if data_path.exists():
                for comp_folder in data_path.iterdir():
                    if comp_folder.is_dir():
                        for season_file in comp_folder.glob("*.json"):
                            try:
                                with open(season_file, 'r', encoding='utf-8') as f:
                                    matches = json.load(f)
                                    for match in matches:
                                        if "Women's" in match['competition']['competition_name']:
                                            continue
                                        
                                        # Filter old matches (exclude 2015-2019)
                                        match_date = match.get('match_date', '')
                                        if match_date < '2020-01-01':
                                            continue
                                        
                                        home_team = match['home_team']['home_team_name']
                                        away_team = match['away_team']['away_team_name']
                                        
                                        if selected_team in home_team:
                                            team_matches.append({
                                                'date': match_date,
                                                'goals_for': match['home_score'],
                                                'goals_against': match['away_score']
                                            })
                                        elif selected_team in away_team:
                                            team_matches.append({
                                                'date': match.get('match_date', ''),
                                                'goals_for': match['away_score'],
                                                'goals_against': match['home_score']
                                            })
                            except:
                                continue
        
        if team_matches:
            # Convert to DataFrame and sort by date
            form_df = pd.DataFrame(team_matches)
            form_df['date'] = pd.to_datetime(form_df['date'])
            form_df = form_df.sort_values('date')
            
            # Calculate rolling averages
            form_df['goals_for_avg'] = form_df['goals_for'].rolling(window=5, min_periods=1).mean()
            form_df['goals_against_avg'] = form_df['goals_against'].rolling(window=5, min_periods=1).mean()
            
            # Create dual-axis chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=form_df['date'],
                y=form_df['goals_for_avg'],
                name='Goals Scored (5-match avg)',
                line=dict(color='#51CF66', width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=form_df['date'],
                y=form_df['goals_against_avg'],
                name='Goals Conceded (5-match avg)',
                line=dict(color='#FF6B6B', width=3),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f"{selected_team} - Performance Trend",
                xaxis_title="Date",
                yaxis_title="Goals (5-match rolling average)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Trend indicator
            recent_form = form_df.tail(5)
            recent_gf = recent_form['goals_for'].mean()
            recent_ga = recent_form['goals_against'].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Recent Goals/Match", f"{recent_gf:.2f}")
            col2.metric("Recent Conceded/Match", f"{recent_ga:.2f}")
            col3.metric("Goal Difference", f"+{recent_gf - recent_ga:.2f}" if recent_gf > recent_ga else f"{recent_gf - recent_ga:.2f}")
        else:
            st.info("No match data available for form analysis")
    except Exception as e:
        st.error(f"Could not load form data: {str(e)}")
else:
    st.info("Select a specific team to view form and momentum")

# Style distribution
st.markdown("---")
st.subheader("🎯 Playing Style Distribution")

if 'primary_style' in data.columns and 'position_category' in data.columns:
    # Show ALL players, including those with Unknown styles
    style_position = data.groupby(['position_category', 'primary_style']).size().reset_index(name='count')
    
    if not style_position.empty:
        # Count players with classified vs unknown styles
        classified_count = len(data[data['primary_style'] != 'Unknown'])
        total_count = len(data)
        
        fig = px.sunburst(
            style_position,
            path=['position_category', 'primary_style'],
            values='count',
            title=f"Team Composition by Position and Style ({classified_count}/{total_count} players with ML-classified styles)",
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Show breakdown by position
        unknown_by_pos = data[data['primary_style'] == 'Unknown'].groupby('position_category').size()
        if len(unknown_by_pos) > 0:
            st.info(f"ℹ️ Players without ML-classified styles by position: " + 
                   ", ".join([f"{pos}: {count}" for pos, count in unknown_by_pos.items()]))
    else:
        st.warning("No player data available")
else:
    st.warning("Playing style data not available")

# Top Performers Table
st.markdown("---")
st.subheader("🏆 Top Performers")

if selected_team != "All Teams" and len(data) > 0:
    # Select key metrics
    perf_cols = ['player_name', 'position_category', 'primary_style', 'goals', 'assists', 
                 'tackles', 'interceptions', 'passes_completed', 'offensive_score', 'defensive_score']
    available_cols = [c for c in perf_cols if c in data.columns]
    
    if len(available_cols) > 3:
        perf_data = data[available_cols].copy()
        
        # Sort by offensive score by default
        if 'offensive_score' in perf_data.columns:
            perf_data = perf_data.sort_values('offensive_score', ascending=False)
        
        # Display top 10
        st.dataframe(
            perf_data.head(10),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("💡 Top 10 players by offensive score. Scroll horizontally to see all stats.")
    else:
        st.info("Insufficient data for performance table")
else:
    st.info("Select a specific team to view top performers")

# Optimal XI Builder (simplified)
st.markdown("---")
st.subheader("⭐ Optimal XI Suggestion")

st.markdown("""
**Based on multi-dimensional scores, here's a balanced lineup:**
""")

# Select top players per position
if 'position_category' in data.columns and 'offensive_score' in data.columns:
    positions_needed = {
        'GK': 1,
        'Defender': 4,
        'Midfielder': 3,
        'Forward': 3
    }
    
    optimal_xi = []
    
    for position, count in positions_needed.items():
        position_players = data[data['position_category'] == position]
        
        if len(position_players) > 0:
            # Sort by overall performance (average of scores)
            if 'offensive_score' in position_players.columns:
                position_players = position_players.copy()
                position_players['overall_score'] = position_players[
                    ['offensive_score', 'creative_score', 'defensive_score']
                ].mean(axis=1)
                
                top_players = position_players.nlargest(count, 'overall_score')
                optimal_xi.append(top_players)
    
    if optimal_xi:
        xi_df = pd.concat(optimal_xi)
        
        # Display in formation
        cols = st.columns(4)
        
        for idx, position in enumerate(['GK', 'Defender', 'Midfielder', 'Forward']):
            with cols[idx]:
                st.markdown(f"**{position}**")
                position_xi = xi_df[xi_df['position_category'] == position]
                
                for _, player in position_xi.iterrows():
                    st.write(f"• {player['player_name']}")
                    if 'overall_score' in player:
                        st.caption(f"Score: {player['overall_score']:.0f}/100")

# Team weaknesses
st.markdown("---")
st.subheader("⚠️ Team Weaknesses & Recommendations")

if 'position_category' in data.columns:
    position_counts = data['position_category'].value_counts()
    
    st.markdown("**Squad Depth:**")
    for position, count in position_counts.items():
        st.write(f"- {position}: {count} players")
    
    # Check balance
    if 'offensive_score' in data.columns and 'defensive_score' in data.columns:
        avg_off = data['offensive_score'].mean()
        avg_def = data['defensive_score'].mean()
        
        st.markdown("**Overall Team Profile:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Offensive Score", f"{avg_off:.0f}/100")
        with col2:
            st.metric("Avg Defensive Score", f"{avg_def:.0f}/100")
        
        # Recommendations - Context-aware logic
        st.markdown("**💡 Recommendations:**")
        
        # Determine team playing style based on score patterns
        score_diff = avg_off - avg_def
        
        if avg_off >= 60 and avg_def >= 60:
            # Balanced high-quality team
            st.success("✅ **Well-balanced squad** with strong offensive and defensive capabilities")
        elif avg_off >= 60 and avg_def < 50:
            # High-scoring, possession-based team
            if score_diff > 15:
                st.info("ℹ️ **Possession-dominant team**: High offensive output with lower defensive activity. This is typical for teams that control games through possession rather than defensive actions.")
            else:
                st.warning("⚠️ **Recommendation**: Consider adding defensive depth for tactical flexibility")
        elif avg_off < 50 and avg_def >= 60:
            # Defensive-minded team
            st.info("ℹ️ **Defensively solid team**: Consider adding creative/offensive players to increase goal threat")
        elif avg_off < 45 and avg_def < 45:
            # Struggling team
            st.error("⚠️ **Recommendation**: Squad needs reinforcement in both offensive and defensive areas")
        else:
            # Average balanced team
            st.success("✅ Squad has good offensive/defensive balance")

# Key Player Dependencies
st.markdown("---")
st.subheader("🎯 Key Player Dependencies")

if selected_team != "All Teams" and len(data) > 0:
    # Calculate team totals
    team_goals = data['goals'].sum() if 'goals' in data.columns else 0
    team_assists = data['assists'].sum() if 'assists' in data.columns else 0
    team_tackles = data['tackles'].sum() if 'tackles' in data.columns else 0
    
    if team_goals > 0 or team_assists > 0 or team_tackles > 0:
        # Calculate contributions
        dependencies = []
        
        for _, player in data.iterrows():
            player_deps = {'player_name': player['player_name'], 'position': player.get('position_category', 'Unknown')}
            
            if team_goals > 0 and 'goals' in data.columns:
                player_deps['goals_pct'] = (player['goals'] / team_goals) * 100
            if team_assists > 0 and 'assists' in data.columns:
                player_deps['assists_pct'] = (player['assists'] / team_assists) * 100
            if team_tackles > 0 and 'tackles' in data.columns:
                player_deps['tackles_pct'] = (player['tackles'] / team_tackles) * 100
            
            # Check if player is a key dependency (>25% in any category)
            max_contribution = max([player_deps.get('goals_pct', 0), 
                                   player_deps.get('assists_pct', 0),
                                   player_deps.get('tackles_pct', 0)])
            
            if max_contribution >= 25:
                player_deps['max_contribution'] = max_contribution
                dependencies.append(player_deps)
        
        if dependencies:
            # Sort by max contribution
            dependencies = sorted(dependencies, key=lambda x: x['max_contribution'], reverse=True)
            
            st.warning(f"⚠️ **{len(dependencies)} key dependencies identified** (players contributing ≥25% in any category)")
            
            # Show top dependencies
            for dep in dependencies[:5]:
                with st.expander(f"🔑 {dep['player_name']} ({dep['position']}) - {dep['max_contribution']:.1f}% max contribution"):
                    cols = st.columns(3)
                    if 'goals_pct' in dep:
                        cols[0].metric("Goals", f"{dep['goals_pct']:.1f}%")
                    if 'assists_pct' in dep:
                        cols[1].metric("Assists", f"{dep['assists_pct']:.1f}%")
                    if 'tackles_pct' in dep:
                        cols[2].metric("Tackles", f"{dep['tackles_pct']:.1f}%")
        else:
            st.success("✅ No critical dependencies - contributions are well distributed")
    else:
        st.info("Insufficient data for dependency analysis")
else:
    st.info("Select a specific team to view key player dependencies")

# Transfer Suggestions
st.markdown("---")
st.subheader("🔄 Transfer Suggestions")

if selected_team != "All Teams" and len(data) > 0:
    # Identify squad gaps
    gaps = []
    
    # Check position depth
    if 'position_category' in data.columns:
        pos_counts = data['position_category'].value_counts()
        for pos in ['Forward', 'Midfielder', 'Defender', 'GK']:
            count = pos_counts.get(pos, 0)
            if count < 3:
                gaps.append({'type': 'depth', 'position': pos, 'current': count, 'needed': 3 - count})
    
    # Check style diversity
    if 'primary_style' in data.columns and 'position_category' in data.columns:
        for pos in data['position_category'].unique():
            pos_players = data[data['position_category'] == pos]
            unique_styles = pos_players['primary_style'].nunique()
            if unique_styles <= 1 and len(pos_players) >= 2:
                gaps.append({'type': 'diversity', 'position': pos, 'styles': unique_styles})
    
    if gaps:
        st.warning(f"📊 **{len(gaps)} squad gaps identified**")
        
        # Load all players from other teams
        try:
            all_players = full_data[full_data['team_name'] != selected_team].copy()
            
            if len(all_players) > 0:
                suggestions = []
                seen_players = set()
                
                for gap in gaps[:3]:  # Limit to top 3 gaps
                    gap_pos = gap['position']
                    
                    # Filter candidates
                    candidates = all_players[all_players['position_category'] == gap_pos].copy()
                    
                    if len(candidates) > 0:
                        # Calculate fit scores
                        candidates['fit_score'] = 0
                        
                        # Position match (already filtered, so +40 points)
                        candidates['fit_score'] += 40
                        
                        # Performance score (30 points max)
                        if 'offensive_score' in candidates.columns:
                            candidates['fit_score'] += (candidates['offensive_score'] / 100) * 15
                        if 'defensive_score' in candidates.columns:
                            candidates['fit_score'] += (candidates['defensive_score'] / 100) * 15
                        
                        # Style diversity bonus (30 points if different style)
                        if gap['type'] == 'diversity' and 'primary_style' in candidates.columns:
                            current_styles = data[data['position_category'] == gap_pos]['primary_style'].unique()
                            candidates['style_bonus'] = candidates['primary_style'].apply(
                                lambda x: 30 if x not in current_styles else 0
                            )
                            candidates['fit_score'] += candidates['style_bonus']
                        
                        # Get top 3 suggestions for this gap
                        top_candidates = candidates.nlargest(3, 'fit_score')
                        
                        for _, candidate in top_candidates.iterrows():
                            p_name = candidate['player_name']
                            if p_name not in seen_players:
                                seen_players.add(p_name)
                                suggestions.append({
                                    'gap_type': gap['type'],
                                    'position': gap_pos,
                                    'player': p_name,
                                    'team': candidate.get('team_name', 'Unknown'),
                                    'style': candidate.get('primary_style', 'Unknown'),
                                    'fit_score': candidate['fit_score'],
                                    'off_score': candidate.get('offensive_score', 0),
                                    'def_score': candidate.get('defensive_score', 0)
                                })
                
                if suggestions:
                    # Sort by fit score
                    suggestions = sorted(suggestions, key=lambda x: x['fit_score'], reverse=True)
                    
                    st.success(f"✅ **{len(suggestions)} transfer suggestions generated**")
                    
                    # Display top 5
                    for i, sug in enumerate(suggestions[:5], 1):
                        gap_label = "Low Depth" if sug['gap_type'] == 'depth' else "Style Diversity"
                        with st.expander(f"#{i} - {sug['player']} ({sug['position']}) - Fit: {sug['fit_score']:.0f}/100"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Current Team:** {sug['team']}")
                                st.write(f"**Playing Style:** {sug['style']}")
                                st.write(f"**Gap Addressed:** {gap_label}")
                            with col2:
                                st.metric("Offensive Score", f"{sug['off_score']:.0f}/100")
                                st.metric("Defensive Score", f"{sug['def_score']:.0f}/100")
                                st.metric("Fit Score", f"{sug['fit_score']:.0f}/100")
                else:
                    st.info("No suitable transfer candidates found in dataset")
            else:
                st.info("Insufficient data from other teams for suggestions")
        except Exception as e:
            st.error(f"Could not generate suggestions: {str(e)}")
    else:
        st.success("✅ No significant squad gaps identified - well-balanced roster")
else:
    st.info("Select a specific team to view transfer suggestions")

# Team Chemistry Network
st.markdown("---")
st.subheader("🔗 Team Chemistry Network")

st.info("""
**How to interpret this network:**
- **Nodes (circles)** = Players, sized by total passes completed
- **Colors** = Position groups (Red: Forwards, Green: Midfielders, Blue: Defenders, Orange: GK)
- **Connections** = Players in the same position group (simplified team chemistry model)
- **Hover** over nodes to see player name, passes, and matches played

This visualization shows how players cluster by position and their passing activity. Larger nodes indicate players with higher passing volume.
""")

if selected_team != "All Teams" and len(data) >= 5:
    try:
        import networkx as nx
        import plotly.graph_objects as go
        
        # Create network based on player statistics
        G = nx.Graph()
        
        # Add nodes (players) with their stats
        for _, player in data.iterrows():
            G.add_node(
                player['player_name'],
                position=player.get('position_category', 'Unknown'),
                passes=player.get('passes_completed', 0),
                matches=player.get('matches_played', 0)
            )
        
        # Add edges between players in same position (simplified chemistry)
        for pos in data['position_category'].unique():
            pos_players = data[data['position_category'] == pos]['player_name'].tolist()
            for i, p1 in enumerate(pos_players):
                for p2 in pos_players[i+1:]:
                    G.add_edge(p1, p2, weight=1)
        
        # Create layout
        pos_layout = nx.spring_layout(G, k=2, iterations=50)
        
        # Prepare edge traces
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none', showlegend=False
        )
        
        for edge in G.edges():
            x0, y0 = pos_layout[edge[0]]
            x1, y1 = pos_layout[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Prepare node traces by position
        position_colors = {
            'Forward': '#FF6B6B',      # Red
            'Midfielder': '#51CF66',   # Green (changed from teal)
            'Defender': '#4DABF7',     # Blue (changed to brighter blue)
            'GK': '#FFA94D',           # Orange
            'Unknown': '#95A5A6'       # Gray
        }
        
        node_traces = []
        for position in data['position_category'].unique():
            pos_players = [n for n in G.nodes() if G.nodes[n]['position'] == position]
            
            node_trace = go.Scatter(
                x=[pos_layout[node][0] for node in pos_players],
                y=[pos_layout[node][1] for node in pos_players],
                mode='markers+text',
                name=position,
                text=[node.split()[-1] for node in pos_players],  # Last name only
                textposition='top center',
                marker=dict(
                    size=[G.nodes[node]['passes']/50 + 10 for node in pos_players],
                    color=position_colors.get(position, '#95A5A6'),
                    line=dict(width=2, color='white')
                ),
                hovertext=[f"{node}<br>Passes: {G.nodes[node]['passes']}<br>Matches: {G.nodes[node]['matches']}" 
                          for node in pos_players],
                hoverinfo='text'
            )
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=[edge_trace] + node_traces)
        fig.update_layout(
            title="Team Chemistry Network (Players grouped by position)",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, width='stretch')
        st.caption("Node size = Total passes completed | Colors = Position groups")
        
    except ImportError:
        st.warning("NetworkX library required for network visualization. Install with: `pip install networkx`")
    except Exception as e:
        st.error(f"Could not generate network: {str(e)}")
else:
    st.info("Select a specific team to view the team chemistry network.")
