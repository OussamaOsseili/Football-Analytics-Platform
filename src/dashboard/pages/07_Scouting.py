"""
Scouting Page - Advanced Player Search & Filtering
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from dashboard.utils import load_css

# Load Global CSS
load_css()

st.title("🔍 Scouting Tool")
st.markdown("### Find the perfect player for your squad")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_scouting_data():
    try:
        # Prefer the ENHANCED dataset with percentiles and styles
        enhanced_path = settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv"
        if enhanced_path.exists():
            df_enhanced = pd.read_csv(enhanced_path)
            
            # Load basic stats to get team_name
            basic_path = settings.PROCESSED_DATA_DIR / "players_season_stats.csv"
            if basic_path.exists():
                df_basic = pd.read_csv(basic_path)
                if 'team_name' in df_basic.columns:
                    # Merge team_name
                    # Group by player to handle duplicates (take first team)
                    teams = df_basic.groupby('player_name')['team_name'].first().reset_index()
                    df_enhanced = df_enhanced.merge(teams, on='player_name', how='left')
            
            # Ensure essential columns exist (fill with 0 if missing)
            for col in ['offensive_score', 'defensive_score', 'creative_score']:
                if col not in df_enhanced.columns:
                    df_enhanced[col] = 0
            if 'primary_style' not in df_enhanced.columns:
                df_enhanced['primary_style'] = 'Unknown'
            if 'team_name' not in df_enhanced.columns:
                df_enhanced['team_name'] = 'Unknown'
                
            return df_enhanced
            
        # Fallback to basic (not ideal, but works)
        return pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_scouting_data()

if data.empty:
    st.error("⚠️ No data available. Please run ETL pipeline.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Filter Criteria")

# 1. Demographics
positions = ['All'] + sorted(data['position_category'].dropna().unique().tolist()) if 'position_category' in data.columns else ['All']
sel_pos = st.sidebar.selectbox("Position", positions)

teams = ['All'] + sorted(data['team_name'].dropna().unique().tolist()) if 'team_name' in data.columns else ['All']
sel_team = st.sidebar.selectbox("Team", teams)

# 2. Playing Style
styles = ['All'] + sorted(data['primary_style'].dropna().unique().tolist())
sel_style = st.sidebar.selectbox("Playing Style", styles)

# 3. Metrics Sliders
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Performance Scores (0-100)**")

min_off = st.sidebar.slider("Offensive Score", 0, 100, 0, key="off_slider")
min_def = st.sidebar.slider("Defensive Score", 0, 100, 0, key="def_slider")
min_create = st.sidebar.slider("Creative Score", 0, 100, 0, key="create_slider")

# 4. Activity
st.sidebar.markdown("---")
min_mins = st.sidebar.slider("Min. Minutes Played", 0, 3000, 500, step=100)

# -----------------------------------------------------------------------------
# FILTERING LOGIC
# -----------------------------------------------------------------------------
filtered = data.copy()

if sel_pos != 'All':
    filtered = filtered[filtered['position_category'] == sel_pos]

if sel_team != 'All':
    filtered = filtered[filtered['team_name'] == sel_team]

if sel_style != 'All':
    filtered = filtered[filtered['primary_style'] == sel_style]

# Numeric Filters
filtered = filtered[
    (filtered['offensive_score'] >= min_off) &
    (filtered['defensive_score'] >= min_def) &
    (filtered['creative_score'] >= min_create) &
    (filtered['minutes_played'] >= min_mins)
]

# -----------------------------------------------------------------------------
# RESULTS AREA
# -----------------------------------------------------------------------------

# Metrics Summary
c1, c2, c3 = st.columns(3)
c1.metric("Players Found", len(filtered))
if not filtered.empty:
    c2.metric("Avg Offensive", f"{filtered['offensive_score'].mean():.1f}")
    c3.metric("Avg Defensive", f"{filtered['defensive_score'].mean():.1f}")

st.markdown("---")

if filtered.empty:
    st.info("No players match these criteria. Try relaxing the filters.")
else:
    # -------------------------------------------------------------------------
    # TOP PERFORMERS (Enhanced Table)
    # -------------------------------------------------------------------------
    st.subheader("🏆 Top Performers")
    
    # Sort controls
    col_sort, col_dl = st.columns([3, 1])
    with col_sort:
        sort_col = st.selectbox("Sort By", 
                               ['offensive_score', 'defensive_score', 'creative_score', 'goals_per90', 'assists_per90'],
                               format_func=lambda x: x.replace('_', ' ').title())
    
    # Prepare Display Data
    display_cols = [
        'player_name', 'team_name', 'position_category', 'primary_style',
        'offensive_score', 'defensive_score', 'creative_score',
        'goals_per90', 'assists_per90', 'minutes_played'
    ]
    
    # Select available cols
    final_cols = [c for c in display_cols if c in filtered.columns]
    
    df_display = filtered[final_cols].sort_values(sort_col, ascending=False).head(50)
    
    # Column Configuration for visual bars
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "player_name": st.column_config.TextColumn("Player", width="medium"),
            "team_name": st.column_config.TextColumn("Team", width="small"),
            "position_category": "Pos",
            "primary_style": st.column_config.TextColumn("Style", width="small"),
            "offensive_score": st.column_config.ProgressColumn(
                "Offensive", format="%d", min_value=0, max_value=100
            ),
            "defensive_score": st.column_config.ProgressColumn(
                "Defensive", format="%d", min_value=0, max_value=100
            ),
            "creative_score": st.column_config.ProgressColumn(
                "Creative", format="%d", min_value=0, max_value=100
            ),
            "goals_per90": st.column_config.NumberColumn("Goals/90", format="%.2f"),
            "assists_per90": st.column_config.NumberColumn("Ast/90", format="%.2f"),
            "minutes_played": st.column_config.NumberColumn("Mins", format="%d"),
        },
        height=500
    )

    # -------------------------------------------------------------------------
    # HIDDEN GEMS (Enhanced Logic)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("💎 Hidden Gems")
    st.caption("High-potential players with very limited playtime (< 200 mins)")
    
    # Filter gems based on Sidebar Context (Position/Style) but ignore sliders
    gems_pool = data.copy()
    
    if sel_pos != 'All':
        gems_pool = gems_pool[gems_pool['position_category'] == sel_pos]
    if sel_style != 'All':
        gems_pool = gems_pool[gems_pool['primary_style'] == sel_style]
        
    # Logic: Lower minutes but high impact in RELEVANT category
    # Calculate a 'Gem Score' weighted by position to avoid anomalies (e.g. Defender with 1 lucky goal)
    def calc_weighted_score(row):
        pos = str(row.get('position_category', ''))
        off = row.get('offensive_score', 0)
        defn = row.get('defensive_score', 0)
        cre = row.get('creative_score', 0)
        
        if 'Defender' in pos or 'GK' in pos:
            # Defenders should be gems for defending/creating, not lucky goals
            return max(defn, cre, off * 0.5) 
        elif 'Forward' in pos or 'Wing' in pos:
            # Forwards rely on off/cre
            return max(off, cre, defn * 0.5)
        else:
            # Midfielders can be anything
            return max(off, defn, cre)

    gems_pool['gem_score'] = gems_pool.apply(calc_weighted_score, axis=1)
    
    gems = gems_pool[
        (gems_pool['minutes_played'] > 45) &
        (gems_pool['minutes_played'] < 200) &
        (gems_pool['gem_score'] > 65)
    ].sort_values('gem_score', ascending=False).head(4)
    
    if not gems.empty:
        # Pre-calculate position averages for comparison
        avg_stats = data.groupby('position_category')[['goals_per90', 'assists_per90', 'offensive_score', 'creative_score', 'defensive_score']].mean()
        
        for i, (_, player) in enumerate(gems.iterrows()):
            # Determine primary score to display
            s_off = player.get('offensive_score',0)
            s_def = player.get('defensive_score',0)
            s_cre = player.get('creative_score',0)
            
            if s_def >= s_off and s_def >= s_cre:
                score_label, score_val = "Def Score", s_def
            elif s_cre >= s_off:
                score_label, score_val = "Cre Score", s_cre
            else:
                score_label, score_val = "Off Score", s_off

            with st.expander(f"💎 {player['player_name']} - {score_label}: {int(score_val)}", expanded=True):
                c_info, c_radar = st.columns([1, 1])
                
                with c_info:
                    st.markdown(f"**Position:** {player.get('position_category', 'Unknown')}")
                    st.markdown(f"**Minutes:** {int(player['minutes_played'])}")
                    st.markdown(f"**{score_label}:** <span style='color:#00ff00; font-weight:bold'>{int(score_val)}</span>", unsafe_allow_html=True)
                    
                    # Scout's Verdict
                    verdict = []
                    pos_cat = player.get('position_category', 'Unknown')
                    pos_avg = avg_stats.loc[pos_cat] if pos_cat in avg_stats.index else None
                    
                    if pos_avg is not None:
                        is_def = 'Defender' in pos_cat or 'GK' in pos_cat
                        is_mid = 'Midfield' in pos_cat
                        is_fwd = 'Forward' in pos_cat or 'Wing' in pos_cat
                        
                        # Strict Verdicts based on primary role
                        if is_fwd:
                            if player['goals_per90'] > pos_avg['goals_per90'] * 1.5:
                                verdict.append(f"⚽ **Goal Machine:** Scores {player['goals_per90']:.2f}/90 (Avg: {pos_avg['goals_per90']:.2f})")
                        
                        elif is_mid:
                            if player['creative_score'] > pos_avg['creative_score'] + 10: # Lower threshold since it's their main job
                                verdict.append(f"🧠 **Elite Vision:** Creative score {player['creative_score']:.0f} vs {pos_avg['creative_score']:.0f} avg")
                        
                        elif is_def:
                            if player['defensive_score'] > pos_avg['defensive_score'] + 10:
                                 verdict.append(f"🛡️ **Wall:** Def score {player['defensive_score']:.0f} vs {pos_avg['defensive_score']:.0f} avg")

                    if not verdict: 
                        # Fallback if they don't excel in their primary role but are good
                        if player['offensive_score'] > 80: verdict.append("🔥 High Offensive Impact")
                        elif player['defensive_score'] > 80: verdict.append("🔒 Solid Defensive Presence")
                        else: verdict.append("✨ Promising Talent")
                    
                    st.info("\n\n".join(verdict))
                    
                with c_radar:
                    # Comparisons Radar
                    if pos_avg is not None:
                        # Dynamic Categories based on position?
                        # Keep standard for now but maybe swap Off/Def?
                        categories = ['Goals/90', 'Assists/90', 'Offensive', 'Defensive', 'Creative']
                        
                        maxes = [1.5, 0.8, 100, 100, 100]
                        p_vals = [
                            min(player.get('goals_per90',0), maxes[0])/maxes[0],
                            min(player.get('assists_per90',0), maxes[1])/maxes[1],
                            player.get('offensive_score',0)/maxes[2],
                            player.get('defensive_score',0)/maxes[3],
                            player.get('creative_score',0)/maxes[4]
                        ]
                        a_vals = [
                            min(pos_avg.get('goals_per90',0), maxes[0])/maxes[0],
                            min(pos_avg.get('assists_per90',0), maxes[1])/maxes[1],
                            pos_avg.get('offensive_score',0)/maxes[2],
                            pos_avg.get('defensive_score',0)/maxes[3],
                            pos_avg.get('creative_score',0)/maxes[4]
                        ]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=p_vals, theta=categories, fill='toself', name='Player', line_color='#00ff00'))
                        fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name='Avg', line_color='#666666'))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                            margin=dict(t=20, b=20, l=30, r=30),
                            height=250,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    else:
        st.info("No hidden gems found currently.")
