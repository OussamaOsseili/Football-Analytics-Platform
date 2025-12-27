"""
Predictions Page - ML-based performance forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

st.set_page_config(page_title="Performance Predictions", layout="wide")
st.title("🔮 Performance Predictions")
st.markdown("### AI-Powered Player Performance Forecasting")

# ================= DATA LOADING (Robust Aggregation) =================
@st.cache_data
def load_match_data_with_names():
    try:
        df = pd.read_csv(settings.PROCESSED_DATA_DIR / "players_match_stats.csv")
        
        team_mapping = {}
        comp_mapping = {}
        data_path = settings.PROJECT_ROOT / "dataset 3" / "data" / "matches"
        
        for comp_folder in data_path.iterdir():
            if comp_folder.is_dir():
                for season_file in comp_folder.glob("*.json"):
                    try:
                        with open(season_file, 'r', encoding='utf-8') as f:
                            matches = json.load(f)
                            for match in matches:
                                if "Women's" in match['competition']['competition_name']:
                                    continue
                                team_mapping[match['home_team']['home_team_id']] = match['home_team']['home_team_name']
                                team_mapping[match['away_team']['away_team_id']] = match['away_team']['away_team_name']
                                comp_full = f"{match['competition']['competition_name']} - {match['season']['season_name']}"
                                if 'match_id' in df.columns:
                                     # This mapping approach is a bit indirect, but we map comp ID to name
                                     pass 
                                # Better: Mapping logic needs to match Player Profile. 
                                # Since we can't easily map match_id -> comp directly without huge dict, 
                                # let's rely on what we have or a simplified version.
                                # Actually, for predictions, we just need the aggregated stats.
                                # The hierarchy search needs the names.
                    except:
                        continue
        
        # Simplified for prediction page: Just ensure we have team names if possible
        if 'team_name' not in df.columns and 'team_id' in df.columns and team_mapping:
             df['team_name'] = df['team_id'].map(team_mapping)
             
        # Create a simple competition name if missing (or load from json if crucial)
        # For now, we will proceed with what's in CSV or basic mapping
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# Reuse global COLS_TO_SUM
COLS_TO_SUM = [
    'goals', 'assists', 'xg', 'xa', 'shots', 'key_passes',
    'passes_completed', 'passes', 'progressive_passes',
    'dribbles_completed', 'carries', 'progressive_carries',
    'tackles', 'interceptions', 'blocks', 'clearances', 'pressures',
    'minutes_played'
]

@st.cache_data
def build_robust_season_data(match_df):
    if match_df.empty: return pd.DataFrame()
    
    existing = [c for c in COLS_TO_SUM if c in match_df.columns]
    agg = match_df.groupby('player_name').agg({c: 'sum' for c in existing}).reset_index()
    
    # Matches / Teams / Position
    meta = match_df.groupby('player_name').agg({
        'minutes_played': 'count', # count rows as matches
        'position': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).rename(columns={'minutes_played': 'matches_played'}).reset_index()
    
    # Merge
    season = pd.merge(agg, meta[['player_name', 'matches_played', 'position']], on='player_name')
    
    # Teams
    if 'team_name' in match_df.columns:
        teams = match_df.groupby('player_name')['team_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        season['team_name'] = season['player_name'].map(teams)
        
    # Comps
    if 'competition_name' in match_df.columns:
        comps = match_df.groupby('player_name')['competition_name'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        season['competition_name'] = season['player_name'].map(comps)

    # Per 90s
    for c in existing:
        if c != 'minutes_played':
            season[f"{c}_per90"] = (season[c] / season['minutes_played'] * 90).fillna(0)
            
    # Position Category
    def cat_pos(p):
        p = str(p).lower()
        if 'keeper' in p: return 'Goalkeeper'
        if 'back' in p or 'defender' in p: return 'Defender'
        if 'midfield' in p: return 'Midfielder'
        if 'wing' in p or 'forward' in p: return 'Forward'
        return 'Unknown'
    season['position_category'] = season['position'].apply(cat_pos)
    
    # Scores (Simplified logic if missing in aggregation)
    # We'll calculate simple proxy scores for the model features
    season['offensive_score'] = (season.get('goals_per90',0)*40 + season.get('assists_per90',0)*30).clip(0, 100)
    season['creative_score'] = (season.get('key_passes_per90',0)*20 + season.get('xa_per90',0)*40).clip(0, 100)
    
    return season

@st.cache_resource
def train_model(data, target_col, feature_cols):
    """Train a simple prediction model"""
    # Prepare data
    X = data[feature_cols].fillna(0)
    y = data[target_col]
    
    # Remove outliers
    X = X[(y >= y.quantile(0.01)) & (y <= y.quantile(0.99))]
    y = y[(y >= y.quantile(0.01)) & (y <= y.quantile(0.99))]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, r2, mae, feature_cols

match_data = load_match_data_with_names()
season_data = build_robust_season_data(match_data)

# Rename for compatibility with rest of script
data = season_data

if data.empty:
    st.warning("⚠️ No data available.")
    st.stop()

# Prediction target selection
st.subheader("🎯 Select Prediction Target")

prediction_options = {
    'Goals per 90 minutes': 'goals_per90',
    'Assists per 90 minutes': 'assists_per90',
    'xG per 90 minutes': 'xg_per90',
    'Offensive Score': 'offensive_score',
    'Creative Score': 'creative_score'
}

selected_target_name = st.selectbox("What do you want to predict?", list(prediction_options.keys()))
target_col = prediction_options[selected_target_name]

# Check if target exists
if target_col not in data.columns:
    st.error(f"Target column '{target_col}' not found. Run ML pipeline first.")
    st.stop()

# Feature selection
st.subheader("📊 Model Training")

# Auto-select features (exclude target and identifiers)
exclude_cols = ['player_id', 'player_name', 'match_id', 'team_id', 'position', 
                'primary_style', 'style_cluster', target_col]

feature_candidates = [col for col in data.columns 
                     if col not in exclude_cols 
                     and data[col].dtype in ['float64', 'int64']
                     and not col.startswith('affinity_')]

# Select top relevant features
if len(feature_candidates) > 10:
    # Use correlation to select top features
    correlations = data[feature_candidates].corrwith(data[target_col]).abs().sort_values(ascending=False)
    feature_cols = correlations.head(10).index.tolist()
else:
    feature_cols = feature_candidates[:10]

# Train model
with st.spinner("Training model..."):
    try:
        model, r2, mae, features_used = train_model(data, target_col, feature_cols)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("MAE", f"{mae:.3f}")
        with col3:
            st.metric("Features Used", len(features_used))
        
        if r2 > 0.6:
            st.success("✅ Good model performance!")
        elif r2 > 0.3:
            st.info("ℹ️ Moderate model performance")
        else:
            st.warning("⚠️ Model performance could be improved")
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.stop()

# Feature importance
st.subheader("📈 Feature Importance")

importance_df = pd.DataFrame({
    'Feature': features_used,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig = go.Figure(go.Bar(
    x=importance_df['Importance'],
    y=importance_df['Feature'],
    orientation='h',
    marker_color='lightblue'
))

fig.update_layout(
    title="Top Features for Prediction",
    xaxis_title="Importance",
    yaxis_title="Feature",
    height=400
)

st.plotly_chart(fig, width='stretch')

# Make prediction for a player
st.markdown("---")
st.subheader("🎯 Make Prediction")

# Hierarchical Selection
with st.expander("🔍 Find Player", expanded=True):
    c_comp, c_team, c_player = st.columns(3)
    
    # 1. Competition
    if 'competition_name' in data.columns:
        comps = sorted([c for c in data['competition_name'].unique() if pd.notna(c)])
        sel_comp = c_comp.selectbox("Competition", ["All Competitions"] + comps, key="pred_comp")
        
        filtered = data.copy()
        if sel_comp != "All Competitions":
            filtered = filtered[filtered['competition_name'] == sel_comp]
    else:
        filtered = data
        sel_comp = "All Competitions" # default
            
    # 2. Team
    if 'team_name' in filtered.columns:
        teams = sorted([t for t in filtered['team_name'].unique() if pd.notna(t)])
        sel_team = c_team.selectbox("Team", ["All Teams"] + teams, key="pred_team")
        
        if sel_team != "All Teams":
            filtered = filtered[filtered['team_name'] == sel_team]
            
    # 3. Player
    players = sorted(filtered['player_name'].unique())
    selected_player = c_player.selectbox("Player", ["None"] + players, key="pred_player")

col1, col2 = st.columns([2, 1])

with col1:
    if selected_player != "None":
        # Get player data (potentially specific to the filtered context if we were strict, 
        # but here 'data' is already robustly aggregated season data.
        # If we wanted context-specific prediction, we'd need to re-aggregate.
        # For now, we use the row from 'filtered' which IS from 'data'.
        # Wait, 'filtered' is just a subset of 'data' (season_data).
        # Since 'data' is aggregation of ALL matches, filtering it by Comp name 
        # (which is just the mode comp) might filter out players who played elsewhere?
        # Actually 'build_robust_season_data' assigns a single 'competition_name' (mode).
        # So filtering by Comp works for primary comp.
        
        player_data = filtered[filtered['player_name'] == selected_player].iloc[0]
        
        # Make prediction
        player_features = player_data[features_used].fillna(0).values.reshape(1, -1)
        prediction = model.predict(player_features)[0]
        actual = player_data[target_col]
        
        st.markdown(f"### {selected_player}")
        st.caption(f"Position: {player_data.get('position_category', 'Unknown')} | Team: {player_data.get('team_name', 'Unknown')}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted " + selected_target_name, f"{prediction:.2f}")
        with col_b:
            st.metric("Actual " + selected_target_name, f"{actual:.2f}", 
                     delta=f"{actual - prediction:.2f}")
        
        # Accuracy
        error_pct = abs(actual - prediction) / (actual + 0.001) * 100
        if error_pct < 10:
            st.success(f"✅ Very accurate prediction (error: {error_pct:.1f}%)")
        elif error_pct < 25:
            st.info(f"ℹ️ Good prediction (error: {error_pct:.1f}%)")
        else:
            st.warning(f"⚠️ Moderate prediction (error: {error_pct:.1f}%)")

with col2:
    if selected_player != "None":
        # Show player style
        if 'primary_style' in player_data:
            st.markdown("**Playing Style:**")
            st.write(player_data['primary_style'])
        
        # Show key stats
        st.markdown("**Key Stats:**")
        if 'goals_per90' in player_data:
            st.write(f"Goals/90: {player_data['goals_per90']:.2f}")
        if 'assists_per90' in player_data:
            st.write(f"Assists/90: {player_data['assists_per90']:.2f}")

# Prediction distribution
st.markdown("---")
st.subheader("📊 Prediction vs Actual Distribution")

# Make predictions for all players
all_features = data[features_used].fillna(0)
all_predictions = model.predict(all_features)

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=data[target_col],
    name='Actual',
    opacity=0.7,
    marker_color='blue'
))

fig.add_trace(go.Histogram(
    x=all_predictions,
    name='Predicted',
    opacity=0.7,
    marker_color='red'
))

fig.update_layout(
    title=f"Distribution: {selected_target_name}",
    xaxis_title=selected_target_name,
    yaxis_title="Count",
    barmode='overlay',
    height=400
)

st.plotly_chart(fig, width='stretch')

# Info box
st.info("""
**How it works:**
- Model: Random Forest Regressor
- Training: 80% of data, Testing: 20%
- Features: Top 10 most correlated metrics
- Evaluation: R² score and Mean Absolute Error

**Use cases:**
- Predict future performance based on current stats
- Identify over/under-performing players
- Scout players with high predicted potential
""")
