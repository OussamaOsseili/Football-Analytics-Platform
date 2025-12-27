"""
Player Routes - API endpoints for player data
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings

router = APIRouter()


def load_data():
    """Load player data"""
    try:
        enhanced_path = settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv"
        if enhanced_path.exists():
            return pd.read_csv(enhanced_path)
        return pd.read_csv(settings.PROCESSED_DATA_DIR / "players_season_stats.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@router.get("/")
async def list_players(
    position: Optional[str] = Query(None, description="Filter by position"),
    style: Optional[str] = Query(None, description="Filter by playing style"),
    min_minutes: int = Query(0, description="Minimum minutes played"),
    limit: int = Query(100, ge=1, le=1000, description="Max results")
):
    """
    Get list of players with optional filters
    """
    data = load_data()
    
    if data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    # Apply filters
    if position and 'position_category' in data.columns:
        data = data[data['position_category'] == position]
    
    if style and 'primary_style' in data.columns:
        data = data[data['primary_style'] == style]
    
    if 'minutes_played' in data.columns:
        data = data[data['minutes_played'] >= min_minutes]
    
    # Limit results
    data = data.head(limit)
    
    # Select columns
    columns = ['player_name', 'position_category', 'matches_played', 'minutes_played',
               'goals_per90', 'assists_per90', 'offensive_score', 'creative_score']
    
    columns = [col for col in columns if col in data.columns]
    
    return {
        "total": len(data),
        "players": data[columns].to_dict('records')
    }


@router.get("/{player_name}")
async def get_player(player_name: str):
    """
    Get detailed player profile
    """
    data = load_data()
    
    if data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    # Find player
    player_data = data[data['player_name'] == player_name]
    
    if player_data.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    
    player = player_data.iloc[0].to_dict()
    
    # Clean NaN values
    player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
    
    return {
        "player": player
    }


@router.get("/{player_name}/similar")
async def get_similar_players(
    player_name: str,
    limit: int = Query(5, ge=1, le=20, description="Number of similar players")
):
    """
    Find similar players using simple distance metric
    """
    data = load_data()
    
    if data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    # Find target player
    target_player = data[data['player_name'] == player_name]
    
    if target_player.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    
    target = target_player.iloc[0]
    
    # Filter same position
    position = target.get('position_category', '')
    same_position = data[data['position_category'] == position].copy()
    
    # Calculate similarity based on key metrics
    feature_cols = ['goals_per90', 'assists_per90', 'tackles_per90', 'xg_per90']
    available_features = [col for col in feature_cols if col in data.columns]
    
    if not available_features:
        raise HTTPException(status_code=500, detail="No features available for similarity calculation")
    
    # Calculate distances
    target_stats = target[available_features].fillna(0).values
    
    similarities = []
    for idx, row in same_position.iterrows():
        if row['player_name'] != player_name:
            row_stats = row[available_features].fillna(0).values
            distance = ((target_stats - row_stats) ** 2).sum() ** 0.5
            similarities.append({
                'player_name': row['player_name'],
                'similarity_score': max(0, 100 - distance * 10),
                'position': row.get('position_category', ''),
                'primary_style': row.get('primary_style', None)
            })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return {
        "player": player_name,
        "similar_players": similarities[:limit]
    }


@router.get("/styles/list")
async def list_playing_styles():
    """
    Get all available playing styles
    """
    data = load_data()
    
    if data.empty or 'primary_style' not in data.columns:
        raise HTTPException(status_code=503, detail="No style data available")
    
    styles = data['primary_style'].value_counts().to_dict()
    
    return {
        "total_styles": len(styles),
        "styles": styles
    }
