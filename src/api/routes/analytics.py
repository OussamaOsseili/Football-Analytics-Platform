"""
Analytics Routes - API endpoints for analytics and statistics
"""
from fastapi import APIRouter, HTTPException
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


@router.get("/summary")
async def get_summary_statistics():
    """
    Get overall platform statistics
    """
    data = load_data()
    
    if data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    stats = {
        "total_players": len(data),
        "total_matches": int(data['matches_played'].sum()) if 'matches_played' in data.columns else 0,
        "total_goals": int(data['goals'].sum()) if 'goals' in data.columns else 0,
        "positions": data['position_category'].nunique() if 'position_category' in data.columns else 0,
        "avg_goals_per90": float(data['goals_per90'].mean()) if 'goals_per90' in data.columns else 0,
        "avg_assists_per90": float(data['assists_per90'].mean()) if 'assists_per90' in data.columns else 0
    }
    
    return stats


@router.get("/leaderboards/{metric}")
async def get_leaderboard(metric: str, limit: int = 10):
    """
    Get top players for a specific metric
    
    Available metrics: goals_per90, assists_per90, xg_per90, offensive_score, etc.
    """
    data = load_data()
    
    if data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    if metric not in data.columns:
        raise HTTPException(status_code=400, detail=f"Metric '{metric}' not available")
    
    # Get top players
    top_players = data.nlargest(limit, metric)[['player_name', 'position_category', metric]]
    
    return {
        "metric": metric,
        "leaderboard": top_players.to_dict('records')
    }


@router.get("/position/{position}/stats")
async def get_position_statistics(position: str):
    """
    Get aggregated statistics for a position
    """
    data = load_data()
    
    if data.empty:
        raise HTTPException(status_code=503, detail="No data available")
    
    position_data = data[data['position_category'] == position]
    
    if position_data.empty:
        raise HTTPException(status_code=404, detail=f"Position '{position}' not found")
    
    # Calculate averages
    numeric_cols = ['goals_per90', 'assists_per90', 'xg_per90', 'tackles_per90', 
                   'offensive_score', 'creative_score', 'defensive_score']
    
    available_cols = [col for col in numeric_cols if col in position_data.columns]
    
    stats = {
        "position": position,
        "player_count": len(position_data),
        "averages": {col: float(position_data[col].mean()) for col in available_cols}
    }
    
    return stats


@router.get("/styles/{style}/players")
async def get_players_by_style(style: str, limit: int = 20):
    """
    Get all players with a specific playing style
    """
    data = load_data()
    
    if data.empty or 'primary_style' not in data.columns:
        raise HTTPException(status_code=503, detail="No style data available")
    
    style_players = data[data['primary_style'] == style]
    
    if style_players.empty:
        raise HTTPException(status_code=404, detail=f"Style '{style}' not found")
    
    # Get top players by offensive score
    sort_col = 'offensive_score' if 'offensive_score' in style_players.columns else 'goals_per90'
    
    if sort_col in style_players.columns:
        top_players = style_players.nlargest(limit, sort_col)
    else:
        top_players = style_players.head(limit)
    
    columns = ['player_name', 'position_category', 'matches_played',
               'goals_per90', 'assists_per90', 'offensive_score']
    
    available_columns = [col for col in columns if col in top_players.columns]
    
    return {
        "style": style,
        "total_players": len(style_players),
        "players": top_players[available_columns].to_dict('records')
    }
