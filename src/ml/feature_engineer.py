"""
Feature Engineering for Football Analytics
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict


class FeatureEngineer:
    """Feature engineering for player analytics"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_per90_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create per-90 minute normalized features"""
        df = df.copy()
        
        if 'minutes_played' not in df.columns or df['minutes_played'].sum() == 0:
            return df
            
        per90_cols = ['goals', 'assists', 'shots', 'xg_total', 'xa_total',
                      'passes_completed', 'progressive_passes', 'tackles', 
                      'interceptions', 'dribbles_completed', 'carries']
        
        for col in per90_cols:
            if col in df.columns:
                df[f'{col}_per90'] = (df[col] / df['minutes_played']) * 90
                
        return df
    
    def create_dimension_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-dimensional performance scores"""
        df = df.copy()
        
        # Offensive dimension
        offensive_metrics = ['goals_per90', 'xg_per90', 'shots_per90']
        df['offensive_score'] = self._calculate_percentile_score(df, offensive_metrics)
        
        # Creative dimension  
        creative_metrics = ['assists_per90', 'xa_per90', 'progressive_passes_per90', 'key_passes_per90']
        df['creative_score'] = self._calculate_percentile_score(df, creative_metrics)
        
        # Defensive dimension
        defensive_metrics = ['tackles_per90', 'interceptions_per90', 'pressures_per90']
        df['defensive_score'] = self._calculate_percentile_score(df, defensive_metrics)
        
        # Work-rate dimension
        workrate_metrics = ['carries_per90', 'progressive_carries_per90', 'duels_won_per90']
        df['workrate_score'] = self._calculate_percentile_score(df, workrate_metrics)
        
        # Discipline (inverse - fewer is better)
        if 'fouls_per90' in df.columns:
            df['discipline_score'] = 100 - df['fouls_per90'].rank(pct=True) * 100
        else:
            df['discipline_score'] = 50  # Neutral
            
        return df
    
    def _calculate_percentile_score(self, df: pd.DataFrame, metrics: List[str]) -> pd.Series:
        """Calculate percentile-based composite score"""
        available_metrics = [m for m in metrics if m in df.columns]
        if not available_metrics:
            return pd.Series(50, index=df.index)
            
        percentiles = pd.DataFrame()
        for metric in available_metrics:
            percentiles[metric] = df[metric].rank(pct=True) * 100
            
        return percentiles.mean(axis=1)
    
    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features"""
        df = df.copy()
        
        # Position encoding
        position_mapping = {
            'Goalkeeper': 'GK',
            'Left Back': 'Defender', 'Right Back': 'Defender',
            'Center Back': 'Defender', 'Left Wing Back': 'Defender',
            'Right Wing Back': 'Defender', 'Left Center Back': 'Defender',
            'Right Center Back': 'Defender',
            'Defensive Midfield': 'Midfielder', 'Center Midfield': 'Midfielder',
            'Left Midfield': 'Midfielder', 'Right Midfield': 'Midfielder',
            'Attacking Midfield': 'Midfielder', 'Left Center Midfield': 'Midfielder',
            'Right Center Midfield': 'Midfielder',
            'Left Defensive Midfield': 'Midfielder', 'Right Defensive Midfield': 'Midfielder',
            'Center Defensive Midfield': 'Midfielder',
            'Left Attacking Midfield': 'Midfielder', 'Right Attacking Midfield': 'Midfielder',
            'Center Attacking Midfield': 'Midfielder',
            'Left Wing': 'Forward', 'Right Wing': 'Forward',
            'Center Forward': 'Forward', 'Secondary Striker': 'Forward',
            'Left Center Forward': 'Midfielder', 'Right Center Forward': 'Forward'
        }
        
        if 'position' in df.columns:
            df['position_category'] = df['position'].map(position_mapping).fillna('Unknown')
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Standardize numerical features"""
        df = df.copy()
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if available_cols:
            df[available_cols] = self.scaler.fit_transform(df[available_cols])
            
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature preparation pipeline"""
        df = self.create_per90_features(df)
        df = self.create_dimension_scores(df)
        df = self.create_position_features(df)
        
        return df
