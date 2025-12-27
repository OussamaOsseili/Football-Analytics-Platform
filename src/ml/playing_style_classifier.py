"""
Automatic Playing Style Classification using Clustering
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List


class PlayingStyleClassifier:
    """Classify players into playing style archetypes"""
    
    # Style definitions by position
    STYLE_ARCHETYPES = {
        'Forward': ['Poacher', 'Inside Forward', 'Complete Forward', 'Target Man', 'Winger'],
        'Midfielder': ['Deep-Lying Playmaker', 'Box-to-Box', 'Ball-Winner', 'Advanced Playmaker'],
        'Defender': ['Ball-Playing Defender', 'No-Nonsense Defender', 'Sweeper', 'Wing-Back'],
        'GK': ['Sweeper-Keeper', 'Traditional Keeper']
    }
    
    def __init__(self):
        self.position_models = {}
        self.scaler = StandardScaler()
        
    def get_position_features(self, position_category: str) -> List[str]:
        """Get relevant features for each position"""
        feature_sets = {
            'Forward': ['goals_per90', 'xg_per90', 'shots_per90', 'dribbles_completed_per90',
                       'progressive_carries_per90', 'aerial_duels_won_per90'],
            'Midfielder': ['assists_per90', 'xa_per90', 'progressive_passes_per90', 'key_passes_per90',
                          'tackles_per90', 'interceptions_per90', 'carries_per90'],
            'Defender': ['tackles_per90', 'interceptions_per90', 'clearances_per90',
                        'passes_completed_per90', 'progressive_passes_per90', 'aerial_duels_won_per90'],
            'GK': ['saves_per90', 'passes_per90', 'long_passes_per90']
        }
        return feature_sets.get(position_category, ['goals_per90', 'assists_per90', 'tackles_per90'])
    
    def cluster_by_position(self, df: pd.DataFrame, position: str, n_clusters: int = None):
        """Cluster players by position"""
        position_df = df[df['position_category'] == position].copy()
        
        if len(position_df) < 10:
            return position_df
        
        # Get features
        features = self.get_position_features(position)
        available_features = [f for f in features if f in position_df.columns]
        
        if not available_features:
            return position_df
        
        # Auto-determine clusters if not specified
        if n_clusters is None:
            n_clusters = min(len(self.STYLE_ARCHETYPES.get(position, [])), len(position_df) // 20 + 2)
            n_clusters = max(2, min(5, n_clusters))
        
        # Scale and cluster
        X = position_df[available_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        position_df['style_cluster'] = kmeans.fit_predict(X_scaled)
        
        # Label clusters
        position_df = self._label_clusters(position_df, position, available_features)
        
        self.position_models[position] = {'model': kmeans, 'features': available_features}
        
        return position_df
    
    def _label_clusters(self, df: pd.DataFrame, position: str, features: List[str]) -> pd.DataFrame:
        """Label clusters with style names based on characteristics"""
        df = df.copy()
        archetypes = self.STYLE_ARCHETYPES.get(position, ['Style A', 'Style B', 'Style C'])
        
        # Analyze cluster characteristics
        cluster_profiles = df.groupby('style_cluster')[features].mean()
        
        # Simple heuristic labeling
        style_labels = {}
        for cluster_id in cluster_profiles.index:
            if cluster_id < len(archetypes):
                style_labels[cluster_id] = archetypes[cluster_id]
            else:
                style_labels[cluster_id] = f"{position} - Style {cluster_id}"
        
        df['primary_style'] = df['style_cluster'].map(style_labels)
        
        return df
    
    def calculate_style_affinity(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """Calculate affinity scores to all style archetypes"""
        df = df.copy()
        
        if position not in self.position_models:
            return df
        
        model_info = self.position_models[position]
        features = model_info['features']
        model = model_info['model']
        
        # Get cluster centers
        centers = model.cluster_centers_
        
        # Calculate distance to each center (affinity = inverse distance)
        X = df[features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        affinities = {}
        archetypes = self.STYLE_ARCHETYPES.get(position, [])
        
        for i, center in enumerate(centers):
            if i < len(archetypes):
                distances = np.linalg.norm(X_scaled - center, axis=1)
                # Convert distance to affinity (0-100 scale)
                max_dist = distances.max() if distances.max() > 0 else 1
                affinities[archetypes[i]] = (1 - distances / max_dist) * 100
        
        for style, scores in affinities.items():
            df[f'affinity_{style.replace(" ", "_").lower()}'] = scores
        
        return df
    
    def classify_all_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify all players by position"""
        results = []
        
        for position in ['Forward', 'Midfielder', 'Defender', 'GK']:
            position_df = self.cluster_by_position(df, position)
            if len(position_df) > 0:
                position_df = self.calculate_style_affinity(position_df, position)
                results.append(position_df)
        
        # CRITICAL FIX: Include players with Unknown position category
        unknown_players = df[~df['position_category'].isin(['Forward', 'Midfielder', 'Defender', 'GK'])].copy()
        if len(unknown_players) > 0:
            unknown_players['primary_style'] = 'Unknown'
            results.append(unknown_players)
        
        if results:
            return pd.concat(results, ignore_index=True)
        return df


if __name__ == "__main__":
    # Test with sample data
    print("Playing Style Classifier ready for integration")
