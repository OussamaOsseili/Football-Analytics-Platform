"""
ETL Pipeline - Data extraction and transformation from StatsBomb JSON
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import settings
from database.database import (
    init_db, SessionLocal, Competition, Team, Match, Player,
    PlayerMatchStats, PlayerSeasonStats
)


class StatsBombETL:
    """Main ETL pipeline for StatsBomb data"""
    
    def __init__(self):
        self.raw_data_dir = settings.RAW_DATA_DIR
        self.competitions_data = []
        self.matches_data = []
        self.events_data = []
        
    def load_competitions(self):
        """Load and filter competitions"""
        print("📊 Loading competitions...")
        comp_file = self.raw_data_dir / "competitions.json"
        with open(comp_file, 'r', encoding='utf-8') as f:
            all_comps = json.load(f)
        
        # Filter selected competitions
        selected = []
        for comp in all_comps:
            comp_name = comp['competition_name']
            season_name = comp['season_name']
            gender = comp.get('competition_gender', 'male')
            
            # Additional filter for Women's competitions
            if gender == 'female' or "Women's" in comp_name:
                continue

            if comp_name in settings.SELECTED_COMPETITIONS:
                if season_name in settings.SELECTED_COMPETITIONS[comp_name]:
                    selected.append(comp)
        
        self.competitions_data = selected
        print(f"✓ Loaded {len(selected)} competition-seasons")
        return selected
    
    def load_matches(self):
        """Load matches for selected competitions"""
        print("⚽ Loading matches...")
        matches_dir = self.raw_data_dir / "matches"
        all_matches = []
        
        for comp in tqdm(self.competitions_data, desc="Competitions"):
            comp_id = comp['competition_id']
            season_id = comp['season_id']
            match_file = matches_dir / str(comp_id) / f"{season_id}.json"
            
            if match_file.exists():
                with open(match_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    
                    # Filter Women's competitions (Global filter)
                    if "Women's" in comp['competition_name']:
                        continue
                        
                    for match in matches:
                        match['_competition'] = comp
                    all_matches.extend(matches)
        
        self.matches_data = all_matches
        print(f"✓ Loaded {len(all_matches)} matches")
        return all_matches
    
    def load_events(self, match_id: int):
        """Load events for a specific match"""
        events_file = self.raw_data_dir / "events" / f"{match_id}.json"
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def calculate_player_match_stats(self, events: List[Dict], match_id: int) -> List[Dict]:
        """Calculate player statistics from match events"""
        player_stats = {}
        
        for event in events:
            player_id = event.get('player', {}).get('id')
            if not player_id:
                continue
            
            if player_id not in player_stats:
                player_stats[player_id] = {
                    'player_id': player_id,
                    'player_name': event.get('player', {}).get('name'),
                    'team_id': event.get('team', {}).get('id'),
                    'position': event.get('position', {}).get('name'),
                    'match_id': match_id,
                    'minutes_played': 0,
                    'shots': 0,
                    'shots_on_target': 0,
                    'goals': 0,
                    'xg': 0.0,
                    'assists': 0,
                    'xa': 0.0,
                    'passes': 0,
                    'passes_completed': 0,
                    'progressive_passes': 0,
                    'key_passes': 0,
                    'dribbles': 0,
                    'dribbles_completed': 0,
                    'carries': 0,
                    'progressive_carries': 0,
                    'tackles': 0,
                    'interceptions': 0,
                    'pressures': 0,
                    'blocks': 0,
                    'clearances': 0,
                    'duels_won': 0,
                    'duels_lost': 0,
                    'aerial_duels_won': 0,
                    'aerial_duels_lost': 0,
                }
            
            stats = player_stats[player_id]
            event_type = event.get('type', {}).get('name')
            
            # Count events
            if event_type == 'Shot':
                stats['shots'] += 1
                shot_outcome = event.get('shot', {}).get('outcome', {}).get('name')
                if shot_outcome in ['Goal', 'Saved', 'Saved To Post']:
                    stats['shots_on_target'] += 1
                if shot_outcome == 'Goal':
                    stats['goals'] += 1
                stats['xg'] += event.get('shot', {}).get('statsbomb_xg', 0.0)
            
            elif event_type == 'Pass':
                stats['passes'] += 1
                if event.get('pass', {}).get('outcome') is None:  # Successful pass
                    stats['passes_completed'] += 1
                if event.get('pass', {}).get('goal_assist'):
                    stats['assists'] += 1
                if event.get('pass', {}).get('shot_assist'):
                    stats['key_passes'] += 1
                # Progressive pass heuristic: forward >10m
                pass_length = event.get('pass', {}).get('length', 0)
                if pass_length > 10:
                    stats['progressive_passes'] += 1
            
            elif event_type == 'Duel':
                outcome = event.get('duel', {}).get('outcome', {}).get('name')
                duel_type = event.get('duel', {}).get('type', {}).get('name')
                if outcome == 'Success':
                    stats['duels_won'] += 1
                    if duel_type == 'Aerial Lost':
                        stats['aerial_duels_won'] += 1
                else:
                    stats['duels_lost'] += 1
                    if duel_type == 'Aerial Lost':
                        stats['aerial_duels_lost'] += 1
            
            elif event_type == 'Dribble':
                stats['dribbles'] += 1
                if event.get('dribble', {}).get('outcome', {}).get('name') == 'Complete':
                    stats['dribbles_completed'] += 1
            
            elif event_type == 'Carry':
                stats['carries'] += 1
                # Progressive carry: forward >5m
                end_loc = event.get('carry', {}).get('end_location', [0, 0])
                start_loc = event.get('location', [0, 0])
                if len(end_loc) >= 2 and len(start_loc) >= 2:
                    forward_distance = end_loc[0] - start_loc[0]
                    if forward_distance > 5:
                        stats['progressive_carries'] += 1
            
            elif event_type in ['Tackle', 'Interception', 'Pressure', 'Block', 'Clearance']:
                stats[event_type.lower() + 's'] += 1
        
        # Estimate minutes played from events
        for pid, stats in player_stats.items():
            stats['minutes_played'] = 90  # Simplified, should be from lineup data
        
        return list(player_stats.values())
    
    def process_all_matches(self):
        """Process all matches and calculate stats"""
        print("🔄 Processing match events...")
        all_player_match_stats = []
        
        for match in tqdm(self.matches_data, desc="Processing matches"):
            match_id = match['match_id']
            events = self.load_events(match_id)
            if events:
                player_stats = self.calculate_player_match_stats(events, match_id)
                all_player_match_stats.extend(player_stats)
        
        print(f"✓ Processed {len(all_player_match_stats)} player-match records")
        return pd.DataFrame(all_player_match_stats)
    
    def aggregate_season_stats(self, player_match_df: pd.DataFrame):
        """Aggregate match stats to season level"""
        print("📈 Aggregating season statistics...")
        
        # First get position for each player (mode)
        player_positions = player_match_df.groupby('player_id')['position'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).to_dict()
        
        # Categorize positions
        position_mapping = {
            'Goalkeeper': 'GK',
            'Left Back': 'Defender', 'Right Back': 'Defender',
            'Center Back': 'Defender', 'Left Wing Back': 'Defender',
            'Right Wing Back': 'Defender',
            'Defensive Midfield': 'Midfielder', 'Center Midfield': 'Midfielder',
            'Left Midfield': 'Midfielder', 'Right Midfield': 'Midfielder',
            'Attacking Midfield': 'Midfielder', 'Left Center Midfield': 'Midfielder',
            'Right Center Midfield': 'Midfielder',
            'Left Wing': 'Forward', 'Right Wing': 'Forward',
            'Center Forward': 'Forward', 'Secondary Striker': 'Forward'
        }
        
        # Group by player and calculate per-90 metrics
        season_stats = player_match_df.groupby(['player_id', 'player_name']).agg({
            'match_id': 'count',
            'minutes_played': 'sum',
            'goals': 'sum',
            'assists': 'sum',
            'xg': 'sum',
            'xa': 'sum',
            'shots': 'sum',
            'passes_completed': 'sum',
            'tackles': 'sum',
            'interceptions': 'sum',
            'progressive_passes': 'sum',
        }).reset_index()
        
        season_stats.columns = ['player_id', 'player_name', 'matches_played', 'minutes_played',
                                'goals', 'assists', 'xg_total', 'xa_total', 'shots',
                                'passes_completed', 'tackles', 'interceptions', 'progressive_passes']
        
        # Add position
        season_stats['position'] = season_stats['player_id'].map(player_positions)
        season_stats['position_category'] = season_stats['position'].map(position_mapping).fillna('Unknown')
        
        # Per-90 metrics
        season_stats['goals_per90'] = (season_stats['goals'] / season_stats['minutes_played']) * 90
        season_stats['assists_per90'] = (season_stats['assists'] / season_stats['minutes_played']) * 90
        season_stats['xg_per90'] = (season_stats['xg_total'] / season_stats['minutes_played']) * 90
        season_stats['xa_per90'] = (season_stats['xa_total'] / season_stats['minutes_played']) * 90
        season_stats['progressive_passes_per90'] = (season_stats['progressive_passes'] / season_stats['minutes_played']) * 90
        season_stats['tackles_per90'] = (season_stats['tackles'] / season_stats['minutes_played']) * 90
        
        print(f"✓ Aggregated {len(season_stats)} player-season records")
        return season_stats
    
    def export_to_csv(self, player_match_df, season_stats_df):
        """Export processed data to CSV"""
        print("💾 Exporting to CSV...")
        
        output_dir = settings.PROCESSED_DATA_DIR
        
        player_match_df.to_csv(output_dir / "players_match_stats.csv", index=False)
        season_stats_df.to_csv(output_dir / "players_season_stats.csv", index=False)
        
        print(f"✓ Exported to {output_dir}")
    
    def run_pipeline(self):
        """Run full ETL pipeline"""
        print("=" * 60)
        print("🚀 Starting ETL Pipeline")
        print("=" * 60)
        
        # Load data
        self.load_competitions()
        self.load_matches()
        
        # Process
        player_match_df = self.process_all_matches()
        season_stats_df = self.aggregate_season_stats(player_match_df)
        
        # Export
        self.export_to_csv(player_match_df, season_stats_df)
        
        print("=" * 60)
        print("✅ ETL Pipeline Complete!")
        print("=" * 60)
        
        return player_match_df, season_stats_df


if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Run ETL
    etl = StatsBombETL()
    player_match_df, season_stats_df = etl.run_pipeline()
    
    print(f"\n📊 Data Summary:")
    print(f"  - Matches: {player_match_df['match_id'].nunique()}")
    print(f"  - Players: {season_stats_df.shape[0]}")
    print(f"  - Avg goals/90: {season_stats_df['goals_per90'].mean():.3f}")
