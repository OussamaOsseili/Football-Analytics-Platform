import json
import pandas as pd
from pathlib import Path
import sys

# Setup paths (assuming running from project root)
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "dataset 3" / "data" / "matches"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

team_mapping = {}
match_comp_mapping = {}
match_home_team_map = {}
match_away_team_map = {}
match_date_map = {}
match_home_score_map = {}
match_away_score_map = {}


print(f"Scanning {DATA_PATH}...")

if not DATA_PATH.exists():
    print("Error: dataset 3 not found locally!")
    sys.exit(1)

count = 0
for comp_folder in DATA_PATH.iterdir():
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
                        
                        if "Women's" in comp_name: continue
                        
                        team_mapping[str(htid)] = match['home_team']['home_team_name']
                        team_mapping[str(atid)] = match['away_team']['away_team_name']
                        
                        comp_full = f"{comp_name} - {match['season']['season_name']}"
                        match_comp_mapping[str(mid)] = comp_full
                        match_home_team_map[str(mid)] = htid
                        match_away_team_map[str(mid)] = atid
                        match_date_map[str(mid)] = mdate
                        match_home_score_map[str(mid)] = match['home_score']
                        match_away_score_map[str(mid)] = match['away_score']
                        count += 1
            except Exception as e:
                print(f"Error reading {season_file}: {e}")

# Save mappings
mappings = {
    'team_mapping': team_mapping,
    'match_comp_mapping': match_comp_mapping,
    'match_home_team_map': match_home_team_map,
    'match_away_team_map': match_away_team_map,
    'match_date_map': match_date_map,
    'match_home_score_map': match_home_score_map,
    'match_away_score_map': match_away_score_map
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / "match_lookups.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(mappings, f)

print(f"Processed {count} matches.")
print(f"Saved mappings to {output_file}")
