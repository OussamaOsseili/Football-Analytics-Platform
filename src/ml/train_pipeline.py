"""
Main ML Training Pipeline - Orchestrates full ML workflow
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import settings
from ml.feature_engineer import FeatureEngineer
from ml.playing_style_classifier import PlayingStyleClassifier


def main():
    """Run complete ML training pipeline"""
    print("="*60)
    print("🤖 ML Training Pipeline")
    print("="*60)
    
    # Load processed data
    print("\n📂 Loading data...")
    season_stats_path = settings.PROCESSED_DATA_DIR / "players_season_stats.csv"
    
    if not season_stats_path.exists():
        print("❌ Error: players_season_stats.csv not found!")
        print("   Run ETL pipeline first: python src/etl/etl_pipeline.py")
        return
    
    df = pd.read_csv(season_stats_path)
    print(f"✓ Loaded {len(df)} player records")
    
    # Feature Engineering
    print("\n🔧 Feature Engineering...")
    engineer = FeatureEngineer()
    df = engineer.prepare_ml_features(df)
    print("✓ Features engineered")
    
    # Playing Style Classification
    print("\n🎯 Classifying playing styles...")
    classifier = PlayingStyleClassifier()
    df = classifier.classify_all_positions(df)
    print("✓ Playing styles classified")
    
    # Save enhanced data
    output_path = settings.PROCESSED_DATA_DIR / "players_season_stats_enhanced.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved enhanced data to: {output_path}")
    
    # Summary statistics
    print("\n📊 Summary:")
    print(f"  - Total players: {len(df)}")
    
    if 'position_category' in df.columns:
        print(f"  - Positions: {df['position_category'].nunique()}")
    
    if 'primary_style' in df.columns:
        print(f"  - Playing styles identified: {df['primary_style'].nunique()}")
        print("\n  Top styles:")
        style_counts = df['primary_style'].value_counts().head(5)
        for style, count in style_counts.items():
            print(f"    • {style}: {count} players")
    
    print("\n✅ ML Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
