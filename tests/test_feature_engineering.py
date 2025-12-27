"""
Unit Tests for Feature Engineering
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ml.feature_engineer import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engineer = FeatureEngineer()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'player_name': ['Player A', 'Player B', 'Player C'],
            'minutes_played': [900, 1800, 270],
            'goals': [10, 15, 3],
            'assists': [5, 8, 2],
            'shots': [40, 60, 15],
            'tackles': [20, 30, 10],
            'position': ['Center Forward', 'Left Wing', 'Center Back']
        })
    
    def test_per90_features(self):
        """Test per-90 minute normalization"""
        result = self.engineer.create_per90_features(self.sample_data)
        
        # Check if per90 columns created
        self.assertIn('goals_per90', result.columns)
        self.assertIn('assists_per90', result.columns)
        
        # Verify calculation
        expected_goals_per90 = (10 / 900) * 90
        self.assertAlmostEqual(result.iloc[0]['goals_per90'], expected_goals_per90, places=2)
    
    def test_dimension_scores(self):
        """Test multi-dimensional scoring"""
        # Add required columns
        self.sample_data['xg_total'] = [8, 12, 2]
        self.sample_data['xa_total'] = [4, 6, 1]
        self.sample_data['progressive_passes'] = [50, 70, 30]
        self.sample_data['interceptions'] = [5, 8, 15]
        self.sample_data['pressures'] = [40, 50, 60]
        
        result = self.engineer.create_dimension_scores(self.sample_data)
        
        # Check if score columns created
        self.assertIn('offensive_score', result.columns)
        self.assertIn('creative_score', result.columns)
        self.assertIn('defensive_score', result.columns)
        
        # Scores should be 0-100
        for col in ['offensive_score', 'creative_score', 'defensive_score']:
            self.assertTrue((result[col] >= 0).all())
            self.assertTrue((result[col] <= 100).all())
    
    def test_position_categorization(self):
        """Test position category assignment"""
        result = self.engineer.create_position_features(self.sample_data)
        
        self.assertIn('position_category', result.columns)
        
        # Check correct categorization
        self.assertEqual(result.iloc[0]['position_category'], 'Forward')
        self.assertEqual(result.iloc[1]['position_category'], 'Forward')
        self.assertEqual(result.iloc[2]['position_category'], 'Defender')
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        result = self.engineer.create_per90_features(empty_df)
        
        self.assertTrue(result.empty)


class TestDataValidation(unittest.TestCase):
    """Test data validation"""
    
    def setUp(self):
        """Set up test data"""
        self.valid_data = pd.DataFrame({
            'player_name': ['Test Player'],
            'goals_per90': [0.5],
            'assists_per90': [0.3]
        })
    
    def test_valid_data(self):
        """Test with valid data"""
        self.assertFalse(self.valid_data.empty)
        self.assertEqual(len(self.valid_data), 1)
    
    def test_missing_values(self):
        """Test detection of missing values"""
        data_with_nan = self.valid_data.copy()
        data_with_nan.loc[0, 'goals_per90'] = np.nan
        
        has_missing = data_with_nan.isnull().any().any()
        self.assertTrue(has_missing)


if __name__ == '__main__':
    unittest.main()
