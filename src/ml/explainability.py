"""
SHAP Explainability Module - Model interpretability
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
from pathlib import Path

# Make SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Explainability features disabled.")
    print("   To enable: pip install shap")

sys.path.append(str(Path(__file__).parent.parent))
from config import settings


class SHAPExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self, model_path: Path, model_type: str = 'regression'):
        """
        Initialize explainer
        
        Args:
            model_path: Path to saved model (.pkl)
            model_type: 'regression' or'classification'
        """
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.features = self.model_data['features']
        self.model_type = model_type
        self.explainer = None
        
    def create_explainer(self, X_background: pd.DataFrame):
        """Create SHAP explainer with background data"""
        print("🔍 Creating SHAP explainer...")
        
        # Use TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        
        print("✓ Explainer ready")
        
    def get_shap_values(self, X: pd.DataFrame):
        """Calculate SHAP values for dataset"""
        if self.explainer is None:
            self.create_explainer(X)
        
        shap_values = self.explainer.shap_values(X)
        
        return shap_values
    
    def plot_summary(self, X: pd.DataFrame, save_path: Path = None):
        """Create SHAP summary plot"""
        print("\n📊 Generating summary plot...")
        
        shap_values = self.get_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, X: pd.DataFrame, save_path: Path = None):
        """Plot feature importance bar chart"""
        print("\n📊 Generating feature importance...")
        
        shap_values = self.get_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return importance_df
    
    def explain_prediction(self, X_single: pd.DataFrame, player_name: str = None):
        """Explain a single prediction with waterfall plot"""
        print(f"\n🔍 Explaining prediction{' for ' + player_name if player_name else ''}...")
        
        shap_values = self.get_shap_values(X_single)
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=X_single.iloc[0].values,
                feature_names=self.features
            ),
            show=True
        )
        plt.tight_layout()
        plt.show()


def main():
    """Demo SHAP analysis"""
    print("="*60)
    print("🔍 SHAP Explainability Analysis")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    data_path = settings.PROCESSED_DATA_DIR / "players_season_stats.csv"
    
    if not data_path.exists():
        print("❌ Data not found")
        return
    
    data = pd.read_csv(data_path)
    
    # Check for regression model
    model_path = settings.MODELS_DIR / "xg_regression_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train models first: python src/ml/train_models.py")
        return
    
    # Load model and get features
    model_data = joblib.load(model_path)
    features = model_data['features']
    
    # Prepare data
    X = data[features].fillna(0).head(100)  # Use first 100 for demo
    
    # Create explainer
    explainer = SHAPExplainer(model_path, model_type='regression')
    explainer.create_explainer(X)
    
    # Generate plots
    output_dir = settings.PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    
    # Summary plot
    explainer.plot_summary(X, save_path=output_dir / "shap_summary.png")
    
    # Feature importance
    importance_df = explainer.plot_feature_importance(X, save_path=output_dir / "shap_importance.png")
    
    print("\n📊 Top 5 Important Features:")
    print(importance_df.head())
    
    print("\n" + "="*60)
    print("✅ SHAP Analysis Complete!")
    print("="*60)
    print(f"\n💾 Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
