# Football Analytics Platform - Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes!

### Step 1: Setup Environment
```powershell
# Navigate to project
cd "C:\Users\ossei\Downloads\PFA PROJECT"

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Process Data
```powershell
# Run ETL pipeline (processes StatsBomb data)
python src/etl/etl_pipeline.py
```

**Expected output**:
- ✅ ~8 competitions loaded
- ✅ ~700-900 matches processed
- ✅ CSV files generated in `data/processed/`

**Time**: ~5-10 minutes depending on your system

### Step 3: Run ML Pipeline
```powershell
# Train ML models and classify playing styles
python src/ml/train_pipeline.py
```

**Expected output**:
- ✅ Features engineered
- ✅ 15+ playing styles classified
- ✅ Enhanced dataset saved

**Time**: ~2-3 minutes

### Step 4: Launch Dashboard
```powershell
# Start Streamlit dashboard
streamlit run src/dashboard/app.py
```

**Access**: http://localhost:8501

## 📊 What You'll See

### Available Pages:
1. **🏠 Overview** - KPIs, top performers, distributions
2. **👤 Player Profile** - Detailed analysis with radar charts
3. **🔄 Comparison** - Multi-player side-by-side
4. **🔍 Scouting** - Advanced filters + CSV export

## 🐛 Troubleshooting

### Error: "No module named 'config'"
```powershell
# Make sure you're in the right directory
cd "C:\Users\ossei\Downloads\PFA PROJECT"
```

### Error: "File not found: players_season_stats.csv"
```powershell
# Run ETL pipeline first
python src/etl/etl_pipeline.py
```

### Dashboard doesn't load
```powershell
# Check port 8501 is free
# Try alternative port:
streamlit run src/dashboard/app.py --server.port 8502
```

## 📈 Next Steps

1. ✅ **Explore data**: Check Overview page for statistics
2. ✅ **Analyze players**: Use Player Profile for detailed insights
3. ✅ **Scout talent**: Use Scouting page with custom filters
4. ✅ **Compare**: Use Comparison page for side-by-side analysis

## 💡 Tips

- **Performance**: First load may be slow as data caches
- **Filters**: Start broad, then narrow down in Scouting
- **Export**: All pages support CSV export of results
- **Refresh**: Re-run ETL if you add new data

## 🎓 For Your PFA Presentation

**Demo Flow**:
1. Show Overview (total stats, visualizations)
2. Pick a famous player → Show Profile page
3. Compare 2-3 players → Show Comparison page  
4. Use Scouting filters → Export results

**Key Points to Highlight**:
- ✅ Complete ETL pipeline (JSON → CSV)
- ✅ Advanced feature engineering
- ✅ ML-based playing style classification
- ✅ Multi-dimensional performance scoring
- ✅ Professional interactive dashboard
- ✅ 2022-2024 data (temporal consistency)

## 📞 Need Help?

Check `README.md` for full documentation.

**Good luck with your PFA! ⚽🚀**
