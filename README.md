# ⚽ Football Analytics Platform

**Advanced Player Performance Analysis & Prediction System**

A comprehensive data science platform for football analytics using StatsBomb Open Data (2022-2024). Features multi-dimensional performance analysis, automatic playing style classification, AI-generated insights, and professional scouting tools.

---

## 🎯 Key Features

### 📊 **Data & Analytics**
- ✅ **700-900 matches** from 5 elite competitions (FIFA World Cup 2022, Ligue 1, Bundesliga, UEFA Euro 2024, etc.)
- ✅ **360° tracking data** for advanced physical metrics
- ✅ **Multi-dimensional scoring** across 5 performance categories
- ✅ **Per-90 minute normalization** for fair comparisons

### 🎯 **Playing Style Classification**
- ✅ **15+ automated archetypes**: Inside Forward, Ball-Playing Defender, Box-to-Box Midfielder, etc.
- ✅ **Position-specific clustering** with ML
- ✅ **Multi-style affinity scoring**

### 🤖 **AI-Powered Insights**
- ✅ **Natural language commentary** generation
- ✅ **Automated standout metric identification**
- ✅ **Peer comparison** rankings
- ✅ **Tactical recommendations**

### 📱 **Interactive Dashboard** (Streamlit)
- 10 comprehensive pages
- Real-time visualizations
- Radar charts, heatmaps, trend analysis
- PDF export capabilities

### 🔬 **Machine Learning**
- Performance prediction models
- Cluster-based player similarity
- Anomaly detection
- Temporal trend forecasting

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 4GB+ RAM
- Git

### Installation

```bash

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env
```

### Run ETL Pipeline

```bash
# Process StatsBomb data
python src/etl/etl_pipeline.py
```

This will:
- Load 5 selected competitions (2022-2024)
- Process  ~700-900 matches
- Calculate player statistics
- Export CSVs to `data/processed/`

### Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run src/dashboard/app.py
```

Navigate to `http://localhost:8501`

---

## 📂 Project Structure

```
PFA PROJECT/
├── src/
│   ├── config.py                 # Configuration management
│   ├── database/                 # SQLAlch emy models
│   ├── etl/                      # Data pipeline
│   │   └── etl_pipeline.py       # Main ETL script
│   ├── ml/                       # Machine learning
│   │   ├── feature_engineer.py
│   │   └── playing_style_classifier.py
│   ├── intelligence/             # AI insights
│   │   └── ai_insights_generator.py
│   └── dashboard/                # Streamlit app
│       ├── app.py                # Main dashboard
│       └── pages/                # Dashboard pages
├── data/
│   └── processed/                # Generated CSVs
├── dataset 3/                    # StatsBomb source data
├── requirements.txt
└── README.md
```

---

## 📊 Data Dictionary

### `players_season_stats.csv`

| Column | Description | Type |
|--------|-------------|------|
| `player_id` | Unique player identifier | int |
| `player_name` | Player full name | str |
| `matches_played` | Matches in season | int |
| `minutes_played` | Total minutes | float |
| `goals_per90` | Goals per 90 minutes | float |
| `xg_per90` | Expected goals per 90 | float |
| `offensive_score` | Offensive dimension 0-100 | float |
| `creative_score` | Creative dimension 0-100 | float |
| `defensive_score` | Defensive dimension 0-100 | float |
| `primary_style` | Playing style archetype | str |

---

## 🎓 Academic Context (PFA)

This project demonstrates:
- ✅ Complete CRISP-DM methodology implementation
- ✅ ETL pipeline design & execution
- ✅ Advanced feature engineering
- ✅ ML model development & evaluation
- ✅ Deployment & visualization
- ✅ Professional documentation

**Differentiation**: Combines player analytics + team analysis + tactical insights + AI commentary in a single platform.

---

## 📄 License

Data: StatsBomb Open Data ([License](https://github.com/statsbomb/open-data))  
Code: MIT License

---

## 👥 Author

Osseili Oussama - https://www.linkedin.com/in/oussama-osseili/
Rochdi Othmane - https://www.linkedin.com/in/othmane-rochdi-b2874628a/
---

## 🙏 Acknowledgments

- **StatsBomb** for providing open football data
- **Streamlit** for dashboard framework
- **scikit-learn** for ML tools
