# Football Analytics API - Quick Test Guide

## 🚀 Start the API

```powershell
# Activate venv
.\venv\Scripts\activate

# Start API server
uvicorn src.api.main:app --reload --port 8000
```

Access: http://localhost:8000

## 📚 API Documentation

Interactive docs: **http://localhost:8000/docs**

## 🧪 Test Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Get All Players
```bash
curl "http://localhost:8000/api/players?limit=10"
```

### 3. Filter by Position
```bash
curl "http://localhost:8000/api/players?position=Forward&limit=5"
```

### 4. Get Player Profile
```bash
curl "http://localhost:8000/api/players/Lionel%20Andrés%20Messi%20Cuccittini"
```

### 5. Find Similar Players
```bash
curl "http://localhost:8000/api/players/Kylian%20Mbappé%20Lottin/similar?limit=5"
```

### 6. Get Summary Stats
```bash
curl "http://localhost:8000/api/analytics/summary"
```

### 7. Get Leaderboard
```bash
curl "http://localhost:8000/api/analytics/leaderboards/goals_per90?limit=10"
```

### 8. List Playing Styles
```bash
curl "http://localhost:8000/api/players/styles/list"
```

### 9. Get Players by Style
```bash
curl "http://localhost:8000/api/analytics/styles/Inside%20Forward/players"
```

## 🔧 Using Python

```python
import requests

# Get top scorers
response = requests.get("http://localhost:8000/api/analytics/leaderboards/goals_per90")
data = response.json()

for player in data['leaderboard']:
    print(f"{player['player_name']}: {player['goals_per90']:.2f} goals/90")
```

## 📊 Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/api/players` | List players (filterable) |
| GET | `/api/players/{name}` | Get player profile |
| GET | `/api/players/{name}/similar` | Find similar players |
| GET | `/api/players/styles/list` | List all styles |
| GET | `/api/analytics/summary` | Platform statistics |
| GET | `/api/analytics/leaderboards/{metric}` | Top players |
| GET | `/api/analytics/position/{pos}/stats` | Position stats |
| GET | `/api/analytics/styles/{style}/players` | Players by style |

## 💡 Tips

- Use `/docs` for interactive testing (Swagger UI)
- URL-encode player names with spaces
- Check response status codes
- Use `limit` parameter to control result size

## 🐛 Troublesh ooting

**Port already in use:**
```powershell
# Use different port
uvicorn src.api.main:app --port 8001
```

**Module not found:**
```powershell
# Make sure you're in project root
cd "C:\Users\ossei\Downloads\PFA PROJECT"
```
