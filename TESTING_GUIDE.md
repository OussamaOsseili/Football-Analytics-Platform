# 🧪 Guide de Test Complet - Football Analytics Platform

## 📋 Checklist de Test

Suivez ces étapes pour tester chaque composant du projet.

---

## ✅ Phase 1: Setup Initial

### 1.1 Installation
```powershell
cd "C:\Users\ossei\Downloads\PFA PROJECT"

# Vérifier que venv existe
dir venv

# Si pas de venv, lancer setup
.\setup.bat
```

**✓ Succès si**: Dossier `venv` créé, packages installés sans erreur

---

## ✅ Phase 2: ETL Pipeline

### 2.1 Lancer ETL
```powershell
python src\etl\etl_pipeline.py
```

**✓ Succès si**:
- Message "✅ ETL Pipeline Complete!"
- Fichiers créés dans `data/processed/`:
  - `players_season_stats.csv`
  - `players_match_stats.csv`
- Aucune erreur critique

**⏱️ Temps estimé**: 5-10 minutes

**Vérification rapide**:
```powershell
dir data\processed
# Devrait afficher les 2 fichiers CSV
```

---

## ✅ Phase 3: ML Training

### 3.1 Entraîner modèles
```powershell
python src\ml\train_pipeline.py
```

**✓ Succès si**:
- Message "✅ ML Pipeline Complete!"
- Fichier créé: `data/processed/players_season_stats_enhanced.csv`
- Styles de jeu identifiés (15+)

### 3.2 Sauvegarder modèles
```powershell
python src\ml\train_models.py
```

**✓ Succès si**:
- Fichiers créés dans `models/`:
  - `xg_regression_model.pkl`
  - `tier_classification_model.pkl`
- R² score affiché (> 0.4 minimum)

---

## ✅ Phase 4: SHAP Explainability

### 4.1 Générer analyses SHAP
```powershell
python src\ml\explainability.py
```

**✓ Succès si**:
- Images créées dans `reports/`:
  - `shap_summary.png`
  - `shap_importance.png`
- Pas d'erreur de dépendances

---

## ✅ Phase 5: Tests Unitaires

### 5.1 Lancer tests
```powershell
.\run_tests.bat
```

**✓ Succès si**:
- Tests passent (peut avoir quelques warnings)
- Aucun FAILED

**Alternative manuelle**:
```powershell
python -m pytest tests\ -v
```

---

## ✅ Phase 6: API FastAPI

### 6.1 Démarrer API
```powershell
.\start_api.bat
```

**✓ Succès si**:
- Message "Application startup complete"
- Accessible sur http://localhost:8000
- Swagger docs: http://localhost:8000/docs

### 6.2 Tester endpoints (dans un nouveau terminal)
```powershell
# Health check
curl http://localhost:8000/health

# Players
curl "http://localhost:8000/api/players?limit=5"

# Analytics
curl http://localhost:8000/api/analytics/summary
```

**✓ Succès si**: Réponses JSON valides (status 200 ou 503 si pas de données)

**Pour arrêter l'API**: Ctrl+C dans le terminal

---

## ✅ Phase 7: Dashboard Streamlit

### 7.1 Lancer dashboard
```powershell
streamlit run src\dashboard\app.py
```

**✓ Succès si**:
- Dashboard s'ouvre dans navigateur (http://localhost:8501)
- Pas d'erreur Python dans terminal

### 7.2 Tester chaque page

#### Page 1: Overview 🏠
- [ ] KPIs affichés (total players, matches, goals)
- [ ] Graphique top scorers
- [ ] Distribution par position

#### Page 2: Player Profile 👤
- [ ] Sélectionner un joueur
- [ ] Radar chart affiché
- [ ] Scores multi-dimensionnels (5 dimensions)
- [ ] AI insights visibles

#### Page 3: Match Analysis ⚔️
- [ ] Page se charge

#### Page 4: Comparison 🔄
- [ ] Sélectionner 2-3 joueurs
- [ ] Radar comparatif affiché
- [ ] Tableau statistiques

#### Page 5: Predictions 🔮
- [ ] Sélectionner métrique à prédire
- [ ] Modèle trained (R² affiché)
- [ ] Feature importance chart
- [ ] Prédiction pour un joueur

#### Page 6: Clusters 🎯
- [ ] Distribution styles affichée
- [ ] PCA 2D visualization
- [ ] Recherche similarité fonctionne

#### Page 7: Scouting 🔍
- [ ] Filtres fonctionnent (position, style, minutes)
- [ ] Résultats mis à jour
- [ ] Export CSV fonctionne
- [ ] Bouton PDF visible

#### Page 8: Team Analysis 🤝
- [ ] Balance offensive/défensive affichée
- [ ] Sunburst chart
- [ ] Optimal XI suggéré

#### Page 9: Tactical Board 📊
- [ ] Formation 4-3-3 affichée
- [ ] Heatmap conceptuel
- [ ] Stats par position

#### Page 10: Temporal Trends 📈
- [ ] Courbes de forme
- [ ] Détection tendances

#### Page 11: Contextual Analysis 🏟️
- [ ] Home vs Away comparaison
- [ ] Performance vs opponent strength

**✓ Toutes les pages**: Pas d'erreur "KeyError" ou "AttributeError"

---

## ✅ Phase 8: PDF Reports

### 8.1 Générer PDF sample
```powershell
.\generate_pdf_report.bat
```

**✓ Succès si**:
- PDF créé dans `reports/`
- Nom: `scout_report_[PlayerName].pdf`
- PDF ouvrable et bien formaté

### 8.2 Tester depuis dashboard
1. Aller sur page **Scouting**
2. Appliquer filtres
3. Cliquer "📄 Generate PDF Report (Top Player)"
4. Vérifier téléchargement PDF

**✓ Succès si**: PDF téléchargé avec radar chart inclus

---

## ✅ Phase 9: Notebook EDA

### 9.1 Ouvrir Jupyter
```powershell
jupyter notebook notebooks/01_EDA.ipynb
```

**✓ Succès si**:
- Notebook s'ouvre
- Cellules exécutables (Shift+Enter)
- Visualisations affichées

---

## 🎯 Résumé des Tests

### Checklist Finale

- [ ] **ETL**: CSV générés
- [ ] **ML Training**: Modèles sauvegardés (.pkl)
- [ ] **SHAP**: Images générées
- [ ] **Tests**: Passent
- [ ] **API**: Répond sur http://localhost:8000
- [ ] **Dashboard**: 11 pages fonctionnelles
- [ ] **PDF**: Rapport généré
- [ ] **Notebook**: EDA exécutable

---

## ⚠️ Troubleshooting

### Erreur: "Module not found"
```powershell
pip install -r requirements.txt
```

### Erreur: "File not found" (CSV)
```powershell
# Relancer ETL
python src\etl\etl_pipeline.py
```

### Dashboard lent
```powershell
# Normal au premier chargement (cache Streamlit)
# Recharger la page
```

### Port déjà utilisé
```powershell
# Pour API, changer port:
uvicorn src.api.main:app --port 8001

# Pour Dashboard:
streamlit run src\dashboard\app.py --server.port 8502
```

---

## 📊 Tests de Performance

### Vérifier temps d'exécution:
- ETL: ~5-10 min ✅
- ML Training: ~2-3 min ✅
- Dashboard load: ~10-20 sec ✅
- PDF generation: ~5-10 sec ✅

---

## ✅ Validation Finale PFA

Pour votre présentation, vérifiez que vous pouvez:

1. **Démontrer ETL**: Montrer CSV générés ✅
2. **Démontrer ML**: Montrer modèles + prédictions ✅
3. **Démontrer Dashboard**: Naviguer 3-4 pages clés ✅
4. **Démontrer API**: Swagger docs ✅
5. **Démontrer PDF**: Exporter rapport ✅

**Score attendu**: 9-10/10 tests passent = **EXCELLENT** 🎯

---

*Bonne chance pour vos tests ! 🚀*
