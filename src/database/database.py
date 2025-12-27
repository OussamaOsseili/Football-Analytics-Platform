"""
Database schema and models for Football Analytics Platform
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings

Base = declarative_base()


class Competition(Base):
    __tablename__ = "competitions"
    
    id = Column(Integer, primary_key=True)
    competition_id = Column(Integer, unique=True, index=True)
    competition_name = Column(String)
    season_name = Column(String)
    country_name = Column(String)
    competition_gender = Column(String)
    has_360_data = Column(Boolean, default=False)
    
    matches = relationship("Match", back_populates="competition")


class Team(Base):
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, unique=True, index=True)
    team_name = Column(String, index=True)
    
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, unique=True, index=True)
    match_date = Column(DateTime)
    competition_db_id = Column(Integer, ForeignKey("competitions.id"))
    home_team_id = Column(Integer, ForeignKey("teams.team_id"))
    away_team_id = Column(Integer, ForeignKey("teams.team_id"))
    home_score = Column(Integer)
    away_score = Column(Integer)
    
    competition = relationship("Competition", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    player_stats = relationship("PlayerMatchStats", back_populates="match")


class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, unique=True, index=True)
    player_name = Column(String, index=True)
    
    match_stats = relationship("PlayerMatchStats", back_populates="player")
    season_stats = relationship("PlayerSeasonStats", back_populates="player")


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"
    
    id = Column(Integer, primary_key=True)
    player_db_id = Column(Integer, ForeignKey("players.id"), index=True)
    match_db_id = Column(Integer, ForeignKey("matches.id"), index=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"))
    
    # Basic info
    position = Column(String, index=True)
    minutes_played = Column(Float)
    
    # Offensive metrics
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    xg = Column(Float, default=0.0)
    assists = Column(Integer, default=0)
    xa = Column(Float, default=0.0)  # Expected assists
    
    # Passing
    passes = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    pass_completion_rate = Column(Float, default=0.0)
    progressive_passes = Column(Integer, default=0)
    key_passes = Column(Integer, default=0)
    
    # Dribbling & Carrying
    dribbles = Column(Integer, default=0)
    dribbles_completed = Column(Integer, default=0)
    carries = Column(Integer, default=0)
    progressive_carries = Column(Integer, default=0)
    
    # Defensive
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    pressures = Column(Integer, default=0)
    blocks = Column(Integer, default=0)
    clearances = Column(Integer, default=0)
    
    # Duels
    duels_won = Column(Integer, default=0)
    duels_lost = Column(Integer, default=0)
    aerial_duels_won = Column(Integer, default=0)
    aerial_duels_lost = Column(Integer, default=0)
    
    # Physical (360)
    distance_covered = Column(Float)
    sprints = Column(Integer)
    
    player = relationship("Player", back_populates="match_stats")
    match = relationship("Match", back_populates="player_stats")


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    
    id = Column(Integer, primary_key=True)
    player_db_id = Column(Integer, ForeignKey("players.id"), index=True)
    competition_db_id = Column(Integer, ForeignKey("competitions.id"), index=True)
    
    matches_played = Column(Integer)
    minutes_played = Column(Float)
    position = Column(String)
    
    # Aggregated metrics (totals)
    goals = Column(Integer)
    assists = Column(Integer)
    xg_total = Column(Float)
    xa_total = Column(Float)
    shots = Column(Integer)
    passes_completed = Column(Integer)
    tackles = Column(Integer)
    interceptions = Column(Integer)
    
    # Per-90 metrics
    goals_per90 = Column(Float)
    assists_per90 = Column(Float)
    xg_per90 = Column(Float)
    xa_per90 = Column(Float)
    progressive_passes_per90 = Column(Float)
    tackles_per90 = Column(Float)
    
    # Multi-dimensional scores
    offensive_score = Column(Float)
    creative_score = Column(Float)
    defensive_score = Column(Float)
    workrate_score = Column(Float)
    discipline_score = Column(Float)
    
    # Playing style classification
    primary_style = Column(String)
    style_affinity_json = Column(Text)  # JSON of style scores
    
    player = relationship("Player", back_populates="season_stats")
    competition = relationship("Competition")


# Database engine and session
engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database schema"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized")


if __name__ == "__main__":
    init_db()
