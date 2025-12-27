"""
AI-Generated Insights - Natural Language Player Analysis
"""
import pandas as pd
from typing import Dict, List


class AIInsightsGenerator:
    """Generate natural language insights about player performance"""
    
    def generate_player_summary(self, player_stats: Dict) -> str:
        """Generate comprehensive player summary"""
        insights = []
        
        # Style description
        if 'primary_style' in player_stats:
            insights.append(f"**Playing Style**: {player_stats['primary_style']}")
        
        # Offensive assessment
        if 'offensive_score' in player_stats:
            score = player_stats['offensive_score']
            if score > 80:
                insights.append(f"🔥 **Elite attacker** (Top {100-score:.0f}% offensive rating)")
            elif score > 60:
                insights.append(f"⚔️ **Strong offensive presence** ({score:.0f}/100 offensive rating)")
        
        # Key metrics with context
        if 'xg_per90' in player_stats and player_stats['xg_per90'] > 0.5:
            insights.append(f"🎯 Exceptional finishing threat: **{player_stats['xg_per90']:.2f} xG per 90**")
        
        if 'progressive_passes_per90' in player_stats and player_stats['progressive_passes_per90'] >5:
            insights.append(f"📈 Creative playmaker: **{player_stats['progressive_passes_per90']:.1f} progressive passes per 90**")
        
        return "\n".join(insights)
    
    def identify_standout_metrics(self, player_stats: pd.Series, all_players: pd.DataFrame) -> List[str]:
        """Identify exceptional statistics with percentile context"""
        standouts = []
        
        metrics_to_check = {
            'goals_per90': ('Goals per 90', '⚽'),
            'xg_per90': ('xG per 90', '🎯'),
            'assists_per90': ('Assists per 90', '🅰️'),
            'tackles_per90': ('Tackles per 90', '🛡️'),
            'progressive_passes_per90': ('Progressive passes per 90', '📈')
        }
        
        for metric, (label, emoji) in metrics_to_check.items():
            if metric in player_stats.index and metric in all_players.columns:
                value = player_stats[metric]
                percentile = (all_players[metric] < value).mean() * 100
                
                if percentile > 90:
                    rank_desc = "Elite" if percentile > 95 else "Excellent"
                    standouts.append(f"{emoji} **{rank_desc}** {label}: {value:.2f} (Top {100-percentile:.0f}%)")
        
        return standouts
    
    def compare_to_peers(self, player_stats: pd.Series, all_players: pd.DataFrame, 
                        position: str, competition: str = None) -> str:
        """Generate peer comparison commentary"""
        # Filter peers (same position)
        peers = all_players[all_players.get('position_category', '') == position]
        
        if len(peers) < 5:
            return ""
        
        # Find ranking in key metric
        if 'xg_per90' in player_stats.index and 'xg_per90' in peers.columns:
            xg_value = player_stats['xg_per90']
            rank = (peers['xg_per90'] > xg_value).sum() + 1
            total = len(peers)
            
            position_name = position if position else "the league"
            return f"📊 Ranks **#{rank}** among {total} {position_name}s in xG per 90"
        
        return ""
    
    def trend_commentary(self, current_stats: Dict, historical_stats: Dict = None) -> str:
        """Generate trend-based commentary"""
        if not historical_stats:
            return ""
        
        trends = []
        
        metrics = ['goals_per90', 'assists_per90', 'progressive_passes_per90']
        for metric in metrics:
            if metric in current_stats and metric in historical_stats:
                current = current_stats[metric]
                historical = historical_stats[metric]
                
                if historical > 0:
                    change_pct = ((current - historical) / historical) * 100
                    
                    if abs(change_pct) > 15:
                        direction = "↗️ Improving" if change_pct > 0 else "↘️ Declining"
                        metric_name = metric.replace('_per90', '').replace('_', ' ').title()
                        trends.append(f"{direction}: **{change_pct:+.0f}%** in {metric_name}")
        
        return " | ".join(trends) if trends else "📊 **Consistent** performance levels"
    
    def tactical_recommendations(self, player_stats: Dict) -> List[str]:
        """Suggest tactical roles based on strengths"""
        recommendations = []
        
        # High offensive score
        if player_stats.get('offensive_score', 0) > 75:
            recommendations.append("💡 Best deployed in attacking roles with freedom to shoot")
        
        # High creative score
        if player_stats.get('creative_score', 0) > 75:
            recommendations.append("💡 Excellent playmaker - position centrally to maximize passing options")
        
        # High defensive score
        if player_stats.get('defensive_score', 0) > 75:
            recommendations.append("💡 Strong defensive presence - ideal for ball-winning roles")
        
        # Balanced profile
        scores = [player_stats.get(f'{dim}_score', 50) for dim in ['offensive', 'creative', 'defensive']]
        if all(50 < score < 75 for score in scores):
            recommendations.append("💡 Well-rounded player - versatile across multiple roles")
        
        return recommendations if recommendations else ["💡 Specialized player - maximize playing time in primary position"]
    
    def generate_full_insights(self, player_stats: pd.Series, all_players: pd.DataFrame) -> Dict[str, any]:
        """Generate complete insights package"""
        player_dict = player_stats.to_dict()
        position = player_stats.get('position_category', 'Unknown')
        
        insights = {
            'summary': self.generate_player_summary(player_dict),
            'standout_metrics': self.identify_standout_metrics(player_stats, all_players),
            'peer_comparison': self.compare_to_peers(player_stats, all_players, position),
            'recommendations': self.tactical_recommendations(player_dict)
        }
        
        return insights


if __name__ == "__main__":
    print("AI Insights Generator ready ✨")
