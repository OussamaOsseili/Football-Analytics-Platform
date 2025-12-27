"""
PDF Report Generator - Professional scouting reports
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import sys
import io

sys.path.append(str(Path(__file__).parent.parent))
from config import settings


class PDFReportGenerator:
    """Generate professional PDF scouting reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Create custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=12,
            spaceAfter=6,
            borderWidth=0,
            borderColor=colors.HexColor('#1f77b4'),
            borderPadding=5
        ))
        
        # Insight style
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            leftIndent=20,
            rightIndent=20
        ))
    
    def create_radar_chart(self, player_data: dict, metrics: list) -> str:
        """Create radar chart and return image path"""
        # Extract values
        labels = []
        values = []
        
        for metric in metrics:
            if metric in player_data and pd.notna(player_data[metric]):
                labels.append(metric.replace('_', ' ').title())
                values.append(float(player_data[metric]))
        
        if not values:
            return None
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            width=400,
            height=400,
            margin=dict(l=80, r=80, t=20, b=20)
        )
        
        # Save to temp file
        temp_path = settings.REPORTS_DIR / f"temp_radar_{datetime.now().timestamp()}.png"
        fig.write_image(str(temp_path), format='png')
        
        return str(temp_path)
    
    def generate_player_report(self, player_data: pd.Series, all_players: pd.DataFrame, 
                               output_path: Path = None) -> Path:
        """
        Generate comprehensive player scouting report
        
        Args:
            player_data: Series with player statistics
            all_players: DataFrame with all players for comparison
            output_path: Optional custom output path
        
        Returns:
            Path to generated PDF
        """
        player_name = player_data.get('player_name', 'Unknown Player')
        
        if output_path is None:
            output_path = settings.REPORTS_DIR / f"scout_report_{player_name.replace(' ', '_')}.pdf"
        
        # Create PDF
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for elements
        story = []
        
        # Title
        title = Paragraph(f"<b>SCOUTING REPORT</b>", self.styles['CustomTitle'])
        story.append(title)
        
        # Player name
        player_title = Paragraph(
            f"<b>{player_name}</b>",
            self.styles['Heading1']
        )
        story.append(player_title)
        story.append(Spacer(1, 12))
        
        # Date and position
        date_str = datetime.now().strftime("%B %d, %Y")
        position = player_data.get('position_category', 'Unknown')
        style = player_data.get('primary_style', 'N/A')
        
        header_data = [
            ['Report Date:', date_str],
            ['Position:', position],
            ['Playing Style:', style],
            ['Matches Played:', str(int(player_data.get('matches_played', 0)))],
            ['Minutes:', f"{player_data.get('minutes_played', 0):.0f}"]
        ]
        
        header_table = Table(header_data, colWidths=[2*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 20))
        
        # Section: Key Statistics
        story.append(Paragraph("KEY STATISTICS", self.styles['SectionHeader']))
        story.append(Spacer(1, 6))
        
        stats_data = [
            ['Metric', 'Value', 'Per 90'],
            ['Goals', str(int(player_data.get('goals', 0))), f"{player_data.get('goals_per90', 0):.2f}"],
            ['Assists', str(int(player_data.get('assists', 0))), f"{player_data.get('assists_per90', 0):.2f}"],
            ['xG', f"{player_data.get('xg_total', 0):.2f}", f"{player_data.get('xg_per90', 0):.2f}"],
            ['Tackles', str(int(player_data.get('tackles', 0))), f"{player_data.get('tackles_per90', 0):.2f}"],
            ['Progressive Passes', str(int(player_data.get('progressive_passes', 0))), 
             f"{player_data.get('progressive_passes_per90', 0):.2f}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Section: Multi-Dimensional Scores
        story.append(Paragraph("PERFORMANCE DIMENSIONS", self.styles['SectionHeader']))
        story.append(Spacer(1, 6))
        
        dimensions = {
            'Offensive': player_data.get('offensive_score', 0),
            'Creative': player_data.get('creative_score', 0),
            'Defensive': player_data.get('defensive_score', 0),
            'Work Rate': player_data.get('workrate_score', 0),
            'Discipline': player_data.get('discipline_score', 0)
        }
        
        dim_data = [['Dimension', 'Score (0-100)', 'Rating']]
        for dim, score in dimensions.items():
            if pd.notna(score):
                rating = self._get_rating(score)
                dim_data.append([dim, f"{score:.0f}", rating])
        
        dim_table = Table(dim_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        dim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(dim_table)
        story.append(Spacer(1, 20))
        
        # Radar chart
        radar_metrics = ['offensive_score', 'creative_score', 'defensive_score', 
                        'workrate_score', 'discipline_score']
        radar_path = self.create_radar_chart(player_data.to_dict(), radar_metrics)
        
        if radar_path and Path(radar_path).exists():
            story.append(Paragraph("PERFORMANCE RADAR", self.styles['SectionHeader']))
            story.append(Spacer(1, 6))
            
            img = Image(radar_path, width=4*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
        
        # AI Insights
        story.append(PageBreak())
        story.append(Paragraph("SCOUT INSIGHTS & RECOMMENDATIONS", self.styles['SectionHeader']))
        story.append(Spacer(1, 12))
        
        # Generate insights
        insights = self._generate_insights(player_data, all_players)
        
        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles['Insight']))
            story.append(Spacer(1, 6))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"<i>Report generated on {date_str} by Football Analytics Platform</i>"
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Cleanup temp radar chart
        if radar_path and Path(radar_path).exists():
            Path(radar_path).unlink()
        
        return output_path
    
    def _get_rating(self, score: float) -> str:
        """Convert numerical score to rating"""
        if score >= 90:
            return "⭐⭐⭐⭐⭐ Elite"
        elif score >= 75:
            return "⭐⭐⭐⭐ Excellent"
        elif score >= 60:
            return "⭐⭐⭐ Good"
        elif score >= 40:
            return "⭐⭐ Average"
        else:
            return "⭐ Below Average"
    
    def _generate_insights(self, player_data: pd.Series, all_players: pd.DataFrame) -> list:
        """Generate AI insights for report"""
        insights = []
        
        # Style insight
        if 'primary_style' in player_data and pd.notna(player_data['primary_style']):
            insights.append(f"Classified as <b>{player_data['primary_style']}</b> based on performance analysis")
        
        # Offensive insight
        if 'offensive_score' in player_data:
            score = player_data['offensive_score']
            if score > 80:
                insights.append(f"<b>Elite attacking threat</b> with offensive score of {score:.0f}/100, placing in top tier")
            elif score > 60:
                insights.append(f"Strong offensive presence with score of {score:.0f}/100")
        
        # Goals/xG comparison
        if 'goals' in player_data and 'xg_total' in player_data:
            goals = player_data['goals']
            xg = player_data['xg_total']
            if goals > xg * 1.2:
                insights.append(f"<b>Clinical finisher</b>: {int(goals)} goals vs {xg:.1f} xG (overperforming by {((goals/xg)-1)*100:.0f}%)")
            elif xg > goals * 1.2:
                insights.append(f"Underperforming xG: {int(goals)} goals from {xg:.1f} expected (potential for improvement)")
        
        # Creative insight
        if 'creative_score' in player_data and player_data['creative_score'] > 70:
            insights.append(f"Exceptional playmaking ability (Creative score: {player_data['creative_score']:.0f}/100)")
        
        # Work rate
        if 'workrate_score' in player_data and player_data['workrate_score'] > 75:
            insights.append("High work rate and intensity - valuable for pressing systems")
        
        # Recommendation
        if 'offensive_score' in player_data and 'defensive_score' in player_data:
            off_score = player_data['offensive_score']
            def_score = player_data['defensive_score']
            
            if off_score > 70 and def_score > 70:
                insights.append("<b>RECOMMENDATION: HIGH PRIORITY TARGET</b> - Complete profile with strong all-around capabilities")
            elif off_score > 75:
                insights.append("<b>RECOMMENDATION: ATTACKING ASSET</b> - Ideal for offensive-minded teams")
            elif def_score > 75:
                insights.append("<b>RECOMMENDATION: DEFENSIVE SPECIALIST</b> - Strong option for defensive solidity")
        
        if not insights:
            insights.append("Solid performer with balanced attributes across multiple dimensions")
        
        return insights


if __name__ == "__main__":
    print("="*60)
    print("📄 PDF Report Generator")
    print("="*60)
    
    # Load data
    from config import settings
    data_path = settings.PROCESSED_DATA_DIR / "players_season_stats.csv"
    
    if not data_path.exists():
        print("❌ Data not found")
        exit(1)
    
    data = pd.read_csv(data_path)
    print(f"\n✓ Loaded {len(data)} players")
    
    # Generate sample report
    generator = PDFReportGenerator()
    
    # Pick a player with good stats
    if 'offensive_score' in data.columns:
        sample_player = data.nlargest(1, 'offensive_score').iloc[0]
    else:
        sample_player = data.iloc[0]
    
    print(f"\n📝 Generating report for: {sample_player['player_name']}")
    
    output_path = generator.generate_player_report(sample_player, data)
    
    print(f"\n✅ Report generated!")
    print(f"💾 Saved to: {output_path}")
