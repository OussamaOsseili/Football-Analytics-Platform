"""
Pitch drawing utilities for tactical visualizations
"""
import plotly.graph_objects as go

def create_pitch(fig=None, orientation='horizontal'):
    """
    Create a modern football pitch with improved styling
    Based on the second image style with rounded corners
    
    Args:
        fig: Plotly figure object (creates new if None)
        orientation: 'horizontal' or 'vertical'
    
    Returns:
        Plotly figure with pitch drawn
    """
    if fig is None:
        fig = go.Figure()
    
    # Pitch dimensions (standard: 105m x 68m)
    pitch_length = 105
    pitch_width = 68
    
    # Colors
    pitch_color = '#2d7a2d'  # Dark green
    line_color = 'white'
    line_width = 2
    
    # Outer boundary with rounded corners
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=pitch_length, y1=pitch_width,
        line=dict(color=line_color, width=line_width),
        fillcolor=pitch_color,
        layer='below'
    )
    
    # Center line
    fig.add_shape(
        type="line",
        x0=pitch_length/2, y0=0,
        x1=pitch_length/2, y1=pitch_width,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Center circle
    fig.add_shape(
        type="circle",
        x0=pitch_length/2 - 9.15, y0=pitch_width/2 - 9.15,
        x1=pitch_length/2 + 9.15, y1=pitch_width/2 + 9.15,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Center spot
    fig.add_shape(
        type="circle",
        x0=pitch_length/2 - 0.3, y0=pitch_width/2 - 0.3,
        x1=pitch_length/2 + 0.3, y1=pitch_width/2 + 0.3,
        line=dict(color=line_color, width=line_width),
        fillcolor=line_color,
        layer='below'
    )
    
    # Penalty areas
    # Left penalty area
    fig.add_shape(
        type="rect",
        x0=0, y0=13.84,
        x1=16.5, y1=54.16,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Left 6-yard box
    fig.add_shape(
        type="rect",
        x0=0, y0=24.84,
        x1=5.5, y1=43.16,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Right penalty area
    fig.add_shape(
        type="rect",
        x0=pitch_length - 16.5, y0=13.84,
        x1=pitch_length, y1=54.16,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Right 6-yard box
    fig.add_shape(
        type="rect",
        x0=pitch_length - 5.5, y0=24.84,
        x1=pitch_length, y1=43.16,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Penalty spots
    fig.add_shape(
        type="circle",
        x0=11 - 0.3, y0=pitch_width/2 - 0.3,
        x1=11 + 0.3, y1=pitch_width/2 + 0.3,
        line=dict(color=line_color, width=line_width),
        fillcolor=line_color,
        layer='below'
    )
    
    fig.add_shape(
        type="circle",
        x0=pitch_length - 11 - 0.3, y0=pitch_width/2 - 0.3,
        x1=pitch_length - 11 + 0.3, y1=pitch_width/2 + 0.3,
        line=dict(color=line_color, width=line_width),
        fillcolor=line_color,
        layer='below'
    )
    
    # Penalty arcs
    # Left arc
    fig.add_shape(
        type="path",
        path=f"M 16.5 {pitch_width/2 - 9.15} A 9.15 9.15 0 0 1 16.5 {pitch_width/2 + 9.15}",
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Right arc
    fig.add_shape(
        type="path",
        path=f"M {pitch_length - 16.5} {pitch_width/2 - 9.15} A 9.15 9.15 0 0 0 {pitch_length - 16.5} {pitch_width/2 + 9.15}",
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Goals
    goal_width = 7.32
    goal_y0 = (pitch_width - goal_width) / 2
    goal_y1 = goal_y0 + goal_width
    
    # Left goal
    fig.add_shape(
        type="rect",
        x0=-2, y0=goal_y0,
        x1=0, y1=goal_y1,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Right goal
    fig.add_shape(
        type="rect",
        x0=pitch_length, y0=goal_y0,
        x1=pitch_length + 2, y1=goal_y1,
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Corner arcs
    corner_radius = 1
    # Bottom-left
    fig.add_shape(
        type="path",
        path=f"M 0 {corner_radius} A {corner_radius} {corner_radius} 0 0 1 {corner_radius} 0",
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    # Top-left
    fig.add_shape(
        type="path",
        path=f"M 0 {pitch_width - corner_radius} A {corner_radius} {corner_radius} 0 0 0 {corner_radius} {pitch_width}",
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    # Bottom-right
    fig.add_shape(
        type="path",
        path=f"M {pitch_length} {corner_radius} A {corner_radius} {corner_radius} 0 0 0 {pitch_length - corner_radius} 0",
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    # Top-right
    fig.add_shape(
        type="path",
        path=f"M {pitch_length} {pitch_width - corner_radius} A {corner_radius} {corner_radius} 0 0 1 {pitch_length - corner_radius} {pitch_width}",
        line=dict(color=line_color, width=line_width),
        layer='below'
    )
    
    # Configure layout
    fig.update_layout(
        xaxis=dict(
            range=[-5, pitch_length + 5],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain='domain'
        ),
        yaxis=dict(
            range=[-5, pitch_width + 5],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='x',
            scaleratio=1,
            constrain='domain'
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent instead of green
        paper_bgcolor='#1e1e1e',  # Dark background like the examples
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
        autosize=True
    )
    
    return fig
