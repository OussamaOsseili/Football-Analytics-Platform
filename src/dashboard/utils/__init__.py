
import streamlit as st

def load_css():
    """Load global CSS styles"""
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
        }
        .stSelectbox, .stMultiSelect {
            background-color: #262730;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #00ff00;
        }
        </style>
    """, unsafe_allow_html=True)
