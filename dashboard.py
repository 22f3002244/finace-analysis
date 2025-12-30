"""
Financial Portfolio Analytics - Interactive Streamlit Dashboard
Real-time portfolio visualization and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader, INDIAN_ASSETS
from src.portfolio_optimizer import PortfolioOptimizer, PortfolioBacktest
from src.risk_metrics import RiskAnalyzer, BenchmarkAnalysis, compare_portfolios

# Page configuration
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean White & Black Theme
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #222;
        font-size: 28px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    h2 {
        color: #222;
        font-size: 20px;
        font-weight: 500;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    h3 {
        color: #222;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 12px;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 500;
        color: #222;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #666;
        font-weight: 400;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 12px;
    }
    
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e8e8e8;
        border-radius: 4px;
        padding: 20px;
        box-shadow: none;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-bottom: 1px solid #fff;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border: 1px solid #fff;
        border-radius: 3px;
        color: #666;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 400;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        color: black;
        border-color: #fff;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: white;
        color: #000;
        border: 1px solid #fff;
        border-radius: 3px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 400;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f5f5f5;
        border-color: #d0d0d0;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #000;
        color: white;
        border-color: #000;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #333;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #fff;
        border-radius: 3px;
    }
    
    /* DataFrames */
    .dataframe {
        font-size: 14px;
        border: 1px solid #fff !important;
    }
    
    .dataframe thead tr th {
        background-color: #fff !important;
        color: #666 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        border-bottom: 1px solid #e8e8e8 !important;
        padding: 12px !important;
    }
    
    .dataframe tbody tr td {
        border-bottom: 1px solid #f0f0f0 !important;
        padding: 12px !important;
        color: #333 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #fff !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e8e8e8;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #222;
        font-size: 16px;
        font-weight: 500;
    }
    
    /* Input fields */
    .stDateInput > div > div,
    .stNumberInput > div > div {
        background-color: white;
        border: 1px solid #fff;
        border-radius: 3px;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #fff;
        border: 1px solid #fff;
        border-radius: 4px;
        color: #fff;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border: 1px solid #fff;
        border-radius: 4px;
        background-color: white;
    }
    
    /* Divider */
    hr {
        margin: 24px 0;
        border: none;
        border-top: 1px solid #fff;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #222 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_portfolio_data(portfolio_type, start_date, end_date):
    """Load and cache portfolio data"""
    loader = DataLoader(start_date=start_date, end_date=end_date)
    return loader.get_portfolio_data(INDIAN_ASSETS[portfolio_type])

@st.cache_data
def optimize_portfolio(returns_df):
    """Optimize portfolio and cache results"""
    optimizer = PortfolioOptimizer(returns_df)
    portfolios = optimizer.get_all_portfolios()
    efficient_frontier = optimizer.efficient_frontier(n_portfolios=500)
    return portfolios, efficient_frontier

@st.cache_data
def backtest_portfolio(prices_df, weights, initial_capital):
    """Backtest portfolio and cache results"""
    backtester = PortfolioBacktest(prices_df, weights, initial_capital)
    backtest_df = backtester.run_backtest()
    metrics = backtester.get_metrics()
    return backtest_df, metrics

def main():
    # Title
    st.title("Financial Portfolio Analytics")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Portfolio selection
    portfolio_type = st.sidebar.selectbox(
        "Portfolio Type",
        ["conservative", "balanced", "aggressive"],
        index=1
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=3650),
        max_value=datetime.now()
    )
    end_date = col2.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    # Initial capital
    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹)",
        min_value=100000,
        max_value=100000000,
        value=1000000,
        step=100000
    )
    
    # Load data button
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Loading data and optimizing portfolios..."):
            # Load data
            data = load_portfolio_data(
                portfolio_type,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Optimize portfolios
            portfolios, efficient_frontier = optimize_portfolio(data['returns'])
            
            # Store in session state
            st.session_state['data'] = data
            st.session_state['portfolios'] = portfolios
            st.session_state['efficient_frontier'] = efficient_frontier
            st.session_state['initial_capital'] = initial_capital
    
    # Check if data is loaded
    if 'data' not in st.session_state:
        st.info("Configure settings in the sidebar and click 'Run Analysis' to begin")
        return
    
    # Retrieve from session state
    data = st.session_state['data']
    portfolios = st.session_state['portfolios']
    efficient_frontier = st.session_state['efficient_frontier']
    initial_capital = st.session_state['initial_capital']
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Portfolio Allocation", 
        "Risk Analysis",
        "Performance",
        "Efficient Frontier"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Type",
                portfolio_type.upper()
            )
        
        with col2:
            st.metric(
                "Number of Assets",
                len(data['tickers'])
            )
        
        with col3:
            st.metric(
                "Data Points",
                f"{len(data['prices']):,}"
            )
        
        with col4:
            st.metric(
                "Analysis Period",
                f"{(end_date - start_date).days} days"
            )
        
        st.markdown("---")
        
        # Asset list
        st.subheader("Assets in Portfolio")
        asset_info = pd.DataFrame({
            'Ticker': data['tickers'],
            'Asset Name': [ticker.replace('.NS', '') for ticker in data['tickers']]
        })
        st.dataframe(asset_info, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Asset Correlation Matrix")
        corr_matrix = data['returns'].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='Blues',
            title="Asset Return Correlations"
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: PORTFOLIO ALLOCATION
    with tab2:
        st.header("Portfolio Allocations")
        
        # Display allocation table
        st.subheader("Optimized Portfolios")
        
        display_cols = ['Type', 'Return', 'Volatility', 'Sharpe'] + data['tickers']
        st.dataframe(
            portfolios[display_cols].style.format({
                'Return': '{:.2f}',
                'Volatility': '{:.2f}',
                'Sharpe': '{:.2f}',
                **{ticker: '{:.2%}' for ticker in data['tickers']}
            }),
            use_container_width=True
        )
        
        # Allocation charts
        st.subheader("Visual Allocation")
        
        portfolio_choice = st.selectbox(
            "Select Portfolio",
            portfolios['Type'].tolist()
        )
        
        selected_portfolio = portfolios[portfolios['Type'] == portfolio_choice].iloc[0]
        weights = selected_portfolio[data['tickers']].values
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=data['tickers'],
            values=weights,
            hole=0.4,
            marker=dict(colors=['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6'])
        )])
        fig.update_layout(
            title=f"{portfolio_choice} - Asset Allocation",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar chart
        fig = go.Figure(data=[go.Bar(
            x=data['tickers'],
            y=weights * 100,
            text=[f'{w*100:.1f}%' for w in weights],
            textposition='auto',
            marker=dict(color='#08519c')
        )])
        fig.update_layout(
            title=f"{portfolio_choice} - Allocation Breakdown",
            xaxis_title="Asset",
            yaxis_title="Allocation (%)",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'},
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: RISK ANALYSIS
    with tab3:
        st.header("Risk Metrics")
        
        # Backtest all portfolios
        risk_data = []
        
        for idx, row in portfolios.iterrows():
            port_type = row['Type']
            weights = row[data['tickers']].values
            
            backtest_df, metrics = backtest_portfolio(
                data['prices'][data['tickers']],
                weights,
                initial_capital
            )
            
            # Calculate risk metrics
            returns = backtest_df['Returns'].dropna()
            analyzer = RiskAnalyzer(returns)
            risk_metrics = analyzer.get_all_metrics()
            
            # Benchmark comparison
            benchmark_returns = data['returns'][data['benchmark']]
            bench_analyzer = BenchmarkAnalysis(returns, benchmark_returns)
            bench_metrics = bench_analyzer.get_all_metrics()
            
            # Combine metrics
            all_metrics = {
                'Portfolio': port_type,
                **metrics,
                **risk_metrics,
                **bench_metrics
            }
            risk_data.append(all_metrics)
        
        risk_df = pd.DataFrame(risk_data)
        
        # Display metrics
        st.subheader("Comprehensive Risk Metrics")
        
        # Key metrics
        key_metrics = ['Portfolio', 'CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 
                      'Sortino Ratio', 'Max Drawdown (%)', 'VaR 95% (%)', 'Alpha (%)', 'Beta']
        
        st.dataframe(
            risk_df[key_metrics].style.format({
                'CAGR (%)': '{:.2f}',
                'Volatility (%)': '{:.2f}',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Max Drawdown (%)': '{:.2f}',
                'VaR 95% (%)': '{:.2f}',
                'Alpha (%)': '{:.2f}',
                'Beta': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Risk comparison chart
        st.subheader("Risk-Return Profile")
        
        fig = go.Figure()
        
        for idx, row in risk_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Volatility (%)']],
                y=[row['CAGR (%)']],
                mode='markers+text',
                name=row['Portfolio'],
                text=row['Portfolio'],
                textposition='top center',
                marker=dict(size=15, color='#08519c')
            ))
        
        fig.update_layout(
            title="Risk-Return Tradeoff",
            xaxis_title="Volatility (%)",
            yaxis_title="CAGR (%)",
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'},
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: PERFORMANCE
    with tab4:
        st.header("Portfolio Performance")
        
        # Select portfolio
        portfolio_choice = st.selectbox(
            "Select Portfolio for Analysis",
            portfolios['Type'].tolist(),
            key="performance_select"
        )
        
        selected_portfolio = portfolios[portfolios['Type'] == portfolio_choice].iloc[0]
        weights = selected_portfolio[data['tickers']].values
        
        # Backtest
        backtest_df, metrics = backtest_portfolio(
            data['prices'][data['tickers']],
            weights,
            initial_capital
        )
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Final Value",
                f"₹{metrics['Final Value']:,.0f}",
                f"{metrics['Total Return (%)']:.2f}%"
            )
        
        with col2:
            st.metric(
                "CAGR",
                f"{metrics['CAGR (%)']:.2f}%"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['Sharpe Ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['Max Drawdown (%)']:.2f}%"
            )
        
        # Portfolio value chart
        st.subheader("Portfolio Value Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_df.index,
            y=backtest_df['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value',
            fill='tozeroy',
            line=dict(color='#2ca02c', width=2),
            fillcolor='rgba(44, 160, 44, 0.1)'
        ))
        
        fig.update_layout(
            title=f"{portfolio_choice} - Historical Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'},
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        
        cumulative = (1 + backtest_df['Returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#d62728', width=1.5),
            fillcolor='rgba(214, 39, 40, 0.1)'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'},
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: EFFICIENT FRONTIER
    with tab5:
        st.header("Efficient Frontier")
        
        # Plot efficient frontier
        fig = go.Figure()
        
        # Random portfolios
        fig.add_trace(go.Scatter(
            x=efficient_frontier['Volatility'] * 100,
            y=efficient_frontier['Return'] * 100,
            mode='markers',
            name='Random Portfolios',
            marker=dict(
                size=5,
                color=efficient_frontier['Sharpe'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))
        
        # Optimized portfolios
        for idx, row in portfolios.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Volatility']],
                y=[row['Return']],
                mode='markers+text',
                name=row['Type'],
                text=row['Type'],
                textposition='top center',
                marker=dict(size=15, symbol='star')
            ))
        
        fig.update_layout(
            title="Efficient Frontier - Risk-Return Space",
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': '#222'},
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Efficient Frontier Interpretation:**
        - Each point represents a possible portfolio combination
        - Color indicates Sharpe ratio (higher is better)
        - Star markers show optimized portfolios
        - Portfolios on the upper-left offer better risk-adjusted returns
        """)

if __name__ == "__main__":
    main()