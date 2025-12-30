# Financial Portfolio Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive portfolio optimization and risk analytics platform implementing Modern Portfolio Theory (MPT) for Indian equity markets. Built with Python, featuring an interactive Streamlit dashboard for real-time portfolio analysis and visualization.


This platform provides institutional-grade portfolio optimization and risk management tools designed for the Indian equity market. By implementing Harry Markowitz's Modern Portfolio Theory alongside comprehensive risk metrics, it enables investors to construct optimal portfolios that maximize returns for a given risk tolerance.

---

## ✨ Key Features

### Portfolio Optimization
- **Multiple Strategies**: Equal-weight, minimum volatility, and maximum Sharpe ratio portfolios
- **Efficient Frontier Generation**: Visual representation of optimal risk-return combinations
- **Constraint-Based Optimization**: Customizable allocation limits and diversification rules
- **Mean-Variance Optimization**: Scipy-powered optimization algorithms

### Risk Management
- **Downside Risk Metrics**: Maximum drawdown, VaR (95%, 99%), CVaR analysis
- **Risk-Adjusted Returns**: Sharpe, Sortino, and Calmar ratios
- **Benchmark Attribution**: Alpha, beta, tracking error, and information ratio
- **Distribution Analysis**: Skewness, kurtosis, and tail risk assessment

### Data & Analytics
- **Real-Time Market Data**: Integration with Yahoo Finance for Indian ETFs
- **Historical Analysis**: 10+ years of backtesting capability
- **Performance Metrics**: CAGR, volatility, and total returns calculation
- **Correlation Analysis**: Asset correlation matrices and diversification benefits

### Visualization & Reporting
- **Interactive Dashboard**: Streamlit-powered web interface
- **Dynamic Charts**: Plotly-based visualizations for efficient frontiers, allocations, and performance
- **Export Capabilities**: CSV exports for all portfolios and metrics
- **Professional Theming**: Clean, minimal design inspired by modern analytics platforms

---

## 🏗 System Architecture

```
┌───────────────────────────────────────────────────────┐
│                   User Interface Layer                │
│            (Streamlit Dashboard / Jupyter)            │
└────────────────────┬──────────────────────────────────┘
                     │
┌────────────────────┴──────────────────────────────────┐
│                  Application Layer                    │
├───────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Portfolio  │  │     Risk     │  │   Backtest   │ │
│  │  Optimizer   │  │   Analyzer   │  │    Engine    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬──────────────────────────────────┘
                     │
┌────────────────────┴──────────────────────────────────┐
│                   Data Layer                          │
├───────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Yahoo Fin.  │  │    Cache     │  │  PostgreSQL  │ │
│  │     API      │  │    Layer     │  │  (Optional)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└───────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- Python 3.8+
- NumPy & Pandas for numerical computing
- SciPy for optimization algorithms
- yFinance for market data

**Frontend:**
- Streamlit for web dashboard
- Plotly for interactive visualizations
- Custom CSS for professional theming

**Data Storage:**
- CSV for exports
- PostgreSQL (optional) for production deployments
- In-memory caching for performance

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Internet connection for market data

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/financial-portfolio-analytics.git
cd financial-portfolio-analytics
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --test
```

---

## Quick Start

### Option 1: Interactive Dashboard

Launch the Streamlit web dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Option 2: Python Script

Run complete portfolio analysis:

```bash
cd src
python main.py
```

### Option 3: Jupyter Notebook

Explore the analysis interactively:

```bash
jupyter notebook notebooks/01_complete_analysis.ipynb
```

---

## 📁 Project Structure

```
financial-portfolio-analytics/
│
├── dashboard.py                 # Streamlit web dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Core application modules
│   ├── data_loader.py          # Market data fetching & preprocessing
│   ├── portfolio_optimizer.py  # MPT optimization algorithms
│   ├── risk_metrics.py         # Risk calculation engine
│   └── main.py                 # Main execution pipeline
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_complete_analysis.ipynb
│
├── output/                      # Generated results
│   ├── portfolio_allocations.csv
│   ├── risk_metrics.csv
│   └── efficient_frontier.csv
│
└── sql/                        # Database schema (optional)
    └── create_tables.sql
```

### Module Descriptions

**`data_loader.py`**
- Fetches historical price data from Yahoo Finance
- Handles Indian ETFs with `.NS` suffix
- Calculates daily returns and cleans missing data
- Manages risk-free rate assumptions

**`portfolio_optimizer.py`**
- Implements mean-variance optimization
- Generates efficient frontier portfolios
- Creates equal-weight, min-vol, and max-sharpe portfolios
- Backtests portfolio performance

**`risk_metrics.py`**
- Calculates 20+ risk and performance metrics
- Benchmark attribution analysis
- Portfolio comparison utilities
- Statistical distribution analysis

**`main.py`**
- End-to-end analysis pipeline
- Orchestrates data loading, optimization, and reporting
- Generates comprehensive output files

---

### Modern Portfolio Theory Implementation

Our implementation follows Markowitz's seminal 1952 framework:

1. **Expected Returns**: Calculated as annualized mean of historical returns
   ```
   E(Rp) = Σ wi * E(Ri)
   ```

2. **Portfolio Variance**: Based on covariance matrix
   ```
   σp² = wᵀ Σ w
   ```

3. **Sharpe Ratio Maximization**: Optimal risk-adjusted returns
   ```
   Sharpe = (Rp - Rf) / σp
   ```

### Asset Universe

The platform supports three portfolio configurations for Indian markets:

**Conservative Portfolio:**
- NIFTYBEES (Nifty 50 ETF)
- GOLDBEES (Gold ETF)
- CPSEETF (PSU ETF)
- BANKBEES (Bank Nifty ETF)

**Balanced Portfolio:**
- NIFTYBEES (Nifty 50)
- JUNIORBEES (Nifty Next 50)
- BANKBEES (Banking Sector)
- GOLDBEES (Gold)
- ITBEES (IT Sector)

**Aggressive Portfolio:**
- JUNIORBEES (Mid Cap)
- BANKBEES (Banking)
- ITBEES (IT)
- PHARMABEES (Pharma)
- AUTOBEES (Auto)

### Optimization Constraints

- Sum of weights = 1 (fully invested)
- No short selling (wi ≥ 0)
- Optional: Maximum position size limits
- Risk-free rate: 7% (Indian G-Sec proxy)

### Risk Metrics Calculated

**Return Metrics:**
- CAGR (Compound Annual Growth Rate)
- Total Return
- Annualized Returns

**Volatility Metrics:**
- Standard Deviation (annualized)
- Downside Deviation
- Maximum Drawdown

**Risk-Adjusted:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Tail Risk:**
- Value at Risk (95%, 99%)
- Conditional VaR (CVaR/ES)

**Benchmark Attribution:**
- Alpha (Jensen's)
- Beta
- Tracking Error
- Information Ratio
- Upside/Downside Capture

**Distribution:**
- Skewness
- Kurtosis

---

## 💻 Usage Examples

### Example 1: Basic Portfolio Optimization

```python
from src.data_loader import DataLoader, INDIAN_ASSETS
from src.portfolio_optimizer import PortfolioOptimizer

# Load data
loader = DataLoader(start_date='2015-01-01')
data = loader.get_portfolio_data(INDIAN_ASSETS['balanced'])

# Optimize
optimizer = PortfolioOptimizer(data['returns'])
portfolios = optimizer.get_all_portfolios()

print(portfolios)
```

### Example 2: Custom Asset Selection

```python
custom_assets = {
    'tickers': ['NIFTYBEES.NS', 'BANKBEES.NS', 'ITBEES.NS'],
    'benchmark': ['NIFTYBEES.NS']
}

data = loader.get_portfolio_data(custom_assets)
optimizer = PortfolioOptimizer(data['returns'])
max_sharpe = optimizer.maximize_sharpe()
```

### Example 3: Backtesting

```python
from src.portfolio_optimizer import PortfolioBacktest

# Define weights
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Backtest
backtester = PortfolioBacktest(
    data['prices'][data['tickers']], 
    weights,
    initial_capital=1000000
)

results = backtester.run_backtest()
metrics = backtester.get_metrics()
```

### Example 4: Risk Analysis

```python
from src.risk_metrics import RiskAnalyzer, BenchmarkAnalysis

# Calculate risk metrics
analyzer = RiskAnalyzer(portfolio_returns)
metrics = analyzer.get_all_metrics()

# Benchmark comparison
bench_analyzer = BenchmarkAnalysis(
    portfolio_returns, 
    benchmark_returns
)
attribution = bench_analyzer.get_all_metrics()
```

---

## 🎨 Dashboard Guide

### Navigation

The dashboard contains five main tabs:

1. **Overview**: Portfolio summary and asset correlations
2. **Portfolio Allocation**: Visual breakdown of optimized portfolios
3. **Risk Analysis**: Comprehensive risk metrics comparison
4. **Performance**: Historical backtest results and drawdown analysis
5. **Efficient Frontier**: Risk-return space visualization

### Configuration Panel

**Left Sidebar Options:**
- Portfolio Type: Conservative, Balanced, or Aggressive
- Date Range: Start and end dates for historical data
- Initial Capital: Starting investment amount (₹)
- Run Analysis: Execute optimization and backtesting

### Key Visualizations

**Correlation Matrix**: Heatmap showing asset return correlations

**Allocation Pie Charts**: Asset weights for each optimized portfolio

**Efficient Frontier**: Scatter plot of risk-return combinations with color-coded Sharpe ratios

**Performance Chart**: Line graph of portfolio value over time

**Drawdown Chart**: Visual representation of portfolio drawdowns

---

## 📖 Technical Documentation

### Data Sources

- **Market Data**: Yahoo Finance API via `yfinance` library
- **ETF Universe**: NSE-listed ETFs with `.NS` suffix
- **Frequency**: Daily close prices (adjusted for splits/dividends)
- **Risk-Free Rate**: Fixed 7% annual (Indian 10Y G-Sec proxy)

### Assumptions

1. No transaction costs or taxes
2. Assets are infinitely divisible
3. Returns follow normal distribution (limitation acknowledged)
4. Past performance used for forward-looking optimization
5. Correlations remain stable over time

### Performance Optimization

- Streamlit caching (`@st.cache_data`) for expensive computations
- Vectorized NumPy operations for speed
- Efficient covariance matrix calculations
- Pre-computed efficient frontiers

### Known Limitations

1. **Normal Distribution Assumption**: Real returns often exhibit fat tails
2. **Historical Data Dependency**: Past ≠ future
3. **Static Correlations**: Correlations change during market stress
4. **No Rebalancing Costs**: Real-world implementations incur costs
5. **Limited Asset Classes**: Currently supports ETFs only

---

## 📈 Results & Performance

### Sample Results (Balanced Portfolio, 2015-2025)

| Portfolio | CAGR | Volatility | Sharpe | Max DD |
|-----------|------|------------|--------|--------|
| Equal Weight | 19.95% | 12.08% | 1.01 | -17.69% |
| Min Volatility | 20.19% | 15.65% | 0.82 | -25.13% |
| Max Sharpe | 20.06% | 12.91% | 0.96 | -18.83% |

### Key Findings

- Portfolio optimization improved Sharpe ratio by 15-20% vs equal-weight
- Maximum drawdown reduced through optimal diversification
- All portfolios outperformed benchmark (Nifty 50) on risk-adjusted basis
- Gold allocation provided effective hedge during market downturns

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 src/
black src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Author

Vedant Konde - vedantkonde09@gmail.com

<p align="center">⭐ Star this repo if you find it helpful!</p>
