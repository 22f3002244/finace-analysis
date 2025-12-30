"""
Financial Portfolio Analytics - Risk Metrics Module
Comprehensive risk measurement and analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional

class RiskAnalyzer:
    """Calculate comprehensive risk metrics for portfolios"""
    
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.07):
        """
        Initialize risk analyzer
        
        Args:
            returns: Series of portfolio returns
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns.dropna()
        self.rf_rate = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        
    def cagr(self) -> float:
        """Calculate Compound Annual Growth Rate"""
        n_years = len(self.returns) / 252
        total_return = (1 + self.returns).prod()
        cagr = (total_return ** (1/n_years)) - 1
        return cagr * 100
    
    def volatility(self) -> float:
        """Calculate annualized volatility"""
        return self.returns.std() * np.sqrt(252) * 100
    
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = self.returns - self.rf_daily
        if excess_returns.std() == 0:
            return 0
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe
    
    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = self.returns - self.rf_daily
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return sortino
    
    def maximum_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min() * 100
        max_dd_idx = drawdown.idxmin()
        
        # Find peak before the trough
        peak_idx = cumulative[:max_dd_idx].idxmax()
        
        return max_dd, peak_idx, max_dd_idx
    
    def value_at_risk(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            confidence_level: Confidence level (0.95 or 0.99)
            
        Returns:
            VaR as percentage
        """
        var = np.percentile(self.returns, (1 - confidence_level) * 100)
        return var * 100
    
    def conditional_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            CVaR as percentage
        """
        var = np.percentile(self.returns, (1 - confidence_level) * 100)
        cvar = self.returns[self.returns <= var].mean()
        return cvar * 100
    
    def skewness(self) -> float:
        """Calculate return distribution skewness"""
        return stats.skew(self.returns)
    
    def kurtosis(self) -> float:
        """Calculate return distribution kurtosis"""
        return stats.kurtosis(self.returns)
    
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)"""
        cagr_val = self.cagr()
        max_dd, _, _ = self.maximum_drawdown()
        
        if max_dd == 0:
            return 0
        
        return cagr_val / abs(max_dd)
    
    def get_all_metrics(self) -> Dict:
        """
        Calculate all risk metrics
        
        Returns:
            Dictionary of all metrics
        """
        max_dd, peak_date, trough_date = self.maximum_drawdown()
        
        metrics = {
            'CAGR (%)': self.cagr(),
            'Volatility (%)': self.volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown (%)': max_dd,
            'Drawdown Peak Date': peak_date,
            'Drawdown Trough Date': trough_date,
            'VaR 95% (%)': self.value_at_risk(0.95),
            'CVaR 95% (%)': self.conditional_var(0.95),
            'VaR 99% (%)': self.value_at_risk(0.99),
            'CVaR 99% (%)': self.conditional_var(0.99),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            'Calmar Ratio': self.calmar_ratio()
        }
        
        return metrics


class BenchmarkAnalysis:
    """Compare portfolio against benchmark"""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                 risk_free_rate: float = 0.07):
        """
        Initialize benchmark analyzer
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            risk_free_rate: Annual risk-free rate
        """
        self.portfolio_returns = portfolio_returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna()
        self.rf_rate = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        
        # Align dates
        common_dates = self.portfolio_returns.index.intersection(self.benchmark_returns.index)
        self.portfolio_returns = self.portfolio_returns[common_dates]
        self.benchmark_returns = self.benchmark_returns[common_dates]
    
    def beta(self) -> float:
        """Calculate portfolio beta"""
        covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0][1]
        benchmark_variance = np.var(self.benchmark_returns)
        
        if benchmark_variance == 0:
            return 0
        
        return covariance / benchmark_variance
    
    def alpha(self) -> float:
        """Calculate Jensen's alpha (annualized)"""
        beta_val = self.beta()
        
        portfolio_return = self.portfolio_returns.mean() * 252
        benchmark_return = self.benchmark_returns.mean() * 252
        
        alpha = portfolio_return - (self.rf_rate + beta_val * (benchmark_return - self.rf_rate))
        return alpha * 100
    
    def tracking_error(self) -> float:
        """Calculate tracking error (annualized)"""
        excess_returns = self.portfolio_returns - self.benchmark_returns
        return excess_returns.std() * np.sqrt(252) * 100
    
    def information_ratio(self) -> float:
        """Calculate information ratio"""
        excess_returns = self.portfolio_returns - self.benchmark_returns
        te = self.tracking_error()
        
        if te == 0:
            return 0
        
        return (excess_returns.mean() * 252 * 100) / te
    
    def up_capture(self) -> float:
        """Calculate upside capture ratio"""
        up_months = self.benchmark_returns > 0
        
        if up_months.sum() == 0:
            return 0
        
        portfolio_up = self.portfolio_returns[up_months].mean()
        benchmark_up = self.benchmark_returns[up_months].mean()
        
        if benchmark_up == 0:
            return 0
        
        return (portfolio_up / benchmark_up) * 100
    
    def down_capture(self) -> float:
        """Calculate downside capture ratio"""
        down_months = self.benchmark_returns < 0
        
        if down_months.sum() == 0:
            return 0
        
        portfolio_down = self.portfolio_returns[down_months].mean()
        benchmark_down = self.benchmark_returns[down_months].mean()
        
        if benchmark_down == 0:
            return 0
        
        return (portfolio_down / benchmark_down) * 100
    
    def get_all_metrics(self) -> Dict:
        """
        Calculate all benchmark comparison metrics
        
        Returns:
            Dictionary of metrics
        """
        return {
            'Beta': self.beta(),
            'Alpha (%)': self.alpha(),
            'Tracking Error (%)': self.tracking_error(),
            'Information Ratio': self.information_ratio(),
            'Upside Capture (%)': self.up_capture(),
            'Downside Capture (%)': self.down_capture()
        }


def compare_portfolios(portfolios_dict: Dict[str, pd.Series], 
                       benchmark: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.07) -> pd.DataFrame:
    """
    Compare multiple portfolios
    
    Args:
        portfolios_dict: Dictionary mapping portfolio names to return series
        benchmark: Optional benchmark return series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, returns in portfolios_dict.items():
        analyzer = RiskAnalyzer(returns, risk_free_rate)
        metrics = analyzer.get_all_metrics()
        metrics['Portfolio'] = name
        
        if benchmark is not None:
            bench_analyzer = BenchmarkAnalysis(returns, benchmark, risk_free_rate)
            bench_metrics = bench_analyzer.get_all_metrics()
            metrics.update(bench_metrics)
        
        results.append(metrics)
    
    df = pd.DataFrame(results)
    cols = ['Portfolio'] + [col for col in df.columns if col != 'Portfolio']
    return df[cols]


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader, INDIAN_ASSETS
    from portfolio_optimizer import PortfolioOptimizer, PortfolioBacktest
    
    # Load data
    loader = DataLoader(start_date='2015-01-01')
    data = loader.get_portfolio_data(INDIAN_ASSETS['balanced'])
    
    # Create portfolios
    optimizer = PortfolioOptimizer(data['returns'])
    portfolios = optimizer.get_all_portfolios()
    
    # Backtest each portfolio
    portfolio_returns = {}
    
    for idx, row in portfolios.iterrows():
        port_type = row['Type']
        weights = row.iloc[4:].values
        
        backtester = PortfolioBacktest(data['prices'][data['tickers']], weights)
        backtest_df = backtester.run_backtest()
        
        portfolio_returns[port_type] = backtest_df['Returns']
    
    # Add benchmark
    benchmark_returns = data['returns'][data['benchmark']]
    
    # Compare all portfolios
    comparison = compare_portfolios(portfolio_returns, benchmark_returns)
    
    print("\n=== Portfolio Risk Comparison ===")
    print(comparison.to_string())