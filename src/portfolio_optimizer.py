"""
Financial Portfolio Analytics - Portfolio Optimizer
Implements Modern Portfolio Theory optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Modern Portfolio Theory optimization engine"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.07):
        """
        Initialize optimizer
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Annual risk-free rate (default 7% for India)
        """
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        self.rf_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights: Array of portfolio weights
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.rf_rate) / portfolio_vol
        
        return portfolio_return, portfolio_vol, sharpe_ratio
    
    def minimize_volatility(self) -> Dict:
        """
        Find minimum volatility portfolio
        
        Returns:
            Dictionary with weights and performance metrics
        """
        def objective(weights):
            return self.portfolio_performance(weights)[1]  # Return volatility
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'type': 'Minimum Volatility'
        }
    
    def maximize_sharpe(self) -> Dict:
        """
        Find maximum Sharpe ratio portfolio
        
        Returns:
            Dictionary with weights and performance metrics
        """
        def objective(weights):
            return -self.portfolio_performance(weights)[2]  # Negative Sharpe (for minimization)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'type': 'Maximum Sharpe'
        }
    
    def equal_weight_portfolio(self) -> Dict:
        """
        Create equal-weighted portfolio
        
        Returns:
            Dictionary with weights and performance metrics
        """
        weights = np.array([1/self.n_assets] * self.n_assets)
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'type': 'Equal Weight'
        }
    
    def efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            n_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with portfolio returns and volatilities
        """
        results = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            
            ret, vol, sharpe = self.portfolio_performance(weights)
            results[0, i] = ret
            results[1, i] = vol
            results[2, i] = sharpe
        
        return pd.DataFrame({
            'Return': results[0],
            'Volatility': results[1],
            'Sharpe': results[2]
        })
    
    def get_all_portfolios(self) -> pd.DataFrame:
        """
        Generate all optimized portfolios
        
        Returns:
            DataFrame with portfolio allocations and metrics
        """
        portfolios = [
            self.equal_weight_portfolio(),
            self.minimize_volatility(),
            self.maximize_sharpe()
        ]
        
        # Create allocation DataFrame
        allocation_df = pd.DataFrame()
        
        for port in portfolios:
            port_data = {
                'Type': port['type'],
                'Return': port['return'],
                'Volatility': port['volatility'],
                'Sharpe': port['sharpe_ratio']
            }
            
            # Add weights
            for i, asset in enumerate(self.returns.columns):
                port_data[asset] = port['weights'][i]
            
            allocation_df = pd.concat([allocation_df, pd.DataFrame([port_data])], ignore_index=True)
        
        return allocation_df


class PortfolioBacktest:
    """Backtest portfolio performance over time"""
    
    def __init__(self, prices: pd.DataFrame, weights: np.ndarray, initial_capital: float = 1000000):
        """
        Initialize backtester
        
        Args:
            prices: DataFrame of asset prices
            weights: Portfolio weights
            initial_capital: Starting capital (default 10 lakh)
        """
        self.prices = prices
        self.weights = weights
        self.initial_capital = initial_capital
        
    def run_backtest(self) -> pd.DataFrame:
        """
        Run portfolio backtest
        
        Returns:
            DataFrame with portfolio value over time
        """
        # Calculate position sizes
        positions = self.weights * self.initial_capital
        shares = positions / self.prices.iloc[0]
        
        # Calculate portfolio value over time
        portfolio_value = (self.prices * shares).sum(axis=1)
        
        backtest_df = pd.DataFrame({
            'Date': self.prices.index,
            'Portfolio_Value': portfolio_value,
            'Returns': portfolio_value.pct_change()
        })
        
        return backtest_df.set_index('Date')
    
    def get_metrics(self) -> Dict:
        """
        Calculate backtest performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        backtest = self.run_backtest()
        returns = backtest['Returns'].dropna()
        
        total_return = (backtest['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        n_years = len(backtest) / 252
        cagr = ((backtest['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/n_years) - 1) * 100
        
        volatility = returns.std() * np.sqrt(252) * 100
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Volatility (%)': volatility,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': (cagr / volatility) if volatility > 0 else 0,
            'Final Value': backtest['Portfolio_Value'].iloc[-1]
        }


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader, INDIAN_ASSETS
    
    # Load data
    loader = DataLoader(start_date='2015-01-01')
    data = loader.get_portfolio_data(INDIAN_ASSETS['balanced'])
    
    # Optimize portfolios
    optimizer = PortfolioOptimizer(data['returns'])
    portfolios = optimizer.get_all_portfolios()
    
    print("\n=== Portfolio Allocations ===")
    print(portfolios.to_string())
    
    # Backtest maximum Sharpe portfolio
    max_sharpe_weights = portfolios[portfolios['Type'] == 'Maximum Sharpe'].iloc[0, 4:].values
    backtester = PortfolioBacktest(data['prices'][data['tickers']], max_sharpe_weights)
    
    metrics = backtester.get_metrics()
    print("\n=== Maximum Sharpe Portfolio Performance ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")