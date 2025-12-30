"""
Financial Portfolio Analytics - Data Loader Module
Handles data collection from Yahoo Finance and FRED
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Fetch and prepare financial market data"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize data loader
        
        Args:
            start_date: Format 'YYYY-MM-DD', defaults to 10 years ago
            end_date: Format 'YYYY-MM-DD', defaults to today
        """
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        
    def fetch_india_etfs(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch Indian ETF price data
        
        Args:
            tickers: List of ticker symbols (e.g., ['NIFTYBEES.NS', 'BANKBEES.NS'])
            
        Returns:
            DataFrame with close prices
        """
        print(f"Fetching data from {self.start_date} to {self.end_date}")
        
        # Download data
        raw_data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        # Handle different DataFrame structures based on number of tickers
        if len(tickers) == 1:
            # Single ticker returns simple DataFrame
            if 'Close' in raw_data.columns:
                data = raw_data[['Close']]
                data.columns = [tickers[0]]
            elif 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']]
                data.columns = [tickers[0]]
            else:
                # Fallback if no expected columns
                data = raw_data.to_frame(name=tickers[0]) if isinstance(raw_data, pd.Series) else raw_data
        else:
            # Multiple tickers return multi-level columns
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Try 'Close' first, then 'Adj Close' as fallback
                if 'Close' in raw_data.columns.get_level_values(0):
                    data = raw_data['Close']
                elif 'Adj Close' in raw_data.columns.get_level_values(0):
                    data = raw_data['Adj Close']
                else:
                    raise ValueError("Neither 'Close' nor 'Adj Close' column found in downloaded data")
            else:
                # Single level columns (shouldn't happen with multiple tickers, but handle it)
                data = raw_data
        
        # Ensure data is a DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        print(f"Downloaded {len(data)} days of data for {len(tickers)} assets")
        return data
    
    def get_risk_free_rate(self) -> pd.DataFrame:
        """
        Fetch risk-free rate (using India 10Y G-Sec as proxy)
        Alternative: Use fixed rate if API unavailable
        
        Returns:
            DataFrame with risk-free rates
        """
        # Using a fixed 7% annual rate as proxy for Indian G-Sec
        # In production, fetch from RBI or Bloomberg
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        rf_rate = pd.DataFrame({
            'risk_free_rate': 0.07  # 7% annual
        }, index=dates)
        
        return rf_rate
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare price data
        
        Args:
            df: Raw price data
            
        Returns:
            Cleaned DataFrame
        """
        # Forward fill missing values (market holidays)
        df = df.ffill()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        print(f"Data cleaned: {len(df)} valid rows")
        return df
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            DataFrame of daily returns
        """
        returns = prices.pct_change().dropna()
        return returns
    
    def get_portfolio_data(self, asset_config: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete dataset for portfolio construction
        
        Args:
            asset_config: Dict with 'tickers' and optionally 'benchmark'
            
        Returns:
            Dictionary with prices, returns, and risk-free rate
        """
        tickers = asset_config.get('tickers', [])
        benchmark = asset_config.get('benchmark', [])
        
        all_tickers = tickers + benchmark
        
        # Fetch price data
        prices = self.fetch_india_etfs(all_tickers)
        prices = self.clean_data(prices)
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Get risk-free rate
        rf_rate = self.get_risk_free_rate()
        
        return {
            'prices': prices,
            'returns': returns,
            'risk_free_rate': rf_rate,
            'tickers': tickers,
            'benchmark': benchmark[0] if benchmark else None
        }


# Example usage and asset configuration
INDIAN_ASSETS = {
    'conservative': {
        'tickers': [
            'NIFTYBEES.NS',      # Nifty 50 ETF
            'GOLDBEES.NS',       # Gold ETF
            'CPSEETF.NS',        # CPSE ETF (PSU)
            'BANKBEES.NS'        # Bank Nifty ETF
        ],
        'benchmark': ['NIFTYBEES.NS']
    },
    'balanced': {
        'tickers': [
            'NIFTYBEES.NS',      # Nifty 50
            'JUNIORBEES.NS',     # Nifty Next 50
            'BANKBEES.NS',       # Banking
            'GOLDBEES.NS',       # Gold
            'ITBEES.NS'          # IT Sector
        ],
        'benchmark': ['NIFTYBEES.NS']
    },
    'aggressive': {
        'tickers': [
            'JUNIORBEES.NS',     # Mid Cap
            'BANKBEES.NS',       # Banking
            'ITBEES.NS',         # IT
            'PHARMABEES.NS',     # Pharma
            'AUTOBEES.NS'        # Auto
        ],
        'benchmark': ['NIFTYBEES.NS']
    }
}


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader(start_date='2015-01-01')
    
    # Fetch balanced portfolio data
    data = loader.get_portfolio_data(INDIAN_ASSETS['balanced'])
    
    print("\n=== Data Summary ===")
    print(f"Price data shape: {data['prices'].shape}")
    print(f"Returns data shape: {data['returns'].shape}")
    print(f"\nAssets: {data['tickers']}")
    print(f"Benchmark: {data['benchmark']}")
    print(f"\nFirst few returns:\n{data['returns'].head()}")
    print(f"\nReturns statistics:\n{data['returns'].describe()}")