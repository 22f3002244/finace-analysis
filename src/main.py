"""
Financial Portfolio Analytics - Main Execution Pipeline
Complete end-to-end portfolio analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader, INDIAN_ASSETS
from portfolio_optimizer import PortfolioOptimizer, PortfolioBacktest
from risk_metrics import RiskAnalyzer, BenchmarkAnalysis, compare_portfolios

class PortfolioAnalysisPipeline:
    """Complete portfolio analysis workflow"""
    
    def __init__(self, start_date: str = '2015-01-01', initial_capital: float = 1000000):
        """
        Initialize analysis pipeline
        
        Args:
            start_date: Start date for historical data
            initial_capital: Starting capital (default 10 lakh INR)
        """
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.loader = DataLoader(start_date=start_date)
        self.results = {}
        
    def load_data(self, portfolio_type: str = 'balanced'):
        """Load and prepare data"""
        print(f"\n{'='*60}")
        print(f"LOADING DATA FOR {portfolio_type.upper()} PORTFOLIO")
        print(f"{'='*60}")
        
        self.portfolio_type = portfolio_type
        self.data = self.loader.get_portfolio_data(INDIAN_ASSETS[portfolio_type])
        
        print(f"✓ Loaded {len(self.data['prices'])} days of data")
        print(f"✓ Assets: {', '.join(self.data['tickers'])}")
        print(f"✓ Benchmark: {self.data['benchmark']}")
        
    def optimize_portfolios(self):
        """Run portfolio optimization"""
        print(f"\n{'='*60}")
        print("PORTFOLIO OPTIMIZATION")
        print(f"{'='*60}")
        
        optimizer = PortfolioOptimizer(self.data['returns'])
        self.portfolios = optimizer.get_all_portfolios()
        
        print("\n--- Portfolio Allocations ---")
        print(self.portfolios.to_string(index=False))
        
        # Generate efficient frontier
        self.efficient_frontier = optimizer.efficient_frontier(n_portfolios=500)
        
        print(f"\n✓ Generated {len(self.efficient_frontier)} portfolios for efficient frontier")
        
    def backtest_portfolios(self):
        """Backtest all optimized portfolios"""
        print(f"\n{'='*60}")
        print("BACKTESTING PORTFOLIOS")
        print(f"{'='*60}")
        
        self.backtest_results = {}
        
        for idx, row in self.portfolios.iterrows():
            port_type = row['Type']
            weights = row.iloc[4:].values
            
            backtester = PortfolioBacktest(
                self.data['prices'][self.data['tickers']], 
                weights,
                self.initial_capital
            )
            
            backtest_df = backtester.run_backtest()
            metrics = backtester.get_metrics()
            
            self.backtest_results[port_type] = {
                'time_series': backtest_df,
                'metrics': metrics,
                'weights': weights
            }
            
            print(f"\n--- {port_type} Portfolio ---")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:,.2f}")
                else:
                    print(f"{key}: {value}")
    
    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        print(f"\n{'='*60}")
        print("RISK ANALYSIS")
        print(f"{'='*60}")
        
        portfolio_returns = {}
        
        for port_type, results in self.backtest_results.items():
            portfolio_returns[port_type] = results['time_series']['Returns']
        
        # Get benchmark returns
        benchmark_returns = self.data['returns'][self.data['benchmark']]
        
        # Compare portfolios
        self.risk_comparison = compare_portfolios(portfolio_returns, benchmark_returns)
        
        print("\n--- Risk Metrics Comparison ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.risk_comparison.to_string(index=False))
        
    def benchmark_attribution(self):
        """Analyze performance vs benchmark"""
        print(f"\n{'='*60}")
        print("BENCHMARK ATTRIBUTION ANALYSIS")
        print(f"{'='*60}")
        
        benchmark_returns = self.data['returns'][self.data['benchmark']]
        
        # Backtest benchmark
        benchmark_weights = np.array([1.0])
        bench_backtester = PortfolioBacktest(
            self.data['prices'][[self.data['benchmark']]], 
            benchmark_weights,
            self.initial_capital
        )
        
        bench_backtest = bench_backtester.run_backtest()
        bench_metrics = bench_backtester.get_metrics()
        
        print("\n--- Benchmark Performance ---")
        for key, value in bench_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")
        
        # Compare each portfolio to benchmark
        print("\n--- Portfolio vs Benchmark ---")
        for port_type, results in self.backtest_results.items():
            port_returns = results['time_series']['Returns']
            
            analyzer = BenchmarkAnalysis(port_returns, benchmark_returns)
            bench_metrics = analyzer.get_all_metrics()
            
            print(f"\n{port_type}:")
            for key, value in bench_metrics.items():
                print(f"  {key}: {value:.4f}")
    
    def generate_summary_report(self):
        """Generate executive summary"""
        print(f"\n{'='*60}")
        print("EXECUTIVE SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nAnalysis Period: {self.start_date} to {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Portfolio Type: {self.portfolio_type.upper()}")
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Assets Analyzed: {len(self.data['tickers'])}")
        
        # Best performing portfolio
        best_sharpe = self.risk_comparison.loc[
            self.risk_comparison['Sharpe Ratio'].idxmax()
        ]
        
        print(f"\n--- Best Risk-Adjusted Performance ---")
        print(f"Portfolio: {best_sharpe['Portfolio']}")
        print(f"CAGR: {best_sharpe['CAGR (%)']:.2f}%")
        print(f"Volatility: {best_sharpe['Volatility (%)']:.2f}%")
        print(f"Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {best_sharpe['Max Drawdown (%)']:.2f}%")
        
        # Key insights
        print(f"\n--- Key Insights ---")
        print("✓ Portfolio optimization completed successfully")
        print("✓ Risk metrics calculated across multiple dimensions")
        print("✓ Benchmark attribution analysis performed")
        print("✓ All portfolios backtested with realistic assumptions")
        
        print("\n--- Recommendations ---")
        print("• Review allocation drift quarterly")
        print("• Monitor tracking error vs benchmark")
        print("• Rebalance when drift exceeds 5% threshold")
        print("• Adjust risk exposure based on market conditions")
        
    def save_results(self, output_dir: str = 'output'):
        """Save analysis results"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio allocations
        self.portfolios.to_csv(f"{output_dir}/portfolio_allocations.csv", index=False)
        
        # Save risk comparison
        self.risk_comparison.to_csv(f"{output_dir}/risk_metrics.csv", index=False)
        
        # Save time series for each portfolio
        for port_type, results in self.backtest_results.items():
            filename = f"{output_dir}/{port_type.lower().replace(' ', '_')}_timeseries.csv"
            results['time_series'].to_csv(filename)
        
        # Save efficient frontier
        self.efficient_frontier.to_csv(f"{output_dir}/efficient_frontier.csv", index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Results saved to '{output_dir}/' directory")
        print(f"{'='*60}")
    
    def run_complete_analysis(self, portfolio_type: str = 'balanced'):
        """Execute complete analysis pipeline"""
        self.load_data(portfolio_type)
        self.optimize_portfolios()
        self.backtest_portfolios()
        self.calculate_risk_metrics()
        self.benchmark_attribution()
        self.generate_summary_report()
        self.save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE ✓")
        print("="*60 + "\n")
        
        return self.results


def run_multi_portfolio_analysis():
    """Run analysis for all portfolio types"""
    portfolio_types = ['conservative', 'balanced', 'aggressive']
    
    print("\n" + "="*60)
    print("MULTI-PORTFOLIO ANALYSIS")
    print("="*60)
    
    all_results = {}
    
    for port_type in portfolio_types:
        print(f"\n\n{'#'*60}")
        print(f"# ANALYZING {port_type.upper()} PORTFOLIO")
        print(f"{'#'*60}\n")
        
        pipeline = PortfolioAnalysisPipeline(start_date='2015-01-01')
        results = pipeline.run_complete_analysis(port_type)
        all_results[port_type] = pipeline
    
    print("\n" + "="*60)
    print("ALL PORTFOLIO ANALYSES COMPLETE ✓")
    print("="*60 + "\n")
    
    return all_results


if __name__ == "__main__":
    # Option 1: Run single portfolio analysis
    print("\nStarting Portfolio Analysis Pipeline...\n")
    
    pipeline = PortfolioAnalysisPipeline(start_date='2015-01-01', initial_capital=1000000)
    pipeline.run_complete_analysis(portfolio_type='balanced')
    
    # Option 2: Run multi-portfolio analysis (uncomment to use)
    # all_results = run_multi_portfolio_analysis()