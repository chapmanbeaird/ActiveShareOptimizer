"""
Utility functions for calculations in the Active Share Optimizer.
"""

def calculate_active_share(portfolio, benchmark):
    """
    Calculate the Active Share:
    ActiveShare = 0.5 * sum(abs(portfolio_weight - benchmark_weight))
    
    The weights in both portfolio and benchmark should sum to 100%.
    
    Parameters:
    - portfolio: Dictionary mapping tickers to weights
    - benchmark: Dictionary mapping tickers to weights
    
    Returns:
    - active_share: The calculated active share as a percentage
    """
    diff_sum = 0
    for ticker in set(portfolio.keys()).union(benchmark.keys()):
        port_weight = portfolio.get(ticker, 0)
        bench_weight = benchmark.get(ticker, 0)
        diff_sum += abs(port_weight - bench_weight)
    
    return diff_sum / 2

def get_sector_weights(stocks_data):
    """
    Calculate sector weights from stock-level data.
    
    Parameters:
    - stocks_data: DataFrame containing stock data with 'Sector' and 'Port. Ending Weight' columns
    
    Returns:
    - sector_weights: Dictionary mapping sectors to their total weights
    """
    sector_weights = {}
    for sector in stocks_data['Sector'].unique():
        sector_stocks = stocks_data[stocks_data['Sector'] == sector]
        sector_weights[sector] = sector_stocks['Port. Ending Weight'].sum()
    
    return sector_weights 