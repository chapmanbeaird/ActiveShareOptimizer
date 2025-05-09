"""
Main entry point for the Active Share Optimizer.
"""

import pandas as pd
from collections import defaultdict
from datetime import datetime

from optimizer.data.loaders import load_portfolio_data_csv, load_constraints, load_optimizer_input_file
from optimizer.models.optimizer import optimize_portfolio_pulp
from optimizer.utils.reporting import save_optimizer_results_to_excel

def main(
    data_file_path='inputs/optimizer_input_file.xlsm',
    constraints_file_path=None,  # Optional for backward compatibility
    num_positions=60,
    target_active_share=0.55,
    sector_tolerance=0.03,
    min_position=0.0,
    max_position=5.0,
    core_rank_limit=3,
    forced_positions=None,
    time_limit=120,
    increment=0.5
):
    """
    Main function to run the optimizer with adjustable parameters.
    
    Parameters:
    - data_file_path: Path to the input file (Excel or CSV)
    - constraints_file_path: Optional path to constraints file (for backward compatibility)
    - num_positions: Exact number of positions required in the portfolio
    - target_active_share: Target Active Share percentage (as a decimal)
    - sector_tolerance: Maximum allowed deviation from benchmark sector weights (as a decimal)
    - min_position: Minimum position size as a percentage (e.g., 1.0)
    - max_position: Maximum position size as a percentage (e.g., 5.0)
    - core_rank_limit: Only consider stocks with Core Model rank <= this value
    - forced_positions: Dictionary {ticker: (min, max)} for positions to force
    - time_limit: Maximum time allowed for the solver (in seconds)
    - increment: Allowed increment for position sizes (e.g., 0.5 for 0.5% increments)
    
    Returns:
    - new_portfolio: Dictionary of ticker to weight for the optimized portfolio
    - added_stocks: List of stocks added to the portfolio
    - optimized_active_share: The new Active Share after optimization
    - output_file: Path to the saved Excel file with results
    """
        # Load all data from single input file
    stocks_data, total_active_share, stocks_to_avoid, sector_constraints, locked_tickers = load_optimizer_input_file(data_file_path)
    
    print(f"Active Share from data: {total_active_share:.2f}%")
    print(f"Loaded {len(stocks_data)} stock entries from {data_file_path}")
    
    print(f"Loaded {len(stocks_to_avoid)} stocks to avoid")
    print(f"Loaded {len(sector_constraints)} sector constraints")
    print(f"Found {len(locked_tickers)} locked tickers that will maintain their current weight")
    
    # Print the input active share for confirmation
    print(f"\nInput Active Share for optimization: {total_active_share:.2f}%")
    
    print(f"Solver timeout set to: {time_limit} seconds")
    if increment is not None:
        print(f"Using position increment size: {increment}%")
    else:
        print("Not enforcing position increment size (using continuous weights)")
        
    # Run the optimizer
    new_portfolio, added_stocks, optimized_active_share = optimize_portfolio_pulp(
        stocks_data, 
        original_active_share=total_active_share,
        num_positions=num_positions,
        target_active_share=target_active_share,
        sector_tolerance=sector_tolerance,
        stocks_to_avoid=stocks_to_avoid,
        sector_constraints=sector_constraints,
        min_position=min_position,
        max_position=max_position,
        core_rank_limit=core_rank_limit,
        forced_positions=forced_positions,
        time_limit=time_limit,
        increment=increment,
        locked_tickers=locked_tickers
    )
    
    
    # Create a results dataframe
    print("\nCreating results spreadsheet...")
    
    # Calculate Active Share for each stock in the original and new portfolio
    original_active_share_by_stock = {}
    new_active_share_by_stock = {}
    
    # First, ensure we have a complete list of all tickers in both portfolios and benchmark
    all_tickers = set()
    for _, row in stocks_data.iterrows():
        all_tickers.add(row['Ticker'])
    
    # Add tickers from new portfolio (if it exists)
    if new_portfolio:
        for ticker in new_portfolio.keys():
            all_tickers.add(ticker)
            
    # Calculate active share contribution for each stock
    for ticker in all_tickers:
        # Get weights (defaulting to 0 if not present)
        bench_weight = 0
        port_weight = 0
        new_weight = new_portfolio.get(ticker, 0) if new_portfolio else 0
        
        # Find the stock in the original data to get benchmark and portfolio weights
        stock_row = stocks_data[stocks_data['Ticker'] == ticker]
        if not stock_row.empty:
            bench_weight = stock_row.iloc[0]['Bench Weight']
            port_weight = stock_row.iloc[0]['Portfolio Weight']
        
        # Calculate Active Share contribution for each stock
        original_active_share_by_stock[ticker] = abs(port_weight - bench_weight) / 2
        new_active_share_by_stock[ticker] = abs(new_weight - bench_weight) / 2
    
    # Calculate total Active Share for verification
    total_original_active_share = sum(original_active_share_by_stock.values())
    total_new_active_share = sum(new_active_share_by_stock.values())
    
    # Print verification of active share calculations
    print(f"\nVerification - Original Active Share: {total_original_active_share:.2f}%")
    print(f"Verification - New Active Share: {total_new_active_share:.2f}%")
    if optimized_active_share is None:
        print("Optimizer - New Active Share: None (no optimal solution found)")
    else:
        print(f"Optimizer - New Active Share: {optimized_active_share:.2f}%")
    
    # Print an explanation if there's a significant difference
    if optimized_active_share is not None and abs(total_new_active_share - optimized_active_share) > 1.0:
        print("\nNote: There is a difference between the optimizer's active share calculation")
        print("and our verification calculation. This may be due to differences in how")
        print("stocks without benchmark weights are handled or rounding differences.")

    # Use the optimizer's active share value for consistency
    results = [{
        'Ticker': 'TOTAL',
        'Sector': '',
        'Sector-and-Subsector': '',
        'Core Rank': 'N/A',
        'Current Weight': sum(stock['Portfolio Weight'] for _, stock in stocks_data.iterrows() if stock['Portfolio Weight'] > 0),
        'New Weight': 100,
        'Benchmark Weight': sum(stock['Bench Weight'] for _, stock in stocks_data.iterrows() if stock['Bench Weight'] > 0),
        'Original Active Share': total_active_share,  # Use the original active share from the data
        'New Active Share': optimized_active_share,   # Use the optimizer's calculated active share
        'Active Share Improvement': total_active_share - optimized_active_share if optimized_active_share is not None else None,
        'Improvement %': (total_active_share - optimized_active_share) / total_active_share * 100 if optimized_active_share is not None and total_active_share > 0 else None,
        'Is New': 'No'
    }]
    
    # Include ALL stocks from the input dataset in the output
    unique_tickers = set()
    
    # First pass: add all portfolio stocks
    for _, row in stocks_data.iterrows():
        ticker = row['Ticker']
        original_as = original_active_share_by_stock.get(ticker, 0)
        new_as = new_active_share_by_stock.get(ticker, 0)
        as_improvement = original_as - new_as
        
        # Check if this stock is a new position added by the optimizer
        is_new_position = ticker in [stock['ticker'] for stock in added_stocks]
        
        # Flag if the stock is in either the original or new portfolio
        in_original = row['Portfolio Weight'] > 0
        in_new = ticker in new_portfolio if new_portfolio else False
        
        # Skip any records without a ticker (like sector headers)
        if pd.isna(ticker):
            continue
            
        # Add to unique tickers set
        unique_tickers.add(ticker)
        
        # For locked tickers, always use the original weight
        is_locked = ticker in locked_tickers
        if is_locked:
            new_weight = row['Portfolio Weight']  # Use original weight for locked tickers
            weight_change = 0  # No change for locked tickers
        else:
            new_weight = new_portfolio.get(ticker, 0)
            weight_change = new_weight - row['Portfolio Weight']
        
        results.append({
            'Ticker': ticker,
            'Sector': row['Sector'],
            'Sector-and-Subsector': row['Sector-and-Subsector'],
            'Core Rank': row['Core Model'],
            'Current Weight': row['Portfolio Weight'],
            'New Weight': new_weight,
            'Benchmark Weight': row['Bench Weight'],
            'Weight Change': weight_change,
            'Original Active Share': original_as,
            'New Active Share': new_as,
            'Active Share Improvement': as_improvement if optimized_active_share is not None else None,
            'Improvement %': (as_improvement / original_as * 100) if optimized_active_share is not None and original_as > 0 else None,
            'In Original Portfolio': 'Yes' if in_original else 'No',
            'In New Portfolio': 'Yes' if in_new else 'No',
            'Is New': 'Yes' if is_new_position else 'No',
            'Is Locked': 'Yes' if is_locked else 'No'
        })
    
    # Second pass: add any stocks in added_stocks that weren't in the original dataset
    # (should be rare but including for completeness)
    for stock in added_stocks:
        ticker = stock['ticker']
        if ticker not in unique_tickers:
            original_as = original_active_share_by_stock.get(ticker, 0)
            new_as = new_active_share_by_stock.get(ticker, 0)
            as_improvement = original_as - new_as
            
            results.append({
                'Ticker': ticker,
                'Sector': stock['sector'],
                'Sector-and-Subsector': stock['subsector'],
                'Core Rank': stock['core_rank'],
                'Current Weight': 0,
                'New Weight': stock['new_weight'],
                'Benchmark Weight': stock['bench_weight'],
                'Weight Change': stock['new_weight'],
                'Original Active Share': original_as,
                'New Active Share': new_as,
                'Active Share Improvement': as_improvement,
                'Improvement %': (as_improvement / original_as * 100) if original_as > 0 else 'N/A',
                'In Original Portfolio': 'No',
                'In New Portfolio': 'Yes',
                'Is New': 'Yes'
            })
    
    # Create results spreadsheet even if no solution was found
    # Sort by sector and weight
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:  # Only sort if we have results
        results_df = results_df.sort_values(by=['Sector', 'New Weight'], ascending=[True, False])
    
    # Create sector analysis data
    sector_analysis = []
    
    # Get unique sectors
    sectors = sorted(list(set([s for s in stocks_data['Sector'].unique() if isinstance(s, str) and s != ''])))
    
    # Get unique subsectors for more detailed analysis
    subsectors = sorted(list(set([s for s in stocks_data['Sector-and-Subsector'].unique() if isinstance(s, str) and s != ''])))
    
    # Calculate sector weights for original portfolio, new portfolio, and benchmark
    for sector in sectors:
        # Original portfolio weight
        original_weight = sum(row['Portfolio Weight'] for _, row in stocks_data.iterrows() 
                            if row['Sector'] == sector and row['Portfolio Weight'] > 0)
        
        # New portfolio weight - use original weight for locked tickers
        new_weight = 0
        for _, row in stocks_data.iterrows():
            if row['Sector'] == sector:
                ticker = row['Ticker']
                if ticker in locked_tickers:
                    new_weight += row['Portfolio Weight']  # Use original weight for locked tickers
                else:
                    new_weight += new_portfolio.get(ticker, 0)  # Use optimized weight for other tickers
        
        # Benchmark weight
        bench_weight = sum(row['Bench Weight'] for _, row in stocks_data.iterrows() 
                          if row['Sector'] == sector and row['Bench Weight'] > 0)
        
        # Calculate differences
        original_diff = original_weight - bench_weight
        new_diff = new_weight - bench_weight
        change = new_weight - original_weight
        
        sector_analysis.append({
            'Sector': sector,
            'Original Weight': original_weight,
            'New Weight': new_weight,
            'Benchmark Weight': bench_weight,
            'Original vs Benchmark': original_diff,
            'New vs Benchmark': new_diff,
            'Weight Change': change,
            'Improvement': abs(original_diff) - abs(new_diff),
        })
    
    # Now do the same for subsector analysis
    subsector_analysis = []
    for subsector in subsectors:
        # Original portfolio weight
        original_weight = sum(row['Portfolio Weight'] for _, row in stocks_data.iterrows() 
                            if row['Sector-and-Subsector'] == subsector and row['Portfolio Weight'] > 0)
        
        # New portfolio weight - use original weight for locked tickers
        new_weight = 0
        for _, row in stocks_data.iterrows():
            if row['Sector-and-Subsector'] == subsector:
                ticker = row['Ticker']
                if ticker in locked_tickers:
                    new_weight += row['Portfolio Weight']  # Use original weight for locked tickers
                else:
                    new_weight += new_portfolio.get(ticker, 0)  # Use optimized weight for other tickers
        
        # Benchmark weight
        bench_weight = sum(row['Bench Weight'] for _, row in stocks_data.iterrows() 
                          if row['Sector-and-Subsector'] == subsector and row['Bench Weight'] > 0)
        
        # Target weight (if specified in constraints)
        target_weight = sector_constraints.get(subsector, bench_weight)
        
        # Calculate differences
        original_diff = original_weight - target_weight
        new_diff = new_weight - target_weight
        change = new_weight - original_weight
        
        subsector_analysis.append({
            'Subsector': subsector,
            'Original Weight': original_weight,
            'New Weight': new_weight,
            'Target Weight': target_weight,
            'Benchmark Weight': bench_weight,
            'Original vs Target': original_diff,
            'New vs Target': new_diff,
            'Weight Change': change,
            'Improvement': abs(original_diff) - abs(new_diff),
        })
    
    # Add a total row
    sector_analysis.append({
        'Sector': 'TOTAL',
        'Original Weight': sum(item['Original Weight'] for item in sector_analysis),
        'New Weight': sum(item['New Weight'] for item in sector_analysis),
        'Benchmark Weight': sum(item['Benchmark Weight'] for item in sector_analysis),
        'Original vs Benchmark': sum(item['Original vs Benchmark'] for item in sector_analysis),
        'New vs Benchmark': sum(item['New vs Benchmark'] for item in sector_analysis),
        'Weight Change': sum(item['Weight Change'] for item in sector_analysis),
        'Improvement': sum(item['Improvement'] for item in sector_analysis),
    })
    # Create dataframes for sector and subsector analysis
    sector_df = pd.DataFrame(sector_analysis)
    subsector_df = pd.DataFrame(subsector_analysis)
    
    # Get current timestamp for the output file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # Save results to Excel with locked tickers information
    output_file = save_optimizer_results_to_excel(
        results_df, 
        sector_df, 
        subsector_df, 
        locked_tickers=locked_tickers,
        output_timestamp=timestamp
    )

    return new_portfolio, added_stocks, optimized_active_share, output_file

if __name__ == "__main__":
    main() 