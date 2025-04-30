#!/usr/bin/env python3
"""
Active Share Optimizer

This script optimizes a portfolio by reducing Active Share while:
- Maintaining similar sector exposures
- Limiting total positions to a specified maximum
- Potentially prioritizing stocks based on qualitative rankings (future enhancement)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import math

def load_portfolio_data(file_path):
    """Load and preprocess the portfolio data from Excel."""
    # Read the Excel file
    # Skip the metadata rows at the beginning
    data = pd.read_excel(file_path, skiprows=6)
    
    # Find the first row that says 'Total' in the first column
    total_row = None
    for i, value in enumerate(data.iloc[:, 0]):
        if value == 'Total':
            total_row = i
            break
    
    if total_row is None:
        raise ValueError("Could not find the 'Total' row in the data")
    
    # Get the overall active share value from the Total row
    total_active_share = float(data.iloc[total_row, data.columns.get_loc('Active Share')])
    
    # Skip the rows with 'Total' and NaN to get the sector and stock data
    data = data.iloc[total_row+2:].reset_index(drop=True)
    
    # Initialize variables to track current sector
    current_sector = None
    sectors = []
    stock_rows = []
    
    # Process each row to identify sectors and stocks
    for i, row in data.iterrows():
        # Check if this is a sector row (no ticker symbol)
        if pd.isna(row['V1Symbol']):
            current_sector = row.iloc[0]  # First column contains sector name
        else:
            # This is a stock row
            stock_row = row.copy()
            stock_row['Sector'] = current_sector
            stock_rows.append(stock_row)
    
    # Create a DataFrame with only the stock rows
    stocks_df = pd.DataFrame(stock_rows)
    
    # Convert weight columns to numeric values
    stocks_df['Bench. Ending Weight'] = pd.to_numeric(stocks_df['Bench. Ending Weight'], errors='coerce').fillna(0)
    stocks_df['Port. Ending Weight'] = pd.to_numeric(stocks_df['Port. Ending Weight'], errors='coerce').fillna(0)
    stocks_df['Active Share'] = pd.to_numeric(stocks_df['Active Share'], errors='coerce').fillna(0)
    
    return stocks_df, total_active_share

def calculate_active_share(portfolio, benchmark):
    """
    Calculate the Active Share:
    ActiveShare = 0.5 * sum(abs(portfolio_weight - benchmark_weight))
    
    The weights in both portfolio and benchmark should sum to 100%.
    """
    diff_sum = 0
    for ticker in set(portfolio.keys()).union(benchmark.keys()):
        port_weight = portfolio.get(ticker, 0)
        bench_weight = benchmark.get(ticker, 0)
        diff_sum += abs(port_weight - bench_weight)
    
    return diff_sum / 2

def get_sector_weights(stocks_data):
    """Calculate sector weights from stock-level data."""
    sector_weights = {}
    for sector in stocks_data['Sector'].unique():
        sector_stocks = stocks_data[stocks_data['Sector'] == sector]
        sector_weights[sector] = sector_stocks['Port. Ending Weight'].sum()
    
    return sector_weights

def optimize_portfolio(stocks_data, original_active_share, max_positions=65, target_active_share=0.55):
    """
    Optimize the portfolio to reduce Active Share while maintaining sector exposure.
    
    The strategy:
    1. Start with current portfolio
    2. Identify underweight positions in the benchmark
    3. Add stocks strategically to reduce Active Share
    4. Maintain sector balance as much as possible
    """
    # Find current portfolio positions (stocks with positive weights)
    portfolio_stocks = stocks_data[stocks_data['Port. Ending Weight'] > 0]
    current_positions = portfolio_stocks['V1Symbol'].nunique()
    print(f"Current unique positions: {current_positions}")
    
    # Calculate the sum of absolute differences for current Active Share
    absolute_diffs = abs(stocks_data['Port. Ending Weight'] - stocks_data['Bench. Ending Weight'])
    calculated_active_share = (absolute_diffs.sum() / 2) 
    print(f"Calculated Active Share: {calculated_active_share:.2f}%")
    print(f"Active Share from data: {original_active_share:.2f}%")
    
    # Use the provided Active Share if calculation differs
    current_active_share = original_active_share / 100  # Convert from percentage
    
    # Get current sector weights
    current_sector_weights = {}
    for sector in stocks_data['Sector'].unique():
        if pd.isna(sector):
            continue
        sector_stocks = stocks_data[stocks_data['Sector'] == sector]
        sector_weight = sector_stocks['Port. Ending Weight'].sum()
        current_sector_weights[sector] = sector_weight
    
    benchmark_sector_weights = {}
    for sector in stocks_data['Sector'].unique():
        if pd.isna(sector):
            continue
        sector_stocks = stocks_data[stocks_data['Sector'] == sector]
        sector_weight = sector_stocks['Bench. Ending Weight'].sum()
        benchmark_sector_weights[sector] = sector_weight
    
    # Print sector differences
    print("\nCurrent sector differences (Portfolio - Benchmark):")
    sector_diffs = {}
    for sector in current_sector_weights.keys():
        bench_weight = benchmark_sector_weights.get(sector, 0)
        diff = current_sector_weights[sector] - bench_weight
        sector_diffs[sector] = diff
        print(f"{sector}: {diff:.2f}%")
    
    # Dictionary to track the current portfolio
    current_portfolio = {}
    for _, row in portfolio_stocks.iterrows():
        current_portfolio[row['V1Symbol']] = row['Port. Ending Weight']
    
    # Find potential stocks to add (stocks in benchmark but not in portfolio)
    benchmark_stocks = stocks_data[stocks_data['Bench. Ending Weight'] > 0]
    potential_additions = []
    
    for _, row in benchmark_stocks.iterrows():
        ticker = row['V1Symbol']
        if ticker not in current_portfolio:
            sector = row['Sector']
            bench_weight = row['Bench. Ending Weight']
            # Calculate how much this would reduce the Active Share
            # The reduction is approximately the benchmark weight value
            # (since we're changing from 0% to the benchmark weight)
            active_share_impact = bench_weight / 2  # Divide by 2 as per Active Share formula
            
            potential_additions.append({
                'ticker': ticker,
                'sector': sector,
                'bench_weight': bench_weight,
                'active_share_impact': active_share_impact
            })
    
    # Sort potential additions by their Active Share impact (largest first)
    potential_additions.sort(key=lambda x: x['active_share_impact'], reverse=True)
    
    # Group by sector
    sector_candidates = defaultdict(list)
    for stock in potential_additions:
        if not pd.isna(stock['sector']):
            sector_candidates[stock['sector']].append(stock)
    
    # Determine the maximum additional positions we can add
    available_positions = max_positions - current_positions
    print(f"\nCurrent positions: {current_positions}")
    print(f"Maximum positions: {max_positions}")
    print(f"Available positions to add: {available_positions}")
    print(f"Found {len(potential_additions)} potential stocks to add from the benchmark")
    
    # Output top 5 candidates with highest Active Share impact
    print("\nTop 5 candidates with highest Active Share impact:")
    for i, stock in enumerate(potential_additions[:5]):
        print(f"{i+1}. {stock['ticker']} ({stock['sector']}): Benchmark weight {stock['bench_weight']:.2f}%, Est. impact {stock['active_share_impact']:.2f}%")
    
    # Start optimization
    new_portfolio = current_portfolio.copy()
    added_stocks = []
    
    # Strategy: Add stocks from sectors where we're most underweight compared to benchmark
    # Sort sectors by how underweight they are (most negative diff first)
    underweight_sectors = [(sector, diff) for sector, diff in sector_diffs.items() if diff < -0.5]
    underweight_sectors.sort(key=lambda x: x[1])
    
    print("\nAdding stocks from underweight sectors first:")
    for sector, diff in underweight_sectors[:3]:  # Focus on the most underweight sectors
        print(f"\nProcessing underweight sector: {sector} (diff: {diff:.2f}%)")
        
        # How many stocks to add from this sector?
        # Allocate based on how underweight the sector is
        allocation = min(math.ceil(abs(diff)), math.ceil(available_positions/3))
        allocation = min(allocation, len(sector_candidates[sector]), available_positions - len(added_stocks))
        
        print(f"Allocating {allocation} positions to {sector}")
        
        # Add the top stocks from this sector
        for i in range(min(allocation, len(sector_candidates[sector]))):
            if len(added_stocks) >= available_positions:
                break
                
            stock = sector_candidates[sector][i]
            new_portfolio[stock['ticker']] = stock['bench_weight']
            added_stocks.append(stock)
            print(f"Added {stock['ticker']} with benchmark weight {stock['bench_weight']:.2f}%")
    
    # If we still have positions available, add the highest impact stocks regardless of sector
    remaining_positions = available_positions - len(added_stocks)
    if remaining_positions > 0:
        print(f"\nAdding {remaining_positions} more high-impact stocks:")
        
        # Filter out stocks we've already added
        added_tickers = [stock['ticker'] for stock in added_stocks]
        remaining_candidates = [stock for stock in potential_additions if stock['ticker'] not in added_tickers]
        
        for i in range(min(remaining_positions, len(remaining_candidates))):
            stock = remaining_candidates[i]
            new_portfolio[stock['ticker']] = stock['bench_weight']
            added_stocks.append(stock)
            print(f"Added {stock['ticker']} ({stock['sector']}) with benchmark weight {stock['bench_weight']:.2f}%")
    
    # Calculate the new weights (normalize to 100%)
    total_weight = sum(new_portfolio.values())
    normalized_portfolio = {ticker: weight/total_weight*100 for ticker, weight in new_portfolio.items()}
    
    # Calculate the new Active Share (estimation)
    active_share_reduction = sum(stock['active_share_impact'] for stock in added_stocks)
    estimated_new_active_share = current_active_share - (active_share_reduction/100)  # Convert from percentage
    
    # Calculate actual new Active Share
    new_active_share_calc = 0
    for ticker in set(normalized_portfolio.keys()).union(benchmark_stocks['V1Symbol']):
        port_weight = normalized_portfolio.get(ticker, 0)
        
        # Find benchmark weight for this ticker
        bench_weight = 0
        ticker_data = benchmark_stocks[benchmark_stocks['V1Symbol'] == ticker]
        if not ticker_data.empty:
            bench_weight = ticker_data.iloc[0]['Bench. Ending Weight']
        
        new_active_share_calc += abs(port_weight - bench_weight)
    
    new_active_share_calc = new_active_share_calc / 2
    
    # Calculate new sector weights
    new_sector_weights = defaultdict(float)
    for ticker, weight in normalized_portfolio.items():
        # Find sector for this ticker
        sector = None
        ticker_data = stocks_data[stocks_data['V1Symbol'] == ticker]
        if not ticker_data.empty:
            sector = ticker_data.iloc[0]['Sector']
        
        # For stocks we just added
        if sector is None:
            for stock in added_stocks:
                if stock['ticker'] == ticker:
                    sector = stock['sector']
                    break
        
        if sector and not pd.isna(sector):
            new_sector_weights[sector] += weight
    
    # Final portfolio stats
    print("\nOptimized Portfolio Summary:")
    print(f"Estimated Final Active Share: {estimated_new_active_share*100:.2f}%")
    print(f"Calculated Final Active Share: {new_active_share_calc:.2f}%")
    print(f"Final number of positions: {len(normalized_portfolio)}")
    print(f"New positions added: {len(added_stocks)}")
    
    # List new positions by sector
    if added_stocks:
        print("\nAdded positions by sector:")
        by_sector = defaultdict(list)
        for stock in added_stocks:
            by_sector[stock['sector']].append(stock)
            
        for sector, stocks in by_sector.items():
            print(f"\n{sector}:")
            for stock in stocks:
                norm_weight = normalized_portfolio[stock['ticker']]
                print(f"  {stock['ticker']}: {norm_weight:.2f}% (Benchmark: {stock['bench_weight']:.2f}%, Impact: {stock['active_share_impact']:.2f}%)")
    
    # Print sector changes
    print("\nSector Analysis:")
    for sector, old_weight in current_sector_weights.items():
        new_weight = new_sector_weights.get(sector, 0)
        bench_weight = benchmark_sector_weights.get(sector, 0)
        print(f"{sector}: {old_weight:.2f}% → {new_weight:.2f}% (Benchmark: {bench_weight:.2f}%, Δ: {new_weight - old_weight:.2f}%)")
    
    return normalized_portfolio, added_stocks

def main():
    """Main function to run the optimizer."""
    file_path = 'Active Share.xlsx'
    stocks_data, total_active_share = load_portfolio_data(file_path)
    
    print(f"Active Share from data: {total_active_share:.2f}%")
    print(f"Loaded {len(stocks_data)} stock entries from {file_path}")
    
    # Set optimization parameters
    max_positions = 65  # Maximum total positions
    target_active_share = 0.55  # Target Active Share (55%)
    
    # Run the optimizer
    new_portfolio, added_stocks = optimize_portfolio(
        stocks_data, 
        original_active_share=total_active_share,
        max_positions=max_positions,
        target_active_share=target_active_share
    )
    
    # You could save the results to a new Excel file
    # Create a results dataframe
    print("\nCreating results spreadsheet...")
    
    # Start with current portfolio data
    results = []
    
    # Calculate Active Share for each stock in the original and new portfolio
    original_active_share_by_stock = {}
    new_active_share_by_stock = {}
    
    for _, row in stocks_data.iterrows():
        ticker = row['V1Symbol']
        port_weight = row['Port. Ending Weight']
        bench_weight = row['Bench. Ending Weight']
        new_weight = new_portfolio.get(ticker, 0)
        
        # Calculate Active Share contribution for each stock
        original_active_share_by_stock[ticker] = abs(port_weight - bench_weight) / 2
        new_active_share_by_stock[ticker] = abs(new_weight - bench_weight) / 2
    
    # Calculate total Active Share for verification
    total_original_active_share = sum(original_active_share_by_stock.values())
    total_new_active_share = sum(new_active_share_by_stock.values())
    
    # Add header row with overall statistics
    results.append({
        'Ticker': 'TOTAL',
        'Sector': '',
        'Current Weight': sum(stock['Port. Ending Weight'] for _, stock in stocks_data.iterrows() if stock['Port. Ending Weight'] > 0),
        'New Weight': 100,
        'Benchmark Weight': sum(stock['Bench. Ending Weight'] for _, stock in stocks_data.iterrows() if stock['Bench. Ending Weight'] > 0),
        'Original Active Share': total_original_active_share,
        'New Active Share': total_new_active_share,
        'Active Share Improvement': total_original_active_share - total_new_active_share,
        'Improvement %': (total_original_active_share - total_new_active_share) / total_original_active_share * 100 if total_original_active_share > 0 else 0,
        'Is New': 'No'
    })
    
    # Include ALL stocks from the input dataset in the output
    unique_tickers = set()
    
    # First pass: add all portfolio stocks
    for _, row in stocks_data.iterrows():
        ticker = row['V1Symbol']
        original_as = original_active_share_by_stock.get(ticker, 0)
        new_as = new_active_share_by_stock.get(ticker, 0)
        as_improvement = original_as - new_as
        
        # Check if this stock is a new position added by the optimizer
        is_new_position = ticker in [stock['ticker'] for stock in added_stocks]
        
        # Flag if the stock is in either the original or new portfolio
        in_original = row['Port. Ending Weight'] > 0
        in_new = ticker in new_portfolio
        
        # Skip any records without a ticker (like sector headers)
        if pd.isna(ticker):
            continue
            
        # Add to unique tickers set
        unique_tickers.add(ticker)
        
        results.append({
            'Ticker': ticker,
            'Sector': row['Sector'],
            'Current Weight': row['Port. Ending Weight'],
            'New Weight': new_portfolio.get(ticker, 0),
            'Benchmark Weight': row['Bench. Ending Weight'],
            'Weight Change': new_portfolio.get(ticker, 0) - row['Port. Ending Weight'],
            'Original Active Share': original_as,
            'New Active Share': new_as,
            'Active Share Improvement': as_improvement,
            'Improvement %': (as_improvement / original_as * 100) if original_as > 0 else 0 if original_as == 0 else 'N/A',
            'In Original Portfolio': 'Yes' if in_original else 'No',
            'In New Portfolio': 'Yes' if in_new else 'No',
            'Is New': 'Yes' if is_new_position else 'No'
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
                'Current Weight': 0,
                'New Weight': new_portfolio.get(ticker, 0),
                'Benchmark Weight': stock['bench_weight'],
                'Weight Change': new_portfolio.get(ticker, 0),
                'Original Active Share': original_as,
                'New Active Share': new_as,
                'Active Share Improvement': as_improvement,
                'Improvement %': (as_improvement / original_as * 100) if original_as > 0 else 'N/A',
                'In Original Portfolio': 'No',
                'In New Portfolio': 'Yes',
                'Is New': 'Yes'
            })
    
    # Sort by sector and weight
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Sector', 'New Weight'], ascending=[True, False])
    
    # Create sector analysis data
    sector_analysis = []
    
    # Get unique sectors
    sectors = sorted(list(set([s for s in stocks_data['Sector'].unique() if isinstance(s, str) and s != ''])))
    sectors.extend(['[Unassigned]', '[Cash]'])  # Add special categories
    
    # Calculate sector weights for original portfolio, new portfolio, and benchmark
    for sector in sectors:
        # Original portfolio weight
        original_weight = sum(row['Port. Ending Weight'] for _, row in stocks_data.iterrows() 
                            if row['Sector'] == sector and row['Port. Ending Weight'] > 0)
        
        # New portfolio weight
        new_weight = sum(new_portfolio.get(row['V1Symbol'], 0) for _, row in stocks_data.iterrows() 
                       if row['Sector'] == sector)
        
        # Benchmark weight
        bench_weight = sum(row['Bench. Ending Weight'] for _, row in stocks_data.iterrows() 
                         if row['Sector'] == sector and row['Bench. Ending Weight'] > 0)
        
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
            'Improvement': abs(original_diff) - abs(new_diff)
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
        'Improvement': sum(item['Improvement'] for item in sector_analysis)
    })
    
    # Create a dataframe for sector analysis
    sector_df = pd.DataFrame(sector_analysis)
    
    # Save both dataframes to Excel (different sheets)
    with pd.ExcelWriter('Optimized_Portfolio.xlsx') as writer:
        results_df.to_excel(writer, sheet_name='Portfolio', index=False)
        sector_df.to_excel(writer, sheet_name='Sector Analysis', index=False)
    
    print("Results saved to 'Optimized_Portfolio.xlsx'")
    
    # You could also save the results to a new Excel file
    # (Implement if needed)

if __name__ == "__main__":
    main()
