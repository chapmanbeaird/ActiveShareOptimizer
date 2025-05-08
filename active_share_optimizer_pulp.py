#!/usr/bin/env python3
"""
Active Share Optimizer using PuLP Linear Programming

This script optimizes a portfolio by reducing Active Share while:
- Maintaining similar sector exposures
- Limiting total positions to a specified maximum
- Applying core model constraints (ranks 1-3)
- Setting position sizes in 0.5% increments between 1% and 5%
- Avoiding specific stocks in the blacklist
- Matching sector-and-subsector weights with target constraints

This version uses PuLP for more systematic optimization compared to the heuristic approach.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pulp
from pulp import COIN_CMD
import os

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

def load_portfolio_data_csv(file_path):
    """Load and preprocess the portfolio data from CSV."""
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Convert weight columns to numeric values
    data['Bench Weight'] = pd.to_numeric(data['Bench Weight'], errors='coerce').fillna(0)
    data['Portfolio Weight'] = pd.to_numeric(data['Portfolio Weight'], errors='coerce').fillna(0)
    data['Active Share'] = pd.to_numeric(data['Active Share'], errors='coerce').fillna(0)
    data['Core Model'] = pd.to_numeric(data['Core Model'], errors='coerce')
    
    # Calculate the total active share directly from the portfolio and benchmark weights
    # This is the correct formula for active share: sum(|portfolio_weight - benchmark_weight|) / 2
    # Since weights are already in percentages (e.g., 1.5 means 1.5%), we don't need to multiply by 100
    absolute_diffs = abs(data['Portfolio Weight'] - data['Bench Weight'])
    total_active_share = (absolute_diffs.sum() / 2)
    
    return data, total_active_share

def load_constraints(file_path):
    """Load the stocks to avoid and sector constraints."""
    # Read the Excel file
    data = pd.read_excel(file_path)
    
    # Get the list of stocks to avoid
    stocks_to_avoid = data['Stocks to Avoid'].dropna().tolist()
    
    # Get the sector constraints
    sector_constraints = {}
    for _, row in data.iterrows():
        if pd.notna(row['Emp Sector & Industry']) and pd.notna(row['Weight']):
            sector_constraints[row['Emp Sector & Industry']] = row['Weight']
    
    return stocks_to_avoid, sector_constraints

def optimize_portfolio_pulp(stocks_data, original_active_share, max_positions=60, target_active_share=0.55, 
                            sector_tolerance=0.03, stocks_to_avoid=None, sector_constraints=None, 
                            min_position=1.0, max_position=5.0, core_rank_limit=3, increment=0.5, forced_positions=None, time_limit=120):
    """
    Optimize the portfolio to reduce Active Share while applying constraints.
    
    Parameters:
    - stocks_data: DataFrame containing stock data
    - original_active_share: Original Active Share percentage
    - max_positions: Maximum number of positions allowed in the portfolio
    - target_active_share: Target Active Share percentage
    - sector_tolerance: Maximum allowed deviation from benchmark sector weights (in decimal)
    - stocks_to_avoid: List of ticker symbols to exclude from the portfolio
    - sector_constraints: Dictionary mapping sector-and-subsector to target weights
    - min_position: Minimum position size as a percentage (e.g., 1.0)
    - max_position: Maximum position size as a percentage (e.g., 5.0)
    - core_rank_limit: Only consider stocks with Core Model rank <= this value (e.g., 1, 2, 3, 4, 5)
    - increment: Allowed increment for position sizes (e.g., 0.5 for 0.5% increments)
    - forced_positions: Dictionary {ticker: (min, max)}. For each ticker, require it to be in the portfolio with weight between min and max.
    
    Returns:
    - optimized_portfolio: Dictionary of ticker to weight for the optimized portfolio
    - added_stocks: List of stocks added to the portfolio
    - new_active_share: The new Active Share after optimization
    """
    # Initialize variables
    if stocks_to_avoid is None:
        stocks_to_avoid = []
    if sector_constraints is None:
        sector_constraints = {}
    if forced_positions is None:
        forced_positions = {}
    
    # Find current portfolio positions (stocks with positive weights)
    portfolio_stocks = stocks_data[stocks_data['Portfolio Weight'] > 0]
    current_positions = portfolio_stocks['Ticker'].nunique()
    print(f"Current unique positions: {current_positions}")
    
    # Calculate the active share from the data
    calculated_active_share = original_active_share 
    print(f"Active Share: {calculated_active_share:.2f}%")
    
    # Get current sector-and-subsector weights
    current_sector_weights = {}
    for sector in stocks_data['Sector-and-Subsector'].unique():
        if pd.isna(sector):
            continue
        sector_stocks = stocks_data[stocks_data['Sector-and-Subsector'] == sector]
        sector_weight = sector_stocks['Portfolio Weight'].sum()
        current_sector_weights[sector] = sector_weight
    
    benchmark_sector_weights = {}
    for sector in stocks_data['Sector-and-Subsector'].unique():
        if pd.isna(sector):
            continue
        sector_stocks = stocks_data[stocks_data['Sector-and-Subsector'] == sector]
        sector_weight = sector_stocks['Bench Weight'].sum()
        benchmark_sector_weights[sector] = sector_weight
    
    # Print sector differences
    print("\nCurrent sector-and-subsector differences (Portfolio - Benchmark):")
    sector_diffs = {}
    for sector in current_sector_weights.keys():
        bench_weight = benchmark_sector_weights.get(sector, 0)
        diff = current_sector_weights[sector] - bench_weight
        sector_diffs[sector] = diff
        print(f"{sector}: {diff:.2f}%")
    
    # Dictionary to track the current portfolio
    current_portfolio = {}
    for _, row in portfolio_stocks.iterrows():
        current_portfolio[row['Ticker']] = row['Portfolio Weight']
    
    # Create a dictionary of all benchmark stocks
    benchmark_stocks_dict = {}
    for _, row in stocks_data.iterrows():
        if row['Bench Weight'] > 0:
            benchmark_stocks_dict[row['Ticker']] = {
                'ticker': row['Ticker'],
                'sector': row['Sector'],
                'subsector': row['Sector-and-Subsector'],
                'bench_weight': row['Bench Weight'],
                'core_rank': row['Core Model']
            }
    
    # Create two lists of stocks:
    # 1. all_stocks - All stocks to consider for active share calculation
    # 2. purchase_eligible - Tracks which stocks can be purchased (core rank <= 3, not in avoid list)
    all_stocks = []
    purchase_eligible = {}  # Dictionary to track which stocks are eligible for purchase
    
    # Add current portfolio stocks
    for ticker, weight in current_portfolio.items():
        stock_data = stocks_data[stocks_data['Ticker'] == ticker]
        if not stock_data.empty:
            sector = stock_data.iloc[0]['Sector']
            subsector = stock_data.iloc[0]['Sector-and-Subsector']
            bench_weight = stock_data.iloc[0]['Bench Weight']
            core_rank = stock_data.iloc[0]['Core Model']
            all_stocks.append({
                'ticker': ticker,
                'sector': sector,
                'subsector': subsector,
                'bench_weight': bench_weight,
                'core_rank': core_rank,
                'current_weight': weight,
                'in_portfolio': 1  # Flag to indicate this stock is already in the portfolio
            })
            
            # Track if this stock is eligible for purchase in the optimized portfolio
            purchase_eligible[ticker] = True
    
    # Add ALL benchmark stocks not already in portfolio to consider for active share calculation
    for ticker, stock_info in benchmark_stocks_dict.items():
        if ticker not in current_portfolio:
            all_stocks.append({
                'ticker': ticker,
                'sector': stock_info['sector'],
                'subsector': stock_info['subsector'],
                'bench_weight': stock_info['bench_weight'],
                'core_rank': stock_info['core_rank'],
                'current_weight': 0,
                'in_portfolio': 0  # Flag to indicate this stock is not in the portfolio
            })
            
            # Only mark as purchase eligible if it meets criteria
            if ticker not in stocks_to_avoid and stock_info['core_rank'] <= core_rank_limit:
                purchase_eligible[ticker] = True
                
    # Make sure any forced positions are purchase eligible regardless of other criteria
    for ticker in forced_positions.keys():
        purchase_eligible[ticker] = True
    
    # Create a dictionary to map tickers to indices in the all_stocks list
    ticker_to_index = {stock['ticker']: i for i, stock in enumerate(all_stocks)}
    
    # Create dictionaries to map sectors and subsectors to lists of stock indices
    sector_to_stocks = defaultdict(list)
    subsector_to_stocks = defaultdict(list)
    for i, stock in enumerate(all_stocks):
        if not pd.isna(stock['sector']):
            sector_to_stocks[stock['sector']].append(i)
        if not pd.isna(stock['subsector']):
            subsector_to_stocks[stock['subsector']].append(i)
    
    # Print information about available positions
    available_positions = max_positions - current_positions
    print(f"\nCurrent positions: {current_positions}")
    print(f"Maximum positions: {max_positions}")
    print(f"Available positions to add: {available_positions}")
    print(f"Found {len(all_stocks) - current_positions} potential stocks to add from the benchmark")
    
    # Create the PuLP model
    model = pulp.LpProblem("Portfolio_Optimization", pulp.LpMinimize)
    
    # Decision variables: 1 if stock is included, 0 otherwise
    include = pulp.LpVariable.dicts("Include", range(len(all_stocks)), cat=pulp.LpBinary)
    
    weight = {}
    if increment is None:
        # Continuous weights (no increment enforcement)
        for i in range(len(all_stocks)):
            weight[i] = pulp.LpVariable(f"Weight_{i}", lowBound=0, upBound=max_position, cat=pulp.LpContinuous)
            # If included, weight >= min_position, else weight == 0
            model += weight[i] >= min_position * include[i], f"min_weight_{i}"
            # If not included, weight == 0
            model += weight[i] <= max_position * include[i], f"max_weight_{i}"
        
        print("Using continuous weights (no increment restriction)")
        
        # Forced positions constraints (no increment)
        for ticker, (minp, maxp) in forced_positions.items():
            if ticker in ticker_to_index:
                idx = ticker_to_index[ticker]
                # Sanity checks
                if not (isinstance(minp, (int, float)) and isinstance(maxp, (int, float))):
                    raise ValueError(f"Invalid forced position bounds for {ticker}: {minp}, {maxp}")
                if minp > maxp:
                    raise ValueError(f"minp > maxp for {ticker}: {minp} > {maxp}")
                # Clip to global bounds
                minp = max(minp, min_position)
                maxp = min(maxp, max_position)
                model += include[idx] == 1, f"force_own_{ticker}"
                if minp > min_position:
                    model += weight[idx] >= minp, f"minpos_{ticker}"
                if maxp < max_position:
                    model += weight[idx] <= maxp, f"maxpos_{ticker}"
                
                print(f"Forcing {ticker} to be included with weight between {minp}% and {maxp}% (continuous)")
                bench_weight = all_stocks[idx]['bench_weight']
                print(f"Benchmark weight for {ticker}: {bench_weight}%")
    else:
        # Enforce increments using binary variables
        increments = [round(x, 4) for x in np.arange(min_position, max_position + increment/2, increment)]
        
        print(f"Using incremental weights with increment: {increment}%")
        print(f"Available increments: {min_position}% to {max_position}% in steps of {increment}%")
        
        increment_bin_vars = {}
        for i in range(len(all_stocks)):
            increment_bin_vars[i] = {}
            for inc in increments:
                increment_bin_vars[i][inc] = pulp.LpVariable(f"IncBin_{i}_{inc}", cat=pulp.LpBinary)
            # The weight is the sum over increments times their binary variable
            weight[i] = pulp.lpSum([inc * increment_bin_vars[i][inc] for inc in increments])
            # Only allow increments if stock is included
            model += pulp.lpSum([increment_bin_vars[i][inc] for inc in increments]) == include[i]
        # Forced positions constraints (increment case) - only add these ONCE per ticker
        for ticker, (minp, maxp) in forced_positions.items():
            if ticker in ticker_to_index:
                idx = ticker_to_index[ticker]
                # Sanity checks
                if not (isinstance(minp, (int, float)) and isinstance(maxp, (int, float))):
                    raise ValueError(f"Invalid forced position bounds for {ticker}: {minp}, {maxp}")
                if minp > maxp:
                    raise ValueError(f"minp > maxp for {ticker}: {minp} > {maxp}")
                # Clip to global bounds
                minp = max(minp, min_position)
                maxp = min(maxp, max_position)
                # Force the stock to be included
                model += include[idx] == 1, f"force_own_{ticker}"
                
                # Create a list of allowed increments for this ticker
                allowed_increments = []
                for inc in increments:
                    if minp <= inc <= maxp:
                        allowed_increments.append(inc)
                    else:
                        model += increment_bin_vars[idx][inc] == 0, f"force_inc_{ticker}_{inc}"
                
                # Ensure exactly one increment is selected for this forced position
                if allowed_increments:
                    model += pulp.lpSum([increment_bin_vars[idx][inc] for inc in allowed_increments]) == 1, f"force_sum_inc_{ticker}"
                    bench_weight = all_stocks[idx]['bench_weight']
                    print(f"Forcing {ticker} to be included with weight between {minp}% and {maxp}%, allowed increments: {allowed_increments}")
                    print(f"Benchmark weight for {ticker}: {bench_weight}%")
                else:
                    print(f"WARNING: No valid increments found for {ticker} between {minp}% and {maxp}%!")
    
    # Add constraints to ensure only purchase-eligible stocks can be selected
    for i, stock in enumerate(all_stocks):
        ticker = stock['ticker']
        if ticker not in purchase_eligible:
            # Force include[i] to be 0 if stock is not eligible for purchase
            model += include[i] == 0
    
    # Weight variables are now defined as a sum over increment binaries for each stock
    
    # Objective: Minimize Active Share
    # Active Share = 0.5 * sum(abs(portfolio_weight - benchmark_weight))
    # Since PuLP can't handle abs() directly, we'll use auxiliary variables
    abs_diff = pulp.LpVariable.dicts("AbsDiff", range(len(all_stocks)), lowBound=0)
    
    # For each stock, abs_diff[i] >= weight[i] - benchmark_weight[i]
    # and abs_diff[i] >= benchmark_weight[i] - weight[i]
    for i in range(len(all_stocks)):
        model += abs_diff[i] >= weight[i] - all_stocks[i]['bench_weight']
        model += abs_diff[i] >= all_stocks[i]['bench_weight'] - weight[i]
    # weight[i] is now a sum over increment binaries
    
    # Create a variable to represent total active share (multiplied by 2 for calculation purposes)
    total_abs_diff = pulp.LpVariable("TotalAbsDiff", lowBound=0)
    
    # Link total_abs_diff to the sum of individual absolute differences
    model += total_abs_diff == pulp.lpSum(abs_diff[i] for i in range(len(all_stocks)))
    
    # Constraint: Active Share should not be lower than target_active_share 
    # Active Share = 0.5 * total_abs_diff
    print(f"Setting minimum Active Share constraint to: {target_active_share * 100:.2f}%")
    # Since weights are already in percentages, target_active_share needs to be multiplied by 100
    # to be in the same scale as weights (e.g., 0.55 * 100 = 55%)
    model += 0.5 * total_abs_diff >= target_active_share * 100
    
    # Objective function: minimize active share while maintaining the constraint
    model += 0.5 * total_abs_diff
    
    # Constraint: Total weight must sum to 100%
    model += pulp.lpSum(weight[i] for i in range(len(all_stocks))) == 100
    
    # Constraint: Total number of positions must not exceed max_positions
    model += pulp.lpSum(include[i] for i in range(len(all_stocks))) <= max_positions

    # Constraint: Filter out stocks in the stocks_to_avoid list
    for ticker in stocks_to_avoid:
        if ticker in ticker_to_index and ticker not in forced_positions:
            i = ticker_to_index[ticker]
            model += include[i] == 0

    # Core rank constraint: Only select stocks with Core Rank <= core_rank_limit
    for i in range(len(all_stocks)):
        ticker = all_stocks[i]['ticker']
        if all_stocks[i]['core_rank'] > core_rank_limit and ticker not in forced_positions:
            model += include[i] == 0

    # abs_diff constraints for active share

    # Sector-and-subsector weights constraints
    for subsector, target_weight in sector_constraints.items():
        if subsector in subsector_to_stocks:
            subsector_weight = pulp.lpSum(weight[i] for i in subsector_to_stocks[subsector])
            model += subsector_weight >= target_weight - sector_tolerance * 100
            model += subsector_weight <= target_weight + sector_tolerance * 100
   
    # Check if the model was solved successfully
    
    # Solve the model
    print(f"\nSolving the optimization model (timeout: {time_limit} seconds)...")
    cbc_path = "/opt/homebrew/bin/cbc"
    if not os.path.exists(cbc_path):
        raise RuntimeError("CBC path not found.")

    # Set strict time limit parameters for the solver
    solver = COIN_CMD(path=cbc_path, msg=True, timeLimit=time_limit, 
                      options=['sec', str(time_limit), 
                              'timeMode', 'elapsed', 
                              'ratioGap', '0.0001',
                              'maxN', '100000000',
                              'maxSolutions', '1',
                              'cuts', 'off'])

    model.solve(solver)
    print("Solver status:", pulp.LpStatus[model.status])


    # Check if the model was solved successfully
    if pulp.LpStatus[model.status] != 'Optimal':
        print(f"Model status: {pulp.LpStatus[model.status]}")
        print("Could not find an optimal solution with the given constraints.")
        optimized_portfolio = {}
        added_stocks = []
        new_active_share = None
        return optimized_portfolio, added_stocks, new_active_share

    # Create the optimized portfolio
    optimized_portfolio = {}
    added_stocks = []
    
    for i in range(len(all_stocks)):
        if pulp.value(include[i]) > 0.5:  # If the stock is included (binary value close to 1)
            ticker = all_stocks[i]['ticker']
            new_weight = pulp.value(weight[i])
            
            # Round to 4 decimal places for display
            optimized_portfolio[ticker] = round(new_weight, 4)
            
            # If this is a new stock (not in the original portfolio)
            if all_stocks[i]['in_portfolio'] == 0:
                added_stocks.append({
                    'ticker': ticker,
                    'sector': all_stocks[i]['sector'],
                    'subsector': all_stocks[i]['subsector'],
                    'bench_weight': all_stocks[i]['bench_weight'],
                    'core_rank': all_stocks[i]['core_rank'],
                    'new_weight': new_weight,
                    'active_share_impact': abs(new_weight - all_stocks[i]['bench_weight']) / 2
                })
    
    # Calculate the new Active Share
    new_active_share = 0
    for i in range(len(all_stocks)):
        ticker = all_stocks[i]['ticker']
        port_weight = optimized_portfolio.get(ticker, 0)
        bench_weight = all_stocks[i]['bench_weight']
        new_active_share += abs(port_weight - bench_weight)
        
    new_active_share = new_active_share / 2
    
    print(f"\nCalculated Active Share in optimizer: {new_active_share:.2f}%")
    
    # Calculate new sector and subsector weights
    new_sector_weights = defaultdict(float)
    new_subsector_weights = defaultdict(float)
    for ticker, weight in optimized_portfolio.items():
        # Find the sector and subsector for this ticker
        for stock in all_stocks:
            if stock['ticker'] == ticker:
                sector = stock['sector']
                subsector = stock['subsector']
                if sector and not pd.isna(sector):
                    new_sector_weights[sector] += weight
                if subsector and not pd.isna(subsector):
                    new_subsector_weights[subsector] += weight
                break
    
    core_rank_counts = defaultdict(int)
    core_rank_weights = defaultdict(float)

    for ticker, weight in optimized_portfolio.items():
        # Find the core rank for this ticker
        for stock in all_stocks:
            if stock['ticker'] == ticker:
                rank = stock['core_rank']
                core_rank_counts[rank] += 1
                core_rank_weights[rank] += weight
                break
    
    # Final portfolio stats
    print("\nOptimized Portfolio Summary:")
    print(f"Original Active Share: {original_active_share:.2f}%")
    print(f"Optimized Active Share: {new_active_share:.2f}%")
    print(f"Improvement: {original_active_share - new_active_share:.2f}%")
    print(f"Final number of positions: {len(optimized_portfolio)}")
    print(f"New positions added: {len(added_stocks)}")
    
    # Print core rank distribution
    print("\nCore Rank Distribution:")
    for rank in sorted(core_rank_counts.keys()):
        print(f"Core Rank {rank}: {core_rank_counts[rank]} stocks ({core_rank_weights[rank]:.2f}%)")
    
    # List new positions by subsector
    if added_stocks:
        print("\nAdded positions by sector:")
        by_subsector = defaultdict(list)
        for stock in added_stocks:
            by_subsector[stock['subsector']].append(stock)
            
        for subsector, stocks in by_subsector.items():
            print(f"\n{subsector}:")
            for stock in stocks:
                print(f"  {stock['ticker']}: {stock['new_weight']:.2f}% (Benchmark: {stock['bench_weight']:.2f}%, Core Rank: {stock['core_rank']})")
    
    # Print sector-and-subsector changes
    print("\nSector-and-Subsector Analysis:")
    for subsector in sorted(set(current_sector_weights.keys()) | set(new_subsector_weights.keys())):
        old_weight = current_sector_weights.get(subsector, 0)
        new_weight = new_subsector_weights.get(subsector, 0)
        bench_weight = benchmark_sector_weights.get(subsector, 0)
        constraint_weight = sector_constraints.get(subsector, None)
        
        if constraint_weight is not None:
            print(f"{subsector}: {old_weight:.2f}% → {new_weight:.2f}% (Target: {constraint_weight:.2f}%, Δ from target: {new_weight - constraint_weight:.2f}%)")
        else:
            print(f"{subsector}: {old_weight:.2f}% → {new_weight:.2f}% (Benchmark: {bench_weight:.2f}%, Δ: {new_weight - old_weight:.2f}%)")

    return optimized_portfolio, added_stocks, new_active_share

def main(
    data_file_path='inputs/active_share_with_core_constraints.csv',
    constraints_file_path='inputs/stocks_to_avoid&sector_constraints.xlsm',
    max_positions=60,
    target_active_share=0.55,
    sector_tolerance=0.03,
    min_position=0.0,
    max_position=5.0,
    core_rank_limit=3,
    forced_positions=None,
    time_limit=120,
    increment=0.5
):
    """Main function to run the optimizer with adjustable parameters."""
    # Load stock data from CSV
    stocks_data, total_active_share = load_portfolio_data_csv(data_file_path)
    
    print(f"Active Share from data: {total_active_share:.2f}%")
    print(f"Loaded {len(stocks_data)} stock entries from {data_file_path}")
    
    # Load constraints from Excel
    stocks_to_avoid, sector_constraints = load_constraints(constraints_file_path)
    
    print(f"Loaded {len(stocks_to_avoid)} stocks to avoid")
    print(f"Loaded {len(sector_constraints)} sector constraints")
    
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
        max_positions=max_positions,
        target_active_share=target_active_share,
        sector_tolerance=sector_tolerance,
        stocks_to_avoid=stocks_to_avoid,
        sector_constraints=sector_constraints,
        max_position=max_position,
        core_rank_limit=core_rank_limit,
        forced_positions=forced_positions,
        time_limit=time_limit,
        increment=increment
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
        
        results.append({
            'Ticker': ticker,
            'Sector': row['Sector'],
            'Sector-and-Subsector': row['Sector-and-Subsector'],
            'Core Rank': row['Core Model'],
            'Current Weight': row['Portfolio Weight'],
            'New Weight': new_portfolio.get(ticker, 0) if new_portfolio else 0,
            'Benchmark Weight': row['Bench Weight'],
            'Weight Change': (new_portfolio.get(ticker, 0) - row['Portfolio Weight']) if new_portfolio else 0,
            'Original Active Share': original_as,
            'New Active Share': new_as,
            'Active Share Improvement': as_improvement if optimized_active_share is not None else None,
            'Improvement %': (as_improvement / original_as * 100) if optimized_active_share is not None and original_as > 0 else None,
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
        
        # New portfolio weight
        new_weight = sum(new_portfolio.get(row['Ticker'], 0) for _, row in stocks_data.iterrows() 
                       if row['Sector'] == sector)
        
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
        
        # New portfolio weight
        new_weight = sum(new_portfolio.get(row['Ticker'], 0) for _, row in stocks_data.iterrows() 
                       if row['Sector-and-Subsector'] == subsector)
        
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
    
    output_file = save_optimizer_results_to_excel(results_df, sector_df, subsector_df)

    return new_portfolio, added_stocks, optimized_active_share, output_file

def save_optimizer_results_to_excel(results_df, sector_df, subsector_df):
    """Save the optimizer results to an Excel file with all sheets."""
    import os
    from datetime import datetime
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_file = os.path.join(outputs_dir, f'Optimized_Portfolio_PuLP_{timestamp}.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name='Portfolio', index=False)
        sector_df.to_excel(writer, sheet_name='Sector Analysis', index=False)
        subsector_df.to_excel(writer, sheet_name='Subsector Analysis', index=False)
    print(f"Results saved to '{output_file}'")
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    return output_file

if __name__ == "__main__":
    main()