"""
Optimization models for the Active Share Optimizer.
"""

import os
import pulp
import numpy as np
from pulp import PULP_CBC_CMD, COIN_CMD
from collections import defaultdict
import pandas as pd
import shutil
import pulp, shutil, subprocess, os, sys
from optimizer.utils import calculate_active_share



def optimize_portfolio_pulp(stocks_data, original_active_share, num_positions=60, target_active_share=0.55,
                            sector_tolerance=0.03, high_level_sector_tolerance=0.03, stocks_to_avoid=None,
                            sector_constraints=None, min_position=1.0, max_position=5.0, core_rank_limit=3,
                            increment=0.5, forced_positions=None, time_limit=120, locked_tickers=None):
    """
    Optimize the portfolio to reduce Active Share while applying constraints.

    Parameters:
    - stocks_data: DataFrame containing stock data
    - original_active_share: Original Active Share percentage
    - num_positions: Exact number of positions required in the portfolio
    - target_active_share: Target Active Share percentage
    - sector_tolerance: Maximum allowed deviation from benchmark subsector weights (in decimal)
    - high_level_sector_tolerance: Maximum allowed deviation from benchmark sector weights (in decimal)
    - stocks_to_avoid: List of ticker symbols to exclude from the portfolio
    - sector_constraints: Dictionary mapping sector-and-subsector to target weights
    - min_position: Minimum position size as a percentage (e.g., 1.0)
    - max_position: Maximum position size as a percentage (e.g., 5.0)
    - core_rank_limit: Only consider stocks with Core Model rank <= this value (e.g., 1, 2, 3, 4, 5)
    - increment: Allowed increment for position sizes (e.g., 0.5 for 0.5% increments)
    - forced_positions: Dictionary {ticker: (min, max)}. For each ticker, require it to be in the portfolio with weight between min and max.
    - time_limit: Maximum time allowed for the solver (in seconds)
    - locked_tickers: Dictionary {ticker: weight}. For each ticker, require it to be in the portfolio with exactly its current weight.

    Returns:
    - optimized_portfolio: Dictionary of ticker to weight for the optimized portfolio
    - added_stocks: List of stocks added to the portfolio
    - new_active_share: The new Active Share after optimization
    - solver_status: The status of the solver after optimization
    """
    # Initialize variables
    if stocks_to_avoid is None:
        stocks_to_avoid = []
    if sector_constraints is None:
        sector_constraints = {}
    if forced_positions is None:
        forced_positions = {}
    if locked_tickers is None:
        locked_tickers = {}
        
    # If using increments, round locked ticker weights to the nearest increment
    if increment is not None and locked_tickers:
        rounded_locked_tickers = {}
        for ticker, weight in locked_tickers.items():
            # Round the weight to the nearest increment
            rounded_weight = round(round(weight / increment) * increment, 4)
            rounded_locked_tickers[ticker] = rounded_weight
            print(f"Rounding locked ticker {ticker} from {weight}% to nearest increment: {rounded_weight}%")
        
        # Replace original locked tickers with rounded ones
        locked_tickers = rounded_locked_tickers
        
    # Automatically add locked tickers to forced positions with exact weights
    for ticker, locked_weight in locked_tickers.items():
        print(f"Locking ticker {ticker} at exact weight: {locked_weight}%")
        # Set min and max to the same value to lock weight, overriding any position size limits
        # Also ensure this overrides any existing forced position constraint
        forced_positions[ticker] = (locked_weight, locked_weight)  # Exact weight preservation
        
        # Verify the ticker exists in the data
        ticker_data = stocks_data[stocks_data['Ticker'] == ticker]
        if ticker_data.empty:
            print(f"WARNING: Locked ticker {ticker} not found in input data!")
        elif ticker_data.iloc[0]['Portfolio Weight'] != locked_weight:
            print(f"WARNING: Locked ticker {ticker} has weight {ticker_data.iloc[0]['Portfolio Weight']} in data, but is being locked at {locked_weight}!")
    
    if locked_tickers:
        print(f"Locked tickers that will maintain exact weight: {', '.join(locked_tickers.keys())}")
    
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

    # Calculate benchmark weights for high-level sectors
    benchmark_high_level_sector_weights = {}
    for sector in stocks_data['Sector'].unique():
        if pd.isna(sector):
            continue
        sector_stocks = stocks_data[stocks_data['Sector'] == sector]
        sector_weight = sector_stocks['Bench Weight'].sum()
        benchmark_high_level_sector_weights[sector] = sector_weight

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
    available_positions = num_positions - current_positions
    print(f"\nCurrent positions: {current_positions}")
    print(f"Total positions: {num_positions}")
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
            ticker = all_stocks[i]['ticker']
            # For regular stocks, apply the standard min/max position constraints
            if ticker not in forced_positions:
                weight[i] = pulp.LpVariable(f"Weight_{i}", lowBound=0, upBound=max_position, cat=pulp.LpContinuous)
                # If included, weight >= min_position, else weight == 0
                model += weight[i] >= min_position * include[i], f"min_weight_{i}"
                # If not included, weight == 0
                model += weight[i] <= max_position * include[i], f"max_weight_{i}"
            else:
                # For forced positions, allow weights outside the standard min/max range
                minp, maxp = forced_positions[ticker]
                weight[i] = pulp.LpVariable(f"Weight_{i}", lowBound=0, upBound=max(maxp, max_position), cat=pulp.LpContinuous)
                # Other constraints will enforce the specific min/max for this ticker
        
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
                
                # No need to clip to global bounds - forced positions override them
                model += include[idx] == 1, f"force_own_{ticker}"
                
                # Check if this is a locked ticker (exact weight required)
                is_locked_ticker = ticker in locked_tickers
                
                if is_locked_ticker:
                    # For locked tickers, force the exact weight
                    model += weight[idx] == minp, f"exact_weight_{ticker}"
                    print(f"Locking {ticker} at exact weight: {minp}% (continuous)")
                else:
                    # For other forced positions, use min/max bounds
                    model += weight[idx] >= minp, f"minpos_{ticker}"
                    model += weight[idx] <= maxp, f"maxpos_{ticker}"
                    print(f"Forcing {ticker} to be included with weight between {minp}% and {maxp}% (continuous)")
                
                bench_weight = all_stocks[idx]['bench_weight']
                print(f"Benchmark weight for {ticker}: {bench_weight}%")
    else:
        # Enforce increments using binary variables
        # Create standard increments based on min_position and max_position
        standard_increments = [round(x, 4) for x in np.arange(min_position, max_position + increment/2, increment)]
        
        # Create a set of all possible increments, including those needed for forced positions
        all_increments = set(standard_increments)
        
        # Add any special increments needed for locked tickers
        for ticker, locked_weight in locked_tickers.items():
            # Add the exact weight of the locked ticker to the set of increments
            all_increments.add(round(locked_weight, 4))
            print(f"Adding exact weight {locked_weight}% for locked ticker {ticker} to available increments")
        
        # Add any special increments needed for forced positions
        for ticker, (minp, maxp) in forced_positions.items():
            # Skip if this is a locked ticker - already handled above
            if ticker in locked_tickers:
                continue
                
            # If the forced position has exact bounds (minp == maxp), add that specific value
            if minp == maxp:
                all_increments.add(round(minp, 4))
            # Otherwise, add increments within the range that might be outside standard bounds
            elif increment is not None:
                # Start from the min position bound or the forced min, whichever is lower
                start_val = min(min_position, minp)
                # Go up to the max position bound or the forced max, whichever is higher
                end_val = max(max_position, maxp) + increment/2
                special_increments = [round(x, 4) for x in np.arange(start_val, end_val, increment)]
                all_increments.update(special_increments)
        
        # Convert to a sorted list
        increments = sorted(list(all_increments))
        
        print(f"Using incremental weights with increment: {increment}%")
        print(f"Available increments: {min(increments)}% to {max(increments)}% in steps of {increment}%")
        
        increment_bin_vars = {}
        for i in range(len(all_stocks)):
            ticker = all_stocks[i]['ticker']
            
            # Check if this is a locked ticker
            is_locked_ticker = ticker in locked_tickers
            
            if is_locked_ticker:
                # For locked tickers, create a continuous weight variable instead of using increments
                locked_weight = locked_tickers[ticker]
                
                # Create a continuous weight variable for this locked ticker
                weight[i] = pulp.LpVariable(f"Weight_{i}", lowBound=locked_weight, upBound=locked_weight, cat=pulp.LpContinuous)
                
                # Force this stock to be included
                model += include[i] == 1, f"force_include_locked_{ticker}"
                
                print(f"Enforcing locked ticker {ticker} at exact weight: {locked_weight}% during optimization")
                
                # Skip the increment binary variables for this ticker
                continue
            
            increment_bin_vars[i] = {}
            
            # For each stock, determine which increments are allowed
            allowed_increments_for_stock = increments
            
            # If this is a forced position, restrict to increments within its bounds
            if ticker in forced_positions:
                minp, maxp = forced_positions[ticker]
                allowed_increments_for_stock = [inc for inc in increments if minp <= inc <= maxp]
            
            # Create binary variables only for allowed increments
            for inc in increments:
                increment_bin_vars[i][inc] = pulp.LpVariable(f"IncBin_{i}_{inc}", cat=pulp.LpBinary)
                
                # If the increment is not allowed for this stock, force its binary var to 0
                if ticker in forced_positions and inc not in allowed_increments_for_stock:
                    model += increment_bin_vars[i][inc] == 0, f"disallow_inc_{i}_{inc}"
            
            # The weight is the sum over increments times their binary variable
            weight[i] = pulp.lpSum([inc * increment_bin_vars[i][inc] for inc in increments])
            
            # Only allow increments if stock is included
            model += pulp.lpSum([increment_bin_vars[i][inc] for inc in increments]) == include[i]
            
        # Forced positions constraints (increment case) - only add these ONCE per ticker
        for ticker, (minp, maxp) in forced_positions.items():
            # Skip if this is a locked ticker - already handled above
            if ticker in locked_tickers:
                continue
            if ticker in ticker_to_index:
                idx = ticker_to_index[ticker]
                # Sanity checks
                if not (isinstance(minp, (int, float)) and isinstance(maxp, (int, float))):
                    raise ValueError(f"Invalid forced position bounds for {ticker}: {minp}, {maxp}")
                if minp > maxp:
                    raise ValueError(f"minp > maxp for {ticker}: {minp} > {maxp}")
                
                # Force the stock to be included
                model += include[idx] == 1, f"force_own_{ticker}"
                
                # Create a list of allowed increments for this ticker
                allowed_increments = [inc for inc in increments if minp <= inc <= maxp]
                
                # Check if this is a locked ticker (exact weight required)
                is_locked_ticker = ticker in locked_tickers
                
                # Ensure exactly one increment is selected for this forced position
                if allowed_increments:
                    model += pulp.lpSum([increment_bin_vars[idx][inc] for inc in allowed_increments]) == 1, f"force_sum_inc_{ticker}"
                    bench_weight = all_stocks[idx]['bench_weight']
                    print(f"Forcing {ticker} to be included with weight between {minp}% and {maxp}%, allowed increments: {allowed_increments}")
                    print(f"Benchmark weight for {ticker}: {bench_weight}%")
                else:
                    # If this is a locked ticker, don't treat this as an error - we'll handle it post-optimization
                    if is_locked_ticker:
                        print(f"Note: Locked ticker {ticker} with weight {minp}% doesn't align with increment pattern. Will be enforced post-optimization.")
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
    
    model += 0.5 * total_abs_diff >= target_active_share * 100
    
    # Objective function: minimize active share while maintaining the constraint
    model += 0.5 * total_abs_diff
    
    # Constraint: Total weight must sum to 100%
    model += pulp.lpSum(weight[i] for i in range(len(all_stocks))) == 100
    
    # Constraint: Total number of positions must be exactly equal to num_positions
    # Count locked tickers separately to provide more flexibility
    if locked_tickers:
        # Count how many locked tickers we have
        num_locked = len(locked_tickers)
        print(f"Adjusting position count constraint to account for {num_locked} locked tickers")
        
        # Get indices of non-locked tickers
        non_locked_indices = [i for i in range(len(all_stocks)) if all_stocks[i]['ticker'] not in locked_tickers]
        
        # The constraint applies only to non-locked tickers
        model += pulp.lpSum(include[i] for i in non_locked_indices) == num_positions - num_locked
    else:
        # No locked tickers, use the original constraint
        model += pulp.lpSum(include[i] for i in range(len(all_stocks))) == num_positions

    # Constraint: Filter out stocks in the stocks_to_avoid list
    for ticker in stocks_to_avoid:
        # Don't exclude a ticker if it's locked or in forced positions
        if ticker in ticker_to_index and ticker not in forced_positions and ticker not in locked_tickers:
            i = ticker_to_index[ticker]
            model += include[i] == 0

    # Core rank constraint: Only select stocks with Core Rank <= core_rank_limit
    for i in range(len(all_stocks)):
        ticker = all_stocks[i]['ticker']
        # Don't apply core rank constraints to locked or forced tickers
        if all_stocks[i]['core_rank'] > core_rank_limit and ticker not in forced_positions and ticker not in locked_tickers:
            model += include[i] == 0

    # Sector-and-subsector weights constraints
    for subsector, target_weight in sector_constraints.items():
        if subsector in subsector_to_stocks:
            subsector_weight = pulp.lpSum(weight[i] for i in subsector_to_stocks[subsector])
            model += subsector_weight >= target_weight - sector_tolerance * 100
            model += subsector_weight <= target_weight + sector_tolerance * 100

    # High-level sector weight constraints
    for sector, bench_weight in benchmark_high_level_sector_weights.items():
        if sector in sector_to_stocks:
            sector_weight = pulp.lpSum(weight[i] for i in sector_to_stocks[sector])
            model += sector_weight >= bench_weight - high_level_sector_tolerance * 100
            model += sector_weight <= bench_weight + high_level_sector_tolerance * 100

    # Solve the model using portable solver detection
    # Try PULP_CBC_CMD first (auto-detects solver path), fall back to common locations
    solver = None
    try:
        solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit)
        model.solve(solver)
    except Exception as e:
        print(f"PULP_CBC_CMD failed: {e}, trying fallback paths...")
        # Try common CBC locations
        cbc_paths = [
            "/opt/homebrew/bin/cbc",  # macOS Homebrew ARM
            "/usr/local/bin/cbc",      # macOS Homebrew Intel / Linux
            "/usr/bin/cbc",            # Linux system
            shutil.which("cbc"),       # System PATH
        ]
        for cbc_path in cbc_paths:
            if cbc_path and os.path.exists(cbc_path):
                try:
                    solver = COIN_CMD(path=cbc_path, msg=True, timeLimit=time_limit)
                    model.solve(solver)
                    break
                except Exception:
                    continue
        if solver is None:
            raise RuntimeError("Could not find CBC solver. Install with: pip install pulp[cbc]")

    # Get the solver status
    solver_status = pulp.LpStatus[model.status]
    print(f"Solver status: {solver_status}")
    
    if solver_status not in ['Optimal', 'Feasible']:
        print(f"Warning: Solver did not find a solution (status: {solver_status})")
        return {}, [], None, solver_status

    if solver_status == 'Feasible':
        print("Note: Solver found a feasible but not proven optimal solution (may be due to time limit)")
    
    # Extract the solution
    optimized_portfolio = {}
    added_stocks = []
    
    for i, stock in enumerate(all_stocks):
        if include[i].value() > 0.5:  # If stock is included
            ticker = stock['ticker']
            weight_value = weight[i].value()
            optimized_portfolio[ticker] = weight_value
            
            # If this stock wasn't in the original portfolio, add it to added_stocks
            if not stock['in_portfolio']:
                added_stocks.append({
                    'ticker': ticker,
                    'sector': stock['sector'],
                    'subsector': stock['subsector'],
                    'new_weight': weight_value,
                    'bench_weight': stock['bench_weight'],
                    'core_rank': stock['core_rank']
                })
    
    # Create benchmark dictionary
    benchmark = {stock['ticker']: stock['bench_weight'] for stock in all_stocks}
    
    # Calculate the new active share using the utility function
    new_active_share = calculate_active_share(optimized_portfolio, benchmark)
    
    print(f"Calculated new active share: {new_active_share:.2f}%")
    
    return optimized_portfolio, added_stocks, new_active_share, solver_status 