"""
Data loading functions for the Active Share Optimizer.
"""

import pandas as pd

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

def load_portfolio_data_csv(file_path):
    """Load and preprocess the portfolio data from CSV."""
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Convert weight columns to numeric values
    data['Bench Weight'] = pd.to_numeric(data['Bench Weight'], errors='coerce').fillna(0)
    data['Portfolio Weight'] = pd.to_numeric(data['Portfolio Weight'], errors='coerce').fillna(0)
    data['Active Share'] = pd.to_numeric(data['Active Share'], errors='coerce').fillna(0)
    data['Core Model'] = pd.to_numeric(data['Core Model'], errors='coerce').fillna(999)

    # Calculate the total active share directly from the portfolio and benchmark weights
    # This is the correct formula for active share: sum(|portfolio_weight - benchmark_weight|) / 2
    # Since weights are already in percentages (e.g., 1.5 means 1.5%), we don't need to multiply by 100
    absolute_diffs = abs(data['Portfolio Weight'] - data['Bench Weight'])
    total_active_share = (absolute_diffs.sum() / 2)

    return data, total_active_share

def load_constraints(file_path):
    """Load the stocks to avoid and sector constraints. Used for backward compatibility."""
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

def load_optimizer_input_file(file_path):
    """
    Load all optimizer input data from a single Excel file.
    This includes:
    - Stock data (active share, benchmark weights, portfolio weights)
    - Stocks to avoid
    - Sector constraints
    - Locked ticker-and-weights
    - Force-include tickers (included but weight not locked)

    Returns:
    - stocks_data: DataFrame containing stock data
    - total_active_share: The calculated total active share
    - stocks_to_avoid: List of ticker symbols to exclude from the portfolio
    - sector_constraints: Dictionary mapping sector-and-subsector to target weights
    - locked_tickers: Dictionary {ticker: weight} of tickers that should not be changed
    - force_include_tickers: Set of tickers to force include (weight determined by optimizer)
    """
    # Read the Excel file
    data = pd.read_excel(file_path, sheet_name=0)

    # Process stock data
    data.columns = data.columns.str.strip()

    # Validate required columns exist
    required_columns = ['Ticker', 'Bench Weight', 'Portfolio Weight', 'Sector', 'Sector-and-Subsector']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in input file: {missing_columns}\n"
            f"Found columns: {list(data.columns)}"
        )

    # Normalize tickers to uppercase for consistent matching
    data['Ticker'] = data['Ticker'].astype(str).str.strip().str.upper()

    # Convert weight columns to numeric values
    data['Bench Weight'] = pd.to_numeric(data['Bench Weight'], errors='coerce').fillna(0)
    data['Portfolio Weight'] = pd.to_numeric(data['Portfolio Weight'], errors='coerce').fillna(0)
    data['Active Share'] = pd.to_numeric(data['Active Share'], errors='coerce').fillna(0)
    # Core Model: fill NaN with 999 so stocks without rank are excluded by default
    # (since core_rank_limit is typically 3-5)
    data['Core Model'] = pd.to_numeric(data['Core Model'], errors='coerce').fillna(999)
    nan_count = (data['Core Model'] == 999).sum()
    if nan_count > 0:
        print(f"Note: {nan_count} stocks have no Core Model rank (will be excluded unless forced)")

    # Validate data quality
    # Check for duplicate tickers
    duplicate_tickers = data[data['Ticker'].duplicated(keep=False)]['Ticker'].unique()
    if len(duplicate_tickers) > 0:
        print(f"Warning: Duplicate tickers found: {list(duplicate_tickers)[:10]}{'...' if len(duplicate_tickers) > 10 else ''}")
        # Keep first occurrence
        data = data.drop_duplicates(subset=['Ticker'], keep='first')
        print(f"  Keeping first occurrence of each duplicate (now {len(data)} stocks)")

    # Check for negative weights
    neg_bench = (data['Bench Weight'] < 0).sum()
    neg_port = (data['Portfolio Weight'] < 0).sum()
    if neg_bench > 0 or neg_port > 0:
        print(f"Warning: Found negative weights - Benchmark: {neg_bench}, Portfolio: {neg_port}")

    # Check weight sums (should be close to 100%)
    bench_sum = data['Bench Weight'].sum()
    port_sum = data[data['Portfolio Weight'] > 0]['Portfolio Weight'].sum()
    if abs(bench_sum - 100) > 1:
        print(f"Warning: Benchmark weights sum to {bench_sum:.2f}% (expected ~100%)")
    if port_sum > 0 and abs(port_sum - 100) > 1:
        print(f"Warning: Portfolio weights sum to {port_sum:.2f}% (expected ~100%)")

    # Calculate the total active share directly from the portfolio and benchmark weights
    # This is the correct formula for active share: sum(|portfolio_weight - benchmark_weight|) / 2
    # Since weights are already in percentages (e.g., 1.5 means 1.5%), we don't need to multiply by 100
    absolute_diffs = abs(data['Portfolio Weight'] - data['Bench Weight'])
    total_active_share = (absolute_diffs.sum() / 2)

    # Get locked tickers where 'Lock Ticker-and-Weight' is 'y' or 'Y'
    locked_tickers = {}

    if 'Lock Ticker-and-Weight' in data.columns:
        locked_data = data[data['Lock Ticker-and-Weight'].astype(str).str.strip().str.upper() == 'Y']

        for _, row in locked_data.iterrows():
            if pd.notna(row['Ticker']) and pd.notna(row['Portfolio Weight']):
                # Ticker already normalized to uppercase earlier
                locked_tickers[row['Ticker']] = row['Portfolio Weight']

    # Get force-include tickers where 'Force Ticker' is 'Y'
    # These tickers will be included in portfolio but weight is not locked
    force_include_tickers = set()

    if 'Force Ticker' in data.columns:
        force_data = data[data['Force Ticker'].astype(str).str.strip().str.upper() == 'Y']

        for _, row in force_data.iterrows():
            if pd.notna(row['Ticker']):
                ticker = row['Ticker']
                # Don't add if already locked (lock takes precedence)
                if ticker not in locked_tickers:
                    force_include_tickers.add(ticker)
    # Get constraints from other sheets in the workbook
    try:
        constraints_data = pd.read_excel(file_path, sheet_name='Constraints')
        
        # Get the list of stocks to avoid (normalize to uppercase for case-insensitive matching)
        if 'Stocks to Avoid' in constraints_data.columns:
            stocks_to_avoid = [str(t).strip().upper() for t in constraints_data['Stocks to Avoid'].dropna().tolist()]
        else:
            stocks_to_avoid = []
        
        # Get the sector constraints
        sector_constraints = {}
        if 'Emp Sector & Industry' in constraints_data.columns and 'Weight' in constraints_data.columns:
            for _, row in constraints_data.iterrows():
                if pd.notna(row['Emp Sector & Industry']) and pd.notna(row['Weight']):
                    sector_constraints[row['Emp Sector & Industry']] = row['Weight']
    except (ValueError, KeyError) as e:
        # Sheet doesn't exist or has wrong format
        print(f"Note: No 'Constraints' sheet found or invalid format: {e}")
        stocks_to_avoid = []
        sector_constraints = {}
    except FileNotFoundError:
        # File doesn't exist (shouldn't happen at this point, but be safe)
        stocks_to_avoid = []
        sector_constraints = {}
    return data, total_active_share, stocks_to_avoid, sector_constraints, locked_tickers, force_include_tickers 