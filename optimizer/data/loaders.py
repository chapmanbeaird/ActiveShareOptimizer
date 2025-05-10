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
    data['Core Model'] = pd.to_numeric(data['Core Model'], errors='coerce')
    
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

    Returns:
    - stocks_data: DataFrame containing stock data
    - total_active_share: The calculated total active share
    - stocks_to_avoid: List of ticker symbols to exclude from the portfolio
    - sector_constraints: Dictionary mapping sector-and-subsector to target weights
    - locked_tickers: Dictionary {ticker: weight} of tickers that should not be changed
    """
    # Read the Excel file
    data = pd.read_excel(file_path, sheet_name=0)
    
    # Process stock data
    data.columns = data.columns.str.strip()
    
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
    
    # Get locked tickers where 'Lock Ticker-and-Weight' is 'y' or 'Y'
    locked_tickers = {}

    if 'Lock Ticker-and-Weight' in data.columns:
        locked_data = data[data['Lock Ticker-and-Weight'].astype(str).str.strip().str.upper() == 'Y']
        
        for _, row in locked_data.iterrows():
            if pd.notna(row['Ticker']) and pd.notna(row['Portfolio Weight']):
                locked_tickers[row['Ticker']] = row['Portfolio Weight']
    # Get constraints from other sheets in the workbook
    try:
        constraints_data = pd.read_excel(file_path, sheet_name='Constraints')
        
        # Get the list of stocks to avoid
        if 'Stocks to Avoid' in constraints_data.columns:
            stocks_to_avoid = constraints_data['Stocks to Avoid'].dropna().tolist()
        else:
            stocks_to_avoid = []
        
        # Get the sector constraints
        sector_constraints = {}
        if 'Emp Sector & Industry' in constraints_data.columns and 'Weight' in constraints_data.columns:
            for _, row in constraints_data.iterrows():
                if pd.notna(row['Emp Sector & Industry']) and pd.notna(row['Weight']):
                    sector_constraints[row['Emp Sector & Industry']] = row['Weight']
    except Exception as e:
        print(f"Warning: Could not load constraints from separate sheet: {e}")
        stocks_to_avoid = []
        sector_constraints = {}
    return data, total_active_share, stocks_to_avoid, sector_constraints, locked_tickers 