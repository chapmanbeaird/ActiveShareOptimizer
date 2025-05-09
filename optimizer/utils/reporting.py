"""
Reporting functions for the Active Share Optimizer.
"""

import os
import pandas as pd
from datetime import datetime

def save_optimizer_results_to_excel(results_df, sector_df, subsector_df, locked_tickers=None, output_timestamp=None):
    """
    Save the optimizer results to an Excel file with all sheets.
    
    Parameters:
    - results_df: DataFrame containing portfolio results
    - sector_df: DataFrame containing sector analysis
    - subsector_df: DataFrame containing subsector analysis
    - locked_tickers: Dictionary of locked tickers and their weights
    - output_timestamp: Optional timestamp to use in the filename
    
    Returns:
    - output_file: Path to the saved Excel file
    """
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Use provided timestamp or generate a new one
    timestamp = output_timestamp or datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_file = os.path.join(outputs_dir, f'Optimized_Portfolio_PuLP_{timestamp}.xlsx')
    
    # Create a DataFrame for locked tickers if provided
    if locked_tickers and len(locked_tickers) > 0:
        locked_df = pd.DataFrame({
            'Ticker': list(locked_tickers.keys()),
            'Locked Weight (%)': list(locked_tickers.values())
        })
        locked_df = locked_df.sort_values(by='Ticker')
    else:
        # Create an empty DataFrame if no locked tickers
        locked_df = pd.DataFrame(columns=['Ticker', 'Locked Weight (%)'])
    
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name='Portfolio', index=False)
        sector_df.to_excel(writer, sheet_name='Sector Analysis', index=False)
        subsector_df.to_excel(writer, sheet_name='Subsector Analysis', index=False)
        locked_df.to_excel(writer, sheet_name='Locked Tickers', index=False)
    
    print(f"Results saved to '{output_file}'")
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    
    return output_file 