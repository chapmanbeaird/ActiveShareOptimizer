import streamlit as st
import pandas as pd
import io
import time
import threading
from active_share_optimizer_pulp import (
    load_portfolio_data_csv,
    load_constraints,
    optimize_portfolio_pulp,
    main as optimizer_main
)

# Global variable to track optimization progress
optimization_time = 0
optimization_running = False

def timer_thread():
    global optimization_time, optimization_running
    optimization_time = 0
    optimization_running = True
    while optimization_running:
        time.sleep(1)
        optimization_time += 1

def main():
    global optimization_time, optimization_running
    
    st.title("Active Share Optimizer (PuLP)")
    st.write("Upload your portfolio and constraints, then adjust parameters to optimize your portfolio interactively.")

    # --- File uploads ---
    portfolio_file = st.file_uploader("Upload Portfolio CSV (e.g. active_share_with_core_constraints.csv)", type=["csv"])
    constraints_file = st.file_uploader("Upload Constraints Excel (e.g. stocks_to_avoid&sector_constraints.xlsm)", type=["xls", "xlsx", "xlsm"])

    st.sidebar.header("Optimizer Parameters")
    max_positions = st.sidebar.slider("Max Positions", min_value=10, max_value=100, value=60, step=1)
    target_active_share = st.sidebar.slider("Target Active Share (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)
    sector_tolerance = st.sidebar.slider("Sector Tolerance (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    min_position = st.sidebar.slider("Min Position Size (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    max_position = st.sidebar.slider("Max Position Size (%)", min_value=0.5, max_value=10.0, value=5.0, step=0.1)
    core_rank_limit = st.sidebar.selectbox("Max Core Model Rank", options=[1, 2, 3, 4, 5], index=2)
    
    # Position size constraint mode selection
    weight_mode = st.sidebar.radio(
        "Position Size Mode:",
        ["Continuous Weights", "Discrete Increments"],
        index=1,
        help="Continuous: Any value between min and max. Discrete: Only specific incremental values."
    )
    
    increment = None
    if weight_mode == "Discrete Increments":
        increment = st.sidebar.slider("Position Increment Size (%)", 
                              min_value=0.01, max_value=1.0, 
                              value=0.5, step=0.01, 
                              help="Position sizes will be multiples of this increment")
        st.sidebar.info(f"Position sizes will be multiples of {increment}% (e.g., {min_position}%, {min_position+increment}%, {min_position+2*increment}%, etc.)")
    else:
        st.sidebar.info("Position sizes can be any value between the min and max position size")
    
    # Add timeout parameter to sidebar
    time_limit = st.sidebar.slider("Solver Timeout (seconds)", min_value=30, max_value=600, value=120, step=30, 
                                help="Maximum time allowed for the solver to find a solution.")
    
    # --- Ticker selection and forced positions ---
    all_tickers = []
    if portfolio_file:
        try:
            portfolio_file.seek(0)
            temp_stocks_data, _ = load_portfolio_data_csv(portfolio_file)
            all_tickers = sorted(temp_stocks_data['Ticker'].unique())
        except Exception:
            pass
    st.sidebar.header("Force Stock Holdings")
    selected_tickers = st.sidebar.multiselect(
        "Search and select tickers to force in portfolio:",
        options=all_tickers,
        help="Type to search. For each selected ticker, you can force ownership and set min/max position size."
    )
    forced_positions = {}
    for ticker in selected_tickers:
        min_val = st.sidebar.number_input(f"{ticker} min %", min_value=0.0, max_value=100.0, value=1.0, step=0.01, key=f"min_{ticker}")
        max_val = st.sidebar.number_input(f"{ticker} max %", min_value=min_val, max_value=100.0, value=5.0, step=0.01, key=f"max_{ticker}")
        forced_positions[ticker] = (min_val, max_val)
    
    if st.button("Run Optimizer"):
        if not portfolio_file or not constraints_file:
            st.error("Please upload both a portfolio CSV and a constraints Excel file.")
            return
        
        # Create a progress display area
        progress_container = st.empty()
        timer_display = st.empty()
        
        # Start the timer thread
        timer_thread_obj = threading.Thread(target=timer_thread)
        timer_thread_obj.daemon = True
        timer_thread_obj.start()
        
        # Show progress message
        progress_container.info("Optimization in progress...")
        
        # Save uploaded files to temporary locations
        import tempfile
        portfolio_file.seek(0)
        constraints_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_portfolio:
            temp_portfolio.write(portfolio_file.read())
            portfolio_path = temp_portfolio.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsm') as temp_constraints:
            temp_constraints.write(constraints_file.read())
            constraints_path = temp_constraints.name
        
        # Create a placeholder for progress information
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run optimizer in a separate thread to keep UI responsive
        optimizer_result = [None]  # Use a list to store the result across threads
        
        def run_optimizer():
            global optimization_running
            try:
                optimizer_result[0] = optimizer_main(
                    data_file_path=portfolio_path,
                    constraints_file_path=constraints_path,
                    max_positions=max_positions,
                    target_active_share=target_active_share/100.0,  # convert to fraction
                    sector_tolerance=sector_tolerance/100.0,        # convert to fraction
                    min_position=min_position,
                    max_position=max_position,
                    core_rank_limit=core_rank_limit,
                    forced_positions=forced_positions,
                    time_limit=time_limit,
                    increment=increment  # Pass the increment parameter
                )
            finally:
                optimization_running = False
        
        optimizer_thread = threading.Thread(target=run_optimizer)
        optimizer_thread.daemon = True
        optimizer_thread.start()
        
        # Update timer display in a loop until optimization completes
        result = None
        try:
            while optimization_running:
                # Update progress bar to show percentage of time limit used
                progress_percent = min(optimization_time / time_limit, 1.0)
                progress_bar.progress(progress_percent)
                
                # Update timer display
                timer_display.info(f"Optimization running... Time elapsed: {optimization_time} seconds (Timeout: {time_limit} seconds)")
                
                # Check if we should stop due to timeout
                if optimization_time >= time_limit:
                    # If we've exceeded the time limit by a significant margin, force stop
                    if optimization_time >= time_limit + 10:
                        st.warning("Forcing solver to stop due to timeout...")
                        optimization_running = False
                        break
                
                time.sleep(0.1)  # More frequent updates for smoother progress bar
            
            # Wait for optimizer thread to complete
            optimizer_thread.join(timeout=10)
            if optimizer_thread.is_alive():
                st.error("Optimizer is still running but taking too long. Please refresh the page and try again with different parameters.")
                return
                
            # Get results from optimizer_main
            if optimizer_result[0] is None:
                st.error("Optimization failed to return a result.")
                return
                
            optimized_portfolio, added_stocks, new_active_share, output_file = optimizer_result[0]
        
        finally:
            # Clear progress displays
            optimization_running = False
            progress_bar.empty()
            status_text.empty()
            timer_display.empty()
            progress_container.empty()
        
        if new_active_share is None:
            st.error("Could not find an optimal solution with the given constraints. Please try adjusting the parameters.")
            
            # Show helpful suggestions for fixing the infeasibility
            st.warning("""
            Common reasons for infeasibility include:
            1. The target active share is too high or too low for the given constraints
            2. Forcing certain stocks creates conflicts with sector constraints
            3. The constraints on position sizes are too restrictive
            
            Try these adjustments:
            - Lower the target active share slightly
            - Increase sector tolerance
            - Adjust or remove forced positions
            - Increase the maximum position size
            """)
            
            # Still display the output file if it was created
            if output_file:
                st.info(f"A partial results file was saved to: {output_file}")
            
            return
            
        st.success(f"Optimization complete! New Active Share: {new_active_share:.2f}%")
        st.write("### Optimized Portfolio")
        results_df = pd.DataFrame(list(optimized_portfolio.items()), columns=["Ticker", "Weight (%)"])
        st.dataframe(results_df)
        st.write("### Added Stocks")
        # Display only the ticker symbols for added stocks (handle both list of dicts and list of strings)
        if added_stocks:
            if isinstance(added_stocks[0], dict):
                st.write(", ".join(stock['ticker'] for stock in added_stocks))
            else:
                st.write(", ".join(str(stock) for stock in added_stocks))
        else:
            st.write("None")

        if output_file:
            st.success(f"Results saved to: {output_file}")
            st.info("For a full breakdown of the optimized portfolio, sector, and subsector analysis, please check the output Excel file. The filename will be timestamped.")
        else:
            st.warning("No output file was generated.")

if __name__ == "__main__":
    main()
