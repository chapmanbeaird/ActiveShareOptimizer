import streamlit as st
import pandas as pd
import altair as alt
import io
import time
import threading
import tempfile
import os
from optimizer import (
    load_optimizer_input_file,
    optimize_portfolio_pulp,
    main as optimizer_main
)

# Global variable to track optimization progress
optimization_time = 0
optimization_running = False
optimization_result = [None]  # Use a list to store the result across threads

def timer_thread():
    global optimization_time, optimization_running
    optimization_time = 0
    optimization_running = True
    while optimization_running:
        time.sleep(1)
        optimization_time += 1

def run_optimizer_thread(data_file_path, num_positions, target_active_share, sector_tolerance,
                        high_level_sector_tolerance, min_position, max_position, core_rank_limit,
                        forced_positions, time_limit, increment):
    global optimization_running, optimization_result
    try:
        optimization_result[0] = optimizer_main(
            data_file_path=data_file_path,
            num_positions=num_positions,
            target_active_share=target_active_share,
            sector_tolerance=sector_tolerance,
            high_level_sector_tolerance=high_level_sector_tolerance,
            min_position=min_position,
            max_position=max_position,
            core_rank_limit=core_rank_limit,
            forced_positions=forced_positions,
            time_limit=time_limit,
            increment=increment
        )
    except Exception as e:
        print(f"Error in optimizer thread: {e}")
        optimization_result[0] = (None, None, None, None, "Error", None)
    finally:
        optimization_running = False

def main():
    global optimization_time, optimization_running, optimization_result
    
    # Password protection
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("Active Share Optimizer")
        password = st.text_input("Enter password to access the application", type="password")
        
        # Try to get password from environment variable first
        expected_password = os.getenv("UNIVERSAL_PASSWORD")
        
        # If not in environment, try to get from secrets
        if expected_password is None:
            try:
                expected_password = st.secrets["general"]["UNIVERSAL_PASSWORD"]
            except (KeyError, AttributeError):
                st.error("Application configuration error. Please contact the administrator.")
                return
        
        if password == expected_password:
            st.session_state.authenticated = True
            st.rerun()
        elif password:  # Only show error if password was entered but was incorrect
            st.error("Incorrect password. Please try again.")
        return
    
    # Initialize session state for toggle states if they don't exist
    if 'show_portfolio' not in st.session_state:
        st.session_state.show_portfolio = True
    if 'show_additions' not in st.session_state:
        st.session_state.show_additions = True
    if 'show_sectors' not in st.session_state:
        st.session_state.show_sectors = True
    if 'optimization_completed' not in st.session_state:
        st.session_state.optimization_completed = False
    if 'optimization_data' not in st.session_state:
        st.session_state.optimization_data = None
    
    st.title("Active Share Optimizer")
    st.write("Upload your portfolio file containing stocks, constraints, and locked tickers, then adjust parameters to optimize your portfolio.")

    # --- File upload ---
    input_file = st.file_uploader("Upload Optimizer Input File (optimizer_input_file.xlsm)", type=["xls", "xlsx", "xlsm"])
    
    # --- Load and preview data ---
    all_tickers = []
    file_data = None  # Store loaded data for reuse

    if input_file:
        try:
            input_file.seek(0)
            temp_stocks_data, file_active_share, file_stocks_to_avoid, file_sector_constraints, file_locked_tickers = load_optimizer_input_file(input_file)
            all_tickers = sorted(temp_stocks_data['Ticker'].unique())

            # Store for later use
            file_data = {
                'stocks_data': temp_stocks_data,
                'active_share': file_active_share,
                'stocks_to_avoid': file_stocks_to_avoid,
                'sector_constraints': file_sector_constraints,
                'locked_tickers': file_locked_tickers
            }

            # --- Pre-Optimization Data Preview ---
            st.subheader("📊 Current Portfolio Summary")

            # Calculate key metrics
            current_positions = len(temp_stocks_data[temp_stocks_data['Portfolio Weight'] > 0])
            benchmark_stocks = len(temp_stocks_data[temp_stocks_data['Bench Weight'] > 0])

            # Display in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Active Share", f"{file_active_share:.2f}%")
            with col2:
                st.metric("Current Positions", current_positions)
            with col3:
                st.metric("Locked Tickers", len(file_locked_tickers))
            with col4:
                st.metric("Stocks to Avoid", len(file_stocks_to_avoid))

            # Show locked tickers if any
            if file_locked_tickers:
                with st.expander(f"🔒 View Locked Tickers ({len(file_locked_tickers)})"):
                    locked_df = pd.DataFrame([
                        {"Ticker": t, "Locked Weight (%)": w}
                        for t, w in sorted(file_locked_tickers.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(locked_df, hide_index=True, use_container_width=True)
                    total_locked = sum(file_locked_tickers.values())
                    st.caption(f"Total locked weight: {total_locked:.2f}%")

        except Exception as e:
            st.error(f"Error loading file: {e}")
            
    st.sidebar.header("Force Stock Holdings")
    selected_tickers = st.sidebar.multiselect(
        "Search and select tickers to force in portfolio:",
        options=all_tickers,
        help="Type to search. For each selected ticker, you can force ownership at a specific weight range."
    )
    
    forced_positions = {}
    for ticker in selected_tickers:
        st.sidebar.markdown(f"### {ticker}")
        min_weight = st.sidebar.slider(
            f"Min Weight for {ticker} (%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key=f"min_{ticker}"
        )
        max_weight = st.sidebar.slider(
            f"Max Weight for {ticker} (%)",
            min_value=min_weight,
            max_value=10.0,
            value=min(min_weight + 1.0, 10.0),
            step=0.5,
            key=f"max_{ticker}"
        )
        forced_positions[ticker] = (min_weight, max_weight)
    
    # --- Optimization parameters ---
    st.sidebar.header("Optimization Parameters")

    with st.sidebar.expander("ℹ️ Parameter Guide"):
        st.markdown("""
        **Total Positions**: Exact number of stocks in the final portfolio.

        **Target Active Share**: Minimum deviation from benchmark

        **Sector Tolerance**: How much each high-level sector can deviate from benchmark. Lower = tighter tracking.

        **Subsector Tolerance**: How much each subsector can deviate from benchmark. Lower = tighter tracking.

        **Position Size Bounds**: Min/max weight per stock.

        **Core Rank Limit**: Only include stocks ranked 1-N by your core model. Lower = higher conviction only.

        **Position Increments**: If enabled, weights snap to 0.5% increments (e.g., 1.0%, 1.5%, 2.0%). Helps with practical implementation.
        """)

    num_positions = st.sidebar.slider(
        "Total Positions",
        min_value=30,
        max_value=100,
        value=60,
        step=1,
        help="Exact number of positions required in the portfolio."
    )
    
    target_active_share = st.sidebar.slider(
        "Target Active Share (%)",
        min_value=30.0,
        max_value=80.0,
        value=55.0,
        step=0.5,
        help="Target Active Share percentage. The optimizer will try to achieve this level of active share."
    ) / 100.0  # Convert to decimal

    high_level_sector_tolerance = st.sidebar.slider(
        "Sector Tolerance (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Maximum allowed deviation from benchmark sector weights (high-level sectors)."
    ) / 100.0  # Convert to decimal

    sector_tolerance = st.sidebar.slider(
        "Subsector Tolerance (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Maximum allowed deviation from benchmark subsector weights."
    ) / 100.0  # Convert to decimal
    
    min_position = st.sidebar.slider(
        "Minimum Position Size (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
        help="Minimum position size as a percentage."
    )
    
    max_position = st.sidebar.slider(
        "Maximum Position Size (%)",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Maximum position size as a percentage."
    )
    
    core_rank_limit = st.sidebar.slider(
        "Core Rank Limit",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Only consider stocks with Core Model rank <= this value."
    )
    
    time_limit = st.sidebar.slider(
        "Solver Time Limit (seconds)",
        min_value=30,
        max_value=300,
        value=120,
        step=30,
        help="Maximum time allowed for the solver."
    )
    
    use_increments = st.sidebar.checkbox(
        "Use Position Increments",
        value=True,
        help="If checked, positions will be in increments of the specified size. If unchecked, continuous weights will be used."
    )
    
    increment = None
    if use_increments:
        increment = st.sidebar.slider(
            "Position Increment Size (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Allowed increment for position sizes."
        )
    
    # --- Feasibility Warnings ---
    if file_data is not None:
        stocks_data = file_data['stocks_data']
        locked_tickers = file_data['locked_tickers']
        stocks_to_avoid = file_data['stocks_to_avoid']

        # Calculate eligible stocks
        eligible_stocks = stocks_data[
            (stocks_data['Bench Weight'] > 0) &
            (stocks_data['Core Model'] <= core_rank_limit) &
            (~stocks_data['Ticker'].isin(stocks_to_avoid))
        ]
        eligible_count = len(eligible_stocks)

        # Check for potential issues
        warnings = []

        if len(locked_tickers) > num_positions:
            warnings.append(f"⚠️ **Too many locked tickers**: {len(locked_tickers)} locked but only {num_positions} positions requested")

        if eligible_count < num_positions:
            warnings.append(f"⚠️ **Not enough eligible stocks**: Only {eligible_count} stocks meet criteria (Core Rank ≤ {core_rank_limit}, not in avoid list) for {num_positions} positions")

        total_locked_weight = sum(locked_tickers.values())
        if total_locked_weight > 100:
            warnings.append(f"⚠️ **Locked weights exceed 100%**: Total locked weight is {total_locked_weight:.1f}%")

        if min_position > max_position:
            warnings.append(f"⚠️ **Invalid position bounds**: Min ({min_position}%) > Max ({max_position}%)")

        if warnings:
            st.warning("### Potential Feasibility Issues\n" + "\n".join(warnings))

    # --- Run optimization button ---
    if st.button("Run Optimization", type="primary"):
        if input_file is None:
            st.error("Please upload an input file.")
        else:
            # Reset global variables
            optimization_running = False
            optimization_time = 0
            optimization_result[0] = None
            
            # Create progress display
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # Save uploaded file to a temporary location
                input_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsm') as temp_input:
                    temp_input.write(input_file.read())
                    input_path = temp_input.name
                
                # Start the timer thread
                optimization_running = True
                timer_thread_obj = threading.Thread(target=timer_thread)
                timer_thread_obj.daemon = True
                timer_thread_obj.start()
                
                # Start the optimizer thread
                optimizer_thread = threading.Thread(
                    target=run_optimizer_thread,
                    args=(input_path, num_positions, target_active_share, sector_tolerance,
                          high_level_sector_tolerance, min_position, max_position, core_rank_limit,
                          forced_positions, time_limit, increment)
                )
                optimizer_thread.daemon = True
                optimizer_thread.start()
                
                # Display progress while optimization is running
                while optimization_running:
                    # Update progress bar
                    progress_percent = min(optimization_time / time_limit, 1.0)
                    progress_bar.progress(progress_percent)
                    progress_placeholder.info(f"Optimization running for {optimization_time} seconds (timeout: {time_limit}s)...")
                    
                    # Check for timeout
                    if optimization_time >= time_limit + 10:  # Give a little extra time
                        st.warning("Optimization taking longer than expected. Checking status...")
                        if not optimizer_thread.is_alive():
                            break
                    
                    time.sleep(0.1)  # Small sleep to prevent UI freezing
                
                # Wait for optimizer thread to complete if it's still running
                if optimizer_thread.is_alive():
                    st.info("Finalizing optimization...")
                    optimizer_thread.join(timeout=10)
                
                # Get the results
                if optimization_result[0] is None:
                    st.error("Optimization failed or was interrupted.")
                    return
                
                # Unpack the results
                new_portfolio, added_stocks, optimized_active_share, output_file, solver_status, original_active_share_from_main = optimization_result[0]
                
                # Store optimization results in session state
                st.session_state.optimization_completed = True
                st.session_state.optimization_data = {
                    'new_portfolio': new_portfolio,
                    'added_stocks': added_stocks,
                    'optimized_active_share': optimized_active_share,
                    'output_file': output_file,
                    'input_path': input_path,
                    'solver_status': solver_status
                }
                
                # Clear progress displays
                progress_placeholder.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                # Stop the timer thread
                optimization_running = False
    
    # Display results if optimization has been completed
    if st.session_state.optimization_completed and st.session_state.optimization_data is not None:
        # Extract data from session state
        new_portfolio = st.session_state.optimization_data['new_portfolio']
        added_stocks = st.session_state.optimization_data['added_stocks']
        optimized_active_share = st.session_state.optimization_data['optimized_active_share']
        output_file = st.session_state.optimization_data['output_file']
        input_path = st.session_state.optimization_data['input_path']
        solver_status = st.session_state.optimization_data.get('solver_status', 'Unknown')
        
        # Display the results
        if solver_status != 'Optimal':
            st.error(f"Optimization status: {solver_status}")
            st.warning("""
            ### The optimization problem is infeasible with the current constraints.
            
            **Try the following to find a feasible solution:**
            - Reduce the target active share
            - Lower the weight of forced positions 
            - Use a wider range for forced position weights
            - Reduce the number of locked tickers
            - Relax sector constraints
            - Use continuous weights instead of increments
            """
            )
        elif len(new_portfolio) > 0 and len(new_portfolio) <= 5:  # Likely a fallback solution with just locked tickers
            # Get the original active share from the data
            _, original_active_share, _, _, _ = load_optimizer_input_file(input_path)
            
            st.warning(f"""
            ### Fallback Solution with Locked Tickers Only
            
            The optimizer could not find a feasible solution with all constraints.
            A fallback portfolio containing only the locked tickers has been created.
            
            **Current constraints that might be causing infeasibility:**
            - Target Active Share: {target_active_share*100:.1f}%
            - Number of positions: {num_positions}
            - Locked tickers: {len(new_portfolio)}
            - Sector constraints with tolerance: {sector_tolerance*100:.1f}%
            - Using increments: {increment is not None}
            """
            )
            
            st.write(f"Original Active Share: {original_active_share:.2f}%")
            st.write(f"Fallback Active Share: {optimized_active_share:.2f}%")
            st.write(f"This is a partial solution with only {len(new_portfolio)} positions instead of the target {num_positions}.")
            
            # Display the output file - moved to top before checkboxes
            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download Fallback Portfolio (Locked Tickers Only)",
                    data=f,
                    file_name=f"Fallback_Portfolio_Locked_Tickers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            # Get the original data for comparison
            original_stocks_data, original_active_share, _, _, original_locked = load_optimizer_input_file(input_path)

            # Calculate turnover metrics
            original_positions = set(original_stocks_data[original_stocks_data['Portfolio Weight'] > 0]['Ticker'].tolist())
            new_positions = set(new_portfolio.keys())
            positions_added = new_positions - original_positions
            positions_removed = original_positions - new_positions

            # Calculate improvement metrics
            improvement_abs = original_active_share - optimized_active_share
            improvement_pct = (improvement_abs / original_active_share * 100) if original_active_share > 0 else 0

            st.success("✅ Optimization Complete!")

            # Display key metrics in columns
            st.subheader("📈 Results Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Active Share",
                    f"{optimized_active_share:.2f}%",
                    delta=f"-{improvement_abs:.2f}%",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Active Share Improvement",
                    f"{improvement_pct:.1f}%",
                    delta="reduction"
                )
            with col3:
                st.metric(
                    "Final Positions",
                    len(new_portfolio),
                    delta=f"{len(new_portfolio) - len(original_positions):+d}"
                )

            # Turnover summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positions Added", len(positions_added))
            with col2:
                st.metric("Positions Removed", len(positions_removed))
            with col3:
                st.metric("Positions Unchanged", len(original_positions & new_positions))
            
            # Display the output file - moved to top before checkboxes
            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download Optimized Portfolio",
                    data=f,
                    file_name=f"Optimized_Portfolio.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
            
            # Get the original data to extract sector information (needed for multiple sections)
            stocks_data, _, _, _, _ = load_optimizer_input_file(input_path)

            # Create mappings for tickers
            ticker_to_sector = {}
            ticker_to_subsector = {}
            ticker_to_bench = {}
            ticker_to_core_rank = {}
            for _, row in stocks_data.iterrows():
                ticker = row['Ticker']
                if pd.notna(ticker):
                    ticker_to_sector[ticker] = row.get('Sector', 'Unknown')
                    ticker_to_subsector[ticker] = row.get('Sector-and-Subsector', 'Unknown')
                    ticker_to_bench[ticker] = row.get('Bench Weight', 0)
                    ticker_to_core_rank[ticker] = row.get('Core Model', 999)

            # Calculate sector weights for new portfolio
            sector_weights = {}
            subsector_weights = {}
            for ticker, weight in new_portfolio.items():
                sector = ticker_to_sector.get(ticker, 'Unknown')
                subsector = ticker_to_subsector.get(ticker, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
                subsector_weights[subsector] = subsector_weights.get(subsector, 0) + weight

            # Calculate benchmark sector weights for comparison
            benchmark_sector_weights = {}
            for _, row in stocks_data.iterrows():
                sector = row.get('Sector', 'Unknown')
                if pd.notna(sector) and row['Bench Weight'] > 0:
                    benchmark_sector_weights[sector] = benchmark_sector_weights.get(sector, 0) + row['Bench Weight']

            # Create sector comparison DataFrame
            all_sectors = sorted(set(sector_weights.keys()) | set(benchmark_sector_weights.keys()))
            sector_comparison = pd.DataFrame({
                "Sector": all_sectors,
                "Portfolio (%)": [sector_weights.get(s, 0) for s in all_sectors],
                "Benchmark (%)": [benchmark_sector_weights.get(s, 0) for s in all_sectors]
            })
            sector_comparison["Difference"] = sector_comparison["Portfolio (%)"] - sector_comparison["Benchmark (%)"]
            sector_comparison = sector_comparison.sort_values(by="Portfolio (%)", ascending=False)

            # Build analysis dataframe for stocks in the optimized portfolio
            analysis_data = []
            for ticker, port_weight in new_portfolio.items():
                bench_weight = ticker_to_bench.get(ticker, 0)
                deviation = port_weight - bench_weight
                active_share_contrib = abs(deviation) / 2  # Each stock's contribution to active share
                core_rank = ticker_to_core_rank.get(ticker, 999)
                analysis_data.append({
                    "Ticker": ticker,
                    "Portfolio (%)": port_weight,
                    "Benchmark (%)": bench_weight,
                    "Deviation": deviation,
                    "Active Share Contrib": active_share_contrib,
                    "Core Rank": core_rank if core_rank != 999 else None
                })
            analysis_df = pd.DataFrame(analysis_data)

            # ============ TABLES SECTION ============

            # Display portfolio holdings
            st.subheader("Optimized Portfolio Holdings")

            # Convert portfolio to DataFrame for better display
            portfolio_df = pd.DataFrame(list(new_portfolio.items()), columns=["Ticker", "Weight (%)"])
            portfolio_df = portfolio_df.sort_values(by="Weight (%)", ascending=False)

            st.dataframe(
                portfolio_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Weight (%)": st.column_config.NumberColumn(
                        "Weight (%)",
                        help="Position weight as a percentage"
                    )
                },
                hide_index=True,
                use_container_width=True
            )

            # Display new additions
            st.subheader("New Additions to Portfolio")

            new_additions = []
            for stock in added_stocks:
                ticker = stock['ticker']
                weight = stock.get('new_weight', new_portfolio.get(ticker, 0))
                sector = stock.get('sector', '')
                subsector = stock.get('subsector', '')
                core_rank = stock.get('core_rank', 'N/A')
                bench_weight = stock.get('bench_weight', 0)

                new_additions.append({
                    "Ticker": ticker,
                    "Weight (%)": weight,
                    "Benchmark Weight (%)": bench_weight,
                    "Sector": sector,
                    "Subsector": subsector,
                    "Core Rank": core_rank
                })

            additions_df = pd.DataFrame(new_additions)

            sector_col = None
            for col in additions_df.columns:
                if 'sector' in col.lower() and not additions_df[col].isna().all():
                    sector_col = col
                    break

            if sector_col and len(additions_df) > 0:
                additions_df = additions_df.sort_values(by=[sector_col, "Weight (%)"], ascending=[True, False])
            elif len(additions_df) > 0:
                additions_df = additions_df.sort_values(by="Weight (%)", ascending=False)

            st.dataframe(
                additions_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Weight (%)": st.column_config.NumberColumn("Weight (%)"),
                    "Sector": st.column_config.TextColumn("Sector"),
                    "Subsector": st.column_config.TextColumn("Subsector"),
                    "Core Rank": st.column_config.NumberColumn("Core Rank"),
                    "Benchmark Weight (%)": st.column_config.NumberColumn("Benchmark Weight (%)")
                },
                hide_index=True,
                use_container_width=True
            )

            # Display sector breakdown tables
            st.subheader("📊 Sector Breakdown")

            sector_df = pd.DataFrame(list(sector_weights.items()), columns=["Sector", "Weight (%)"])
            sector_df = sector_df.sort_values(by="Weight (%)", ascending=False)

            subsector_df = pd.DataFrame(list(subsector_weights.items()), columns=["Subsector", "Weight (%)"])
            subsector_df = subsector_df.sort_values(by="Weight (%)", ascending=False)

            st.write("By Sector:")
            st.dataframe(
                sector_df,
                column_config={
                    "Sector": st.column_config.TextColumn("Sector"),
                    "Weight (%)": st.column_config.NumberColumn("Weight (%)")
                },
                hide_index=True,
                use_container_width=True
            )

            st.write("By Subsector:")
            st.dataframe(
                subsector_df,
                column_config={
                    "Subsector": st.column_config.TextColumn("Subsector"),
                    "Weight (%)": st.column_config.NumberColumn("Weight (%)")
                },
                hide_index=True,
                use_container_width=True
            )

            # ============ CHARTS SECTION ============
            st.subheader("📈 Portfolio Analysis Charts")

            # 1. Portfolio vs Benchmark by Sector
            st.write("**Portfolio vs Benchmark by Sector:**")
            chart_data = sector_comparison[["Sector", "Benchmark (%)", "Portfolio (%)"]].melt(
                id_vars="Sector", var_name="Type", value_name="Weight (%)"
            )
            sector_order = sector_comparison.sort_values("Portfolio (%)")["Sector"].tolist()
            sector_chart = alt.Chart(chart_data).mark_bar().encode(
                y=alt.Y("Sector:N", sort=sector_order, title=None),
                x=alt.X("Weight (%):Q"),
                yOffset="Type:N",
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Benchmark (%)", "Portfolio (%)"], range=["#1f77b4", "#2ca02c"]))
            ).properties(height=450)
            st.altair_chart(sector_chart, use_container_width=True)

            # 2. Active Share Contribution Chart
            st.write("**Top Active Share Contributors:**")
            top_contributors = analysis_df.nlargest(15, "Active Share Contrib")[["Ticker", "Active Share Contrib", "Deviation", "Portfolio (%)", "Benchmark (%)"]]

            contrib_chart = alt.Chart(top_contributors).mark_bar().encode(
                y=alt.Y("Ticker:N", sort=alt.EncodingSortField(field="Active Share Contrib", order="descending"), title=None),
                x=alt.X("Active Share Contrib:Q", title="Contribution to Active Share (%)"),
                color=alt.condition(
                    alt.datum.Deviation > 0,
                    alt.value("#2ca02c"),
                    alt.value("#d62728")
                ),
                tooltip=[
                    alt.Tooltip("Ticker:N"),
                    alt.Tooltip("Portfolio (%):Q", title="Portfolio Weight"),
                    alt.Tooltip("Benchmark (%):Q", title="Benchmark Weight"),
                    alt.Tooltip("Deviation:Q", title="Over/Underweight"),
                    alt.Tooltip("Active Share Contrib:Q", title="Active Share Contrib")
                ]
            ).properties(height=400)
            st.altair_chart(contrib_chart, use_container_width=True)

            # 4. Position Size Distribution
            st.write("**Position Size Distribution:**")
            position_bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 100]
            position_labels = ["0-0.5%", "0.5-1%", "1-1.5%", "1.5-2%", "2-2.5%", "2.5-3%", "3-4%", "4-5%", ">5%"]
            analysis_df["Size Bucket"] = pd.cut(analysis_df["Portfolio (%)"], bins=position_bins, labels=position_labels, right=False)
            size_dist = analysis_df.groupby("Size Bucket", observed=True).size().reset_index(name="Count")

            size_chart = alt.Chart(size_dist).mark_bar(color="#1f77b4").encode(
                x=alt.X("Size Bucket:N", sort=position_labels, title="Position Size", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Number of Positions"),
                tooltip=["Size Bucket", "Count"]
            ).properties(height=300)
            st.altair_chart(size_chart, use_container_width=True)

            # 5. Core Model Rank Distribution
            st.write("**Core Model Rank Distribution:**")
            ranked_df = analysis_df[analysis_df["Core Rank"].notna()].copy()
            if len(ranked_df) > 0:
                ranked_df["Core Rank"] = ranked_df["Core Rank"].astype(int)
                rank_dist = ranked_df.groupby("Core Rank", observed=True).size().reset_index(name="Count")
                rank_dist = rank_dist.sort_values("Core Rank")

                rank_chart = alt.Chart(rank_dist).mark_bar(color="#9467bd").encode(
                    x=alt.X("Core Rank:O", title="Core Model Rank", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q", title="Number of Holdings"),
                    tooltip=["Core Rank", "Count"]
                ).properties(height=300)
                st.altair_chart(rank_chart, use_container_width=True)
            else:
                st.info("No Core Model rank data available for portfolio holdings.")

if __name__ == "__main__":
    main()
