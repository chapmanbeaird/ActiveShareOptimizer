import streamlit as st
import pandas as pd
import io
from active_share_optimizer_pulp import (
    load_portfolio_data_csv,
    load_constraints,
    optimize_portfolio_pulp
)

def main():
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
    enforce_increment = st.sidebar.checkbox("Enforce Increment Size?", value=True)
    increment = None
    if enforce_increment:
        increment = st.sidebar.slider("Position Increment Size (%)", min_value=0.01, max_value=1.0, value=0.5, step=0.01, help="All position sizes will be multiples of this increment between the min and max.")
    if st.button("Run Optimizer"):
        if not portfolio_file or not constraints_file:
            st.error("Please upload both a portfolio CSV and a constraints Excel file.")
            return
        stocks_data, total_active_share = load_portfolio_data_csv(portfolio_file)
        stocks_to_avoid, sector_constraints = load_constraints(constraints_file)
        optimized_portfolio, added_stocks, new_active_share = optimize_portfolio_pulp(
            stocks_data,
            total_active_share,
            max_positions=max_positions,
            target_active_share=target_active_share/100.0,  # convert to fraction
            sector_tolerance=sector_tolerance/100.0,        # convert to fraction
            stocks_to_avoid=stocks_to_avoid,
            sector_constraints=sector_constraints,
            min_position=min_position,
            max_position=max_position,
            core_rank_limit=core_rank_limit,
            increment=increment
        )
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

        st.info("For a full breakdown of the optimized portfolio, sector, and subsector analysis, please check the output Excel file in the 'outputs' folder: 'Optimized_Portfolio_PuLP.xlsx'.")

if __name__ == "__main__":
    main()
