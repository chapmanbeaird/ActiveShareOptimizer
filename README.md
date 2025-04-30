# Active Share Optimizer

This project is an **interactive portfolio optimizer** that helps you adjust your stock portfolio to achieve a target Active Share while respecting sector, position size, and other constraints. It provides an easy-to-use **Streamlit web interface** for uploading your portfolio and constraints, tuning parameters, and downloading results.

---

## What does this code do?
- **Loads your current portfolio** and benchmark weights from a CSV file.
- **Loads constraints** (like stocks to avoid and sector targets) from an Excel file.
- **Lets you interactively adjust parameters** (like target Active Share, sector tolerance, position size, etc.) in a web app.
- **Optimizes the portfolio** using linear programming (PuLP) to meet your constraints and goals.
- **Displays the optimized portfolio** and allows you to download detailed results.

---

## Main Parameters (in the Streamlit sidebar):
- **Max Positions**: Maximum number of stocks allowed in the portfolio.
- **Target Active Share (%)**: How different you want your portfolio to be from the benchmark.
- **Sector Tolerance (%)**: How much each sector weight can deviate from the benchmark.
- **Min/Max Position Size (%)**: Smallest/largest allowed size for any single stock.
- **Max Core Model Rank**: Only include stocks with a core rank less than or equal to this value.
- **Enforce Increment Size?**: If checked, all positions must be multiples of the increment size (e.g., 0.5%). If unchecked, position sizes can be any value within min/max.
- **Position Increment Size (%)**: The step size for allowed position sizes (if increment is enforced).

---

## How to Run the App

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the web app**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** to the URL shown in the terminal (usually http://localhost:8501).

4. **Upload your files**:
   - Portfolio CSV (e.g., `active_share_with_core_constraints.csv`)
   - Constraints Excel (e.g., `stocks_to_avoid&sector_constraints.xlsm`)

5. **Adjust parameters** in the sidebar and click "Run Optimizer".

6. **View and download results**:
   - See the optimized portfolio and added stocks in the app.
   - Download the full results from the `outputs/Optimized_Portfolio_PuLP.xlsx` file.

---

## File Structure
- `streamlit_app.py`: The web interface.
- `active_share_optimizer_pulp.py`: The optimization logic.
- `requirements.txt`: Python dependencies.
- `outputs/`: Folder where results are saved.

---

## Notes
- Make sure your portfolio and constraints files match the expected formats (see example files or code comments).
- All optimization is done locally; no data leaves your computer.

---
