# Active Share Optimizer

An interactive portfolio optimization tool that helps portfolio managers adjust their stock holdings to achieve a target Active Share while respecting sector, position size, and other constraints. Powered by PuLP and the CBC Solver for mixed-integer linear programming optimization.

## 🌟 Features

- **Portfolio Analysis**: Load your current portfolio, benchmark weights, and constraints from a single Excel file
- **Ticker Locking**: Lock specific stocks to maintain their exact current weights
- **Interactive UI**: Adjust all parameters through an intuitive Streamlit web interface
- **Advanced Optimization**: Uses mixed-integer linear programming for optimal stock selection
- **Position Size Controls**: Choose between continuous weights or discrete increments
- **Forced Holdings**: Force specific stocks to be included within custom weight ranges
- **Flexible Timeout**: Control how long the optimizer runs to balance speed and solution quality
- **Detailed Reports**: Export comprehensive Excel reports with portfolio, sector, and subsector analysis

## 📋 Requirements

- Python 3.8 or higher
- CBC Solver (COIN-OR Branch and Cut solver)
- Python packages: streamlit, pandas, numpy, pulp, openpyxl

## 🚀 Installation

### Mac Installation

1. **Install Python (if not already installed)**
   ```bash
   brew install python@3.11
   ```

2. **Install CBC Solver**
   ```bash
   brew install cbc
   ```

3. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ActiveShareOptimizer.git
   cd ActiveShareOptimizer
   ```

4. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

5. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Windows Installation

1. **Install Python**
   - Download and install Python from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Install CBC Solver**
   - Download the latest CBC binary from [GitHub Releases](https://github.com/coin-or/Cbc/releases)
   - Extract the ZIP file to a location like `C:\Program Files\CBC`
   - Add the CBC binary directory to your system PATH:
     - Open "Edit the system environment variables" from Control Panel
     - Click "Environment Variables"
     - Edit the "Path" variable and add the CBC bin directory (e.g., `C:\Program Files\CBC\bin`)

3. **Clone the repository**
   - Download and install Git from [git-scm.com](https://git-scm.com/download/win)
   - Open Command Prompt or PowerShell and run:
   ```
   git clone https://github.com/yourusername/ActiveShareOptimizer.git
   cd ActiveShareOptimizer
   ```

4. **Create a virtual environment (recommended)**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

5. **Install Python dependencies**
   ```
   pip install -r requirements.txt
   ```

### Alternative Installation: Using Conda (Mac or Windows)

1. **Install Miniconda**
   - Download and install from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)

2. **Create and activate a conda environment**
   ```bash
   conda create -n activeshare python=3.11
   conda activate activeshare
   ```

3. **Install CBC solver and dependencies**
   ```bash
   conda install -c conda-forge coincbc
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application

1. **Start the web app**
   ```bash
   # For Mac/Linux
   streamlit run app.py
   
   # For Windows
   python -m streamlit run app.py
   ```

2. **Open your browser** to the URL shown in the terminal (usually http://localhost:8501)

3. **Upload your file**:
   - Optimizer Input Excel (e.g., `optimizer_input_file.xlsm`)

4. **Adjust parameters in the sidebar** and click "Run Optimizer"

## 📊 Input File Format

### Optimizer Input Excel Format
Your Excel file (`optimizer_input_file.xlsm`) should include:

#### Main Sheet:
- `Company Name` - Stock name
- `Ticker` - Stock ticker symbol
- `Portfolio Weight` - Current weight in your portfolio (%)
- `Bench Weight` - Weight in the benchmark (%)
- `Sector` - Stock sector
- `Sector-and-Subsector` - More detailed sector/industry classification
- `Core Model` - Ranking or scoring of stocks (lower is better)
- `Lock ticker-and-weight` - Add 'Y' to maintain the current weight for specific tickers

#### Optional Constraints Sheet:
- `Stocks to Avoid` column - List of tickers to exclude
- `Emp Sector & Industry` column - Sector constraints to target
- `Weight` column - Target weight for the sector

## 📊 Ticker Locking Feature

The new "Lock ticker-and-weight" functionality allows you to:

1. Mark specific stocks with 'Y' in the "Lock ticker-and-weight" column
2. The optimizer will maintain the exact current weight for these stocks
3. This is useful for positions you don't want to buy or sell
4. Locked tickers override other constraints (they will be included regardless of Core Model ranking)

## ⚙️ Optimizer Parameters

| Parameter | Description |
|-----------|-------------|
| **Total Positions** | Exact number of positions required in the portfolio |
| **Target Active Share (%)** | Minimum difference from benchmark (higher = more different) |
| **Sector Tolerance (%)** | How much sector weights can deviate from targets |
| **Min/Max Position Size (%)** | Bounds for individual stock position sizes |
| **Max Core Model Rank** | Only include stocks with Core rank <= this value |
| **Position Size Mode** | Choose continuous weights or discrete increments |
| **Position Increment Size (%)** | Step size for position sizing (if using discrete mode) |
| **Solver Timeout (seconds)** | Maximum time allowed for optimization |

## 🗂️ Directory Structure

```
ActiveShareOptimizer/
├── app.py                   # Streamlit web interface
├── run_optimizer.py         # Command-line entry point
├── optimizer/               # Core optimizer package
│   ├── __init__.py          # Package exports
│   ├── main.py              # Main orchestration logic
│   ├── data/                # Data loading modules
│   │   ├── __init__.py
│   │   └── loaders.py       # Functions to load portfolio data
│   ├── models/              # Optimization models
│   │   ├── __init__.py
│   │   └── optimizer.py     # PuLP optimization implementation
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── calculations.py  # Active share calculations
│       └── reporting.py     # Results reporting functions
├── requirements.txt         # Python dependencies
├── README.md                # This documentation
├── inputs/                  # Example input files
│   └── optimizer_input_file.xlsm   # Consolidated input file with all data
└── outputs/                 # Generated optimization results
    └── Optimized_Portfolio_PuLP_*.xlsx
```

## 🔧 Troubleshooting

### Common Issues

1. **CBC Solver not found**
   - Make sure CBC is installed and in your system PATH
   - For Windows, check that you added the CBC bin directory to the PATH
   - For macOS, verify installation with `brew info cbc`

2. **Infeasible solution**
   - Try increasing sector tolerance
   - Lower the target active share
   - Adjust or remove forced positions
   - Check if the combination of constraints is too restrictive

3. **Streamlit interface not loading**
   - Check if port 8501 is already in use
   - Try running with a different port: `streamlit run app.py --server.port 8502`

4. **Slow optimization**
   - Increase the timeout setting for complex problems
   - Consider reducing the number of constraints
   - Simplify sector constraints

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.