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

- Python 3.8 or higher (Python 3.11 recommended for best compatibility)
- Python packages: streamlit, pandas, numpy, pulp, coinor-cbc, openpyxl

## 🚀 Installation

### Using Setup Scripts (Recommended)

We provide setup scripts for both Mac/Linux and Windows that will automatically create a Python 3.11 virtual environment and install all dependencies.

#### For Mac/Linux:
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ActiveShareOptimizer.git
   cd ActiveShareOptimizer
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup_mac.sh
   ./setup_mac.sh
   ```

3. **Activate the environment**
   ```bash
   source activeshare_env_py311/bin/activate
   ```

#### For Windows:
1. **Clone the repository**
   ```cmd
   git clone https://github.com/yourusername/ActiveShareOptimizer.git
   cd ActiveShareOptimizer
   ```

2. **Run the setup script**
   ```cmd
   setup_windows.bat
   ```

3. **Activate the environment**
   ```cmd
   activeshare_env_py311\Scripts\activate
   ```

### Manual Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ActiveShareOptimizer.git
   cd ActiveShareOptimizer
   ```

2. **Create a virtual environment with Python 3.11**
   ```bash
   # For Mac/Linux
   python3.11 -m venv activeshare_env_py311
   source activeshare_env_py311/bin/activate
   
   # For Windows
   python3.11 -m venv activeshare_env_py311
   activeshare_env_py311\Scripts\activate
   ```

3. **Install all dependencies (including the CBC solver)**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Alternative Installation: Using Conda

1. **Install Miniconda**
   - Download and install from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)

2. **Create and activate a conda environment**
   ```bash
   conda create -n activeshare python=3.11 coin-or-cbc pulp pandas numpy scipy openpyxl xlrd matplotlib -c conda-forge
   conda activate activeshare
   ```

3. **Install remaining dependencies**
   ```bash
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
├── setup_mac.sh             # Setup script for Mac/Linux
├── setup_windows.bat        # Setup script for Windows
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

1. **Solver issues**
   - The `coinor-cbc` package should automatically install the CBC solver
   - If you encounter solver issues with Python 3.13+, use Python 3.11 instead (see setup scripts)
   - For compatibility issues, try the conda installation method

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