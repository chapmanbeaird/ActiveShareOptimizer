#!/usr/bin/env bash
# Setup script for Active Share Optimizer on Mac/Linux
# Creates a Python 3.11 virtual environment, installs all dependencies, and bootstraps the CBC solver.

set -euo pipefail

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Check for Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}Python 3.11 is not installed.${NC}"
    echo -e "Please install Python 3.11 before continuing."
    echo -e "  • On macOS: ${YELLOW}brew install python@3.11${NC}"
    echo -e "  • On Linux: use your distro's package manager or pyenv"
    exit 1
fi

echo -e "${YELLOW}Creating Python 3.11 virtual environment…${NC}"
python3.11 -m venv activeshare_env_py311

echo -e "${YELLOW}Activating the environment…${NC}"
# shellcheck disable=SC1091
source activeshare_env_py311/bin/activate

echo -e "${YELLOW}Upgrading pip & installing Python packages…${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}Installing CBC solver…${NC}"
if command -v brew &> /dev/null; then
    brew install cbc
elif command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y coinor-cbc
else
    echo -e "${RED}Could not find Homebrew or apt-get. Please install the CBC solver manually.${NC}"
    exit 1
fi

echo -e "${YELLOW}Verifying installation…${NC}"
python - <<PYCODE
import pulp
print("PuLP version:", pulp.__version__)
print("CBC available:", pulp.PULP_CBC_CMD().available())
PYCODE

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "To activate: ${YELLOW}source activeshare_env_py311/bin/activate${NC}"
echo -e "To run the app: ${YELLOW}streamlit run app.py${NC}"
