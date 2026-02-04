#!/bin/bash
set -e

VENV_NAME="ml-predictor"
PYTHON_VERSION="python3"
START_TIME=$(date +%s)

# colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# box
BOX_H="━"
BOX_V="┃"
BOX_TL="┏"
BOX_TR="┓"
BOX_BL="┗"
BOX_BR="┛"

# header box
print_header() {
    local text="$1"
    local width=60
    echo -e "${CYAN}${BOX_TL}$(printf "${BOX_H}%.0s" {1..58})${BOX_TR}${NC}"
    printf "${CYAN}${BOX_V}${NC} ${BOLD}%-56s${NC} ${CYAN}${BOX_V}${NC}\n" "$text"
    echo -e "${CYAN}${BOX_BL}$(printf "${BOX_H}%.0s" {1..58})${BOX_BR}${NC}"
    echo ""
}

# header
print_step() {
    local step="$1"
    local desc="$2"
    echo -e "${BLUE}━━━ ${BOLD}[$step]${NC} ${BLUE}$desc${NC}"
}

# clear 
clear
print_header "Nutri-Score Preprocessing Pipeline"

# 1. py check version
print_step "1/5" "Checking Python environment"
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${YELLOW}      Python 3 not found${NC}"
    exit 1
fi
echo -e "      ${GREEN}${NC} $($PYTHON_VERSION --version)"
echo ""

# 2. venv setup
print_step "2/5" "Setting up virtual environment"
if [ -d "$VENV_NAME" ]; then
    echo -e "      ${GREEN}${NC} Using existing venv '${VENV_NAME}'"
else
    $PYTHON_VERSION -m venv $VENV_NAME
    echo -e "      ${GREEN}${NC} Created venv '${VENV_NAME}'"
fi
echo ""

# 3. dependency install
print_step "3/5" "Installing dependencies"
source $VENV_NAME/bin/activate

# uv
if ! command -v uv &> /dev/null; then
    pip install --upgrade pip --quiet
    pip install uv --quiet
fi

# uv use
uv pip install -e . --quiet 2>&1 | grep -v "Running setup.py" || true
echo -e "      ${GREEN}${NC} Dependencies installed"
echo ""

# 4. data download
print_step "4/5" "Downloading Open Food Facts data"
python scripts/download_data.py
echo ""

# 5. preprocessing
print_step "5/5" "Running preprocessing pipeline"
python scripts/run_preprocessing.py
echo ""

# summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${GREEN}${BOLD}Pipeline Complete${NC}"
echo -e "  Time: ${BOLD}${MINUTES}m ${SECONDS}s${NC}"
echo -e "  Output: ${BOLD}data/splits/${NC}"
echo ""
