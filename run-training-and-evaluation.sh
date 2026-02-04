#!/bin/bash
set -e

VENV_NAME="ml-predictor"
START_TIME=$(date +%s)

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

BOX_H="━"
BOX_V="┃"
BOX_TL="┏"
BOX_TR="┓"
BOX_BL="┗"
BOX_BR="┛"

print_header() {
    local text="$1"
    echo -e "${CYAN}${BOX_TL}$(printf "${BOX_H}%.0s" {1..68})${BOX_TR}${NC}"
    printf "${CYAN}${BOX_V}${NC} ${BOLD}%-66s${NC} ${CYAN}${BOX_V}${NC}\n" "$text"
    echo -e "${CYAN}${BOX_BL}$(printf "${BOX_H}%.0s" {1..68})${BOX_BR}${NC}"
    echo ""
}

print_step() {
    local step="$1"
    local desc="$2"
    echo -e "${BLUE}━━━ ${BOLD}[$step]${NC} ${BLUE}$desc${NC}"
}

clear
print_header "Nutri-Score Model Training & Evaluation"

if [ ! -d "$VENV_NAME" ]; then
    echo -e "${RED}Error: Virtual environment '${VENV_NAME}' not found.${NC}"
    echo -e "${YELLOW}Please run ./run-preprocessing.sh first.${NC}"
    exit 1
fi

if [ ! -d "data/splits" ]; then
    echo -e "${RED}Error: Preprocessed data not found in data/splits/${NC}"
    echo -e "${YELLOW}Please run ./run-preprocessing.sh first.${NC}"
    exit 1
fi

source $VENV_NAME/bin/activate

print_step "1/3" "Select model to train"
echo ""
echo -e "      ${BOLD}Available models:${NC}"
echo -e "      ${CYAN}1)${NC} Logistic Regression"
echo -e "      ${CYAN}2)${NC} KNN"
echo -e "      ${CYAN}3)${NC} SVM"
echo -e "      ${CYAN}4)${NC} Random Forest"
echo -e "      ${CYAN}5)${NC} XGBoost"
echo ""
echo -e -n "      ${BOLD}Enter choice [1-5]:${NC} "
read choice

case $choice in
    1)
        MODEL_NAME="logistic_regression"
        MODEL_DISPLAY="Logistic Regression"
        ;;
    2)
        MODEL_NAME="knn"
        MODEL_DISPLAY="KNN"
        ;;
    3)
        MODEL_NAME="svm"
        MODEL_DISPLAY="SVM"
        ;;
    4)
        MODEL_NAME="random_forest"
        MODEL_DISPLAY="Random Forest"
        ;;
    5)
        MODEL_NAME="xgboost"
        MODEL_DISPLAY="XGBoost"
        ;;
    *)
        echo -e "${RED}      Invalid choice. Please select 1-5.${NC}"
        exit 1
        ;;
esac

echo -e "      ${GREEN}Selected: ${BOLD}${MODEL_DISPLAY}${NC}"
echo ""

print_step "2/3" "Training ${MODEL_DISPLAY}"
echo ""
python scripts/train_model.py --model $MODEL_NAME
echo ""

print_step "3/3" "Evaluating ${MODEL_DISPLAY}"
echo ""
MODEL_PATH="models/trained/${MODEL_NAME}/${MODEL_NAME}_v1.joblib"
python scripts/evaluate_model.py \
    --model-path $MODEL_PATH \
    --model-type $MODEL_NAME \
    --show-report \
    --show-confusion-matrix
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${GREEN}${BOLD}Workflow Complete${NC}"
echo -e "  Model: ${BOLD}${MODEL_DISPLAY}${NC}"
echo -e "  Time: ${BOLD}${MINUTES}m ${SECONDS}s${NC}"
echo -e "  Saved: ${MODEL_PATH}"
echo ""
echo -e "${GREEN}${BOLD}Next steps:${NC}"
echo -e "  • View results: cat models/trained/${MODEL_NAME}/${MODEL_NAME}_v1_metadata.json"
echo -e "  • Re-evaluate: python scripts/evaluate_model.py --model-path ${MODEL_PATH} --model-type ${MODEL_NAME}"
echo ""
