#!/bin/bash

# Helper script to set up Python 3.11 for FaceFusion
# This is recommended for better package compatibility

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Setting up Python 3.11 for FaceFusion...${NC}"
echo ""

if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}Error: pyenv is not installed${NC}"
    echo -e "${YELLOW}Install pyenv with: brew install pyenv${NC}"
    exit 1
fi

# Check if Python 3.11 is already installed
if pyenv versions | grep -q "3.11"; then
    echo -e "${GREEN}Python 3.11 is already installed${NC}"
    PYTHON311_VERSION=$(pyenv versions | grep "3.11" | head -1 | awk '{print $1}' | tr -d '* ')
    echo -e "${GREEN}Found: ${PYTHON311_VERSION}${NC}"
else
    echo -e "${YELLOW}Installing Python 3.11.14...${NC}"
    pyenv install 3.11.14
    echo -e "${GREEN}✓ Python 3.11.14 installed${NC}"
fi

# Set local Python version for this project
echo -e "${YELLOW}Setting Python 3.11 as local version for this project...${NC}"
cd "$(dirname "$0")"
pyenv local 3.11.14
echo -e "${GREEN}✓ Python 3.11.14 set as local version${NC}"
echo ""
echo -e "${GREEN}Now run: ./setup_and_run.sh${NC}"

