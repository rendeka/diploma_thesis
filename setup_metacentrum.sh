#!/bin/bash

module add python/python-3.10.4-intel-19.0.4-sc7snnf

VENV_PATH=~/venv

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python3 -m venv $VENV_PATH
else
    echo "Virtual environment already exists at $VENV_PATH."
fi

source $VENV_PATH/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir --upgrade pip setuptools
pip install --no-cache-dir -r requirements.txt
