#!/usr/bin/env bash
set -e

# NOTE: dm_control requires MuJoCo. See https://github.com/google-deepmind/dm_control
# for installation instructions before running this script.

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. To activate the environment in your shell, run:"
echo "  source .venv/bin/activate"
