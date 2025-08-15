#!/bin/bash

# ARIMA Backend Startup Script
echo "🚀 Starting ARIMA Forecasting Backend..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "🐍 Detected Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "arima_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv arima_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source arima_env/bin/activate

# Upgrade pip first
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import flask, statsmodels, pandas, numpy; print('All packages installed successfully!')"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export FLASK_ENV=development
export DEBUG=True

# Start the service
echo "🎯 Starting ARIMA service on http://localhost:8080"
python app.py
