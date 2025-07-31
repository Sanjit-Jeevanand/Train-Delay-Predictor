#!/bin/bash

echo "Setting up Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Ensuring necessary data directories exist..."
mkdir -p data/raw/crowding
mkdir -p data/processed
mkdir -p data/final
mkdir -p models

echo "Setup complete. To activate the environment, run 'source venv/bin/activate'."
echo "To run the data processing pipeline, run the scripts in src/ in order (e.g., 'python src/01_raw_data_ingestion.py')."
echo "To run the Streamlit app, run 'streamlit run app.py' from the project root."