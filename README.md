# 🚇 London Tube Crowding Predictor

## Project Overview

This project develops a machine learning model to predict crowding levels on specific links (segments between stations) of the London Underground network. The goal is to provide real-time crowding predictions to help commuters make informed travel decisions. The solution includes data processing, feature engineering, model training, and an interactive Streamlit web application for predictions.

## Features

* **Data Ingestion & Preprocessing:** Handles raw TfL NUMBAT data, transforming it into a clean, long-format dataset.
* **Intelligent Crowding Definition:** Defines crowding levels (`Low`, `Medium`, `High`, `Very High`) based on line-specific percentiles of "Load Per Train," providing a nuanced and data-driven target variable.
* **Feature Engineering:** Extracts crucial time-based features (`quarter_of_day_index`, `DayOfWeek`) and leverages geographical/structural features (`Order`, `From NLC`, `To NLC`, `Line`, `Dir`).
* **Robust Machine Learning Model:** Utilizes a tuned `RandomForestClassifier` to predict crowding levels, optimized using Optuna for superior performance, especially on minority (crowded) classes.
* **Interactive Web Application:** A user-friendly Streamlit app allows users to input specific journey details (Line, Stations, Time) and receive real-time crowding predictions.
* **Automated Inputs:** The Streamlit app intelligently determines `Direction`, `To Station`, and `Link Order` based on user selections for a streamlined experience.

## Data Source

The project utilizes the **Transport for London (TfL) NUMBAT dataset**, which provides detailed statistics on usage and travel patterns across the London Underground, London Overground, Docklands Light Railway, and Elizabeth Line. The data covers 15-minute intervals throughout the traffic day.

*(Note: Specific raw data files are not included in the repository if they are large or proprietary, but the processing scripts expect them in the `data/raw/crowding/` directory.)*

## Project Structure


TubeCrowdPredictor/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── crowding/       # Original raw data files (e.g., NBT23FRI_outputs.xlsx - Link_Loads.csv)
│   ├── processed/      # Intermediate processed files
│   └── final/          # Final processed data, encoders, and mappings
│       ├── extracted_nlc_station_mapping.csv
│       ├── final_merged_link_data_cleaned.csv
│       ├── data_encoded.csv
│       ├── onehot_encoder.pkl
│       └── ordinal_encoder.pkl
├── models/             # Trained machine learning models
│   └── random_forest_model.pkl
├── src/                # Python scripts for data processing and model training
│   ├── 01_raw_data_ingestion.py
│   ├── 02_data_transformation_and_quarter_index.py
│   ├── 03_link_data_merge_and_crowding_definition.py
│   ├── 04_station_nlc_mapping_extraction.py
│   ├── 05_final_data_preparation_and_encoding.py
│   └── 06_model_training_and_tuning.py
├── app/
│   └──   app.py        # Streamlit web application
└── setup.sh            # (Optional) Script for environment setup


## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Sanjit-Jeevanand/TubeCrowdPredictor.git](https://github.com/Sanjit-Jeevanand/TubeCrowdPredictor.git)
    cd TubeCrowdPredictor
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Obtain Raw Data:**
    Place your raw TfL NUMBAT Excel output files (e.g., `NBT23FRI_outputs.xlsx - Link_Loads.csv`, etc., for all days of the week) into the `data/raw/crowding/` directory.

## How to Run the Project

Follow these steps to process the data, train the model, and launch the Streamlit application.

1.  **Data Processing & Encoding:**
    Run the `data_processing.py` script to clean, transform, and prepare the data, and to save the necessary encoders and mappings.
    ```bash
    python src/01_raw_data_ingestion.py
    python src/02_data_transformation_and_quarter_index.py
    python src/03_link_data_merge_and_crowding_definition.py
    python src/04_station_nlc_mapping_extraction.py
    python src/05_final_data_preparation_and_encoding.py
    ```
    *These scripts will generate files in `data/processed/` and `data/final/`.*

2.  **Model Training & Tuning:**
    Run the `model_training.py` script to train the RandomForest model with Optuna-tuned hyperparameters and save the best model.
    ```bash
    python src/06_model_training_and_tuning.py
    ```
    *This script will save `random_forest_model.pkl` in the `models/` directory.*

3.  **Launch the Streamlit Application:**
    ```bash
    streamlit run app/app.py
    ```
    *This will open the application in your web browser.*

## Project Methodology (Phases)

The project was structured into five distinct phases:

* **Phase 1: Data Exploration and Understanding:** Initial data review, transformation from wide to long format, and definition of line-specific crowding levels based on `Load_Per_Train` percentiles.
* **Phase 2: Data Cleaning and Preprocessing:** Aggregation of core link data, handling of missing values, dropping of redundant identifiers, and categorical encoding (One-Hot and Ordinal) with encoder persistence.
* **Phase 3: Feature Engineering:** Focus on `Order`, `From NLC`, `To NLC`, `quarter_of_day_index`, `Line`, `Dir`, and `DayOfWeek` as the core predictive features available at inference time. (More advanced features like lags and rolling statistics were considered for future work due to real-time availability constraints).
* **Phase 4: Model Building and Evaluation:** Selection and training of a `RandomForestClassifier`. Hyperparameter tuning using Optuna and addressing class imbalance with `class_weight='balanced'` led to significant performance improvements (Accuracy: 0.92, Macro F1: 0.85).
* **Phase 5: Prediction Tool and Deployment:** Development of an interactive Streamlit application for real-time crowding predictions, demonstrating practical model deployment.

## Model Performance (Tuned RandomForest)

| Metric            | Score    |
| :---------------- | :------- |
| Accuracy          | 0.9202   |
| Macro Avg F1-Score | 0.85     |
| Weighted Avg F1-Score | 0.92     |

**Classification Report:**

| Class     | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| Low       | 0.98      | 0.96   | 0.97     | 111323  |
| Medium    | 0.77      | 0.83   | 0.80     | 23098   |
| High      | 0.80      | 0.83   | 0.81     | 17322   |
| Very High | 0.78      | 0.88   | 0.83     | 5774    |

## Future Enhancements

* **Integrate Auxiliary Data:** Incorporate `Station_Entries`, `Station_Exits`, `Station_Flows`, `Line_Boarders`, etc., to enrich the feature set and potentially further improve model accuracy.
* **Advanced Feature Engineering:** Explore lag features (carefully considering real-time availability), rolling statistics, and external data (weather, events, disruptions).
* **Model Optimization:** Experiment with other advanced models like LightGBM or CatBoost, and fine-tune prediction thresholds.
* **Deployment Scaling:** Deploy the application to cloud platforms (e.g., Streamlit Cloud, AWS, GCP) for wider accessibility.
* **User Interface Improvements:** Add more visualizations within the app, user feedback mechanisms, and historical data insights.

