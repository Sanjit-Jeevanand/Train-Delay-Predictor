import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Define the final data path (relative to the project root, assuming this script is in src/)
FINAL_DATA_PATH = "./data/final/"
FINAL_ENCODED_FILE_NAME = "data_encoded.csv" # The final encoded data from 05_final_data_preparation_and_encoding.py
MODEL_PATH = "./models/" # Directory to save the trained model

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# --- Main script execution ---
if __name__ == "__main__":
    print("\n--- Step 6: Model Training and Tuning ---")

    # --- 1. Load the encoded dataset ---
    print("\n--- Loading the Final Encoded Data for Model Training ---")
    try:
        final_data_path_full = os.path.join(FINAL_DATA_PATH, FINAL_ENCODED_FILE_NAME)
        df_ml = pd.read_csv(final_data_path_full)
        print(f"Encoded dataset '{FINAL_ENCODED_FILE_NAME}' loaded successfully.")
        print("Loaded Data shape:", df_ml.shape)
    except FileNotFoundError:
        print(f"Error: '{FINAL_ENCODED_FILE_NAME}' not found at {final_data_path_full}.")
        print("Please ensure '05_final_data_preparation_and_encoding.py' was run successfully.")
        exit()
    except Exception as e:
        print(f"An error occurred loading the data: {e}")
        exit()

    # --- 2. Handle Missing Values (Imputation) ---
    # This is a safety check for any remaining NaNs in numerical features.
    print("\n--- Handling Missing Values (Imputation) ---")
    numerical_cols_with_nan = df_ml.select_dtypes(include=np.number).columns[df_ml.select_dtypes(include=np.number).isnull().any()].tolist()
    
    for col in numerical_cols_with_nan:
        print(f"Imputing NaNs in '{col}' with its median.")
        df_ml[col].fillna(df_ml[col].median(), inplace=True)

    print("NaN counts after imputation:")
    print(df_ml.isnull().sum()[df_ml.isnull().sum() > 0]) # Show any remaining NaNs

    # --- 3. Define Features (X) and Target (y) ---
    print("\n--- Defining Features (X) and Target (y) ---")

    # Target variable
    TARGET_COLUMN = 'Crowding_Level'

    # Features (X): All columns except the target and the Load/Freq related columns
    # This list MUST match the 'features_excluded_from_X_in_training' in 05_final_data_preparation_and_encoding.py
    features_to_exclude = [
        'Crowding_Level', 
        'Load_Value', 'Load_Per_Train', 'Total_load', 'Frequency_Value', 'Total_freq'
    ] 
    
    X = df_ml.drop(columns=features_to_exclude, errors='ignore')
    y = df_ml[TARGET_COLUMN]

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"First 5 feature columns: {X.columns[:5].tolist()} ...")

    # Ensure y is integer type for classification (after ordinal encoding)
    if y.dtype == 'float64':
        y = y.astype(int)
    
    # --- 4. Train-Test Split ---
    print("\n--- Splitting Data into Training and Testing Sets (80/20 split) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- 5. Model Training with Optuna Tuned Hyperparameters ---
    print("\n--- Training RandomForestClassifier Model with Optuna Tuned Hyperparameters ---")
    
    # Define the best parameters found by Optuna (from previous Optuna run output)
    # These parameters are specific to the RandomForest model trained on the features available at prediction time.
    tuned_params = {
        'n_estimators': 343, 
        'max_features': 1.0, 
        'max_depth': 37, 
        'min_samples_split': 7, 
        'min_samples_leaf': 1, 
        'bootstrap': True
    }

    # Instantiate the model with the tuned parameters
    model = RandomForestClassifier(
        **tuned_params, 
        random_state=42, 
        class_weight='balanced', # Keep class_weight for imbalance
        n_jobs=-1 # Use all available CPU cores
    )
    
    model.fit(X_train, y_train)
    print("RandomForestClassifier model trained successfully with tuned parameters.")

    # --- 6. Model Evaluation ---
    print("\n--- Evaluating Tuned Model Performance ---")
    y_pred = model.predict(X_test)

    print("\nAccuracy Score:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    target_names = ['Low', 'Medium', 'High', 'Very High']
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Tuned RandomForest)')
    plt.show()

    print("\n--- Feature Importance ---")
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_15_features = feature_importances.nlargest(15)
    print(top_15_features)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_15_features.values, y=top_15_features.index, palette='viridis')
    plt.title('Top 15 Feature Importances (Tuned RandomForest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    print("\n--- Tuned RandomForest Model Building and Evaluation Complete ---")

    # --- Save the tuned model ---
    # This will be the 'random_forest_model.pkl' loaded by the Streamlit app.
    joblib.dump(model, os.path.join(MODEL_PATH, 'random_forest_model.pkl'))
    print(f"\nTuned RandomForest model saved to: {os.path.join(MODEL_PATH, 'random_forest_model.pkl')}")

else:
    print("DataFrame is empty, skipping ML model building.")
    print("\n--- Final Encoded DataFrame saved successfully. ---")