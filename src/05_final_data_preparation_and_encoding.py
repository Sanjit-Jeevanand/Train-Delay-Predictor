import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import joblib # Import joblib to save/load models and transformers

# Define paths (relative to the project root, assuming this script is in src/)
PROCESSED_DATA_PATH = "./data/processed/"
FINAL_DATA_PATH = "./data/final/"

# Ensure final data directory exists
os.makedirs(FINAL_DATA_PATH, exist_ok=True)

# --- Main script execution ---
if __name__ == "__main__":
    print("\n--- Step 5: Final Data Preparation and Encoding ---")

    # --- Load the latest processed DataFrame (output from 03_link_data_merge_and_crowding_definition.py) ---
    print("\n--- Loading Final Merged Link Data for Final Preparation ---")
    try:
        # This file now contains 'Crowding_Level' and has already dropped 'From ASC', 'To ASC'.
        # It also has Load_Per_Train (imputed) and Crowding_Level (filled).
        file_to_load = os.path.join(PROCESSED_DATA_PATH, 'final_merged_link_data_with_crowding_percentile.csv')
        df_processed = pd.read_csv(file_to_load)
        print(f"Dataset '{os.path.basename(file_to_load)}' loaded successfully.")
        print("Loaded Data shape:", df_processed.shape)
    except FileNotFoundError:
        print(f"Error: '{os.path.basename(file_to_load)}' not found.")
        print("Please ensure '03_link_data_merge_and_crowding_definition.py' ran correctly.")
        exit()
    except Exception as e:
        print(f"An error occurred loading the data: {e}")
        exit()
    df_processed.drop(columns=['Link_load', 'Link_freq'], inplace=True, errors='ignore')  # Drop Link_load if it exists
    output_file_name = 'data.csv' 
    output_path_full = os.path.join(FINAL_DATA_PATH, output_file_name)
    df_processed.to_csv(output_path_full, index=False)
    print(f"\nDataFrame saved to {output_path_full}")

    # --- Handle Missing Values (Imputation) for any remaining NaNs ---
    # This is a safety check, as most should be handled by previous steps.
    print("\n--- Handling remaining Missing Values (Imputation) ---")
    numerical_cols_with_nan = df_processed.select_dtypes(include=np.number).columns[df_processed.select_dtypes(include=np.number).isnull().any()].tolist()
    
    for col in numerical_cols_with_nan:
        print(f"Imputing NaNs in '{col}' with its median.")
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

    print("NaN counts after imputation:")
    print(df_processed.isnull().sum()[df_processed.isnull().sum() > 0])


    # --- Drop 'From Station' and 'To Station' columns ---
    # These are human-readable names, not needed for the model's numerical input.
    # NLCs are kept as features.
    print("\n--- Dropping 'From Station' and 'To Station' columns ---")
    columns_to_drop_station_names = ['From Station', 'To Station']
    df_processed.drop(columns=columns_to_drop_station_names, inplace=True, errors='ignore')
    print(f"Columns dropped: {columns_to_drop_station_names}")

    # --- Drop features NOT available at prediction time (and not features) ---
    # These are the columns that were used to define Crowding_Level or were intermediate,
    # but will NOT be inputs for the final prediction model.
    print("\n--- Dropping columns NOT available at prediction time (and not features) ---")
    columns_to_drop_for_model_input = [
        'Load_Value', 'Load_Per_Train', 'Total_load', 'Frequency_Value', 'Total_freq'
    ] 
    df_processed.drop(columns=columns_to_drop_for_model_input, inplace=True, errors='ignore')
    print(f"Columns dropped: {columns_to_drop_for_model_input}")

    print("\nDataFrame head after final column drops:")
    print(df_processed.head())
    print("\nDataFrame columns after final column drops:")
    print(df_processed.columns.tolist())


    # --- Identify categorical columns for encoding ---
    nominal_cols = ['Line', 'Dir', 'DayOfWeek']
    ordinal_col = 'Crowding_Level'

    nominal_cols_to_encode = [col for col in nominal_cols if col in df_processed.columns]
    
    if ordinal_col not in df_processed.columns:
        print(f"Warning: Ordinal column '{ordinal_col}' not found. Ordinal encoding will be skipped.")
        ordinal_col_to_encode = None
    else:
        ordinal_col_to_encode = ordinal_col

    print(f"\nCategorical columns for One-Hot Encoding: {nominal_cols_to_encode}")
    if ordinal_col_to_encode:
        print(f"Categorical column for Ordinal Encoding: {ordinal_col_to_encode}")

    # --- Perform One-Hot Encoding and SAVE the encoder ---
    if nominal_cols_to_encode:
        print("\n--- Performing One-Hot Encoding and Saving Encoder ---")
        encoder_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder_onehot.fit(df_processed[nominal_cols_to_encode])
        joblib.dump(encoder_onehot, os.path.join(FINAL_DATA_PATH, 'onehot_encoder.pkl'))
        print(f"OneHotEncoder saved to {os.path.join(FINAL_DATA_PATH, 'onehot_encoder.pkl')}")
        
        encoded_features = encoder_onehot.transform(df_processed[nominal_cols_to_encode])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder_onehot.get_feature_names_out(nominal_cols_to_encode), index=df_processed.index)
        df_encoded = df_processed.drop(columns=nominal_cols_to_encode).copy()
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        print(f"One-Hot Encoding complete. New shape: {df_encoded.shape}")
        
    else:
        df_encoded = df_processed.copy() # If no nominal cols, just copy for next step
        print("No nominal columns found to apply OneHotEncoder.")

    # --- Perform Ordinal Encoding and SAVE the encoder ---
    if ordinal_col_to_encode:
        print("\n--- Performing Ordinal Encoding for Crowding_Level and Saving Encoder ---")
        crowding_categories_order = ['Low', 'Medium', 'High', 'Very High']
        
        # Ensure column is string type and fill any NaNs/None with 'Low' before fitting/transforming
        df_encoded[ordinal_col_to_encode] = df_encoded[ordinal_col_to_encode].astype(str)
        df_encoded[ordinal_col_to_encode].replace('None', 'Low', inplace=True)
        df_encoded[ordinal_col_to_encode].fillna('Low', inplace=True)

        df_encoded[ordinal_col_to_encode] = pd.Categorical(
            df_encoded[ordinal_col_to_encode], categories=crowding_categories_order, ordered=True
        )

        encoder_ordinal = OrdinalEncoder(categories=[crowding_categories_order], handle_unknown='use_encoded_value', unknown_value=-1)
        encoder_ordinal.fit(df_encoded[[ordinal_col_to_encode]])
        joblib.dump(encoder_ordinal, os.path.join(FINAL_DATA_PATH, 'ordinal_encoder.pkl'))
        print(f"OrdinalEncoder saved to {os.path.join(FINAL_DATA_PATH, 'ordinal_encoder.pkl')}")

        df_encoded[ordinal_col_to_encode] = encoder_ordinal.transform(df_encoded[[ordinal_col_to_encode]])
        print(f"Ordinal Encoding complete. Values for Crowding_Level: {df_encoded[ordinal_col_to_encode].unique()}")
    else:
        print(f"Target column '{ordinal_col}' not found. Skipping ordinal encoding.")


    print("\n--- Encoded DataFrame Head ---")
    print(df_encoded.head())
    print("\n--- Encoded DataFrame Info ---")
    df_encoded.info()

    # --- Save the final encoded DataFrame ---
    output_file_name = 'data_encoded.csv' 
    output_path_full = os.path.join(FINAL_DATA_PATH, output_file_name)
    df_encoded.to_csv(output_path_full, index=False)
    print(f"\nEncoded DataFrame saved to {output_path_full}")

    print("\n--- Step 5: Final Data Preparation and Encoding Complete ---")
