import pandas as pd
import numpy as np
import os

# Define paths (relative to the project root, assuming this script is in src/)
PROCESSED_DATA_PATH = "./data/processed/"
FINAL_DATA_PATH = "./data/final/" # Used for saving final processed data like NLC mapping
os.makedirs(FINAL_DATA_PATH, exist_ok=True) # Ensure final data directory exists

# --- Main script execution ---
if __name__ == "__main__":
    print("\n--- Step 3: Link Data Merge and Crowding Definition ---")

    # --- Load the previously processed long-format data (from 02_data_transformation_and_quarter_index.py) ---
    print("\n--- Loading Processed Long-Format Data ---")
    try:
        link_loads_long = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'link_loads_long.csv'))
        link_frequencies_long = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'link_frequencies_long.csv'))
        print("Long-format Link_Loads and Link_Frequencies loaded successfully.")
    except FileNotFoundError:
        print("Error: Processed long-format CSVs not found. Please ensure '02_data_transformation_and_quarter_index.py' ran correctly.")
        exit()
    except Exception as e:
        print(f"An error occurred loading processed data: {e}")
        exit()

    # --- Perform the INNER merge of Link_Loads and Link_Frequencies DataFrames ---
    print("\n--- Performing INNER merge of Link_Loads and Link_Frequencies DataFrames ---")

    if not link_loads_long.empty and not link_frequencies_long.empty:
        # Define the exact columns to merge on, EXCLUDING 'Total' and the 'Early' through 'Late' columns
        # because they have different values between the two dataframes for the same logical 'key'.
        merge_on_cols = [
            'Line', 'Dir', 'Order', 
            'From NLC', 'From ASC', 'From Station', # Keep these for now to extract NLC mapping later
            'To NLC', 'To ASC', 'To Station', # Keep these for now to extract NLC mapping later
            'DayOfWeek', 'quarter_of_day_index' # These are the true common identifiers
        ]
        
        # Filter merge_on_cols to ensure only columns actually present in both DFs are used
        actual_merge_on_cols_loads = [col for col in merge_on_cols if col in link_loads_long.columns]
        actual_merge_on_cols_freqs = [col for col in merge_on_cols if col in link_frequencies_long.columns]

        final_merge_cols = list(set(actual_merge_on_cols_loads).intersection(set(actual_merge_on_cols_freqs)))
        final_merge_cols.sort() # Sort for consistent output

        if len(final_merge_cols) != len(merge_on_cols):
            missing_in_loads = set(merge_on_cols) - set(link_loads_long.columns)
            missing_in_freqs = set(merge_on_cols) - set(link_frequencies_long.columns)
            print(f"Warning: Not all specified merge columns are present in both DataFrames.")
            if missing_in_loads: print(f"Missing in Link_Loads: {missing_in_loads}")
            if missing_in_freqs: print(f"Missing in Link_Frequencies: {missing_in_freqs}")
            print(f"Proceeding with available common merge columns: {final_merge_cols}")
        else:
            print(f"Merging on columns: {final_merge_cols}")

        merged_link_data = pd.merge(
            link_loads_long, 
            link_frequencies_long, 
            on=final_merge_cols, 
            how='inner', 
            suffixes=('_load', '_freq') 
        )

        print("\n--- Merged Link Data (head) ---")
        print(merged_link_data.head())
        print("\n--- Merged Link Data (info) ---")
        merged_link_data.info()
        print("\nMerged Link Data shape:", merged_link_data.shape)

    else:
        print("One or both Link DataFrames are empty, skipping merge.")
        exit() # Exit if merge cannot proceed

    # --- Recalculate 'Load_Per_Train' feature ---
    print("\n--- Recalculating 'Load_Per_Train' Feature ---")

    df_final_crowding_feature = merged_link_data.copy()

    if 'Frequency_Value' in df_final_crowding_feature.columns and 'Load_Value' in df_final_crowding_feature.columns:
        df_final_crowding_feature['Frequency_Value_clean'] = df_final_crowding_feature['Frequency_Value'].replace(0, np.nan)
        df_final_crowding_feature['Load_Per_Train'] = df_final_crowding_feature['Load_Value'] / df_final_crowding_feature['Frequency_Value_clean']
        df_final_crowding_feature.drop(columns=['Frequency_Value_clean'], inplace=True)
        print("Load_Per_Train recalculated.")
    else:
        print("Required 'Load_Value' or 'Frequency_Value' columns missing for Load_Per_Train calculation.")
        exit() # Exit if Load_Per_Train cannot be calculated

    # --- Calculate Line-Specific Percentiles for 'Load_Per_Train' ---
    print("\n--- Calculating Line-Specific Percentile Thresholds ---")

    line_percentiles = df_final_crowding_feature.groupby('Line')['Load_Per_Train'].quantile([0.30, 0.60, 0.85]).unstack()
    line_percentiles.columns = ['p30', 'p60', 'p85'] 

    print("\nLine-Specific Percentile Thresholds for Load_Per_Train:")
    print(line_percentiles)

    # --- Apply Line-Specific Crowding Levels and Name the Column 'Crowding_Level' ---
    print("\n--- Applying Line-Specific Crowding Levels and Renaming Column ---")

    def get_crowding_level_line_specific(row):
        line = row['Line']
        load_per_train_value = row['Load_Per_Train']

        if load_per_train_value == 0:
            return 'Low'

        if line in line_percentiles.index:
            p30 = line_percentiles.loc[line, 'p30']
            p60 = line_percentiles.loc[line, 'p60']
            p85 = line_percentiles.loc[line, 'p85']
        else:
            return None 

        if pd.isna(load_per_train_value):
            return None 
        elif load_per_train_value <= p30:
            return 'Low'
        elif load_per_train_value <= p60:
            return 'Medium'
        elif load_per_train_value <= p85:
            return 'High'
        else:
            return 'Very High'

    df_final_crowding_feature['Crowding_Level'] = df_final_crowding_feature.apply(get_crowding_level_line_specific, axis=1)

    print("\nSample data with new 'Crowding_Level' feature:")
    print(df_final_crowding_feature[[
        'Line', 'DayOfWeek', 'quarter_of_day_index', 'Load_Value', 'Frequency_Value', 'Load_Per_Train', 'Crowding_Level'
    ]].head(10))

    # --- Save the dataset with the new 'Crowding_Level' column ---
    # This file will be loaded by 04_station_nlc_mapping_extraction.py and 05_final_data_preparation_and_encoding.py
    output_file_name = 'final_merged_link_data_with_crowding_percentile.csv'
    
    # Fill NaNs in Load_Per_Train and Crowding_Level before saving for consistency
    df_final_crowding_feature['Load_Per_Train'].fillna(0, inplace=True)
    df_final_crowding_feature['Crowding_Level'].fillna('Low', inplace=True) # Ensure no NaNs in target
    
    # Drop ASC columns here as they are not needed for subsequent steps and are not features
    df_final_crowding_feature.drop(columns=['From ASC', 'To ASC'], inplace=True, errors='ignore')

    df_final_crowding_feature.to_csv(os.path.join(PROCESSED_DATA_PATH, output_file_name), index=False)
    print(f"\nFinal DataFrame with 'Crowding_Level' saved to {os.path.join(PROCESSED_DATA_PATH, output_file_name)}")

    print("\n--- Step 3: Link Data Merge and Crowding Definition Complete ---")
