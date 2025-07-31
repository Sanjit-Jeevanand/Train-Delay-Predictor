import pandas as pd
import numpy as np
import os

# Define paths (relative to the project root, assuming this script is in src/)
RAW_DATA_PATH = "./data/raw/crowding/"
PROCESSED_DATA_PATH = "./data/processed/"

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# --- Function to load raw Excel data (reused from 01_raw_data_ingestion.py) ---
def load_raw_excel_data(file_name, sheet_names):
    """
    Load data from an Excel file for specified sheets.
    This function is intended for raw data loading, typically from NBT23{Day}_outputs.xlsx.
    """
    file_path = os.path.join(RAW_DATA_PATH, file_name)
    return pd.read_excel(file_path, sheet_name=sheet_names, header=2)

# --- Core transformation function: Melts, extracts time, and drops old time columns ---
def melt_and_process_time_data(df, id_vars_to_keep, value_name):
    """
    Melts the DataFrame from wide to long format, extracts time features (hour, minute, quarter_of_day_index),
    and drops the original time interval columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame (e.g., all_link_loads).
    id_vars_to_keep (list): List of columns that should remain as identifiers after melting.
    value_name (str): The name for the new column containing the melted values (e.g., 'Load_Value').
    
    Returns:
    pd.DataFrame: The transformed DataFrame in long format.
    """
    # Identify columns that represent time intervals (e.g., '0000-0015' onwards)
    time_columns = [col for col in df.columns if isinstance(col, str) and '-' in col and col[0].isdigit()]

    if not time_columns:
        print(f"Warning: No time interval columns found for value_name '{value_name}'. Skipping melt.")
        return pd.DataFrame() 
    
    # Filter id_vars_to_keep to only include columns actually present in the dataframe
    actual_id_vars = [col for col in id_vars_to_keep if col in df.columns]
    
    # Ensure DayOfWeek is in id_vars if it's in the dataframe and not already in the list
    if 'DayOfWeek' in df.columns and 'DayOfWeek' not in actual_id_vars:
        actual_id_vars.append('DayOfWeek')

    # Perform the melt operation
    df_melted = pd.melt(df, 
                        id_vars=actual_id_vars, 
                        value_vars=time_columns, 
                        var_name='Time_Interval', # Keep Time_Interval temporarily to extract hour/minute
                        value_name=value_name)
    
    # Extract Start_Hour and Start_Minute from 'Time_Interval'
    df_melted[['Start_Time_Str', 'End_Time_Str']] = df_melted['Time_Interval'].str.split('-', expand=True)
    df_melted['Start_Hour'] = df_melted['Start_Time_Str'].str[:2].astype(int)
    df_melted['Start_Minute'] = df_melted['Start_Time_Str'].str[2:].astype(int)
    
    # Calculate quarter_of_day_index
    df_melted['quarter_of_day_index'] = (df_melted['Start_Hour'] * 4) + (df_melted['Start_Minute'] // 15)
    
    # Drop all intermediate and original time columns that are no longer needed
    df_melted = df_melted.drop(columns=['Start_Time_Str', 'End_Time_Str', 'Time_Interval', 'Start_Hour', 'Start_Minute'], errors='ignore')
    
    return df_melted

# --- Main script execution ---
if __name__ == "__main__":
    print("\n--- Step 2: Data Transformation and Quarter Index Creation ---")

    # Define the explicit ID columns that need to be preserved from the raw Link data
    # These are the non-time columns from the raw Link_Loads/Link_Frequencies sheets.
    LINK_ID_COLUMNS_TO_PRESERVE = [
        'Link', 'Line', 'Dir', 'Order', 
        'From NLC', 'From ASC', 'From Station', 
        'To NLC', 'To ASC', 'To Station', 
        'Total', 'Early', 'AM Peak', 'Midday', 'PM Peak', 'Evening', 'Late'
    ]

    # Load the concatenated raw Link data (output from 01_raw_data_ingestion.py)
    try:
        all_link_loads = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'all_link_loads.csv'))
        all_link_frequencies = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'all_link_frequencies.csv'))
        print("Raw Link_Loads and Link_Frequencies loaded successfully.")
    except FileNotFoundError:
        print("Error: Raw Link_Loads or Link_Frequencies CSVs not found in processed directory.")
        print("Please ensure '01_raw_data_ingestion.py' was run successfully.")
        exit()
    except Exception as e:
        print(f"An error occurred loading raw link data: {e}")
        exit()

    # Process Link_Loads
    if not all_link_loads.empty:
        print("\n--- Transforming Link_Loads Data to Long Format ---")
        link_loads_long = melt_and_process_time_data(all_link_loads, 
                                                     id_vars_to_keep=LINK_ID_COLUMNS_TO_PRESERVE, 
                                                     value_name='Load_Value')
        
        if not link_loads_long.empty:
            print("Transformed Link_Loads Data (head):")
            print(link_loads_long.head())
            print("Transformed Link_Loads Data Info:")
            link_loads_long.info()
            link_loads_long.to_csv(os.path.join(PROCESSED_DATA_PATH, 'link_loads_long.csv'), index=False)
            print(f"Transformed Link_Loads data saved to {os.path.join(PROCESSED_DATA_PATH, 'link_loads_long.csv')}")
        else:
            print("Link_Loads DataFrame is empty after transformation.")
    else:
        print("All Link Loads DataFrame is empty, skipping transformation.")

    # Process Link_Frequencies
    if not all_link_frequencies.empty:
        print("\n--- Transforming Link_Frequencies Data to Long Format ---")
        link_frequencies_long = melt_and_process_time_data(all_link_frequencies, 
                                                           id_vars_to_keep=LINK_ID_COLUMNS_TO_PRESERVE, 
                                                           value_name='Frequency_Value')
        
        if not link_frequencies_long.empty:
            print("Transformed Link_Frequencies Data (head):")
            print(link_frequencies_long.head())
            print("Transformed Link_Frequencies Data Info:")
            link_frequencies_long.info()
            link_frequencies_long.to_csv(os.path.join(PROCESSED_DATA_PATH, 'link_frequencies_long.csv'), index=False)
            print(f"Transformed Link_Frequencies data saved to {os.path.join(PROCESSED_DATA_PATH, 'link_frequencies_long.csv')}")
        else:
            print("Link_Frequencies DataFrame is empty after transformation.")
    else:
        print("All Link Frequencies DataFrame is empty, skipping transformation.")

    print("\n--- Step 2: Data Transformation and Quarter Index Creation Complete ---")
