import pandas as pd

RAW_DATA = "./data/raw/crowding/"
PROCESSED_DATA_PATH = "./data/processed/" # Define a path for processed data

# Ensure the processed data directory exists
import os
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_data(file_name, sheet_names):
    """
    Load data from an Excel file for specified sheets.
    
    Parameters:
    file_name (str): The name of the Excel file to load.
    sheet_names (list or None): A list of sheet names to load, or None to load all sheets.
    
    Returns:
    dict: A dictionary of DataFrames, where keys are sheet names and values are DataFrames.
    """
    file_path = RAW_DATA + file_name
    # When sheet_name is a list, pd.read_excel returns a dictionary of DataFrames
    # When header=2, it means the 3rd row (0-indexed) is used as the header.
    return pd.read_excel(file_path, sheet_name=sheet_names, header=2)

weekday = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

# Dictionaries to store dataframes for Link_Loads and Link_Frequencies for each day
link_loads_data = {}
link_frequencies_data = {}

for day in weekday:
    # Changed filename as per common practice for "outputs" files
    file_name = f"NBT23{day}_outputs.xlsx" 
    try:
        daily_data = load_data(file_name, sheet_names=['Link_Loads', 'Link_Frequencies'])
        
        if 'Link_Loads' in daily_data:
            # Add a 'DayOfWeek' column to identify the day
            df_load = daily_data['Link_Loads']
            df_load['DayOfWeek'] = day
            link_loads_data[day] = df_load
            print(f"Data for Link_Loads on {day} loaded successfully.")
        else:
            print(f"Sheet 'Link_Loads' not found in {file_name}.")

        if 'Link_Frequencies' in daily_data:
            # Add a 'DayOfWeek' column to identify the day
            df_freq = daily_data['Link_Frequencies']
            df_freq['DayOfWeek'] = day
            link_frequencies_data[day] = df_freq
            print(f"Data for Link_Frequencies on {day} loaded successfully.")
        else:
            print(f"Sheet 'Link_Frequencies' not found in {file_name}.")

    except FileNotFoundError:
        print(f"File {file_name} not found. Please check the path or file name.")
    except pd.errors.EmptyDataError:
        print(f"File {file_name} is empty. Please check the file content.")
    except Exception as e:
        print(f"An error occurred while loading {file_name}: {e}")

# Concatenate all Link_Loads DataFrames
if link_loads_data: # Check if dictionary is not empty
    all_link_loads = pd.concat(link_loads_data.values(), ignore_index=True)
    link_loads_output_path = os.path.join(PROCESSED_DATA_PATH, 'all_link_loads.csv')
    all_link_loads.to_csv(link_loads_output_path, index=False)
    print(f"\nAll Link_Loads data merged and saved to {link_loads_output_path}")
else:
    print("\nNo Link_Loads data was loaded to merge.")

# Concatenate all Link_Frequencies DataFrames
if link_frequencies_data: # Check if dictionary is not empty
    all_link_frequencies = pd.concat(link_frequencies_data.values(), ignore_index=True)
    link_frequencies_output_path = os.path.join(PROCESSED_DATA_PATH, 'all_link_frequencies.csv')
    all_link_frequencies.to_csv(link_frequencies_output_path, index=False)
    print(f"All Link_Frequencies data merged and saved to {link_frequencies_output_path}")
else:
    print("No Link_Frequencies data was loaded to merge.")

print(all_link_frequencies.describe())
print(all_link_loads.describe())