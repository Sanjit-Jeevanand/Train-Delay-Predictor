import pandas as pd
import os

# Define paths
PROCESSED_DATA_PATH = "./data/processed/"

# --- Load the latest processed DataFrame ---
print("\n--- Loading Final Merged Link Data (to extract NLC-Station mapping) ---")
try:
    file_to_load = os.path.join(PROCESSED_DATA_PATH, 'final_merged_link_data_with_crowding_percentile.csv')
    df_processed = pd.read_csv(file_to_load)
    print(f"Dataset '{os.path.basename(file_to_load)}' loaded successfully.")
    print("Loaded Data shape:", df_processed.shape)
except FileNotFoundError:
    print(f"Error: '{os.path.basename(file_to_load)}' not found.")
    print("Please ensure the previous step ran correctly and saved the file.")
    df_processed = pd.DataFrame()
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    df_processed = pd.DataFrame()
    exit()

# --- Extract NLC to Station Name Mapping from the current data ---
print("\n--- Extracting Unique NLC-Station Name Pairs ---")

if not df_processed.empty:
    # Extract pairs from 'From' columns
    from_stations = df_processed[['From NLC', 'From Station']].copy()
    from_stations.rename(columns={'From NLC': 'NLC', 'From Station': 'Station_Name'}, inplace=True)

    # Extract pairs from 'To' columns
    to_stations = df_processed[['To NLC', 'To Station']].copy()
    to_stations.rename(columns={'To NLC': 'NLC', 'To Station': 'Station_Name'}, inplace=True)

    # Concatenate and get unique pairs
    combined_stations = pd.concat([from_stations, to_stations]).drop_duplicates().reset_index(drop=True)

    # Sort for better readability (optional)
    combined_stations.sort_values(by='Station_Name', inplace=True)

    print("\nExtracted NLC-Station Name Mapping (head):")
    print(combined_stations.head())
    print(f"\nTotal unique NLC-Station Name pairs found: {combined_stations.shape[0]}")

    # --- Save this new mapping ---
    output_mapping_file = os.path.join(PROCESSED_DATA_PATH, 'extracted_nlc_station_mapping.csv')
    combined_stations.to_csv(output_mapping_file, index=False)
    print(f"\nExtracted NLC-Station Name mapping saved to {output_mapping_file}")

else:
    print("DataFrame is empty, cannot extract NLC-Station mapping.")

print("\n--- NLC-Station Mapping Extraction Complete ---")