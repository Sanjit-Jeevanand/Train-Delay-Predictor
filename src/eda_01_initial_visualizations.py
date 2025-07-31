import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths (relative to the project root, assuming this script is in src/)
PROCESSED_DATA_PATH = "./data/processed/"

# --- Main script execution ---
if __name__ == "__main__":
    print("\n--- Step: Initial Visualizations (EDA) ---")

    # --- Load the previously processed long-format data (from 02_data_transformation_and_quarter_index.py) ---
    print("\n--- Loading Processed Long-Format Data ---")
    try:
        link_loads_long = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'link_loads_long.csv'))
        link_frequencies_long = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'link_frequencies_long.csv'))
        print("Long-format Link_Loads and Link_Frequencies loaded successfully.")
    except FileNotFoundError:
        print("Error: Processed long-format CSVs not found. Please ensure '02_data_transformation_and_quarter_index.py' ran correctly.")
        link_loads_long = pd.DataFrame()
        link_frequencies_long = pd.DataFrame()
        exit()
    except Exception as e:
        print(f"An error occurred loading processed data: {e}")
        link_loads_long = pd.DataFrame()
        link_frequencies_long = pd.DataFrame()
        exit()


    # --- Initial Visualizations ---

    if not link_loads_long.empty:
        print("\n--- Generating Visualizations for Link_Loads ---")

        # Ensure 'DayOfWeek' is a categorical type for proper ordering in plots
        weekday_order = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        link_loads_long['DayOfWeek'] = pd.Categorical(link_loads_long['DayOfWeek'], categories=weekday_order, ordered=True)

        # --- MODIFIED: Use quarter_of_day_index for plotting ---
        # No need to calculate Time_in_Minutes from Start_Hour/Minute as they are dropped.
        # quarter_of_day_index is already available.

        # 3.1. Average Link Load over 24 hours
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=link_loads_long.groupby('quarter_of_day_index')['Load_Value'].mean().reset_index(), 
                     x='quarter_of_day_index', y='Load_Value')
        plt.title('Average Link Load Profile Over 24 Hours (All Days)')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Load Value')
        # Custom x-ticks to show hours for readability (96 quarters = 24 hours)
        plt.xticks(ticks=np.arange(0, 96, 4), labels=[f'{h:02d}:00' for h in range(24)])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3.2. Average Link Load per Day of Week over 24 hours
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=link_loads_long.groupby(['DayOfWeek', 'quarter_of_day_index'])['Load_Value'].mean().reset_index(), 
                     x='quarter_of_day_index', y='Load_Value', hue='DayOfWeek', marker='o', lw=1.5)
        plt.title('Average Link Load Profile by Day of Week')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Load Value')
        plt.xticks(ticks=np.arange(0, 96, 8), labels=[f'{h:02d}:00' for h in range(0, 24, 2)]) # Show every 2 hours
        plt.grid(True)
        plt.legend(title='Day of Week')
        plt.tight_layout()
        plt.show()

        # 3.3. Top N Busiest Links (based on total load)
        link_loads_long['Link_ID'] = link_loads_long['From NLC'].astype(str) + '-' + link_loads_long['To NLC'].astype(str)
        top_n = 5 
        busiest_links = link_loads_long.groupby('Link_ID')['Load_Value'].sum().nlargest(top_n).index

        plt.figure(figsize=(15, 8))
        for link_id in busiest_links:
            link_data = link_loads_long[link_loads_long['Link_ID'] == link_id]
            link_profile = link_data.groupby('quarter_of_day_index')['Load_Value'].mean().reset_index()
            sns.lineplot(data=link_profile, x='quarter_of_day_index', y='Load_Value', label=link_id, lw=2)
        
        plt.title(f'Average Load Profile for Top {top_n} Busiest Links')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Load Value')
        plt.xticks(ticks=np.arange(0, 96, 8), labels=[f'{h:02d}:00' for h in range(0, 24, 2)])
        plt.grid(True)
        plt.legend(title='Link ID')
        plt.tight_layout()
        plt.show()

    else:
        print("Link_Loads DataFrame is empty, skipping visualizations.")

    if not link_frequencies_long.empty:
        print("\n--- Generating Visualizations for Link_Frequencies ---")

        link_frequencies_long['DayOfWeek'] = pd.Categorical(link_frequencies_long['DayOfWeek'], categories=weekday_order, ordered=True)
        # --- MODIFIED: Use quarter_of_day_index for plotting ---
        
        # 3.4. Average Link Frequency over 24 hours
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=link_frequencies_long.groupby('quarter_of_day_index')['Frequency_Value'].mean().reset_index(), 
                     x='quarter_of_day_index', y='Frequency_Value')
        plt.title('Average Link Frequency Profile Over 24 Hours (All Days)')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Frequency Value')
        plt.xticks(ticks=np.arange(0, 96, 4), labels=[f'{h:02d}:00' for h in range(25)])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3.5. Average Link Frequency per Day of Week over 24 hours
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=link_frequencies_long.groupby(['DayOfWeek', 'quarter_of_day_index'])['Frequency_Value'].mean().reset_index(), 
                     x='quarter_of_day_index', y='Frequency_Value', hue='DayOfWeek', marker='o', lw=1.5)
        plt.title('Average Link Frequency Profile by Day of Week')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Frequency Value')
        plt.xticks(ticks=np.arange(0, 96, 8), labels=[f'{h:02d}:00' for h in range(0, 24, 2)])
        plt.grid(True)
        plt.legend(title='Day of Week')
        plt.tight_layout()
        plt.show()

    else:
        print("Link_Frequencies DataFrame is empty, skipping visualizations.")

    print("\n--- Step: Initial Visualizations (EDA) Complete ---")
