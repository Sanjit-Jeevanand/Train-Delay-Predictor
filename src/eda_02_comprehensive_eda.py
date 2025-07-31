import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define the final data path
PROCESSED_DATA_PATH = "./data/final/"
# Loading the non-encoded file (the one before final_encoded_tube_data.csv)
FINAL_FILE_NAME = "data.csv" 

# --- Load the non-encoded dataset ---
print("\n--- Starting EDA: Loading the Non-Encoded Data ---")
try:
    final_data_path_full = os.path.join(PROCESSED_DATA_PATH, FINAL_FILE_NAME)
    df_eda = pd.read_csv(final_data_path_full)
    print(f"Non-encoded dataset '{FINAL_FILE_NAME}' loaded successfully for EDA.")
    print("Loaded Data shape:", df_eda.shape)
except FileNotFoundError:
    print(f"Error: '{FINAL_FILE_NAME}' not found at {final_data_path_full}.")
    print("Please ensure the file exists at the specified path.")
    df_eda = pd.DataFrame()
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    df_eda = pd.DataFrame()
    exit()

if not df_eda.empty:
    # --- 1. Basic Information and Summary Statistics ---
    print("\n--- Basic DataFrame Info ---")
    df_eda.info()

    print("\n--- Descriptive Statistics for Numerical Features ---")
    print(df_eda.describe())

    # --- 2. Distribution Plots for Key Numerical Features ---
    print("\n--- Visualizing Distributions of Key Numerical Features ---")

    numerical_features = ['Total_load', 'Load_Value', 'Total_freq', 'Frequency_Value', 'Load_Per_Train']

    plt.figure(figsize=(18, 10))
    for i, col in enumerate(numerical_features):
        if col in df_eda.columns:
            plt.subplot(2, 3, i + 1)
            sns.histplot(df_eda[col].dropna(), kde=True) 
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Special case for Load_Per_Train distribution for crowding insights
    if 'Load_Per_Train' in df_eda.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_eda['Load_Per_Train'].dropna(), bins=50, kde=True)
        plt.title('Distribution of Load_Per_Train')
        plt.xlabel('Load Per Train')
        plt.ylabel('Count')
        plt.show()

    # --- Distribution of Categorical Features (including the target variable) ---
    print("\n--- Visualizing Distributions of Categorical Features ---")
    categorical_features = ['Line', 'Dir', 'DayOfWeek', 'Crowding_Level']

    plt.figure(figsize=(18, 10))
    for i, col in enumerate(categorical_features):
        if col in df_eda.columns:
            plt.subplot(2, 2, i + 1)
            if col == 'Crowding_Level':
                # Ensure correct order for Crowding_Level if it's strings ('Low', 'Medium', etc.)
                crowding_order = ['Low', 'Medium', 'High', 'Very High']
                sns.countplot(x=col, data=df_eda, palette='viridis', order=[c for c in crowding_order if c in df_eda[col].unique()])
            elif col == 'DayOfWeek':
                day_order = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
                sns.countplot(x=col, data=df_eda, palette='viridis', order=[d for d in day_order if d in df_eda[col].unique()])
            else:
                sns.countplot(x=col, data=df_eda, palette='viridis')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    # --- 3. Correlation Matrix (only for numerical features) ---
    print("\n--- Correlation Matrix of Numerical Features ---")
    numerical_cols_for_corr = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude NLCs and Order from correlation if they are just IDs, not quantities
    cols_to_exclude_from_corr = ['From NLC', 'To NLC', 'Order'] 
    features_for_corr_matrix = [col for col in numerical_cols_for_corr if col not in cols_to_exclude_from_corr]
    
    correlation_matrix = df_eda[features_for_corr_matrix].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5) 
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

    # --- 4. Relationships with Target Variable (Crowding_Level) ---
    print("\n--- Visualizing Relationships with Crowding_Level ---")

    # Load Per Train vs. Crowding_Level (Box Plot)
    if 'Load_Per_Train' in df_eda.columns and 'Crowding_Level' in df_eda.columns:
        plt.figure(figsize=(10, 6))
        crowding_order = ['Low', 'Medium', 'High', 'Very High']
        sns.boxplot(x='Crowding_Level', y='Load_Per_Train', data=df_eda, palette='viridis', order=crowding_order)
        plt.title('Load Per Train vs. Crowding Level')
        plt.xlabel('Crowding Level')
        plt.ylabel('Load Per Train')
        plt.show()

    # Average Load_Per_Train by quarter_of_day_index and Crowding_Level
    if 'quarter_of_day_index' in df_eda.columns and 'Load_Per_Train' in df_eda.columns and 'Crowding_Level' in df_eda.columns:
        plt.figure(figsize=(15, 7))
        # Ensure 'Crowding_Level' is ordered for hue
        df_eda['Crowding_Level'] = pd.Categorical(df_eda['Crowding_Level'], categories=crowding_order, ordered=True)
        
        sns.lineplot(x='quarter_of_day_index', y='Load_Per_Train', hue='Crowding_Level', 
                     data=df_eda.groupby(['quarter_of_day_index', 'Crowding_Level'])['Load_Per_Train'].mean().reset_index(), 
                     palette='viridis', hue_order=crowding_order)
        plt.title('Average Load Per Train by Time of Day and Crowding Level')
        plt.xlabel('Quarter of Day Index')
        plt.ylabel('Average Load Per Train')
        plt.xticks(range(0, 96, 4), [f'{h:02d}:00' for h in range(24)]) # Label every hour
        plt.grid(True)
        plt.show()
    
    # Average Crowding Level by DayOfWeek
    if 'DayOfWeek' in df_eda.columns and 'Crowding_Level' in df_eda.columns:
        day_order = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        # Convert Crowding_Level to numerical for averaging if it's not already
        # Assuming Crowding_Level is string ('Low', 'Medium', etc.) at this stage
        crowding_to_num = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
        df_eda['Crowding_Level_Num'] = df_eda['Crowding_Level'].map(crowding_to_num)

        avg_crowding_by_day = df_eda.groupby('DayOfWeek')['Crowding_Level_Num'].mean().reindex(day_order).reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='DayOfWeek', y='Crowding_Level_Num', data=avg_crowding_by_day, palette='cividis', order=day_order)
        plt.title('Average Crowding Level by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Crowding Level (0=Low to 3=Very High)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        df_eda.drop(columns=['Crowding_Level_Num'], inplace=True, errors='ignore') # Drop temporary column


    print("\n--- EDA Complete ---")

else:
    print("DataFrame is empty, skipping EDA.")