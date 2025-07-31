# --- Streamlit Application (app.py) ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths (adjust if your project structure is different)
FINAL_DATA_PATH = "./data/final/"
PROCESSED_DATA_PATH = "./data/processed/" # Needed for NLC mapping and EDA data
MODEL_PATH = "./models/"

# Ensure directories exist
os.makedirs(FINAL_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True) # Ensure processed data path exists
os.makedirs(MODEL_PATH, exist_ok=True)

# --- Load Model and Encoders ---
@st.cache_resource # Cache the model and encoders to avoid reloading on every rerun
def load_resources():
    # Load Model
    try:
        model = joblib.load(os.path.join(MODEL_PATH, 'random_forest_model.pkl'))
    except FileNotFoundError:
        st.error(f"Error: Model file 'random_forest_model.pkl' not found in {MODEL_PATH}. Please ensure it's trained and saved.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Load OneHotEncoder
    try:
        onehot_encoder = joblib.load(os.path.join(FINAL_DATA_PATH, 'onehot_encoder.pkl'))
    except FileNotFoundError:
        st.error(f"Error: OneHotEncoder file 'onehot_encoder.pkl' not found in {FINAL_DATA_PATH}. Please run the encoder saving script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading OneHotEncoder: {e}")
        st.stop()

    # Load OrdinalEncoder
    try:
        ordinal_encoder = joblib.load(os.path.join(FINAL_DATA_PATH, 'ordinal_encoder.pkl'))
    except FileNotFoundError:
        st.error(f"Error: OrdinalEncoder file 'ordinal_encoder.pkl' not found in {FINAL_DATA_PATH}. Please run the encoder saving script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading OrdinalEncoder: {e}")
        st.stop()
    # --- Infer Model Feature Columns from a sample of the encoded data ---
    try:
        sample_encoded_df = pd.read_csv(os.path.join(FINAL_DATA_PATH, 'data_encoded.csv'))
        features_excluded_from_X_in_training = [
            'Crowding_Level',
            'Load_Value', 'Load_Per_Train', 'Total_load', 'Frequency_Value', 'Total_freq'
        ]
        model_features_columns = sample_encoded_df.drop(columns=features_excluded_from_X_in_training, errors='ignore').columns.tolist()

    except FileNotFoundError:
        st.error(f"Error: 'data_encoded.csv' not found in {FINAL_DATA_PATH}. This file is needed to infer model feature columns.")
        st.stop()
    except Exception as e:
        st.error(f"Error inferring model feature columns: {e}")
        st.stop()

    # --- Load NLC to Station Name mapping ---
    nlc_mapping = pd.DataFrame()
    station_name_to_nlc = {}
    nlc_to_station_name = {}
    try:
        nlc_mapping = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'extracted_nlc_station_mapping.csv'))
        nlc_to_station_name = nlc_mapping.set_index('NLC')['Station_Name'].to_dict()
        station_name_to_nlc = nlc_mapping.set_index('Station_Name')['NLC'].to_dict()
    except FileNotFoundError:
        st.warning(f"Warning: 'extracted_nlc_station_mapping.csv' not found. Station name selection will not be available.")
    except Exception as e:
        st.warning(f"Warning: Error loading NLC-Station mapping: {e}. Station name selection might be limited.")

    # --- Load original link data for dynamic filtering ---
    link_data_for_lookup = pd.DataFrame()
    try:
        # This file should contain original Line, Dir, From/To NLC/Station, Order
        link_data_for_lookup = pd.read_csv(os.path.join(FINAL_DATA_PATH, 'data.csv'))
        link_data_for_lookup['From NLC'] = link_data_for_lookup['From NLC'].astype(int)
        link_data_for_lookup['To NLC'] = link_data_for_lookup['To NLC'].astype(int)
    except FileNotFoundError:
        st.error(f"Error: 'data.csv' not found in {FINAL_DATA_PATH}. Dynamic station/order selection will not work.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading link data for lookup: {e}")
        st.stop()

    # --- Load data for EDA tab ---
    eda_data = pd.DataFrame()
    try:
        # Load the non-encoded data with Crowding_Level for EDA
        eda_data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'final_merged_link_data_with_crowding_percentile.csv'))
        # Ensure Crowding_Level is categorical for plotting
        crowding_order = ['Low', 'Medium', 'High', 'Very High']
        eda_data['Crowding_Level'] = pd.Categorical(eda_data['Crowding_Level'], categories=crowding_order, ordered=True)
    except FileNotFoundError:
        st.warning(f"Warning: EDA data 'final_merged_link_data_with_crowding_percentile.csv' not found. EDA tab will be limited.")
    except Exception as e:
        st.warning(f"Warning: Error loading EDA data: {e}. EDA tab might be limited.")


    return model, onehot_encoder, ordinal_encoder, model_features_columns, nlc_to_station_name, station_name_to_nlc, link_data_for_lookup, eda_data

model, onehot_encoder, ordinal_encoder, model_features_columns, nlc_to_station_name, station_name_to_nlc, link_data_for_lookup, eda_data = load_resources()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Tube Crowding Predictor", layout="wide") # Use wide layout for tabs

st.title("🚇 London Tube Crowding Predictor")
st.markdown("Predict the crowding level for a specific Tube link at a given time.")

# Create tabs
tab1, eda_tab = st.tabs(["Crowding Prediction", "Exploratory Data Analysis (EDA)"])

with tab1:
    st.header("Input Details")

    # Input for categorical features (original categories)
    line_categories = onehot_encoder.categories_[0].tolist()
    dayofweek_categories = onehot_encoder.categories_[2].tolist()

    selected_line = st.selectbox("Select Line", line_categories)
    selected_dayofweek = st.selectbox("Select Day of Week", dayofweek_categories)

    # --- Filter Direction options based on selected Line ---
    possible_dirs_df = link_data_for_lookup[
        (link_data_for_lookup['Line'] == selected_line)
    ]['Dir'].unique()

    possible_dirs = sorted([d for d in possible_dirs_df if pd.notna(d)])

    if not possible_dirs:
        st.warning(f"No directions found for {selected_line}. Please adjust line selection.")
        st.stop()

    selected_dir = st.selectbox("Select Direction", possible_dirs)


    # User input for From Station (using names for user-friendliness)
    if nlc_to_station_name and station_name_to_nlc and not link_data_for_lookup.empty:
        # Filter 'From Stations' based on selected Line and Direction
        possible_from_stations_df = link_data_for_lookup[
            (link_data_for_lookup['Line'] == selected_line) &
            (link_data_for_lookup['Dir'] == selected_dir)
        ]['From Station'].unique()
        possible_from_stations = sorted([s for s in possible_from_stations_df if pd.notna(s)])

        if not possible_from_stations:
            st.warning(f"No 'From Stations' found for {selected_line} {selected_dir}. Please adjust selections.")
            st.stop()
        selected_from_station_name = st.selectbox("Select From Station", possible_from_stations)
        selected_from_nlc = station_name_to_nlc.get(selected_from_station_name)

        if selected_from_nlc is None:
            st.error(f"Error: NLC not found for '{selected_from_station_name}'.")
            st.stop()

        # --- Filter 'To Station' options based on 'From Station' and Line and Direction ---
        possible_to_nlcs_df = link_data_for_lookup[
            (link_data_for_lookup['Line'] == selected_line) &
            (link_data_for_lookup['Dir'] == selected_dir) &
            (link_data_for_lookup['From NLC'] == selected_from_nlc)
        ]['To NLC'].unique()

        if len(possible_to_nlcs_df) == 1:
            selected_to_nlc = int(possible_to_nlcs_df[0])
            to_station_name_display = nlc_to_station_name.get(selected_to_nlc, "Unknown To Station")
            # # st.info(f"Automatically determined To Station: **{to_station_name_display}** (NLC: {selected_to_nlc})")
        elif len(possible_to_nlcs_df) > 1:
            selected_to_nlc = int(possible_to_nlcs_df[0])
            st.warning(f"Multiple 'To Stations' found for this link. Using the first one: **{nlc_to_station_name.get(int(possible_to_nlcs_df[0]), 'Unknown')}**.")
        else:
            st.error("Could not determine 'To NLC' for the selected 'From Station' and Line/Direction. Please check data consistency.")
            st.stop()


        # --- Automatically choose 'Order' based on selections ---
        matching_link_details_df = link_data_for_lookup[
            (link_data_for_lookup['Line'] == selected_line) &
            (link_data_for_lookup['Dir'] == selected_dir) &
            (link_data_for_lookup['From NLC'] == selected_from_nlc) &
            (link_data_for_lookup['To NLC'] == selected_to_nlc)
        ]['Order'].unique()

        if len(matching_link_details_df) == 1:
            selected_order = int(matching_link_details_df[0])
            # st.info(f"Automatically determined Link Order: **{selected_order}**")
        elif len(matching_link_details_df) > 1:
            selected_order = int(matching_link_details_df[0])
            st.warning(f"Multiple orders found for this link. Using: **{selected_order}**. Consider refining data.")
        else:
            st.error("Could not determine Link Order for the selected stations. Please check data consistency.")
            st.stop()

    else: # Fallback if mappings or lookup data are not loaded
        st.warning("Station name mapping or link lookup data not loaded. Falling back to manual NLC/Order/Direction input.")
        selected_from_nlc = st.number_input("Enter From NLC (e.g., 570)", min_value=0, value=570, key="from_nlc_manual")
        selected_to_nlc = st.number_input("Enter To NLC (e.g., 628)", min_value=0, value=628, key="to_nlc_manual")
        selected_dir = st.selectbox("Select Direction (manual fallback)", onehot_encoder.categories_[1].tolist())
        selected_order = st.number_input("Enter Link Order (e.g., 1)", min_value=1, value=1, step=1)


    st.subheader("Time of Day")
    selected_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=8, step=1)
    selected_minute = st.number_input("Minute (0-59)", min_value=0, max_value=59, value=0, step=1)

    # Calculate quarter_of_day_index
    quarter_of_day_index = (selected_hour * 4) + (selected_minute // 15)
    # st.info(f"Calculated Quarter of Day Index: **{quarter_of_day_index}**")


    # --- Prediction Button ---
    if st.button("Predict Crowding Level"):
        # --- Prepare input for prediction ---
        input_df_raw = pd.DataFrame({
            'Order': [selected_order],
            'From NLC': [selected_from_nlc],
            'To NLC': [selected_to_nlc],
            'quarter_of_day_index': [quarter_of_day_index],
            'Line': [selected_line],
            'Dir': [selected_dir],
            'DayOfWeek': [selected_dayofweek]
        })

        categorical_input_for_ohe = input_df_raw[['Line', 'Dir', 'DayOfWeek']]
        encoded_categorical_features = onehot_encoder.transform(categorical_input_for_ohe)
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical_features,
            columns=onehot_encoder.get_feature_names_out(['Line', 'Dir', 'DayOfWeek']),
            index=input_df_raw.index
        )

        final_input_for_model = pd.DataFrame(0, index=[0], columns=model_features_columns)
        # Populate numerical features that are direct inputs
        numerical_cols_to_populate = [
            'Order', 'From NLC', 'To NLC', 'quarter_of_day_index'
        ]
        for col in numerical_cols_to_populate:
            if col in final_input_for_model.columns:
                final_input_for_model[col] = input_df_raw[col]
        for col in encoded_categorical_df.columns:
            if col in final_input_for_model.columns:
                final_input_for_model[col] = encoded_categorical_df[col]

        final_input_for_model = final_input_for_model[model_features_columns]


        # --- Make Prediction ---
        try:
            prediction_encoded = model.predict(final_input_for_model)
            predicted_crowding_level = ordinal_encoder.inverse_transform(prediction_encoded.reshape(-1, 1))[0][0]
            emoji_map = {
                'Low': '🟢',
                'Medium': '🟡',
                'High': '🟠',
                'Very High': '🔴'
            }
            crowding_emoji = emoji_map.get(predicted_crowding_level, '⚪')

            color_map = {
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red',
                'Very High': 'darkred'
            }
            text_color = color_map.get(predicted_crowding_level, 'black')

            st.markdown(f"**Predicted Crowding Level: <span style='color:{text_color};'>{crowding_emoji} {predicted_crowding_level}</span>**", unsafe_allow_html=True)
            # st.balloons() # Removed balloons for less distraction

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)

with eda_tab:
    st.header("Exploratory Data Analysis")

    if eda_data.empty:
        st.warning("EDA data not loaded. Please ensure 'final_merged_link_data_with_crowding_percentile.csv' exists in the processed data path.")
    else:
        # --- Distribution of Categorical Features ---
        st.subheader("Distributions of Categorical Features")
        categorical_features = ['Line', 'Dir', 'DayOfWeek', 'Crowding_Level']

        plt.figure(figsize=(18, 10))
        for i, col in enumerate(categorical_features):
            if col in eda_data.columns:
                plt.subplot(2, 2, i + 1)
                if col == 'Crowding_Level':
                    crowding_order = ['Low', 'Medium', 'High', 'Very High']
                    sns.countplot(x=col, data=eda_data, palette='viridis', order=[c for c in crowding_order if c in eda_data[col].unique()])
                elif col == 'DayOfWeek':
                    day_order_full = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
                    sns.countplot(x=col, data=eda_data, palette='viridis', order=[d for d in day_order_full if d in eda_data[col].unique()])
                else:
                    sns.countplot(x=col, data=eda_data, palette='viridis')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close(plt.gcf())
        link_loads_long = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'link_loads_long.csv'))
        # 3.2. Average Link Load per Day of Week over 24 hours
        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=link_loads_long.groupby(['DayOfWeek', 'quarter_of_day_index'])['Load_Value'].mean().reset_index(),
            x='quarter_of_day_index',
            y='Load_Value',
            hue='DayOfWeek',
            marker='o',
            lw=1.5
        )
        plt.title('Average Link Load Profile by Day of Week')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Load Value')
        plt.xticks(
            ticks=np.arange(0, 96, 8),
            labels=[f'{h:02d}:00' for h in range(0, 24, 2)]
        )  # Every 2 hours
        plt.grid(True)
        plt.legend(title='Day of Week')
        plt.tight_layout()
        plt.show()

        
        st.subheader("Average Load per Train by Categorical Features")

        categorical_features = ['Line', 'Dir', 'DayOfWeek']
        day_order_full = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

        plt.figure(figsize=(20, 12))
        for i, col in enumerate(categorical_features):
            if col in eda_data.columns:
                plt.subplot(2, 2, i + 1)
                order = None
                if col == 'DayOfWeek':
                    order = [d for d in day_order_full if d in eda_data[col].unique()]
                
                sns.barplot(
                    data=eda_data,
                    x=col,
                    y='Load_Value',
                    estimator='mean',
                    palette='viridis',
                    order=order
                )
                plt.title(f'Average Load per Train by {col}')
                plt.xlabel(col)
                plt.ylabel('Average Load')
                plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close(plt.gcf())

        st.subheader("3.2. Average Link Load Profile by Day of Week")

        avg_load_by_day = (
            link_loads_long.groupby(['DayOfWeek', 'quarter_of_day_index'])['Load_Value']
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=avg_load_by_day,
            x='quarter_of_day_index',
            y='Load_Value',
            hue='DayOfWeek',
            marker='o',
            lw=1.5
        )
        plt.title('Average Link Load Profile by Day of Week')
        plt.xlabel('Time of Day (Quarter Index)')
        plt.ylabel('Average Load Value')
        plt.xticks(
            ticks=np.arange(0, 96, 8),
            labels=[f'{h:02d}:00' for h in range(0, 24, 2)]
        )
        plt.grid(True)
        plt.legend(title='Day of Week')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        # --- 3.3. Top N Busiest Links (based on total load) ---
        st.subheader("3.3. Top N Busiest Links")

        # Create Link_ID column if not already present
        link_loads_long['Link_ID'] = link_loads_long['From NLC'].astype(str) + '-' + link_loads_long['To NLC'].astype(str)

        top_n = st.slider("Select Top N Links", min_value=3, max_value=20, value=5)

        busiest_links = (
            link_loads_long.groupby('Link_ID')['Load_Value']
            .sum()
            .nlargest(top_n)
            .index
        )

        top_links_df = link_loads_long[link_loads_long['Link_ID'].isin(busiest_links)]

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=top_links_df.groupby('Link_ID')['Load_Value'].sum().reset_index(),
            x='Link_ID',
            y='Load_Value',
            palette='magma'
        )
        plt.title(f'Top {top_n} Busiest Links by Total Load')
        plt.xlabel('Link ID')
        plt.ylabel('Total Load Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()


        # --- Correlation Matrix (only for numerical features) ---
        st.subheader("Correlation Matrix of Numerical Features")
        numerical_cols_for_corr = eda_data.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude NLCs and Order from correlation if they are just IDs, not quantities
        cols_to_exclude_from_corr = ['From NLC', 'To NLC', 'Order']
        features_for_corr_matrix = [col for col in numerical_cols_for_corr if col not in cols_to_exclude_from_corr]

        if features_for_corr_matrix: # Ensure there are columns to plot
            correlation_matrix = eda_data[features_for_corr_matrix].corr()

            fig_corr, ax_corr = plt.subplots(figsize=(14, 12))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
            ax_corr.set_title('Correlation Matrix of Numerical Features')
            st.pyplot(fig_corr)
            plt.close(fig_corr)
        # else:
            # st.info("No suitable numerical features found for correlation matrix after exclusions.")

        # --- Relationships with Target Variable (Crowding_Level) ---
        st.subheader("Relationships with Crowding Level")

        # Average Load_Per_Train by quarter_of_day_index and Crowding_Level
        if 'quarter_of_day_index' in eda_data.columns and 'Load_Per_Train' in eda_data.columns and 'Crowding_Level' in eda_data.columns:
            fig_avg_lpt_quarter, ax_avg_lpt_quarter = plt.subplots(figsize=(15, 7))
            crowding_order = ['Low', 'Medium', 'High', 'Very High'] # Ensure order is consistent
            eda_data['Crowding_Level'] = pd.Categorical(eda_data['Crowding_Level'], categories=crowding_order, ordered=True) # Re-ensure categorical order for hue

            sns.lineplot(x='quarter_of_day_index', y='Load_Per_Train', hue='Crowding_Level',
                         data=eda_data.groupby(['quarter_of_day_index', 'Crowding_Level'])['Load_Per_Train'].mean().reset_index(),
                         palette='viridis', hue_order=crowding_order, ax=ax_avg_lpt_quarter)
            ax_avg_lpt_quarter.set_title('Average Load Per Train by Time of Day and Crowding Level')
            ax_avg_lpt_quarter.set_xlabel('Quarter of Day Index')
            ax_avg_lpt_quarter.set_ylabel('Average Load Per Train')
            ax_avg_lpt_quarter.set_xticks(range(0, 96, 4)) # Label every hour
            ax_avg_lpt_quarter.set_xticklabels([f'{h:02d}:00' for h in range(24)])
            ax_avg_lpt_quarter.grid(True)
            st.pyplot(fig_avg_lpt_quarter)
            plt.close(fig_avg_lpt_quarter)
        # else:
            # st.info("Cannot plot 'Average Load Per Train by Time of Day and Crowding Level': required columns missing.")

st.markdown("---")
st.markdown("Developed for the Tube Crowding Predictor Project.")