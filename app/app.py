# --- Streamlit Application (app.py) ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define paths (adjust if your project structure is different)
FINAL_DATA_PATH = "./data/final/" 
MODEL_PATH = "./models/" 

# Ensure directories exist
os.makedirs(FINAL_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# --- Load Model and Encoders ---
@st.cache_resource # Cache the model and encoders to avoid reloading on every rerun
def load_resources():
    try:
        # Load the model. This should be the one saved as 'random_forest_model.pkl'
        model = joblib.load(os.path.join(MODEL_PATH, 'random_forest_model.pkl'))
        # st.success("Random Forest Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: Model file 'random_forest_model.pkl' not found in {MODEL_PATH}. Please ensure it's trained and saved.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    try:
        onehot_encoder = joblib.load(os.path.join(FINAL_DATA_PATH, 'onehot_encoder.pkl'))
        # st.success("OneHotEncoder loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: OneHotEncoder file 'onehot_encoder.pkl' not found in {FINAL_DATA_PATH}. Please run the encoder saving script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading OneHotEncoder: {e}")
        st.stop()

    try:
        ordinal_encoder = joblib.load(os.path.join(FINAL_DATA_PATH, 'ordinal_encoder.pkl'))
        # st.success("OrdinalEncoder loaded successfully!")
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

        # st.success("Model feature columns inferred successfully from 'data_encoded.csv'!")
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
        nlc_mapping = pd.read_csv(os.path.join(FINAL_DATA_PATH, 'extracted_nlc_station_mapping.csv'))
        nlc_to_station_name = nlc_mapping.set_index('NLC')['Station_Name'].to_dict()
        station_name_to_nlc = nlc_mapping.set_index('Station_Name')['NLC'].to_dict()
        # st.success("NLC-Station mapping loaded successfully!")
    except FileNotFoundError:
        st.warning(f"Warning: 'extracted_nlc_station_mapping.csv' not found. Station name selection will not be available.")
    except Exception as e:
        st.warning(f"Warning: Error loading NLC-Station mapping: {e}. Station name selection might be limited.")

    # --- Load original link data for dynamic filtering ---
    link_data_for_lookup = pd.DataFrame()
    try:
        link_data_for_lookup = pd.read_csv(os.path.join(FINAL_DATA_PATH, 'final_merged_link_data_cleaned.csv'))
        link_data_for_lookup['From NLC'] = link_data_for_lookup['From NLC'].astype(int)
        link_data_for_lookup['To NLC'] = link_data_for_lookup['To NLC'].astype(int)
        # st.success("Link data for dynamic lookup loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: 'final_merged_link_data_cleaned.csv' not found in {FINAL_DATA_PATH}. Dynamic station/order selection will not work.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading link data for lookup: {e}")
        st.stop()


    return model, onehot_encoder, ordinal_encoder, model_features_columns, nlc_to_station_name, station_name_to_nlc, link_data_for_lookup

model, onehot_encoder, ordinal_encoder, model_features_columns, nlc_to_station_name, station_name_to_nlc, link_data_for_lookup = load_resources()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Tube Crowding Predictor", layout="centered")

st.title("🚇 London Tube Crowding Predictor")
st.markdown("Predict the crowding level for a specific Tube link at a given time.")

# --- User Input Section ---
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
    
    # --- NEW: Informative message for single option ---
    # if len(possible_from_stations) == 1:
    #     st.info(f"Only one 'From Station' available for {selected_line} {selected_dir} due to line structure.")

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
        # st.info(f"Automatically determined To Station: **{to_station_name_display}** (NLC: {selected_to_nlc})")
    elif len(possible_to_nlcs_df) > 1:
        # st.warning(f"Multiple 'To Stations' found for this link. Using the first one: **{nlc_to_station_name.get(int(possible_to_nlcs_df[0]), 'Unknown')}**.")
        selected_to_nlc = int(possible_to_nlcs_df[0])
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
        # st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e) 

st.markdown("---")
st.markdown("Developed for the Tube Crowding Predictor Project.")

