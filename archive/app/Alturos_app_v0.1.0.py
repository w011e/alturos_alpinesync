from pre_processing import PreProcessor
from modelling import Model
from post_processing import PostProcessor
from visualisation import Plotting, Mapping
import streamlit as st
import logging
import os


# # # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(_name
# )

# Determine the absolute path to the data file
current_folder = os.path.dirname(__file__)



# #Define paths to data and model files
data_paths = {
    '57': os.path.join(current_folder, "..", "..", "data", "processed", "df_57_labeled_on_lift.csv"),
    '95': os.path.join(current_folder, "..", "..", "data", "processed", "df_95_labeled_on_lift.csv"),
    '135': os.path.join(current_folder, "..", "..", "data", "processed", "df_135_labeled_on_lift.csv"),
    '166': os.path.join(current_folder, "..", "..", "data", "processed", "df_166m_labeled_on_lift.csv.csv"),
    '310': os.path.join(current_folder, "..", "..", "data", "processed", "df_310_labeled_on_lift_v4.csv"),
    '290': os.path.join(current_folder, "..", "..", "data", "processed", "df_290_labeled_on_lift.csv"),
    '206': os.path.join(current_folder, "..", "..", "data", "processed", "df_206_labeled_on_lift.csv"),
    'Andermatt_Gondelbahn_Gutsch': os.path.join(current_folder, "..", "..", "data", "processed", "df_Andermatt_Gondelbahn_Gutsch_2024_02_04_no_lift_down.csv"),
    'Natschen_18': os.path.join(current_folder, "..", "..", "data", "processed", "df_Natschen_18_not_lift_down.csv"),
    'Natschen_46': os.path.join(current_folder, "..", "..", "data", "processed", "df_Natschen_46_no_lift_down.csv"),
}

model_path = os.path.join(current_folder, "..", "..", "models", "rf_v_0.4.pkl")

image_path = os.path.join(current_folder, "..", "..", "data", "images", "header.png")

#Create instances from classes
preprocessor = PreProcessor()
model = Model()
plotting = Plotting()
mapping = Mapping()
postprocessor = PostProcessor()

# Add header image
header_image = image_path
st.image(header_image, use_column_width=True)  

# #Title of the app
# st.title('Alturos Lift Identifaction')

# #Dropdown menu for file selection
selected_data_label = st.selectbox('Choose a tracked ski ride:', list(data_paths.keys()))

# Button to start the prediction process
if st.button('Analyse the data'):
    with st.spinner('Processing data and generating predictions...'):
        try:
            # Load data
            data = preprocessor.import_data(data_paths[selected_data_label])

#           # Preprocessing and feature engineering
            data = preprocessor.convert_datetime(data)
            resampled_data = preprocessor.reduce_sampling_rate(data)
            data_ready_for_prediction = preprocessor.add_engineered_features(resampled_data)

            # Modelling
            rfc = model.load(model_path)
            prediction = model.predict_on_features(rfc, data_ready_for_prediction)

            # Post-Processing

            # Update predictions with mask
            prediction_masked, event_log = postprocessor.generate_misclassification_mask(prediction)

            #Generate on lift event assignments
            prediction_events, continuous_events_dict = postprocessor.on_lift_event_identification(prediction_masked, event_log)

            # Visualisation
            fig_mask = plotting.predictions(prediction_masked, target_column='mask')
            fig_event = plotting.predictions(prediction_events, target_column='event')

            # Display the plot with results
            st.plotly_chart(fig_mask)
            st.plotly_chart(fig_event)
            

        except Exception as e:
            st.error(f'An error occurred: {e}')