from pre_processing import PreProcessor
from modelling import Model
from post_processing import PostProcessor
from visualisation import Plotting, Mapping
from ux import UX 
from simulation_animation import SimulationAnim
import folium


import streamlit as st
import os
import time

import plotly.graph_objects as go

st.set_page_config(page_title="AlpineSync", 
                   page_icon="⛷️", 
                   initial_sidebar_state="expanded")

# Determine the absolute path to the data file
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))

#Define paths to data and model files
data_paths = {
    '7 Apr 2023': os.path.join(parent_dir, "data", "processed", "df_95_labeled_on_lift.csv"), # 95
    '8 Apr 2023': os.path.join(parent_dir, "data", "processed", "df_310_labeled_on_lift_v4.csv"), # 310
    '4 Feb 2024': os.path.join(parent_dir, "data", "processed", "df_Natschen_18_not_lift_down.csv"), # Natschen_18
    '10 Feb 2024': os.path.join(parent_dir, "data", "raw", "v5_20240210_093434_131m.csv"), # 131
    '11 Feb 2024': os.path.join(parent_dir, "data", "raw", "v5_20240211_113607_100m.csv"), # 100
    }

model_path = os.path.join(parent_dir, "models", "rf_v_0.4.pkl")
logo_path = os.path.join(parent_dir, "data", "images", "Alturos-logo_white.png")
image_path = os.path.join(parent_dir, "data", "images", "Alturos_frontpage.png")
legend_path =os.path.join(parent_dir, "data", "images", "Map_legend.png")
lifts_db = os.path.join(parent_dir, "data", "lift_data", "lifts_db_v0.1.csv")


#Create instances from classes
preprocessor = PreProcessor()
model = Model()
plotting = Plotting()
mapping = Mapping()
postprocessor = PostProcessor()
ux = UX()
st.write()
# Add Alturos logo
st.image(logo_path, use_column_width=False) 
# Add header image
st.image(image_path, use_column_width=True)  

# Dropdown menu for file selection
selected_data_label = st.selectbox('Choose one of your tracked ski rides for analysis:', list(data_paths.keys()))

if st.button('Analyse'):
    with st.spinner('Processing data ...'):
        try:
            # Load data
            data = preprocessor.import_data(data_paths[selected_data_label])

            # Preprocessing and feature engineering
            data = preprocessor.convert_datetime(data)
            resampled_data = preprocessor.reduce_sampling_rate(data)
            data_ready_for_prediction = preprocessor.add_engineered_features(resampled_data)

            # Modelling
            rfc = model.load(model_path)
            prediction = model.predict_on_features(rfc, data_ready_for_prediction)

            # Post-Processing
            prediction_masked, event_log = postprocessor.generate_misclassification_mask(prediction)
            prediction_events = postprocessor.on_lift_event_identification(prediction_masked, event_log)

            # Store processed data and predictions for later use
            st.session_state.prediction_masked = prediction_masked
            st.session_state.prediction_events = prediction_events

            # Inform the user that the analysis is complete
            st.success('Analysis complete!')

        except Exception as e:
            st.error(f'An error occurred: {e}')

# Show slft statistics
if 'prediction_masked' in st.session_state and 'prediction_events' in st.session_state:

    # Lift statistics 
    try:
            
        prediction_events = st.session_state.prediction_events
            
        # Load database with information on lifts 
        lifts_db = preprocessor.import_data(lifts_db)
        statistics, number_of_lift_rides = ux.get_lift_names_and_count(prediction_events, lifts_db)
            
        # Print number of times lift used today 
        st.write(f"<b>{number_of_lift_rides} lift rides today:</b>", unsafe_allow_html=True)
        for lift_name, count in statistics.items():
                st.write(f"- {lift_name} was used {count} times.")

        # Line break 
        st.write(' ')

        # Calculate alt_climbed_on_lift
        total_alt_climbed_on_lift = ux.alt_climbed_on_lift(lifts_db, statistics)
        # Print number of times lift used today 
        st.write(f'Total altitude climbed on lifts: {total_alt_climbed_on_lift} m')


        # Calculate time_spent_on_lift
        total_time_spent_on_lift = ux.time_spent_on_lift(lifts_db, statistics)
        # Print times spent on lift today 
        st.write(f"Total time spent on lifts: {total_time_spent_on_lift}")


        # Line break 
        st.write(' ')


        # Display buttons for further actions
        st.write('<b>Get more insights into your tracked ride:</b>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f'An error occurred during lift statistics: {e}')


    # Split the layout into  columns
    left_column, middle_column, right_column = st.columns(3)
    
    # Button for visualization
    if left_column.button('Altitude graph'):
        try:
            prediction_masked = st.session_state.prediction_masked
            # prediction_events = st.session_state.prediction_events

            # Visualisation
            fig_mask = plotting.predictions(prediction_masked, target_column='mask')
            # fig_events = plotting.predictions(prediction_events, target_column='event')

            # Display the plot with results
            st.plotly_chart(fig_mask)
            # st.plotly_chart(fig_events)

        except Exception as e:
            st.error(f'An error occurred during visualization: {e}')

    #Button for mapping features
    if middle_column.button("Map lifts"):

        # Heading
        st.write('<b>Lift Rides On Map</b>', unsafe_allow_html=True)

        with st.spinner('Mapping workout ...'):
            try:
                prediction_masked = st.session_state.prediction_masked

                # Generate Folium map
                folium_map = mapping.map_lifts_and_other_movement(prediction_masked) 

                #Get the HTML representation of the folium map
                map_html = folium_map._repr_html_()

                #Define the map width and height
                map_width, map_height = 715, 715

                # Modify the HTML to set the iframe size
                map_html = map_html.replace(
                    '<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;">',
                    f'<div style="width:100%;"><div style="position:relative;width:{map_width}px;height:{map_height}px;">'
                )

                # Display the modified HTML
                st.components.v1.html(map_html, width=map_width, height=map_height)
                st.image(legend_path, use_column_width=False)

            except Exception as e:
                st.error(f'An error occurred during map visualization: {e}')
            
            
    # Button for simulation
    if right_column.button('Simulation'):
        try:
            # Import data and preprocess
            animation_data = preprocessor.import_data(data_paths[selected_data_label])
            ani_converted_data = preprocessor.convert_datetime(animation_data)
            ani_resampled_data = preprocessor.reduce_sampling_rate(ani_converted_data)
            ani_data_ready_for_prediction = preprocessor.add_engineered_features(ani_resampled_data)
            
            #load model and set threshold for mask values
            rfc = model.load(model_path)
            threshold = 0.3
            
            # Initialize class and preload variables
            simulation_anim = SimulationAnim(ani_data_ready_for_prediction)

            # Create initial Plotly figure
            fig = go.Figure(data=go.Scatter(x=[], 
                                            y=[], 
                                            mode='markers'), 
                                            layout=dict(xaxis=dict(title="Time", tickangle=45),
                                                    yaxis=dict(title="Elevation (m)"))
                                                )
            # Display the Plotly chart in Streamlit
            st.write("## Simulation")
            chart = st.plotly_chart(fig)

            # Loop to update animation frames
            for frame in range(simulation_anim.num_chunks):
                try:
                    # Initialize graph variables
                    aggregate_x = []
                    aggregate_y = []
                    aggregate_colors = []
                    # Loop to update animation frames
                    while True:
                        chunk_data = simulation_anim.update()
                        if chunk_data is None:
                            break   
                        
                        # Modelling
                        ani_prediction = model.predict_on_features(rfc, chunk_data)
                        # Post-Process predictions
                        mean_value = ani_prediction['predicted'].mean()
                        mask_value =  1 if mean_value >= threshold else  0
                        ani_prediction['mask'] = mask_value
                        processed_chunk = ani_prediction
                        
                        # Extract x, y, and color data from the chunk
                        x_chunk = processed_chunk['Timestamp']
                        y_chunk = processed_chunk['Alt(m)']
                        colors_chunk = processed_chunk['mask']

                        # Aggregate data for the frame
                        aggregate_x.extend(x_chunk)
                        aggregate_y.extend(y_chunk)
                        aggregate_colors.extend(colors_chunk)            
                    
                        # Update the Plotly chart with the aggregated data
                        fig.data[0].x = aggregate_x
                        fig.data[0].y = aggregate_y
                        fig.data[0].marker.color = aggregate_colors

                        chart.plotly_chart(fig, use_container_width=True)
                        
                        # Add a small delay to control animation speed
                        time.sleep(0.3)
                except Exception as e:
                    st.error(f'An error occurred generating the frames: {e}')                 
        except:
            st.error(f'An error occurred: {e}')

