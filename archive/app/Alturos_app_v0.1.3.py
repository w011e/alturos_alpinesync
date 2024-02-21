from pre_processing import PreProcessor
from modelling import Model
from post_processing import PostProcessor
from visualisation import Plotting, Mapping
from ux import UX 
from real_time_simulation import DataAnimation

import streamlit as st
import logging
import os

##################
## plotting dependencies

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Test dep
from sine_animationv013 import SineAnim
import plotly.graph_objects as go
import time
###################


# # # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(_name
# )

# Determine the absolute path to the data file
current_folder = os.path.dirname(__file__)

#Define paths to data and model files
data_paths = {
    '7 Apr 2023': os.path.join(current_folder, "..", "..", "data", "processed", "df_95_labeled_on_lift.csv"), # 95
    '8 Apr 2023': os.path.join(current_folder, "..", "..", "data", "processed", "df_310_labeled_on_lift_v4.csv"), # 310
    '4 Feb 2024': os.path.join(current_folder, "..", "..", "data", "processed", "df_Natschen_18_not_lift_down.csv"), # Natschen_18
    '10 Feb 2024': os.path.join(current_folder, "..", "..", "data", "raw", "v5_20240210_093434_131m.csv"), # 131
    '11 Feb 2024': os.path.join(current_folder, "..", "..", "data", "raw", "v5_20240211_113607_100m.csv"), # 100
    }
model_path = os.path.join(current_folder, "..", "..", "models", "rf_v_0.4.pkl")
logo_path = os.path.join(current_folder, "..", "..", "data", "images", "Alturos-logo_white.png")
image_path = os.path.join(current_folder, "..", "..", "data", "images", "Alturos_frontpage.png")
lifts_db = os.path.join(current_folder, "..", "..", "data", "lift_data", "lifts_db_v0.1.csv")

#Create instances from classes
preprocessor = PreProcessor()
model = Model()
plotting = Plotting()
mapping = Mapping()
postprocessor = PostProcessor()
ux = UX()

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

    # Button for mapping features
    if middle_column.button("Map lifts"):
        try:
            prediction_masked = st.session_state.prediction_masked

            # Heading
            st.write('<b>Lift Rides On Map</b>', unsafe_allow_html=True)

            # Generate Folium map
            folium_map = mapping.lift_rides(prediction_masked, column='mask') 

            # Display the Folium map
            st.components.v1.html(folium_map._repr_html_(), width=700, height=425)

        except Exception as e:
            st.error(f'An error occurred during map visualization: {e}')

    # Button for simulation
    if right_column.button('Simulation'):
        try:
            # Load data
            animation_data = preprocessor.import_data(data_paths[selected_data_label])
            #st.dataframe(animation_data)        

            # Initialize class 
            sine_anim = SineAnim(animation_data)

            # Generate initial data
            x, y, colors = sine_anim.generate_data()
        
            # Create the figure and add the initial trace
            fig = go.Figure(data=go.Scatter(x=x, 
                                            y=y, 
                                            mode='markers',
                                            marker=dict(color=colors),
                                            ))


            # Customize the layout
            fig.update_layout(
                title='Real Time prediction',
                xaxis_title='Time',
                yaxis_title='Altitude (m)',
                width=700,
                height=425,
                legend=dict(title='Status', traceorder='normal')
            )

            # Update function for animation frames
            # Create the animation
            frames = len(x)
            # Display animation in Streamlit
            st.write("## Animation")
            chart_sine = st.plotly_chart(fig)

            # Loop to update animation frames
            for i in range(1, frames + 1):
                # Update frame
                x_frame, y_frame, colors_frames = sine_anim.update(i)
                fig.data[0].x = x_frame
                fig.data[0].y = y_frame
                fig.data[0].marker.color = colors_frames
                # Update the Plotly chart in Streamlit
                chart_sine.plotly_chart(fig, use_container_width=True)
                # Pause for a short duration to control animation speed
                time.sleep(0.001)
            # # ########################
                


            # ##################
            # # This generates a correctly colored graph
            # x = animation_data['Timestamp'].values
            # y = animation_data['Alt(m)'].values
            # colors = animation_data['on_lift'].values 
            # # Create the figure
            # fig = go.Figure(data=go.Scatter(x=x, 
            #                                 y=y, 
            #                                 mode='markers', 
            #                                 marker=dict(color=colors)
            #                                 ))
            # # Display the Plotly chart in Streamlit
            # st.write("## Sine Wave Animation")
            # st.plotly_chart(fig, use_container_width=True)
            # ##################

        except:
            # st.error(f'An error occurred: {e}')
            st.write('Horse Shit')




            # # Initialise class for animation
            # data_animation = DataAnimation(animation_data, model_path)

            # ani = animation.FuncAnimation(data_animation.fig, data_animation.update, frames=data_animation.total_chunks, interval=200, blit=False, repeat=False)

            # # Show animation 
            # st.pyplot(data_animation.fig)

            # #######################
            # # This shows a plotly animated sinewave it works
            # import plotly.graph_objects as go
            # import time

            # # Create a function to generate data for the animation
            # def generate_data():
            #     x = np.linspace(0, 2*np.pi, 100)
            #     y = np.sin(x)
            #     return x, y

            # # Generate initial data
            # x, y = generate_data()

            # # Create the figure and add the initial trace
            # fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))

            # # Update function for animation frames
            # def update(frame):
            #     x, y = generate_data()
            #     fig.data[0].x = x[:frame]
            #     fig.data[0].y = y[:frame]
            #     return fig

            # # Create the animation
            # frames = len(x)

            # # Display animation in Streamlit
            # st.write("## Sine Wave Animation")
            # chart = st.plotly_chart(fig)

            # # Loop to update animation frames
            # for i in range(1, frames + 1):
            #     # Update frame
            #     fig = update(i)
            #     # Update the Plotly chart in Streamlit
            #     chart.plotly_chart(fig, use_container_width=True)
            #     # Pause for a short duration to control animation speed
            #     time.sleep(0.4)
            # ########################

            #######################
            # This shows a plotly animated sinewave it works


