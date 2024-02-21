from pre_processing import PreProcessor
from modelling import Model
from post_processing import PostProcessor
from visualisation import Plotting, Mapping
from ux import UX 
from simulation_animation import SimulationAnim

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
intro_pic = os.path.join(parent_dir, "data", "images", "pejo_skiing_chairlift_trentino.jpg")

# Team profile pics
Tosin_profile_path = os.path.join(parent_dir, "data", "images", "Sebastian_profile.png")
Paul_profile_path = os.path.join(parent_dir, "data", "images", "Sebastian_profile.png")
Raphael_profile_path = os.path.join(parent_dir, "data", "images", "Sebastian_profile.png")
Sebastian_profile_path = os.path.join(parent_dir, "data", "images", "Sebastian_profile.png")

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


tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Analysis", "Simulation", "About us"])

with tab1:
   st.image(intro_pic)
   st.markdown('''

## AlpineSync 
### Elevating the Ski Experience with Data Science

As skiers navigate the slopes of the world's ski resorts, their experiences have evolved into data-centric adventures. They are seeking a deeper understanding of their performance by utilizing technology to access detailed analytics and performance metrics. This implies monitoring progress, evaluating achievements, and capturing key moments.

#### The Idea
Utilize sensor data from skier’s mobile devices to automatically track ski-lift usage within the Alturos Skiline mobile app. This can serve as an additional way to collect data when ticket scanners are inoperational, and could potentially replace them altogether. These machine learning algorithms will enable real-time data analysis on mobile devices and differentiate skier’s utilization of various ski-lifts.

#### The Company
Alturos Destinations specializes in crafting and implementing state-of-the-art digitalization strategies that redefine the landscape of tourism. The Skiline App revolutionizes the skiing experience by capturing and analyzing skiers' data to provide personalized statistics and memories of their mountain adventures.
str
#### The Approach
Our endeavor focuses on harnessing sensor data from mobile devices to accurately identify ski lift usage.

##### Incoming Data
We utilized sensor data from two popular mobile apps: Sensor Logger for iOS devices and Sensors Toolbox for Android devices. This data formed the foundation for our lift detection algorithm.

##### Data Management
To train our models effectively, we utilized labeled data for supervised machine learning from 12 different sessions and two geographical locations. Datasets were then prepared by incorporating outlier detection and engineered features to maximize predictive performance.

##### Modeling
We explored various machine learning models, prioritizing accuracy metrics and selecting a Random Forest Classifier as the most suitable model for lift detection.

##### Post-Processing
We achieved a 96% accuracy rate with our Random Forest Classifier (RFC) model. Through the implementation of post-processing strategies, we enhanced the continuity of classified events and minimized misclassifications. These refinements contribute to a more consistent user experience and improve overall accuracy.

##### UX Development
Our focus extended beyond algorithmic development to user experience enhancement. We plotted and mapped ski-lift events, simulated real-time predictions, and generated lift usage statistics.

#### The AlpineSync App
We created a Streamlit App that serves as a Proof of Concept (POC) platform for visualization and interaction with our models. [AlpineSync (Link TBA)]

#### Future Vision
The incorporation of mapping features by leveraging Google Maps APIs and continual refinement of the models remain key objectives. Future initiatives may include:

- Model training and validation using datasets from diverse geographical areas
- Validation of the pipeline with live data input
- Integration of logging functionalities into functions
- Implementation of anticipated user experience enhancements within the Streamlit application
- Implementation of a clustering algorithm for grouping lift rides based on their origin and destination points

#### Project Owners
Tosin Aderanti, Paul Biesold, Raphael Penayo Schwarz, Sebastian Rozo

''')
with tab2:
    st.header("Analysis")
        #Dropdown menu for file selection
    selected_data_label = st.selectbox('Choose one of your tracked ski rides for analysis:', list(data_paths.keys()))
    if st.button('Analyze'):
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
            st.write(f"<b> You rode {number_of_lift_rides} lifts today:</b>", unsafe_allow_html=True)
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
    left_column, right_column = st.columns(2)
    
    # Button for visualization
    if left_column.button('Altitude graph'):
        with st.spinner('Graphing data ...'):
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
if right_column.button("Map lifts"):
    with st.spinner('Mapping data ...'):
        try:
            prediction_masked = st.session_state.prediction_masked

            # Heading
            st.write('<b>Lift Rides On Map</b>', unsafe_allow_html=True)
            st.image(legend_path, use_column_width=True)
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

        except Exception as e:
            st.error(f'An error occurred during map visualization: {e}')
            
with tab3:
    st.header("Simulation")
    selected_data_label = st.selectbox('Choose one of your tracked ski rides for simulation:', list(data_paths.keys()))
    if st.button('Simulate'):
        with st.spinner('Hitting the slopes ⛷️...'):
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
                fig = go.Figure()

                fig.add_trace(go.Scatter(mode='markers', 
                                         marker=dict(size=4, color='#c93a92'), 
                                         name='On lift'))
                
                fig.add_trace(go.Scatter(mode='markers', 
                                         marker=dict(size=4, color='#3e748c'), 
                                         name='Not on lift'))

                fig.update_layout(title="Real Time Simulation")
                
                chart = st.plotly_chart(fig)

                # Loop to update animation frames
                for frame in range(simulation_anim.num_chunks):
                    try:
                        # Initialize graph variables
                        aggregate_x_on = []
                        aggregate_y_on = []
                        aggregate_colors_on = []

                        # Initialize graph variables
                        aggregate_x_off = []
                        aggregate_y_off = []
                        aggregate_colors_off = []

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
                            processed_chunk_on = ani_prediction[ani_prediction['mask'] == 1]
                            processed_chunk_off = ani_prediction[ani_prediction['mask'] == 0]
                            
                            # Extract x, y, and color data from the chunk
                            x_chunk_on = processed_chunk_on['Timestamp']
                            y_chunk_on = processed_chunk_on['Alt(m)']

                            # Extract x, y, and color data from the chunk
                            x_chunk_off = processed_chunk_off['Timestamp']
                            y_chunk_off = processed_chunk_off['Alt(m)']

                            # Aggregate data for the frame
                            aggregate_x_on.extend(x_chunk_on)
                            aggregate_y_on.extend(y_chunk_on)
                                                        
                            aggregate_x_off.extend(x_chunk_off)
                            aggregate_y_off.extend(y_chunk_off)          
                        
                            # Update the Plotly chart with the aggregated data
                            fig.data[0].x = aggregate_x_on
                            fig.data[0].y = aggregate_y_on
                            fig.data[0].marker.color = '#c93a92'

                            fig.data[1].x = aggregate_x_off
                            fig.data[1].y = aggregate_y_off
                            fig.data[1].marker.color = '#3e748c'

                            chart.plotly_chart(fig, use_container_width=True)
                            
                            # Add a small delay to control animation speed
                            time.sleep(0.05)

                    except Exception as e:
                        st.error(f'An error occurred generating the frames: {e}')                 
            except:
                st.error(f'An error occurred: {e}')

with tab4:
    st.header("About us")
    st.image(Tosin_profile_path)
    st.markdown('''
                ### Tosin Aderanti:
               ''')
    st.image(Paul_profile_path)
    st.markdown('''
                ### Paul Biesold:
               ''')
    st.image(Raphael_profile_path)
    st.markdown('''
                ### Raphael Penayo Schwarz:
               ''')
    st.image(Sebastian_profile_path)
    st.markdown('''
                ### Sebastian Rozo:
                Sebastian is awesome
               ''')