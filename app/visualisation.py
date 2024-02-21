import pandas as pd
import folium
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import streamlit as st

class Plotting:

    def predictions(self, df, target_column='predicted'):
        # Define the plot title based on the target column
        if target_column == 'on_lift':
            plot_title = 'Predictions'
        elif target_column == 'mask':
            plot_title = 'Classification Of Lift Rides'
        elif target_column == 'event':
            plot_title = 'Lift Events'
        else:
            plot_title = 'Predictions'

        # Check if 'Timestamp' column exists and is in datetime format
        if 'Timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            # Create a scatter plot for Altitude over Time, colored by target_column with an accessible color scheme
            fig = go.Figure()

            # Split the dataframe based on the target_column categories
            if target_column == 'mask':
                df_on_lift = df[df[target_column] == 1]
                df_not_on_lift = df[df[target_column] == 0]

                # Add traces for each category
                fig.add_trace(go.Scatter(x=df_on_lift['Timestamp'], y=df_on_lift['Alt(m)'],
                                        mode='markers', marker=dict(size=4, color='#c93a92'), name='On lift'))
                fig.add_trace(go.Scatter(x=df_not_on_lift['Timestamp'], y=df_not_on_lift['Alt(m)'],
                                        mode='markers', marker=dict(size=4, color='#3e748c'), name='Not on lift'))
            else:
                fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Alt(m)'],
                                        mode='markers', marker=dict(size=6), name='Status'))

            # Customize the layout
            fig.update_layout(
                title=plot_title,
                xaxis_title='Time',
                yaxis_title='Altitude (m)',
                width=700,
                height=425,
                legend=dict(title=None, traceorder='normal')
            )

            # Show the plot
            return fig

        else:
            print("Warning: DataFrame's 'Timestamp' column is not in datetime format and must be converted first.")





    def total_alt_over_time(self, df, plot_title='Total Tracked Altitude Over Time'):
        """
        Plot the total tracked altitude over time using a line plot.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to plot.
        - plot_title (str): The title of the plot.

        Returns:
        - None
        """
        # Create a line plot using Plotly
        fig = go.Figure()

        # Add a trace for altitude over time
        fig.add_trace(go.Scatter(x=df['Timestamp'],
                                 y=df['Alt(m)'],
                                 mode='lines',
                                 name='Altitude'))

        # Update layout
        fig.update_layout(title=plot_title,
                          xaxis_title='Timestamp',
                          yaxis_title='Altitude (m)')

        # Show plot
        fig.show()

class Mapping:

    def tracked_movement(self, df, zoom_start=11):
        """
        Map all tracked movement based on latitude and longitude from GPS data.

        Parameters:
        - df (DataFrame): The DataFrame containing GPS data with 'Lat' and 'Long' columns.
        - zoom_start (int): The initial zoom level of the map.

        Returns:
        - folium.Map: The map object displaying the tracked movement.
        """
        
        # Create a map centered on the mean latitude and longitude
        map_center = [df['Lat'].mean(), df['Long'].mean()]
        movement_on_map = folium.Map(location=map_center, zoom_start=zoom_start)
        # Add CircleMarkers for each data point
        for index, row in df.iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Long']], radius=5, color='blue', fill=True, fill_color='blue').add_to(movement_on_map)
        # Display the map
        return movement_on_map

    def map_lifts_and_other_movement(self, df, column='mask', zoom_start=14):
        """
        Map lift rides in red and all other movement in blue based on a specified column.

        Parameters:
        - df (DataFrame): The DataFrame containing GPS data with 'Lat' and 'Long' columns, and the specified column indicating lift rides.
        - column (str): The column indicating lift rides (default is 'on_lift').
        - zoom_start (int): The initial zoom level of the map.

        Returns:
        - folium.Map: The map object displaying the lift rides and other movement.
        """
        # Create a map centered on the mean latitude and longitude
        map_center = [df['Lat'].mean(), df['Long'].mean()]
        tracking_map = folium.Map(location=map_center, zoom_start=zoom_start,
                                  tiles='https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png', 
                                  attr='Map data © OpenStreetMap contributors, CC-BY-SA, Tiles by Stamen Design', 
                                  width=700, height=700, 
                                   )

        # Plot data points with not_on_lift  types
        not_on_lift = df[df[column] != 1]
        for _, row in not_on_lift.iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Long']], radius=2, color='#3e748c', fill=True, fill_color='#3e748c', tooltip=str(row['Timestamp'])).add_to(tracking_map)

        # Plot data points with on_lift type
        on_lift = df[df[column] == 1]
        for _, row in on_lift.iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Long']], 
                                radius=0.5, 
                                color='#c93a92', 
                                fill=True, 
                                fill_color='#c93a92', 
                                tooltip=str(row['Timestamp'])).add_to(tracking_map)
        
        # Return the map object
        return tracking_map

    def lift_rides(self, df, column='on_lift', zoom_start=13):
        """
        Map only lift rides based on a specified column.

        Parameters:
        - df (DataFrame): The DataFrame containing GPS data with 'Lat' and 'Long' columns, and the specified column indicating lift rides.
        - column (str): The column indicating lift rides (default is 'on_lift').
        - zoom_start (int): The initial zoom level of the map.

        Returns:
        - folium.Map: The map object displaying only lift rides.
        """
    
        # Create a map centered on the mean latitude and longitude
        map_center = [df['Lat'].mean(), df['Long'].mean()]
        lift_map = folium.Map(location=map_center, zoom_start=zoom_start)

        # Plot data points with on_lift type
        on_lift = df[df[column] == 1]
        for _, row in on_lift.iterrows():
            folium.CircleMarker(location=[row['Lat'],
                                        row['Long']],
                                        radius=5,
                                        color='red',
                                        fill=True,
                                        fill_color='red',
                                        tooltip=str(row['Timestamp'])).add_to(lift_map)

        # Return the map object
        return lift_map

# class Mapping:
    def tracked_movement(self, df, zoom_start=12):
        """
        Map all tracked movement based on latitude and longitude from GPS data.

        Parameters:
        - df (DataFrame): The DataFrame containing GPS data with 'Lat' and 'Long' columns.
        - zoom_start (int): The initial zoom level of the map.

        Returns:
        - folium.Map: The map object displaying the tracked movement.
        """
        
        # Create a map centered on the mean latitude and longitude
        map_center = [df['Lat'].mean(), df['Long'].mean()]
        movement_on_map = folium.Map(location=map_center, zoom_start=zoom_start)
        # Add CircleMarkers for each data point
        for index, row in df.iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Long']], radius=5, color='blue', fill=True, fill_color='blue').add_to(movement_on_map)
        # Display the map
        return movement_on_map

    def lifts_and_other_movement(self, df, column='on_lift', zoom_start=15):
        """
        Map lift rides in red and all other movement in blue based on a specified column.

        Parameters:
        - df (DataFrame): The DataFrame containing GPS data with 'Lat' and 'Long' columns, and the specified column indicating lift rides.
        - column (str): The column indicating lift rides (default is 'on_lift').
        - zoom_start (int): The initial zoom level of the map.

        Returns:
        - folium.Map: The map object displaying the lift rides and other movement.
        """
        
        # Create a map centered on the mean latitude and longitude
        map_center = [df['Lat'].mean(), df['Long'].mean()]
        tracking_map = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Plot data points with on_lift type in red
        on_lift = df[df[column] == 1]
        for _, row in on_lift.iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Long']], radius=5, color='red', fill=True, fill_color='red', tooltip=str(row['Timestamp'])).add_to(tracking_map)
        
        # Plot data points with not_on_lift types in blue
        not_on_lift = df[df[column] != 1]
        for _, row in not_on_lift.iterrows():
            folium.CircleMarker(location=[row['Lat'], row['Long']], radius=5, color='blue', fill=True, fill_color='blue', tooltip=str(row['Timestamp'])).add_to(tracking_map)
        
        # Return the map object
        return tracking_map

    def lift_rides(self, df, column='on_lift', zoom_start=13):
        """
        Map only lift rides based on a specified column.

        Parameters:
        - df (DataFrame): The DataFrame containing GPS data with 'Lat' and 'Long' columns, and the specified column indicating lift rides.
        - column (str): The column indicating lift rides (default is 'on_lift').
        - zoom_start (int): The initial zoom level of the map.

        Returns:
        - folium.Map: The map object displaying only lift rides.
        """
    
        # Create a map centered on the mean latitude and longitude
        map_center = [df['Lat'].mean(), df['Long'].mean()]
        lift_map = folium.Map(location=map_center, zoom_start=zoom_start)

        # Plot data points with on_lift type
        on_lift = df[df[column] == 1]
        for _, row in on_lift.iterrows():
            folium.CircleMarker(location=[row['Lat'],
                                        row['Long']],
                                        radius=5,
                                        color='red',
                                        fill=True,
                                        fill_color='red',
                                        tooltip=str(row['Timestamp'])).add_to(lift_map)

        # Return the map object
        return lift_map