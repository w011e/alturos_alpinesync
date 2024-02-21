import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
from sklearn.neighbors import BallTree

class UX:

    def get_lift_names_and_count(self, df, lifts_db):
        
        df = df[df['event']!=0]
        
        # Initialise counter for lift usage
        lift_usage_counter = {}

        # Group df by  'event' column
        df_grouped = df.groupby('event')

        # Iterate over each group
        for event, group in df_grouped:
            # Extract start and end coordinates from the first and last rows of the group
            start_row = group.iloc[0]
            end_row = group.iloc[-1]

            # Extract start coordinates of events
            start_coords = (start_row['Lat'], start_row['Long'])

            # Convert start coordinates to radians
            start_coords_rad = np.radians([start_coords])

            # get hold of start and end alt
            start_alt = start_row['Alt(m)']
            end_alt = end_row['Alt(m)']

            # compare start_alt and end_alt to decide if start_coords should be compared to top_coord or base_coord
            if start_alt < end_alt:
                # Convert lift base locations to radians
                lift_base_locations_rad = np.radians([[lift['base_latitude'], lift['base_longitude']] for _, lift in lifts_db.iterrows()])
                # Use BallTree to find the nearest lift for the start coordinates
                base_tree = BallTree(lift_base_locations_rad, metric='haversine')
                _, base_indices = base_tree.query(start_coords_rad, k=1)
                # Get the lift name for the nearest lift to the start coordinates
                base_lift_name = lifts_db.iloc[base_indices.flatten()[0]]['lift_name']
                # Update lift usage counter
                lift_usage_counter[base_lift_name] = lift_usage_counter.get(base_lift_name, 0) + 1
            elif start_alt > end_alt:
                # Convert lift top locations to radians
                lift_top_locations_rad = np.radians([[lift['top_latitude'], lift['top_longitude']] for _, lift in lifts_db.iterrows()])
                # Use BallTree to find the nearest lift for the start coordinates
                top_tree = BallTree(lift_top_locations_rad, metric='haversine')
                _, top_indices = top_tree.query(start_coords_rad, k=1)
                # Get the lift name for the nearest lift to the start coordinates
                top_lift_name = lifts_db.iloc[top_indices.flatten()[0]]['lift_name']
                # Update lift usage counter
                lift_usage_counter[top_lift_name] = lift_usage_counter.get(top_lift_name, 0) + 1

        number_of_lift_rides = 0
        for _, count in lift_usage_counter.items():
            number_of_lift_rides += count

        return lift_usage_counter, number_of_lift_rides

    def alt_climbed_on_lift(self, lifts_db, lift_usage_counter):
        """
        Calculate the total altitude climbed based on lift usage.

        Parameters:
        - lifts_db (pandas DataFrame): DataFrame containing lift information.
        - lift_usage_counter (dict): Dictionary with lift names as keys and the number of times each lift was used as values.

        Returns:
        - total_alt (float): Total altitude climbed.
        """
        total_alt = 0 

        # Loop through each lift in the lift usage counter
        for lift_name, count in lift_usage_counter.items():
            # Filter lift data based on the current lift name
            lift_data = lifts_db[lifts_db['lift_name'] == lift_name]

            # Iterate over each row in the filtered lift data
            for _, lift in lift_data.iterrows():           
                # Calculate the lift length
                lift_length = lift['top_station(m)'] - lift['base_station(m)']
                
                # Update total altitude climbed by adding the product of lift length and count
                total_alt += lift_length * int(count)

        return total_alt



            
    def time_spent_on_lift(self, lifts_db, lift_usage_counter):

        total_time = timedelta()

        for lift_name, count in lift_usage_counter.items():
            # Filter lift data based on the current lift name
            lift_data = lifts_db[lifts_db['lift_name'] == lift_name]

            # Iterate over each row in the filtered lift data
            for _, lift in lift_data.iterrows():
                # Calculate the lift length
                time_on_lift = pd.to_datetime(lift['transit_time'], unit='ns')
                
                # Extract time components
                lift_hours = time_on_lift.hour
                lift_minutes = time_on_lift.minute
                lift_seconds = time_on_lift.second
                
                # Convert time components to timedelta
                time_on_lift_delta = timedelta(hours=lift_hours, minutes=lift_minutes, seconds=lift_seconds)
                
                # Add the calculated lift length multiplied by the count to the total time
                total_time += time_on_lift_delta * count

        return total_time
