import pandas as pd

class PreProcessor:

    def import_data(self, file_path):
        """
        Import data from the specified file path.

        Parameters:
        - file_path (str): The file path of the data to import.

        Returns:
        - pd.DataFrame: The imported DataFrame.
        """
        # Read data from the specified file path
        df_raw = pd.read_csv(file_path)

        # Return the imported DataFrame
        return df_raw

    #updated convert_datetime function
    def convert_datetime(self, df, inplace=False):
        """
        Convert the 'Timestamp' column in a DataFrame to datetime format.

        Args:
            df (pd.DataFrame): DataFrame containing the 'Timestamp' column.
            inplace (bool): Whether to modify the original DataFrame or create a copy.

        Returns:
            pd.DataFrame: The DataFrame with the 'Timestamp' column converted to datetime format.
        """
        # Check if 'Timestamp' column exists
        if 'Timestamp' not in df.columns:
            raise ValueError("Column 'Timestamp' not found in DataFrame.")

        # Convert 'Timestamp' column to datetime format
        if inplace:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns').dt.floor('s')
        else:
            df = df.copy()
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns').dt.floor('s')

        return df

    def reduce_sampling_rate(self, df):
        """
        Reduce the sampling rate of the DataFrame by resampling to seconds.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing timestamped data.

        Returns:
        - pd.DataFrame: The DataFrame with reduced sampling rate.
        """
        # Set the 'Timestamp' column as the DataFrame's index
        df.set_index('Timestamp', inplace=True)
        
        # Resample the DataFrame to seconds and keep the first value of each second
        df_resampled = df.resample('S').first()
        
        # Reset the index to create a standard numerical index
        df_resampled.reset_index(inplace=True)
        
        # Option to export to CSV
        # df_resampled.to_csv('df_206_reduced_sample_rate.csv', index=False)
        
        return df_resampled


    def add_engineered_features(self, df, alt_threshold=20, speed_threshold=3, inplace=False):
        """
        Calculate the altitude change, speed change, and course change between consecutive rows and remove outliers.

        Args:
            df (pd.DataFrame): DataFrame containing the altitude, speed, and course data.
            alt_threshold (float): Threshold value for altitude change outlier detection.
            speed_threshold (float): Threshold value for speed change outlier detection.
            inplace (bool): Whether to modify the original DataFrame or create a copy.

        Returns:
            tuple: A tuple containing the modified DataFrame and a dictionary with shape information.
        """
        # Check if required columns exist
        required_columns = ['Alt(m)', 'Speed(m/s)', 'Course']
        if not all(col in df.columns for col in required_columns):
            logger.error("Required columns not found in DataFrame.")
            return None, {}

        # Store the initial DataFrame size
        initial_size = len(df)

        # Calculate changes
        df['Alt(m)_change'] = df['Alt(m)'].diff().fillna(0)
        df['Speed(m/s)_change'] = df['Speed(m/s)'].diff().fillna(0)
        df['Course_change'] = df['Course'].diff().fillna(0)

        # Remove outliers
        mask = (df['Alt(m)_change'].abs() <= alt_threshold) & \
            (df['Speed(m/s)_change'].abs() <= speed_threshold)
        filtered_df = df[mask] if inplace else df.copy()[mask]

        # Reset the index of the filtered DataFrame and drop null values
        filtered_df = filtered_df.dropna()
        filtered_df = filtered_df.reset_index(drop=True)

        # # Log the shape after outlier removal
        # logger.info("Shape before outlier removal: %d", initial_size)
        # logger.info("Shape after outlier removal: %d", len(filtered_df))

        return filtered_df