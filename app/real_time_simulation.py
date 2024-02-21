import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates


class DataAnimation:
    def __init__(self, df, file_path_to_model):
        self.df = df
        self.file_path_to_model = file_path_to_model
        self.chunk_size = 60
        self.data_agg = pd.DataFrame()
        self.fig, self.ax = plt.subplots()
        self.data_generator = DataAnimation.get_next_chunk(self, self.df, self.chunk_size)
        self.total_chunks = len(df) // 60
        # self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.total_chunks, interval=200, blit=False, repeat=False)
        # plt.show()

    def update(self, i):   
        # Get the next chunk of data
        data_chunk = next(self.data_generator)
        processed_chunk = DataAnimation.complete_pipeline(self, data_chunk, file_path_to_model)

        # Concatenate the processed chunk with the aggregated data
        self.data_agg = pd.concat([self.data_agg, processed_chunk])

        # Update the plot with the new data
        sc = self.ax.scatter(self.data_agg['Timestamp'], self.data_agg['Alt(m)'], c=self.data_agg['color'])

        # Set the axes limits dynamically based on the new data
        x_min = self.data_agg['Timestamp'].min() - pd.Timedelta(minutes=5)
        x_max = self.data_agg['Timestamp'].max() + pd.Timedelta(minutes=5)
        y_min = self.data_agg['Alt(m)'].min() - 100
        y_max = self.data_agg['Alt(m)'].max() + 100

        # Set the axes limits if necessary
        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([y_min, y_max])

        # Update the x-axis major locator and formatter
        #self.ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        #Set tick mark angle
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')

        # Add labels and title if necessary
        self.ax.set_title(f'Real-time Data Analysis (minute: {i})')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Altitude (m)')

         # Add legend with custom labels
        legend_labels = {'red': 'on_lift'}
        self.ax.legend(handles=sc.legend_elements()[0], labels=legend_labels)

        # Return the scatter plot object to blit
        return sc

    # Define your custom function
    def complete_pipeline(self, data_chunk, file_path_to_model):
        # Define colors for the categories
        colors = {1: 'red',  0: 'blue'}
        #file_path_to_model = '../../models/rf_v_0.4.pkl'
        df = add_engineered_features(data_chunk)
        df = convert_datetime(df)
        # Feature selection
        features = select_features(df)
        rfc=load_model(file_path_to_model)
        # Make predictions
        df = predict_on_features(rfc, df, features)
        df, event_log = generate_misclassification_mask(df)
        df['color'] = df['mask'].map(colors)
        return df

    # Placeholder for your data stream setup
    def get_next_chunk(self, df, chunk_size):
        total_chunks = len(df) // chunk_size
        for i in range(total_chunks):
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            data_chunk = df.iloc[start_index:end_index]
            yield data_chunk









# class DataAnimation:
#     def __init__(self, df):
#         self.df = df
#         # Initialize the data generator here if needed, or prepare it in another method
#         self.chunk_size = 60
#         self.data_agg = pd.DataFrame()
#         self.fig, self.ax = plt.subplots()
#         # This will call the method to prepare the data generator
#         self.prepare_data_generator()

#     def prepare_data_generator(self):
#         # Setup the data generator here, so it's ready to use in animate
#         self.data_generator = self.get_next_chunk(self.df, self.chunk_size)

#     def the_thing(self, file_path_to_model):
#         self.file_path_to_model = file_path_to_model
#         total_chunks = len(self.df) // self.chunk_size
#         plt.switch_backend('Agg')  # Keep this if you're generating files or remove if testing interactively
#         ani = animation.FuncAnimation(self.fig, self.animate, frames=total_chunks, interval=200, blit=False, repeat=False)
#         return ani

#     def animate(self, i):
#         # Make sure to use 'self' to access the instance variables
#         try:
#             data_chunk = next(self.data_generator)
#         except StopIteration:
#             return
#         processed_chunk = self.complete_pipeline(data_chunk, self.file_path_to_model)

#         # Update data_agg
#         self.data_agg = pd.concat([self.data_agg, processed_chunk])

#         # Update plot
#         # Your plotting code here

#     def get_next_chunk(self, df, chunk_size):
#         for start in range(0, len(df), chunk_size):
#             yield df.iloc[start:start + chunk_size]
