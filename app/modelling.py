import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import joblib

class Model:

    # ???
    # def __init__(self, features_to_use):
    #     self.features_to_use = features_to_use


    def load(self, file_path_to_model):
        """
        Load a pretrained machine learning model from the given file path.

        Parameters:
        - file_path_to_model (str): The file path to the saved model.

        Returns:
        - model: The loaded machine learning model.
        """
        # Load model
        return joblib.load(file_path_to_model)

    def predict_on_features(self, model, df):
        #    V2 combined old select features and predict on features functions
        """
        Select a subset of features from a DataFrame and make a prediction based on those features, returns full dataframe including prediction.

        Args:
            df (pd.DataFrame): DataFrame containing the features.

        Returns:
            pd.DataFrame: A DataFrame containing only the selected features.
        """
        features_to_use = ['accelX(g)', 'accelY(g)', 'accelZ(g)', 'accelUserX(g)', 'accelUserY(g)',
                        'accelUserZ(g)', 'gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)',
                        'Roll(rads)', 'Pitch(rads)', 'Yaw(rads)', 'Lat', 'Long', 'Speed(m/s)',
                        'HorizontalAccuracy(m)', 'VerticalAccuracy(m)', 'Course', 'calMagX(µT)',
                        'calMagY(µT)', 'calMagZ(µT)', 'Alt(m)_change',
                        'Speed(m/s)_change', 'Course_change']

        # Check if all features exist in the DataFrame
        missing_features = [feature for feature in features_to_use if feature not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        # Select the features
        X = df.copy()[features_to_use]

        #predict on selected features
        predictions = model.predict(X)
        df.loc[:, 'predicted'] = predictions

        return df


    def show_hyperparameters(self, model):
        # show hyperparameters
        return model.get_params()
