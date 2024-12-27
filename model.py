import pandas as pd
from joblib import load  # Using joblib for better model serialization

def load_model():
    """Load the trained model."""
    return load('f1_pipe.joblib')  # Ensure the correct model file is loaded

def predict_winner(pipe, input_data):
    """Predict the winner based on the highest probability."""
    probabilities = pipe.predict_proba(input_data)
    winner_index = probabilities[:, 1].argmax()  # Index of the highest win probability
    win_probability = round(probabilities[winner_index, 1] * 100)
    return input_data.iloc[winner_index]['driver_name'], win_probability

def prepare_input_data(driver_name, constructor, track, grid_position, laps_completed, laps_remaining):
    """Prepare the input data for prediction."""
    return pd.DataFrame({
        'driver_name': [driver_name],
        'constructor': [constructor],
        'track': [track],
        'grid_position': [grid_position],
        'laps_completed': [laps_completed],
        'laps_remaining': [laps_remaining]
    })
