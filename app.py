from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained model
pipe = load('f1_pipe.joblib')  # Ensure the correct model file is loaded

# Load driver and constructor data
drivers_constructors_df = pd.read_csv('F1_2025_Drivers_Constructors.csv')

# Extract the top 20 drivers and their constructors
drivers_on_grid = drivers_constructors_df[['Driver', 'Constructor']].drop_duplicates().head(20)

@app.route('/')
def home():
    # Pass driver-constructor pairs to the template
    driver_options = drivers_on_grid.to_dict('records')
    return render_template('index.html', driver_options=driver_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input from the form
        track = request.form['track']
        drivers = request.form.getlist('drivers[]')  # List of selected driver names
        positions = request.form.getlist('positions[]')  # List of starting positions
        current_lap = int(request.form['current_lap'])
        total_laps = int(request.form['total_laps'])

        # Validate input
        if len(drivers) != 20 or len(positions) != 20:
            return render_template('index.html', error="Please provide exactly 20 drivers and their starting positions.")

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            "track": [track] * 20,
            "driver": drivers,
            "starting_position": [int(pos) for pos in positions],
            "current_lap": [current_lap] * 20,
            "total_laps": [total_laps] * 20
        })

        # Make predictions
        results = pipe.predict_proba(input_data)
        predictions = [
            {
                "driver_name": driver,
                "win_probability": round(result[1] * 100, 2)
            }
            for driver, result in zip(drivers, results)
        ]

        # Sort predictions by win probability
        predictions = sorted(predictions, key=lambda x: x['win_probability'], reverse=True)

        # Highlight top 10 and podium finishers
        top_10 = predictions[:10]
        podium = predictions[:3]

        # Render results
        return render_template(
            'result.html',
            track=track,
            current_lap=current_lap,
            total_laps=total_laps,
            predictions=predictions,
            top_10=top_10,
            podium=podium
        )

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
