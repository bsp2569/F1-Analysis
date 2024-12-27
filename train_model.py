import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load your training data
data = pd.read_csv('results.csv')

# Handle missing or non-numeric values
data = data.replace('\\N', pd.NA)
data = data.dropna(subset=['grid', 'laps', 'milliseconds'])  # Drop rows with missing values
data[['grid', 'laps', 'milliseconds']] = data[['grid', 'laps', 'milliseconds']].astype(float)  # Convert to float

# Prepare features and target
X = data[['grid', 'laps', 'milliseconds']]  # Example features
y = (data['positionOrder'] == 1).astype(int)  # Binary target: 1 for winner, 0 otherwise

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
dump(model, 'f1_pipe.joblib')
print("Model trained and saved as f1_pipe.joblib")
