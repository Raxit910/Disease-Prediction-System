import pandas as pd
import joblib
import sys
import json

# Load the trained Random Forest model
rf_model = joblib.load('models/liver.pkl')

# Expected features for liver disease prediction
expected_features = ['age', 'gender', 'tb', 'db', 'alkphos', 'sgpt', 'sgot', 'tp', 'alb', 'a/g']

# Function to make predictions
def make_predictions(input_data):
    try:
        # Parse the input data from JSON
        input_dict = json.loads(input_data)
        
        # Extract the relevant features for liver disease prediction
        input_data_prepared = {feature: input_dict.get(feature) for feature in expected_features}
        
        # Check for any missing features
        missing_features = [feature for feature in expected_features if input_data_prepared[feature] is None]
        if missing_features:
            return json.dumps({'error': f'Missing features: {", ".join(missing_features)}'})
        
        # Prepare the input data as a DataFrame
        df = pd.DataFrame(input_data_prepared, index=[0])
        
        # Make predictions
        predictions = rf_model.predict(df)
        
        # Convert predictions to a JSON string and return
        return json.dumps(predictions.tolist())
    
    except json.JSONDecodeError:
        return json.dumps({'error': 'Invalid JSON format'})
    except Exception as e:
        # Return an error message if an exception occurs
        return json.dumps({'error': str(e)})

# Main function to handle command li
